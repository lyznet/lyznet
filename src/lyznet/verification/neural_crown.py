"""CROWN adapter for one proposed neural Lyapunov funnel."""

from dataclasses import dataclass
import math
from time import perf_counter

import numpy as np
import sympy as sp

from lyznet.neural import (
    CenteredNeuralLyapunov,
    QuadraticAnchoredNeuralLyapunov,
)
from lyznet.constraints import (
    normalise_state_constraints,
    normalise_viable_faces,
)
from lyznet.sets import RestrictedSublevelSet, SublevelSet

from ._crown import (
    CrownRunner,
    expression_graph,
    load_crown,
    normalise_device,
    require_supported_expressions,
)
from .core.neural_lyapunov_crown import (
    anchored_annular_decrease,
    anchored_boundary_viability,
    anchored_bootstrap_containment,
    anchored_boundary_separation,
    anchored_constrained_annular_decrease,
    anchored_constrained_boundary_separation,
    anchored_constrained_boundary_viability,
    anchored_equilibrium_membership,
    anchored_state_constraint_viability,
    annular_decrease,
    boundary_viability,
    boundary_separation,
    constrained_annular_decrease,
    constrained_boundary_separation,
    constrained_boundary_viability,
    equilibrium_membership,
    quadratic_bootstrap_containment,
    state_constraint_viability,
)
from .quadratic import (
    VerificationClaim,
    VerificationStatus,
)


@dataclass(frozen=True)
class NeuralVerificationResult:
    """Formal evidence for a neural reach-and-stay set."""

    system: object
    certificate: object
    claim: VerificationClaim
    status: VerificationStatus
    local_set: object | None
    verified_set: object | None
    obligations: tuple
    backend: str
    elapsed_time: float
    assessment: object | None
    state_constraints: tuple
    viable_faces: tuple

    @property
    def verified_level(self):
        return None if self.verified_set is None else self.verified_set.level

    @property
    def is_verified(self):
        return self.status is VerificationStatus.VERIFIED


def _failed_status(obligations):
    if any(
        item.status is VerificationStatus.UNKNOWN for item in obligations
    ):
        return VerificationStatus.UNKNOWN
    return VerificationStatus.COUNTEREXAMPLE


def _anchored_latent_and_numerator(torch, candidate, states, dynamics=None):
    displacement = states - candidate.centre
    quadratic = states[:, :1] * 0
    for row in range(candidate.state_dimension):
        for column in range(candidate.state_dimension):
            quadratic = quadratic + (
                candidate.matrix[row, column]
                * displacement[:, row:row + 1]
                * displacement[:, column:column + 1]
            )
    neural, neural_gradient = candidate.model.value_and_gradient(states)
    correction = neural - candidate.centre_output
    latent = candidate.scale * quadratic * torch.exp(correction)
    if dynamics is None:
        return latent, None
    quadratic_gradient = 2 * displacement @ candidate.matrix
    numerator = torch.sum(
        (quadratic_gradient + quadratic * neural_gradient) * dynamics,
        dim=1,
        keepdim=True,
    )
    return latent, numerator


def _candidate_graph(torch, candidate):
    class CandidateGraph(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.candidate = candidate

        def forward(self, states):
            if isinstance(
                self.candidate, QuadraticAnchoredNeuralLyapunov
            ):
                value, _ = _anchored_latent_and_numerator(
                    torch, self.candidate, states
                )
                return value
            return self.candidate(states)

    return CandidateGraph()


def _containment_graph(torch, candidate, quadratic):
    class ContainmentGraph(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.candidate = candidate
            self.register_buffer(
                "matrix",
                torch.tensor(np.asarray(quadratic.matrix, dtype=float)),
            )
            self.register_buffer(
                "centre",
                torch.tensor(np.asarray(quadratic.centre, dtype=float)),
            )

        def forward(self, states):
            value = self.candidate(states)
            displacement = states - self.centre
            quadratic_value = torch.sum(
                (displacement @ self.matrix) * displacement,
                dim=1,
                keepdim=True,
            )
            return torch.cat((value, quadratic_value), dim=1)

    return ContainmentGraph()


def _correction_graph(torch, candidate):
    class CorrectionGraph(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = candidate.model
            self.register_buffer(
                "centre_output", candidate.centre_output.clone()
            )

        def forward(self, states):
            return self.model(states) - self.centre_output

    return CorrectionGraph()


def _decrease_graph(torch, candidate, dynamics):
    class DecreaseGraph(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.candidate = candidate
            self.dynamics = dynamics

        def forward(self, states):
            vector_field = self.dynamics(states)
            if isinstance(
                self.candidate, QuadraticAnchoredNeuralLyapunov
            ):
                value, derivative = _anchored_latent_and_numerator(
                    torch, self.candidate, states, vector_field
                )
            else:
                value, gradient = self.candidate.value_and_gradient(states)
                derivative = torch.sum(
                    gradient * vector_field, dim=1, keepdim=True
                )
            return torch.cat((value, derivative), dim=1)

    return DecreaseGraph()


def _concatenated_graph(torch, *graphs):
    class ConcatenatedGraph(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.graphs = torch.nn.ModuleList(graphs)

        def forward(self, states):
            return torch.cat(
                tuple(graph(states) for graph in self.graphs), dim=1
            )

    return ConcatenatedGraph()


def _face_graph(torch, candidate, dynamics, coordinate, outward_sign):
    class FaceGraph(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.candidate = candidate
            self.dynamics = dynamics

        def forward(self, states):
            if isinstance(
                self.candidate, QuadraticAnchoredNeuralLyapunov
            ):
                value, _ = _anchored_latent_and_numerator(
                    torch, self.candidate, states
                )
            else:
                value = self.candidate(states)
            outward_rate = (
                outward_sign
                * self.dynamics(states)[:, coordinate:coordinate + 1]
            )
            return torch.cat((value, outward_rate), dim=1)

    return FaceGraph()


def _result(
    system,
    candidate,
    status,
    obligations,
    started,
    *,
    local_level=None,
    outer_level=None,
    assessment=None,
    state_constraints=(),
    viable_faces=(),
):
    restricted = bool(state_constraints or viable_faces)

    def make_set(level):
        if level is None:
            return None
        if restricted:
            return RestrictedSublevelSet(
                candidate,
                level,
                system.domain,
                system.states,
                state_constraints,
            )
        return SublevelSet(candidate, level)

    local_set = make_set(local_level)
    return NeuralVerificationResult(
        system=system,
        certificate=candidate,
        claim=VerificationClaim.STABLE_EQUILIBRIUM_ROA,
        status=status,
        local_set=local_set,
        verified_set=(
            make_set(outer_level)
            if status is VerificationStatus.VERIFIED
            else None
        ),
        obligations=tuple(obligations),
        backend="crown",
        elapsed_time=perf_counter() - started,
        assessment=assessment,
        state_constraints=tuple(state_constraints),
        viable_faces=tuple(viable_faces),
    )


def verify_neural_lyapunov_crown(
    system,
    candidate,
    bootstrap,
    *,
    local_level,
    outer_level,
    assessment=None,
    state_constraints=(),
    viable_faces=(),
    state_constraint_band=1e-3,
    state_constraint_viability_margin=1e-5,
    face_viability_margin=1e-5,
    timeout=None,
    device="auto",
    batch_size=64,
    proof_margin=1e-6,
    verbose=False,
):
    """Prove one neural funnel from a verified quadratic inner set."""

    if bootstrap.status is not VerificationStatus.VERIFIED:
        raise ValueError("bootstrap must be a formally verified result.")
    if bootstrap.claim not in (
        VerificationClaim.LYAPUNOV,
        VerificationClaim.STABLE_EQUILIBRIUM_ROA,
    ):
        raise ValueError("bootstrap has the wrong formal claim.")
    if type(candidate) not in (
        CenteredNeuralLyapunov,
        QuadraticAnchoredNeuralLyapunov,
    ):
        raise TypeError("candidate is not a supported neural architecture.")
    state_constraints = normalise_state_constraints(
        system, state_constraints
    )
    viable_faces = normalise_viable_faces(system, viable_faces)
    for value, label in (
        (state_constraint_band, "state_constraint_band"),
        (
            state_constraint_viability_margin,
            "state_constraint_viability_margin",
        ),
        (face_viability_margin, "face_viability_margin"),
    ):
        if not np.isfinite(float(value)) or float(value) <= 0:
            raise ValueError("{} must be positive and finite.".format(label))
    local_level = float(local_level)
    outer_level = float(outer_level)
    if not 0 < local_level <= outer_level:
        raise ValueError("Require 0 < local_level <= outer_level.")
    if proof_margin <= 0 or proof_margin >= local_level:
        raise ValueError("proof_margin must lie strictly between 0 and c1.")
    anchored = isinstance(candidate, QuadraticAnchoredNeuralLyapunov)
    if anchored and outer_level + proof_margin >= 1:
        raise ValueError("Anchored neural levels plus margin must be below one.")
    lie_derivatives = tuple(
        sp.expand(sum(
            sp.diff(constraint, state) * dynamics
            for state, dynamics in zip(
                system.states, system.symbolic.dynamics
            )
        ))
        for constraint in state_constraints
    )
    require_supported_expressions(
        system.states,
        tuple(system.symbolic.dynamics)
        + state_constraints
        + lie_derivatives,
    )
    fingerprint = candidate.proof_fingerprint()
    started = perf_counter()
    torch, api = load_crown()
    crown_device = normalise_device(torch, device)
    if verbose:
        print("Neural verification: CROWN on {}.".format(crown_device))
    runner = CrownRunner(
        torch=torch,
        api=api,
        device=crown_device,
        timeout=timeout,
        batch_size=batch_size,
        verbose=verbose,
    )
    bounds = system.domain.bounds
    candidate_graph = _candidate_graph(torch, candidate)
    if anchored:
        inner_lower_latent = math.atanh(local_level - proof_margin)
        inner_upper_latent = math.atanh(local_level + proof_margin)
        outer_upper_latent = math.atanh(outer_level + proof_margin)

        def membership_condition(output):
            return anchored_equilibrium_membership(
                output, inner_lower_latent=inner_lower_latent
            )
    else:
        def membership_condition(output):
            return equilibrium_membership(
                output,
                local_level=local_level,
                proof_margin=proof_margin,
            )

    equilibrium_bounds = tuple(
        (float(value), float(value)) for value in system.equilibrium
    )
    membership = runner.solve(
        "neural_equilibrium_membership",
        candidate_graph,
        1,
        membership_condition,
        equilibrium_bounds,
    )
    obligations = [membership]
    if membership.status is not VerificationStatus.VERIFIED:
        return _result(
            system, candidate, _failed_status(obligations), obligations,
            started,
            assessment=assessment,
            state_constraints=state_constraints,
            viable_faces=viable_faces,
        )

    quadratic_set = bootstrap.verified_set
    if anchored:
        minimum_correction = math.log(
            inner_upper_latent
            / (
                float(candidate.scale.detach().cpu())
                * (quadratic_set.level - proof_margin)
            )
        )
        containment_graph = _correction_graph(torch, candidate)

        def containment_condition(output):
            return anchored_bootstrap_containment(
                output, minimum_correction=minimum_correction
            )
        containment_dimension = 1
    else:
        containment_graph = _containment_graph(
            torch, candidate, quadratic_set.certificate
        )

        def containment_condition(output):
            return quadratic_bootstrap_containment(
                output,
                local_level=local_level,
                quadratic_level=quadratic_set.level,
                proof_margin=proof_margin,
            )
        containment_dimension = 2

    containment = runner.solve(
        "neural_inner_inside_quadratic_bootstrap",
        containment_graph,
        containment_dimension,
        containment_condition,
        bounds,
    )
    obligations.append(containment)
    if containment.status is not VerificationStatus.VERIFIED:
        return _result(
            system, candidate, _failed_status(obligations), obligations,
            started,
            assessment=assessment,
            state_constraints=state_constraints,
            viable_faces=viable_faces,
        )
    if outer_level == local_level and not state_constraints:
        result = _result(
            system,
            candidate,
            VerificationStatus.VERIFIED,
            obligations,
            started,
            local_level=local_level,
            outer_level=outer_level,
            assessment=assessment,
            state_constraints=state_constraints,
            viable_faces=viable_faces,
        )
        if candidate.proof_fingerprint() != fingerprint:
            raise RuntimeError("The candidate changed during verification.")
        return result

    dynamics = expression_graph(
        torch, system.states, tuple(system.symbolic.dynamics)
    )
    constraint_graph = (
        expression_graph(torch, system.states, state_constraints)
        if state_constraints
        else None
    )
    decrease_graph = _decrease_graph(torch, candidate, dynamics)
    if constraint_graph is not None:
        decrease_graph = _concatenated_graph(
            torch, decrease_graph, constraint_graph
        )
    if anchored:
        def decrease_condition(output):
            if state_constraints:
                return anchored_constrained_annular_decrease(
                    output,
                    inner_lower_latent=inner_lower_latent,
                    outer_upper_latent=outer_upper_latent,
                    proof_margin=proof_margin,
                    constraint_count=len(state_constraints),
                )
            return anchored_annular_decrease(
                output,
                inner_lower_latent=inner_lower_latent,
                outer_upper_latent=outer_upper_latent,
                proof_margin=proof_margin,
            )
    else:
        def decrease_condition(output):
            if state_constraints:
                return constrained_annular_decrease(
                    output,
                    local_level=local_level,
                    outer_level=outer_level,
                    proof_margin=proof_margin,
                    constraint_count=len(state_constraints),
                )
            return annular_decrease(
                output,
                local_level=local_level,
                outer_level=outer_level,
                proof_margin=proof_margin,
            )
    if outer_level > local_level:
        decrease = runner.solve(
            "neural_constrained_annular_decrease"
            if state_constraints
            else "neural_annular_decrease",
            decrease_graph,
            2 + len(state_constraints),
            decrease_condition,
            bounds,
        )
        obligations.append(decrease)
        if decrease.status is not VerificationStatus.VERIFIED:
            return _result(
                system,
                candidate,
                _failed_status(obligations),
                obligations,
                started,
                local_level=local_level,
                assessment=assessment,
                state_constraints=state_constraints,
                viable_faces=viable_faces,
            )

    for constraint_index, lie_derivative in enumerate(lie_derivatives):
        lie_graph = expression_graph(
            torch, system.states, (lie_derivative,)
        )
        active_boundary_graph = _concatenated_graph(
            torch, candidate_graph, constraint_graph, lie_graph
        )

        def constraint_condition(output, index=constraint_index):
            if anchored:
                return anchored_state_constraint_viability(
                    output,
                    outer_upper_latent=outer_upper_latent,
                    constraint_index=index,
                    constraint_count=len(state_constraints),
                    boundary_band=state_constraint_band,
                    viability_margin=state_constraint_viability_margin,
                )
            return state_constraint_viability(
                output,
                outer_level=outer_level,
                proof_margin=proof_margin,
                constraint_index=index,
                constraint_count=len(state_constraints),
                boundary_band=state_constraint_band,
                viability_margin=state_constraint_viability_margin,
            )

        constraint_proof = runner.solve(
            "neural_state_constraint_{}_viability".format(
                constraint_index + 1
            ),
            active_boundary_graph,
            2 + len(state_constraints),
            constraint_condition,
            bounds,
        )
        obligations.append(constraint_proof)
        if constraint_proof.status is not VerificationStatus.VERIFIED:
            return _result(
                system,
                candidate,
                _failed_status(obligations),
                obligations,
                started,
                local_level=local_level,
                assessment=assessment,
                state_constraints=state_constraints,
                viable_faces=viable_faces,
            )

    if outer_level == local_level:
        result = _result(
            system,
            candidate,
            VerificationStatus.VERIFIED,
            obligations,
            started,
            local_level=local_level,
            outer_level=outer_level,
            assessment=assessment,
            state_constraints=state_constraints,
            viable_faces=viable_faces,
        )
        if candidate.proof_fingerprint() != fingerprint:
            raise RuntimeError("The candidate changed during verification.")
        return result

    viable_face_set = set(viable_faces)
    for coordinate, (lower, upper) in enumerate(bounds):
        for side_name, side in (("lower", lower), ("upper", upper)):
            face = list(bounds)
            face[coordinate] = (side, side)
            if (coordinate, side_name) in viable_face_set:
                outward_sign = -1.0 if side_name == "lower" else 1.0
                face_graph = _face_graph(
                    torch,
                    candidate,
                    dynamics,
                    coordinate,
                    outward_sign,
                )
                if constraint_graph is not None:
                    face_graph = _concatenated_graph(
                        torch, face_graph, constraint_graph
                    )

                def face_condition(output):
                    if anchored and state_constraints:
                        return anchored_constrained_boundary_viability(
                            output,
                            outer_upper_latent=outer_upper_latent,
                            viability_margin=face_viability_margin,
                            constraint_count=len(state_constraints),
                        )
                    if anchored:
                        return anchored_boundary_viability(
                            output,
                            outer_upper_latent=outer_upper_latent,
                            viability_margin=face_viability_margin,
                        )
                    if state_constraints:
                        return constrained_boundary_viability(
                            output,
                            outer_level=outer_level,
                            proof_margin=proof_margin,
                            viability_margin=face_viability_margin,
                            constraint_count=len(state_constraints),
                        )
                    return boundary_viability(
                        output,
                        outer_level=outer_level,
                        proof_margin=proof_margin,
                        viability_margin=face_viability_margin,
                    )

                output_dimension = 2 + len(state_constraints)
            else:
                face_graph = candidate_graph
                if constraint_graph is not None:
                    face_graph = _concatenated_graph(
                        torch, candidate_graph, constraint_graph
                    )

                def face_condition(output):
                    if anchored and state_constraints:
                        return anchored_constrained_boundary_separation(
                            output,
                            outer_upper_latent=outer_upper_latent,
                            constraint_count=len(state_constraints),
                        )
                    if anchored:
                        return anchored_boundary_separation(
                            output, outer_upper_latent=outer_upper_latent
                        )
                    if state_constraints:
                        return constrained_boundary_separation(
                            output,
                            outer_level=outer_level,
                            proof_margin=proof_margin,
                            constraint_count=len(state_constraints),
                        )
                    return boundary_separation(
                        output,
                        outer_level=outer_level,
                        proof_margin=proof_margin,
                    )
                output_dimension = 1 + len(state_constraints)
            boundary = runner.solve(
                "neural_boundary_x{}_{}".format(
                    coordinate + 1, side_name
                ),
                face_graph,
                output_dimension,
                face_condition,
                tuple(face),
            )
            obligations.append(boundary)
            if boundary.status is not VerificationStatus.VERIFIED:
                return _result(
                    system,
                    candidate,
                    _failed_status(obligations),
                    obligations,
                    started,
                    local_level=local_level,
                    assessment=assessment,
                    state_constraints=state_constraints,
                    viable_faces=viable_faces,
                )

    result = _result(
        system,
        candidate,
        VerificationStatus.VERIFIED,
        obligations,
        started,
        local_level=local_level,
        outer_level=outer_level,
        assessment=assessment,
        state_constraints=state_constraints,
        viable_faces=viable_faces,
    )
    if candidate.proof_fingerprint() != fingerprint:
        raise RuntimeError("The candidate changed during verification.")
    return result


__all__ = ["NeuralVerificationResult", "verify_neural_lyapunov_crown"]
