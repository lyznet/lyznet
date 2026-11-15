"""Dependency-free option records for neural Lyapunov learning."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PINNCorrectionStage:
    """One explicit correction phase for a published PINN recipe."""

    steps: int
    learning_rate: float
    data_weight: float = 1.0
    pde_weight: float = 1.0
    decrease_weight: float = 0.0
    boundary_weight: float = 0.0
    boundary_lower_bound: float | None = None
    condition_source: str = "uniform"
    condition_target_bounds: tuple[float, float] | None = None
    condition_seed: int | None = None
    boundary_seed: int | None = None
    data_batch_size: int | None = None

    def __post_init__(self):
        if self.steps < 1:
            raise ValueError("Correction-stage steps must be positive.")
        if self.learning_rate <= 0:
            raise ValueError(
                "Correction-stage learning_rate must be positive."
            )
        if self.condition_source not in ("uniform", "target_data"):
            raise ValueError(
                "condition_source must be 'uniform' or 'target_data'."
            )
        if (
            self.condition_source == "target_data"
            and self.condition_target_bounds is None
        ):
            raise ValueError(
                "target_data stages require condition_target_bounds."
            )


__all__ = ["PINNCorrectionStage"]
