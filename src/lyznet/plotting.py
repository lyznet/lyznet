"""Compact plotting for two-dimensional Lyapunov workflow results."""

from pathlib import Path

import numpy as np
import sympy as sp

from .sets import RestrictedSublevelSet, SublevelSet
from .systems import ControlAffineSystem


def plot_lyapunov_result(
    result,
    *,
    surface=False,
    phase_portrait=True,
    show=False,
    save=None,
    resolution=300,
):
    """Plot a two-dimensional candidate and its formally verified levels."""

    if result.system.state_dimension != 2 or result.system.domain is None:
        raise ValueError("Plotting currently requires a bounded 2D system.")
    if not isinstance(resolution, int) or isinstance(resolution, bool):
        raise TypeError("resolution must be an integer.")
    if resolution < 50:
        raise ValueError("resolution must be at least 50.")
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as error:
        raise ImportError(
            "Plotting requires `pip install -e '.[plot]'`."
        ) from error

    bounds = result.system.domain.bounds
    first = np.linspace(bounds[0][0], bounds[0][1], resolution)
    second = np.linspace(bounds[1][0], bounds[1][1], resolution)
    x1, x2 = np.meshgrid(first, second)
    points = np.column_stack((x1.ravel(), x2.ravel()))
    values = result.candidate.value(points).reshape(x1.shape)
    state_constraints = tuple(
        getattr(
            result.verification,
            "state_constraints",
            getattr(result, "state_constraints", ()),
        )
        if result.verification is not None
        else getattr(result, "state_constraints", ())
    )
    constraint_maximum = None
    if state_constraints:
        constraint_values = []
        for expression in state_constraints:
            function = sp.lambdify(
                result.system.states, expression, modules="numpy"
            )
            evaluated = np.asarray(function(x1, x2), dtype=float)
            if evaluated.ndim == 0:
                evaluated = np.full_like(x1, evaluated.item())
            constraint_values.append(evaluated)
        constraint_maximum = np.maximum.reduce(constraint_values)
        plotted_values = np.ma.masked_where(
            constraint_maximum > 1, values
        )
    else:
        plotted_values = values
    style = {
        "font.size": 15,
        "axes.labelsize": 17,
        "axes.titlesize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
    }
    with mpl.rc_context(style):
        if surface:
            figure = plt.figure(figsize=(12, 5))
            surface_axis = figure.add_subplot(1, 2, 1, projection="3d")
            axis = figure.add_subplot(1, 2, 2)
            surface_axis.plot_surface(
                x1, x2, values, cmap="viridis", linewidth=0
            )
            surface_axis.set(
                xlabel=r"$x_1$",
                ylabel=r"$x_2$",
                zlabel=r"$V(x)$",
                title="Lyapunov function",
            )
        else:
            figure, axis = plt.subplots(figsize=(6.5, 5.5))

        axis.set_facecolor("0.82")
        filled = axis.contourf(
            x1, x2, plotted_values, levels=30, cmap="viridis", alpha=0.78
        )
        figure.colorbar(filled, ax=axis, label=r"$V(x)$")
        if phase_portrait:
            dynamics = _plot_dynamics(result, points)
            axis.streamplot(
                first,
                second,
                dynamics[:, 0].reshape(x1.shape),
                dynamics[:, 1].reshape(x2.shape),
                color="0.35",
                density=0.9,
                linewidth=0.7,
                arrowsize=0.8,
            )
        local = (
            None
            if result.verification is None
            else getattr(result.verification, "local_set", None)
        )
        verified = result.verified_set
        legend_handles = []
        set_types = (SublevelSet, RestrictedSublevelSet)
        finite_values = plotted_values.compressed() if np.ma.isMaskedArray(
            plotted_values
        ) else plotted_values.ravel()
        minimum_value = float(np.min(finite_values))
        maximum_value = float(np.max(finite_values))
        if isinstance(local, set_types) and (
            not isinstance(verified, set_types)
            or local.level != verified.level
        ) and minimum_value <= local.level <= maximum_value:
            axis.contour(
                x1,
                x2,
                plotted_values,
                levels=[local.level],
                colors=["white"],
                linestyles=["--"],
                linewidths=[1.8],
            )
            legend_handles.append(Line2D(
                [0],
                [0],
                color="white",
                linestyle="--",
                linewidth=1.8,
                label="Local",
            ))
        if (
            isinstance(verified, set_types)
            and minimum_value <= verified.level <= maximum_value
        ):
            axis.contour(
                x1,
                x2,
                plotted_values,
                levels=[verified.level],
                colors=["#c62828"],
                linewidths=[2.6],
            )
            legend_handles.append(Line2D(
                [0],
                [0],
                color="#c62828",
                linewidth=2.6,
                label="Verified",
            ))
        if constraint_maximum is not None:
            axis.contour(
                x1,
                x2,
                constraint_maximum,
                levels=[1],
                colors=["black"],
                linewidths=[2.0],
            )
            legend_handles.append(Line2D(
                [0], [0], color="black", linewidth=2.0, label="Constraint"
            ))
        viable_faces = tuple(
            getattr(
                result.verification,
                "viable_faces",
                getattr(result, "viable_faces", ()),
            )
            if result.verification is not None
            else getattr(result, "viable_faces", ())
        )
        if isinstance(verified, RestrictedSublevelSet) and viable_faces:
            _plot_viable_faces(
                axis,
                result,
                viable_faces,
                resolution=max(400, resolution),
            )
            legend_handles.append(Line2D(
                [0],
                [0],
                color="#1565c0",
                linewidth=3.0,
                label="Viable boundary",
            ))
        if legend_handles:
            axis.legend(handles=legend_handles, loc="best")
        axis.set(
            xlabel=r"$x_1$",
            ylabel=r"$x_2$",
            title="Verified level set",
            xlim=bounds[0],
            ylim=bounds[1],
        )
        figure.tight_layout()
        if save is not None:
            path = Path(save).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, bbox_inches="tight")
        if show:
            plt.show()
    return figure


def _plot_dynamics(result, points):
    system = result.system
    if not isinstance(system, ControlAffineSystem):
        return system.numpy.dynamics(points)
    deviations = (
        points - np.asarray(result.candidate.centre, dtype=float)
    ) @ result.candidate.gain.T
    equilibrium_input = np.asarray(
        tuple(map(float, system.equilibrium_input))
    )
    controls = deviations + equilibrium_input
    return system.numpy.dynamics(points, controls)


def _plot_viable_faces(axis, result, viable_faces, resolution):
    bounds = result.system.domain.bounds
    verified = result.verified_set
    for coordinate, side_name in viable_faces:
        other = 1 - coordinate
        varying = np.linspace(
            bounds[other][0], bounds[other][1], resolution
        )
        fixed = bounds[coordinate][0 if side_name == "lower" else 1]
        points = np.empty((resolution, 2))
        points[:, coordinate] = fixed
        points[:, other] = varying
        mask = verified.contains(points)
        horizontal = points[:, 0]
        vertical = points[:, 1]
        axis.plot(
            np.ma.masked_where(~mask, horizontal),
            np.ma.masked_where(~mask, vertical),
            color="#1565c0",
            linewidth=3.0,
        )


__all__ = ["plot_lyapunov_result"]
