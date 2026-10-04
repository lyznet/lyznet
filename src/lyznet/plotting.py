"""Compact plotting for two-dimensional Lyapunov workflow results."""

from pathlib import Path

import numpy as np

from .sets import SublevelSet
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

        filled = axis.contourf(
            x1, x2, values, levels=30, cmap="viridis", alpha=0.78
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
        if isinstance(local, SublevelSet) and (
            not isinstance(verified, SublevelSet)
            or local.level != verified.level
        ):
            axis.contour(
                x1,
                x2,
                values,
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
        if isinstance(verified, SublevelSet):
            axis.contour(
                x1,
                x2,
                values,
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


__all__ = ["plot_lyapunov_result"]
