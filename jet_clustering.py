"""Plotting and export helpers for jet reconstruction."""

import colorsys
import math
from jet_reconstruction import (
    ALGORITHMS,
    ClusteringStep,
    EVENT_X_MAX,
    EVENT_X_MIN,
    GHOST_COUNT,
    GHOST_PT,
    GridCell,
    Jet,
    Particle,
    cluster,
    clustering_histories,
    clustering_steps,
    generate_event,
    generate_grid_event,
)


ALGORITHM_TITLES = {
    "anti-kt": r"anti-$k_t$",
    "kt": r"$k_t$",
    "cambridge-aachen": "Cambridge/Aachen",
}


def _jet_colors(jets):
    """Assign colors by descending pT, preserving finalization order for ties."""
    import matplotlib.pyplot as plt

    ordered = sorted(jets, key=lambda jet: jet.pt, reverse=True)
    if len(ordered) <= 20:
        palette = plt.get_cmap("tab20")
        return {jet: palette(i) for i, jet in enumerate(ordered)}
    return {
        jet: colorsys.hsv_to_rgb(i / len(ordered), 0.75, 0.85)
        for i, jet in enumerate(ordered)
    }


def plot_comparison(particles, radius=0.7, y_scale="log"):
    """Plot all three algorithms with a 'log' or 'linear' pT axis."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if y_scale not in ("log", "linear"):
        raise ValueError("y_scale must be 'log' or 'linear'")

    results = {name: cluster(particles, name, radius) for name in ALGORITHMS}
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, sharey=True,
                             layout="constrained")
    for ax, (algorithm, jets) in zip(axes, results.items()):
        colors = _jet_colors(jets)
        for jet in jets:
            color = colors[jet]
            for index in jet.constituents:
                particle = particles[index]
                if particle.kind == "ghost":
                    ax.scatter(particle.x, 0.025, color=color, marker="|", s=100,
                               transform=ax.get_xaxis_transform(), zorder=5)
                    continue
                ax.vlines(particle.x, 0, particle.pt, color=color, alpha=0.7)
                ax.scatter(particle.x, particle.pt, color=color,
                           marker="o" if particle.kind == "hard" else ".",
                           s=45 if particle.kind == "hard" else 22, zorder=3)
            ax.scatter(jet.x, jet.pt, color=color, marker="D", s=55,
                       edgecolor="black", linewidth=0.6, zorder=4)
        ax.set_title(f"{ALGORITHM_TITLES[algorithm]}: {len(jets)} jets")
        ax.set_ylabel(r"$p_T$ (arbitrary units)")
        ax.set_yscale(y_scale)
        ax.grid(axis="y", alpha=0.2)
        if y_scale == "linear":
            ax.set_ylim(bottom=0)

    axes[-1].set_xlabel("Angular position x (non-periodic)")
    legend = [
        Line2D([], [], color="black", marker="o", linestyle="None", label="Hard particle"),
        Line2D([], [], color="black", marker=".", linestyle="None", label="Soft particle"),
        Line2D([], [], color="black", marker="D", linestyle="None", label="Jet axis and total pT"),
    ]
    if any(p.kind == "ghost" for p in particles):
        legend.append(Line2D([], [], color="black", marker="|", linestyle="None",
                             label="Ghost (colored by jet; position only)"))
    axes[0].legend(handles=legend, loc="upper right")
    fig.suptitle(f"Sequential-recombination jets in one dimension (R = {radius:g})")
    return fig, axes, results


def plot_grid_comparison(cells, grid_size=12, radius=0.7):
    """Compare input objects and final jets on a non-periodic 2D grid."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size <= 0:
        raise ValueError("grid_size must be a positive integer")
    occupied = set()
    for cell in cells:
        if (isinstance(cell.column, bool) or not isinstance(cell.column, int)
                or isinstance(cell.row, bool) or not isinstance(cell.row, int)
                or not (0 <= cell.column < grid_size and 0 <= cell.row < grid_size)):
            raise ValueError("grid cell coordinates must be integer indices within the grid")
        if (cell.column, cell.row) in occupied:
            raise ValueError("grid cells must not overlap")
        occupied.add((cell.column, cell.row))

    width = (EVENT_X_MAX - EVENT_X_MIN) / grid_size
    particles = [
        Particle(EVENT_X_MIN + (cell.column + 0.5) * width, cell.pt, cell.kind,
                 EVENT_X_MIN + (cell.row + 0.5) * width)
        for cell in cells
    ]
    results = {name: cluster(particles, name, radius) for name in ALGORITHMS}
    input_jets = tuple(
        Jet(particle.x, particle.pt, (index,), particle.kind, particle.y)
        for index, particle in enumerate(particles)
    )
    rows = [("Input particles", input_jets), *results.items()]
    fig = plt.figure(figsize=(13, 18), layout="constrained")
    axes = []
    for row, (label, jets) in enumerate(rows):
        plane = fig.add_subplot(4, 2, 2 * row + 1)
        bars = fig.add_subplot(4, 2, 2 * row + 2, projection="3d")
        axes.append((plane, bars))
        jet_colors = _jet_colors(jets)
        for jet in jets:
            color = jet_colors[jet]
            for index in jet.constituents:
                cell = cells[index]
                x = EVENT_X_MIN + cell.column * width
                y = EVENT_X_MIN + cell.row * width
                plane.add_patch(Rectangle((x, y), width, width, facecolor=color,
                                          edgecolor="white", linewidth=0.5))
                bars.bar3d(x, y, 0, width * 0.9, width * 0.9, cell.pt,
                           color=color, shade=False, edgecolor="black", linewidth=0.1)
            plane.scatter(jet.x, jet.y, marker="x", color="black", s=32, zorder=3)
        ticks = [EVENT_X_MIN + i * width for i in range(grid_size + 1)]
        plane.set_xticks(ticks, minor=True)
        plane.set_yticks(ticks, minor=True)
        plane.grid(which="minor", color="lightgray", linewidth=0.4)
        plane.tick_params(which="minor", bottom=False, left=False)
        plane.set(xlim=(EVENT_X_MIN, EVENT_X_MAX), ylim=(EVENT_X_MIN, EVENT_X_MAX),
                  xlabel="x", ylabel="y",
                  title=(f"{label}: {len(jets)} objects" if row == 0
                         else f"{ALGORITHM_TITLES[label]}: {len(jets)} jets"))
        plane.set_aspect("equal")
        bars.set(xlim=(EVENT_X_MIN, EVENT_X_MAX), ylim=(EVENT_X_MIN, EVENT_X_MAX),
                 xlabel="x", ylabel="y", zlabel=r"cell $p_T$",
                 title=(f"{label}: cell pT" if row == 0
                        else f"{ALGORITHM_TITLES[label]}: cell pT"))
        bars.set_zlim(0, max((cell.pt for cell in cells), default=1) * 1.1)
    fig.suptitle("2D grid: input objects and jet-colored cells (crosses = axes)")
    return fig, axes, results


def _draw_step(fig, axes, histories, step_index, color_by_final_jet=True):
    """Draw a shared iteration, holding completed algorithms at their final state."""
    from matplotlib.lines import Line2D

    all_objects = [
        obj for history in histories.values() for obj in history[0].active
    ]
    final_jets = [jet for history in histories.values() for jet in history[-1].final]
    if all_objects:
        xmin = min(obj.x for obj in all_objects)
        xmax = max(obj.x for obj in all_objects)
        padding = max((xmax - xmin) * 0.06, 0.2)
        xlimits = (xmin - padding, xmax + padding)
    else:
        xlimits = (-1, 1)
    ymax = max((jet.pt for jet in final_jets), default=1) * 1.15

    for ax, (algorithm, history) in zip(axes, histories.items()):
        ax.clear()
        current_index = min(step_index, len(history) - 1)
        state = history[current_index]
        jet_colors = _jet_colors(history[-1].final)
        constituent_colors = {
            index: jet_colors[jet]
            for jet in history[-1].final for index in jet.constituents
        }
        unused_indices = {
            obj.constituents[0] for obj in state.active if len(obj.constituents) == 1
        }
        used_particles = tuple(
            obj for obj in history[0].active if obj.constituents[0] not in unused_indices
        )
        active_merged = tuple(obj for obj in state.active if len(obj.constituents) > 1)
        unused_particles = tuple(
            obj for obj in history[0].active if obj.constituents[0] in unused_indices
        )
        for objects, color, marker, annotate, used in (
            (used_particles, "gray", "o", False, True),
            (unused_particles, "tab:blue", "o", True, False),
            (active_merged, "tab:blue", "o", True, False),
            (state.final, None, "D", True, False),
        ):
            for obj in objects:
                if marker == "D":
                    object_color = jet_colors[obj]
                elif color_by_final_jet and not used:
                    object_color = constituent_colors[obj.constituents[0]]
                else:
                    object_color = color
                if obj.kind == "ghost" and marker != "D":
                    ax.scatter(obj.x, 0.025, color=object_color, marker="|", s=100,
                               transform=ax.get_xaxis_transform(), zorder=4)
                    continue
                ax.vlines(obj.x, 0, obj.pt, color=object_color, alpha=0.6)
                ax.scatter(obj.x, obj.pt, color=object_color, marker=marker,
                           s=75 if used else 35,
                           zorder=2 if used else 3)
                if annotate:
                    ax.annotate(str(len(obj.constituents)), (obj.x, obj.pt),
                                xytext=(3, 3), textcoords="offset points", fontsize=7)
        status = "complete" if not state.active else f"{len(state.active)} active"
        ax.set_title(f"{ALGORITHM_TITLES[algorithm]}: step {current_index} "
                     f"({status}, {len(state.final)} final jets)", fontsize=10)
        ax.text(0.01, 0.97, state.action, transform=ax.transAxes,
                va="top", fontsize=8, wrap=True)
        ax.set_xlim(*xlimits)
        ax.set_ylim(0, ymax)
        ax.set_ylabel(r"$p_T$")
        ax.grid(axis="y", alpha=0.2)

    axes[-1].set_xlabel("Angular position x (non-periodic)")
    axes[0].legend(handles=[
        Line2D([], [], color="gray", marker="o", linestyle="None",
               label="Used original particle"),
        Line2D([], [], color="black" if color_by_final_jet else "tab:blue",
               marker="o", linestyle="None",
               label="Active object (color by final jet)" if color_by_final_jet
               else "Active object"),
        Line2D([], [], color="black", marker="D", linestyle="None",
               label="Final jet (color by jet)"),
        *([Line2D([], [], color="black", marker="|", linestyle="None",
                  label="Ghost (position only)")]
          if any(obj.kind == "ghost" for obj in all_objects) else []),
    ], loc="upper right", fontsize=8)
    fig.suptitle(f"Clustering iteration {step_index} (numbers = constituent counts)")


def _step_figure(histories):
    import matplotlib.pyplot as plt

    with plt.ioff():
        return plt.subplots(3, 1, figsize=(12, 9), sharex=True, sharey=True,
                            layout="constrained")


def reconstruction_widget(histories, color_by_final_jet=True):
    """Return an interactive slider and plot for precomputed histories."""
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    from IPython.display import display

    slider = widgets.IntSlider(
        value=0, min=0, max=max(len(history) - 1 for history in histories.values()),
        description="Iteration", continuous_update=False,
        style={"description_width": "initial"},
        layout=widgets.Layout(width="80%"),
    )
    color_toggle = widgets.Checkbox(
        value=color_by_final_jet, description="Color by final jet",
        indent=False, layout=widgets.Layout(width="auto"),
    )
    output = widgets.Output()

    def show_step(change=None):
        fig, axes = _step_figure(histories)
        try:
            _draw_step(fig, axes, histories, slider.value, color_toggle.value)
            with output:
                output.clear_output(wait=True)
                display(fig)
        finally:
            plt.close(fig)

    slider.observe(show_step, names="value")
    color_toggle.observe(show_step, names="value")
    show_step()
    return widgets.VBox([slider, color_toggle, output])


def _save_reconstruction(histories, filename, writer, dpi, color_by_final_jet):
    import matplotlib.pyplot as plt

    fig, axes = _step_figure(histories)
    try:
        with writer.saving(fig, str(filename), dpi=dpi):
            for step_index in range(max(len(history) for history in histories.values())):
                _draw_step(fig, axes, histories, step_index, color_by_final_jet)
                writer.grab_frame()
    finally:
        plt.close(fig)


def save_reconstruction_gif(histories, filename, fps=2, dpi=80, color_by_final_jet=True):
    """Save every shared iteration to a GIF, including the initial state."""
    from matplotlib.animation import PillowWriter

    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    _save_reconstruction(histories, filename, PillowWriter(fps=fps), dpi,
                         color_by_final_jet)


def save_reconstruction_mp4(histories, filename, fps=2, dpi=80, color_by_final_jet=True):
    """Save each iteration as one MP4 frame; fps sets the playback speed."""
    from matplotlib.animation import FFMpegWriter

    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if not FFMpegWriter.isAvailable():
        raise RuntimeError("MP4 export requires ffmpeg; install it and make it available on PATH")
    writer = FFMpegWriter(
        fps=fps, codec="libx264",
        extra_args=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p"],
    )
    _save_reconstruction(histories, filename, writer, dpi, color_by_final_jet)
