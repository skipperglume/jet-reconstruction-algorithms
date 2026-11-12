import math
import json
import shutil
import subprocess
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from jet_clustering import (
    ALGORITHMS, EVENT_X_MAX, EVENT_X_MIN, GHOST_COUNT, GHOST_PT, GridCell, Particle,
    _draw_step, _step_figure, cluster, clustering_histories, clustering_steps,
    generate_event, generate_grid_event, plot_comparison, plot_grid_comparison,
    reconstruction_widget, save_reconstruction_gif,
    save_reconstruction_mp4,
)


class EventTests(unittest.TestCase):
    def test_counts_scales_and_reproducibility(self):
        particles = generate_event(n_hard=3, n_soft=12, seed=123)
        self.assertEqual(particles, generate_event(n_hard=3, n_soft=12, seed=123))
        self.assertEqual(sum(p.kind == "hard" for p in particles), 3)
        self.assertEqual(sum(p.kind == "soft" for p in particles), 12)
        self.assertTrue(all(15 <= p.pt <= 30 for p in particles if p.kind == "hard"))
        self.assertTrue(all(1 <= p.pt <= 4 for p in particles if p.kind == "soft"))
        self.assertTrue(all(-3 <= p.x <= 3 for p in particles))

    def test_invalid_counts(self):
        for count in (-1, 1.5, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                generate_event(n_hard=count)

    def test_optional_uniform_ghosts_preserve_event(self):
        baseline = generate_event(n_hard=3, n_soft=12, seed=123)
        event = generate_event(n_hard=3, n_soft=12, seed=123, include_ghosts=True)
        self.assertEqual(event[:len(baseline)], baseline)
        ghosts = event[len(baseline):]
        self.assertEqual(len(ghosts), GHOST_COUNT)
        spacing = (EVENT_X_MAX - EVENT_X_MIN) / GHOST_COUNT
        for i, particle in enumerate(ghosts):
            self.assertEqual(particle.kind, "ghost")
            self.assertEqual(particle.pt, GHOST_PT)
            self.assertAlmostEqual(particle.x, EVENT_X_MIN + (i + 0.5) * spacing)
        self.assertLess(max(p.pt for p in ghosts), min(p.pt for p in baseline))
        with self.assertRaises(ValueError):
            generate_event(include_ghosts=1)


class ClusteringTests(unittest.TestCase):
    def test_empty_and_separated_particles(self):
        particles = [Particle(0, 10, "hard"), Particle(2, 1, "soft")]
        for algorithm in ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                self.assertEqual(cluster([], algorithm), [])
                jets = cluster(particles, algorithm, radius=1)
                self.assertEqual([jet.constituents for jet in jets], [(0,), (1,)])

    def test_merge_and_conservation(self):
        particles = [Particle(0, 10, "hard"), Particle(0.5, 2, "soft"),
                     Particle(3, 1, "soft")]
        for algorithm in ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                jets = cluster(particles, algorithm, radius=1)
                self.assertEqual(len(jets), 2)
                self.assertEqual(jets[0].pt, 12)
                self.assertAlmostEqual(jets[0].x, 1 / 12)
                self.assertEqual(sorted(i for j in jets for i in j.constituents), [0, 1, 2])
                self.assertEqual(sum(j.pt for j in jets), sum(p.pt for p in particles))

    def test_exact_radius_tie_goes_to_beam(self):
        particles = [Particle(0, 10, "hard"), Particle(1, 1, "soft")]
        for algorithm in ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                self.assertEqual(len(cluster(particles, algorithm, radius=1)), 2)

    def test_algorithms_can_produce_different_jets(self):
        particles = [Particle(0.3, 5, "soft"), Particle(1.9, 1, "soft"),
                     Particle(0.8, 30, "hard"), Particle(1.6, 1, "soft")]
        expected = {
            "anti-kt": [(0, 2, 3), (1,)],
            "kt": [(0, 1, 2, 3)],
            "cambridge-aachen": [(0, 2), (1, 3)],
        }
        for algorithm, groups in expected.items():
            with self.subTest(algorithm=algorithm):
                jets = cluster(particles, algorithm, radius=1)
                self.assertEqual(sorted(tuple(sorted(j.constituents)) for j in jets), groups)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            cluster([], "unknown")
        for radius in (0, -1, math.inf, math.nan):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                cluster([], radius=radius)
        for particle in (Particle(0, 0, "soft"), Particle(math.inf, 1, "hard"),
                         Particle(0, math.nan, "soft"),
                         Particle(0, 1, "hard", math.inf)):
            with self.subTest(particle=particle), self.assertRaises(ValueError):
                cluster([particle])


class GridTests(unittest.TestCase):
    def test_generation_counts_positions_and_validation(self):
        cells = generate_grid_event(4, 20, grid_size=6, seed=10)
        self.assertEqual(cells, generate_grid_event(4, 20, grid_size=6, seed=10))
        self.assertEqual(len({(cell.column, cell.row) for cell in cells}), 24)
        self.assertEqual(sum(cell.kind == "hard" for cell in cells), 4)
        self.assertTrue(all(0 <= cell.column < 6 and 0 <= cell.row < 6 for cell in cells))
        self.assertTrue(all(15 <= cell.pt <= 30 for cell in cells if cell.kind == "hard"))
        self.assertTrue(all(1 <= cell.pt <= 4 for cell in cells if cell.kind == "soft"))
        for kwargs in ({"grid_size": 0}, {"grid_size": True}, {"grid_size": 2, "n_soft": 5},
                       {"n_hard": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                generate_grid_event(**kwargs)

    def test_clustering_uses_both_coordinates_and_weighted_centroid(self):
        particles = [Particle(0, 10, "hard", 0), Particle(0.3, 2, "soft", 0.4),
                     Particle(0, 1, "soft", 3)]
        for algorithm in ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                jets = cluster(particles, algorithm, radius=1)
                self.assertEqual(len(jets), 2)
                self.assertAlmostEqual(jets[0].x, 0.05)
                self.assertAlmostEqual(jets[0].y, 0.4 / 6)
                self.assertEqual(jets[0].pt, 12)
                self.assertEqual(sorted(jets[0].constituents), [0, 1])
                self.assertEqual(jets[1].constituents, (2,))

    def test_grid_plot_jet_colors_and_bar_heights(self):
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_rgba
        from mpl_toolkits.mplot3d import Axes3D
        from jet_clustering import _jet_colors

        cells = [GridCell(0, 0, 12, "hard"), GridCell(1, 0, 2, "soft"),
                 GridCell(3, 3, 3, "soft")]
        calls = []
        original_bar3d = Axes3D.bar3d

        def capture_bar(ax, x, y, z, dx, dy, dz, *args, **kwargs):
            calls.append((x, y, dz, to_rgba(kwargs["color"])))
            return original_bar3d(ax, x, y, z, dx, dy, dz, *args, **kwargs)

        with patch.object(Axes3D, "bar3d", new=capture_bar):
            fig, axes, results = plot_grid_comparison(cells, grid_size=4, radius=2)
        try:
            fig.canvas.draw()
            self.assertEqual(len(axes), 4)
            self.assertEqual(len(calls), 4 * len(cells))
            self.assertEqual(len(axes[0][0].patches), len(cells))
            self.assertEqual(axes[0][0].get_title(), "Input particles: 3 objects")
            for i, ((plane, bars), jets) in enumerate(zip(axes[1:], results.values()), 1):
                colors = _jet_colors(jets)
                expected = [(index, to_rgba(colors[jet])) for jet in jets
                            for index in jet.constituents]
                self.assertEqual(len(plane.patches), len(cells))
                self.assertEqual(bars.name, "3d")
                for patch_obj, call, (index, color) in zip(
                    plane.patches, calls[i * len(cells):(i + 1) * len(cells)], expected
                ):
                    cell = cells[index]
                    self.assertEqual(patch_obj.get_facecolor(), color)
                    self.assertEqual(call[2], cell.pt)
                    self.assertEqual(call[3], color)
                    self.assertEqual((patch_obj.get_x(), patch_obj.get_y()), call[:2])
            self.assertEqual(len({i for jet in results["anti-kt"] for i in jet.constituents}),
                             len(cells))
        finally:
            plt.close(fig)

    def test_empty_and_invalid_grid(self):
        import matplotlib.pyplot as plt

        fig, axes, results = plot_grid_comparison([], grid_size=2)
        try:
            fig.canvas.draw()
            self.assertEqual(len(axes), 4)
            self.assertTrue(all(not jets for jets in results.values()))
        finally:
            plt.close(fig)
        for cells in ([GridCell(2, 0, 1, "soft")],
                      [GridCell(0, 0, 1, "soft"), GridCell(0, 0, 2, "hard")],
                      [GridCell(0, 0, 0, "soft")]):
            with self.subTest(cells=cells), self.assertRaises(ValueError):
                plot_grid_comparison(cells, grid_size=2)


class ComparisonPlotTests(unittest.TestCase):
    def test_scales_preserve_jets_and_render_without_limit_warnings(self):
        import matplotlib.pyplot as plt

        events = [
            [],
            [Particle(0, 10, "hard"), Particle(0.2, GHOST_PT, "ghost")],
            [Particle(0, GHOST_PT, "ghost")],
        ]
        for particles in events:
            results_by_scale = {}
            for scale in ("log", "linear"):
                with self.subTest(particles=particles, scale=scale):
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error", category=UserWarning)
                        fig, axes, results = plot_comparison(particles, y_scale=scale)
                        try:
                            fig.canvas.draw()
                            for ax in axes:
                                self.assertEqual(ax.get_yscale(), scale)
                                bottom, top = ax.get_ylim()
                                self.assertGreater(top, bottom)
                                if scale == "log":
                                    self.assertGreater(bottom, 0)
                                else:
                                    self.assertEqual(bottom, 0)
                            results_by_scale[scale] = results
                        finally:
                            plt.close(fig)
            self.assertEqual(results_by_scale["log"], results_by_scale["linear"])

    def test_default_and_invalid_scale(self):
        import matplotlib.pyplot as plt

        fig, axes, _ = plot_comparison([Particle(0, 2, "soft")])
        try:
            self.assertTrue(all(ax.get_yscale() == "log" for ax in axes))
        finally:
            plt.close(fig)
        with self.assertRaisesRegex(ValueError, "y_scale"):
            plot_comparison([], y_scale="invalid")


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.particles = [Particle(0, 10, "hard"), Particle(0.5, 2, "soft"),
                          Particle(3, 1, "soft")]
        self.histories = clustering_histories(self.particles, radius=1)

    def test_initial_intermediate_and_final_states(self):
        for algorithm, history in self.histories.items():
            with self.subTest(algorithm=algorithm):
                self.assertEqual(len(history), len(self.particles) + 1)
                self.assertEqual(len(history[0].active), 3)
                self.assertEqual(history[0].final, ())
                self.assertEqual(history[0].action, "Initial particles")
                for index, state in enumerate(history):
                    self.assertEqual(len(state.active), 3 - index)
                    objects = state.active + state.final
                    self.assertEqual(sorted(i for obj in objects for i in obj.constituents),
                                     [0, 1, 2])
                    self.assertEqual(sum(obj.pt for obj in objects), 13)
                self.assertEqual(
                    sorted(history[-1].final, key=lambda jet: jet.pt, reverse=True),
                    cluster(self.particles, algorithm, radius=1),
                )
                self.assertTrue(any("Merged" in state.action for state in history))
                self.assertTrue(any("Finalized" in state.action for state in history))

    def test_empty_event_and_invalid_input(self):
        for history in clustering_histories([]).values():
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0].active, ())
            self.assertEqual(history[0].final, ())
        with self.assertRaises(ValueError):
            list(clustering_steps([], radius=0))

    def test_only_used_original_particles_render_gray(self):
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_rgba
        from matplotlib.collections import PathCollection

        particles = [Particle(0, 3, "hard"), Particle(0.1, 2, "soft"),
                     Particle(0.2, 1, "soft"), Particle(4, 4, "hard")]
        histories = clustering_histories(particles, radius=1)
        fig, axes = _step_figure(histories)
        try:
            for step in range(5):
                _draw_step(fig, axes, histories, step, color_by_final_jet=False)
                for ax, history in zip(axes, histories.values()):
                    state = history[step]
                    unused = {obj.constituents[0] for obj in state.active
                              if len(obj.constituents) == 1}
                    expected = {(p.x, p.pt) for i, p in enumerate(particles)
                                if i not in unused}
                    points = [artist for artist in ax.collections
                              if isinstance(artist, PathCollection)]
                    gray = [p for p in points if tuple(p.get_facecolors()[0]) == to_rgba("gray")]
                    self.assertEqual(len(gray), len(expected))
                    self.assertEqual({tuple(p.get_offsets()[0]) for p in gray}, expected)
                    active_merged = [obj for obj in state.active if len(obj.constituents) > 1]
                    self.assertEqual(len(points), len(particles) + len(active_merged)
                                     + len(state.final))
            self.assertIn("Used original particle",
                          [text.get_text() for text in axes[0].get_legend().get_texts()])
        finally:
            plt.close(fig)

    def test_slider_starts_at_zero_and_reaches_longest_history(self):
        histories = dict(self.histories)
        histories["anti-kt"] = (histories["anti-kt"][0], histories["anti-kt"][-1])
        with patch("IPython.display.display") as display:
            widget = reconstruction_widget(histories)
            slider = widget.children[0]
            self.assertEqual(slider.value, 0)
            self.assertEqual(slider.max, 3)
            slider.value = slider.max
            self.assertEqual(display.call_count, 2)
            axes = display.call_args.args[0].axes
            self.assertIn("step 1", axes[0].get_title())
            self.assertIn("step 3", axes[1].get_title())
            toggle = widget.children[1]
            self.assertTrue(toggle.value)
            toggle.value = False
            self.assertEqual(display.call_count, 3)
            widget.close()

    def test_active_objects_use_final_jet_colors_and_used_particles_are_gray(self):
        import matplotlib.pyplot as plt
        from matplotlib.collections import PathCollection
        from matplotlib.colors import to_rgba
        from jet_clustering import _jet_colors

        particles = [Particle(0.3, 5, "soft"), Particle(1.9, GHOST_PT, "ghost"),
                     Particle(0.8, 30, "hard"), Particle(1.6, 1, "soft")]
        histories = clustering_histories(particles, radius=1)
        fig, axes = _step_figure(histories)
        try:
            for step in range(len(particles) + 1):
                _draw_step(fig, axes, histories, step)
                for ax, history in zip(axes, histories.values()):
                    state = history[step]
                    palette = _jet_colors(history[-1].final)
                    colors = {index: to_rgba(palette[jet])
                              for jet in history[-1].final for index in jet.constituents}
                    unused = {o.constituents[0] for o in state.active
                              if len(o.constituents) == 1}
                    used = [o for o in history[0].active if o.constituents[0] not in unused]
                    original_active = [o for o in history[0].active
                                       if o.constituents[0] in unused]
                    merged_active = [o for o in state.active if len(o.constituents) > 1]
                    points = [p for p in ax.collections if isinstance(p, PathCollection)]
                    objects = used + original_active + merged_active + list(state.final)
                    self.assertEqual(len(points), len(objects))
                    for i, (point, obj) in enumerate(zip(points, objects)):
                        expected = (to_rgba("gray") if i < len(used)
                                    else colors[obj.constituents[0]])
                        self.assertEqual(tuple(point.get_facecolors()[0]), expected)
                        if obj.kind == "ghost" and i < len(objects) - len(state.final):
                            self.assertEqual(tuple(point.get_offsets()[0]), (obj.x, 0.025))
        finally:
            plt.close(fig)

    def test_final_jet_colors_are_stable_and_match_comparison(self):
        import matplotlib.pyplot as plt
        from matplotlib.collections import PathCollection

        particles = [Particle(0, 2, "soft"), Particle(3, 10, "hard"),
                     Particle(6, 5, "soft"), Particle(9, 5, "soft")]
        histories = clustering_histories(particles, radius=1)
        comparison, comparison_axes, results = plot_comparison(particles, radius=1)
        fig, axes = _step_figure(histories)
        try:
            for step in range(1, len(particles) + 1):
                _draw_step(fig, axes, histories, step, color_by_final_jet=False)
                for ax, comparison_ax, (algorithm, history) in zip(
                    axes, comparison_axes, histories.items()
                ):
                    comparison_points = [item for item in comparison_ax.collections
                                         if isinstance(item, PathCollection)]
                    # Every jet here has one particle, followed by its jet marker.
                    expected = {
                        jet: tuple(comparison_points[2 * i + 1].get_facecolors()[0])
                        for i, jet in enumerate(results[algorithm])
                    }
                    points = [item for item in ax.collections
                              if isinstance(item, PathCollection)]
                    final_points = points[-len(history[step].final):]
                    actual = [tuple(point.get_facecolors()[0]) for point in final_points]
                    self.assertEqual(actual, [expected[jet] for jet in history[step].final])
                    self.assertEqual(len(set(actual)), len(actual))
        finally:
            plt.close(fig)
            plt.close(comparison)

    def test_more_than_twenty_jets_have_distinct_colors(self):
        from jet_clustering import Jet, _jet_colors

        jets = [Jet(float(i), 1, (i,)) for i in range(25)]
        colors = _jet_colors(jets)
        self.assertEqual(len(set(colors.values())), len(jets))

    def test_gif_contains_initial_and_final_frames(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clustering.gif"
            save_reconstruction_gif(self.histories, path, fps=2, dpi=35)
            with Image.open(path) as image:
                self.assertEqual(image.n_frames, 4)
                self.assertEqual(image.info["duration"], 500)
            with self.assertRaises(ValueError):
                save_reconstruction_gif(self.histories, path, fps=0)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),
                         "ffmpeg and ffprobe are required to test MP4 output")
    def test_mp4_contains_all_frames_at_requested_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clustering.mp4"
            save_reconstruction_mp4(self.histories, path, fps=2, dpi=35,
                                    color_by_final_jet=False)
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                 "-show_entries", "stream=codec_name,nb_read_frames,avg_frame_rate,duration",
                 "-of", "json", str(path)],
                check=True, capture_output=True, text=True,
            )
            video = json.loads(probe.stdout)["streams"][0]
            self.assertEqual(video["codec_name"], "h264")
            self.assertEqual(int(video["nb_read_frames"]), 4)
            self.assertEqual(video["avg_frame_rate"], "2/1")
            self.assertAlmostEqual(float(video["duration"]), 2)

    def test_mp4_validates_rate_and_missing_ffmpeg(self):
        with self.assertRaisesRegex(ValueError, "fps"):
            save_reconstruction_mp4(self.histories, "unused.mp4", fps=0)
        with patch("matplotlib.animation.FFMpegWriter.isAvailable", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "ffmpeg"):
                save_reconstruction_mp4(self.histories, "unused.mp4")

    def test_ghosts_cluster_and_appear_in_both_plots(self):
        import matplotlib.pyplot as plt
        from matplotlib.collections import PathCollection
        from matplotlib.colors import to_rgba

        particles = [Particle(0, 10, "hard"), Particle(0.2, GHOST_PT, "ghost"),
                     Particle(2, GHOST_PT, "ghost")]
        histories = clustering_histories(particles, radius=0.7)
        fig, axes, jets = plot_comparison(particles, radius=0.7)
        try:
            for ax, reconstructed in zip(axes, jets.values()):
                self.assertEqual(sorted(i for jet in reconstructed for i in jet.constituents),
                                 [0, 1, 2])
                markers = [p for p in ax.collections if isinstance(p, PathCollection)
                           and tuple(p.get_offsets()[0]) in {(0.2, 0.025), (2, 0.025)}]
                self.assertEqual(len(markers), 2)
        finally:
            plt.close(fig)

        fig, axes = _step_figure(histories)
        try:
            for step in (0, len(particles)):
                _draw_step(fig, axes, histories, step, color_by_final_jet=False)
                expected_color = to_rgba("tab:blue" if step == 0 else "gray")
                for ax in axes:
                    markers = [p for p in ax.collections if isinstance(p, PathCollection)
                               and tuple(p.get_offsets()[0]) in {(0.2, 0.025), (2, 0.025)}]
                    self.assertEqual(len(markers), 2)
                    self.assertTrue(all(tuple(p.get_facecolors()[0]) == expected_color
                                        for p in markers))
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
