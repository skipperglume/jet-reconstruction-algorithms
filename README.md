# Jet reconstruction examples

Open [jet_reconstruction.ipynb](jet_reconstruction.ipynb) in Jupyter or VS Code,
select a Python kernel, and run all cells. Set `n_hard`, `n_soft`, `radius`, and
`seed` in the parameters cell to explore different events. Set
`include_ghosts = True` to add 240 evenly spaced ghosts across [-3, 3], each
with pT 1e-8. The option defaults to false and does not change the generated
hard and soft particles.

Set `y_scale` to `"log"` (the default) or `"linear"` in the parameters cell
to choose the pT axis scale for the final comparison plot. The step-by-step
widget, GIF and MP4 retain their linear pT axes.

Install the dependencies into your Python environment with
`python -m pip install -r requirements.txt` if needed. Run the tests with
`python -m unittest -v`.

The step-by-step widget starts at iteration zero. Each step applies one
clustering operation per algorithm. By default, active particles and intermediate
objects are colored by their eventual final jet, as are final-jet diamonds.
This uses the completed history, so colors reveal the outcome in advance;
they match the final comparison within each algorithm and are assigned
independently between algorithms. Used original particles remain at their
original positions in gray. Disable **Color by final jet** in the widget to
show active objects in blue and used originals in gray instead. Intermediate
objects are shown only while active or as final jets. A final jet may overlap
its original particle.
The slider ends at the longest history and
holds any earlier-finished algorithms on their final state. The annotations
give constituent counts. Iterations are not physical time.

With ghosts enabled, tick marks near the bottom of each final comparison panel
are colored by the jet that captured each ghost. This samples a one-dimensional
catchment footprint: each ghost represents 0.025 units of x, not a physical
two-dimensional area. The jet summary also reports the ghost count and
sampled x-length for each jet. Ghost-only jets are included, and ghost pT is small but
nonzero. Tick height is for visibility rather than pT. In the step view, ghost
ticks use the same color mode as the other particles. More particles also
mean more steps and a slower GIF export.

To export the same steps as a GIF, set `export_gif = True` in the notebook's
GIF cell and run it. Adjust `gif_path` and `fps` there as needed. GIF export
uses final-jet colors by default; pass `color_by_final_jet=False` to use the
blue/gray view. The GIF is
written relative to the notebook kernel's working directory.

For MP4 export, install FFmpeg separately and ensure its `ffmpeg` executable
is on the notebook kernel's PATH. Set `export_mp4 = True` in the notebook's
MP4 cell and adjust the path and `fps` if needed. Each iteration is one frame,
starting at iteration zero; `fps=2` displays each step for half a second.
Like the GIF exporter, `save_reconstruction_mp4` accepts
`color_by_final_jet=False` for the blue/gray view. Large events can take time
to render. The timestamps represent playback time, not physical collision time.

The notebook compares anti-kt, kt, and Cambridge/Aachen using an intentionally
simplified, non-periodic one-dimensional angular coordinate and scalar-pT
recombination. It is not a replacement for a physical rapidity–azimuth
clustering implementation such as FastJet.

The notebook also contains an independent 2D grid example. Change `grid_size`,
`grid_n_hard`, `grid_n_soft`, `grid_radius`, or `grid_seed` in its code cell.
The number of particles must fit in the grid, with at most one particle per
cell. Each algorithm has a top-down jet-colored cell map and a 3D view whose
bar heights show cell pT; black crosses in the map show jet axes. Both spatial
coordinates are non-periodic and clustering uses Euclidean 2D distances and
pT-weighted centroid recombination, not physical four-vector recombination.
The grid example does not add ghosts or use the 1D iteration widget.
