"""Educational one- and two-dimensional sequential-recombination algorithms."""

from dataclasses import dataclass
import math
import random


@dataclass(frozen=True)
class Particle:
    x: float
    pt: float
    kind: str
    y: float = 0.0


@dataclass(frozen=True)
class Jet:
    x: float
    pt: float
    constituents: tuple[int, ...]
    kind: str = "composite"
    y: float = 0.0


@dataclass(frozen=True)
class ClusteringStep:
    active: tuple[Jet, ...]
    final: tuple[Jet, ...]
    action: str


@dataclass(frozen=True)
class GridCell:
    column: int
    row: int
    pt: float
    kind: str


ALGORITHMS = {"anti-kt": -1, "kt": 1, "cambridge-aachen": 0}
GHOST_COUNT = 60 * 4  # Number of ghost particles to add
GHOST_PT = 1e-8  # Transverse momentum of ghost particles
# Event x-coordinate range for particle generation
EVENT_X_MIN = 0  # Minimum x-coordinate for particle generation
EVENT_X_MAX = 3  # Maximum x-coordinate for particle generation


def _validate_counts(n_hard, n_soft):
    for name, count in (("n_hard", n_hard), ("n_soft", n_soft)):
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{name} must be a non-negative integer")


def generate_event(n_hard=4, n_soft=60, seed=7, include_ghosts=False):
    """Generate hard and soft particles, optionally adding evenly spaced ghosts."""
    _validate_counts(n_hard, n_soft)
    if not isinstance(include_ghosts, bool):
        raise ValueError("include_ghosts must be a boolean")
    rng = random.Random(seed)
    particles = [
        Particle(rng.uniform(EVENT_X_MIN, EVENT_X_MAX), rng.uniform(*pt_range), kind)
        for kind, count, pt_range in (
            ("hard", n_hard, (15, 30)),
            ("soft", n_soft, (1, 4)),
        )
        for _ in range(count)
    ]
    if include_ghosts:
        spacing = (EVENT_X_MAX - EVENT_X_MIN) / GHOST_COUNT
        particles.extend(
            Particle(EVENT_X_MIN + (i + 0.5) * spacing, GHOST_PT, "ghost")
            for i in range(GHOST_COUNT)
        )
    return particles


def generate_grid_event(n_hard=4, n_soft=30, grid_size=12, seed=7):
    """Generate one hard or soft particle per occupied cell of a square grid."""
    _validate_counts(n_hard, n_soft)
    if isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size <= 0:
        raise ValueError("grid_size must be a positive integer")
    if n_hard + n_soft > grid_size ** 2:
        raise ValueError("n_hard + n_soft must not exceed the number of grid cells")
    rng = random.Random(seed)
    locations = rng.sample(range(grid_size ** 2), n_hard + n_soft)
    return [
        GridCell(location % grid_size, location // grid_size,
                 rng.uniform(15, 30) if i < n_hard else rng.uniform(1, 4),
                 "hard" if i < n_hard else "soft")
        for i, location in enumerate(locations)
    ]


def clustering_steps(particles, algorithm="anti-kt", radius=0.7):
    """Yield the initial state and each sequential-recombination state.

    Distances use non-periodic x and y; y defaults to zero for 1D events.
    Scalar pT is summed and positions use pT-weighted centroids, not four-vectors.
    """
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm!r}")
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    if any(not math.isfinite(p.x) or not math.isfinite(p.y)
           or not math.isfinite(p.pt) or p.pt <= 0
           for p in particles):
        raise ValueError("particle positions must be finite and pT must be finite and positive")

    power = ALGORITHMS[algorithm]
    active = [Jet(p.x, p.pt, (i,), p.kind, p.y) for i, p in enumerate(particles)]
    jets = []

    yield ClusteringStep(tuple(active), tuple(jets), "Initial particles")
    while active:
        weights = [jet.pt ** (2 * power) for jet in active]
        best_distance = min(weights)
        beam_index = weights.index(best_distance)
        pair = None

        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                distance = min(weights[i], weights[j]) * (
                    ((active[i].x - active[j].x) / radius) ** 2
                    + ((active[i].y - active[j].y) / radius) ** 2
                )
                if distance < best_distance:
                    best_distance = distance
                    pair = (i, j)

        if pair is None:
            jet = active.pop(beam_index)
            jets.append(jet)
            action = (f"Finalized object at x={jet.x:.2f}, pT={jet.pt:.2f} "
                      f"(beam distance {best_distance:.3g})")
        else:
            i, j = pair
            first, second = active[i], active[j]
            pt = first.pt + second.pt
            merged = Jet(
                (first.x * first.pt + second.x * second.pt) / pt,
                pt,
                first.constituents + second.constituents,
                y=(first.y * first.pt + second.y * second.pt) / pt,
            )
            active.pop(j)
            active.pop(i)
            active.append(merged)
            action = (f"Merged objects at x={first.x:.2f} and {second.x:.2f} "
                      f"(pair distance {best_distance:.3g})")

        yield ClusteringStep(tuple(active), tuple(jets), action)


def cluster(particles, algorithm="anti-kt", radius=0.7):
    """Return final jets ordered by decreasing pT."""
    for step in clustering_steps(particles, algorithm, radius):
        pass
    return sorted(step.final, key=lambda jet: jet.pt, reverse=True)


def clustering_histories(particles, radius=0.7):
    """Calculate all three histories, including step zero."""
    return {
        name: tuple(clustering_steps(particles, name, radius))
        for name in ALGORITHMS
    }
