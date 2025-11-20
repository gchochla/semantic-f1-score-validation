#!/usr/bin/env python
"""
Study A.1 — Synthetic (construct validity)

This script generates a synthetic label graph, constructs similarity matrices,
simulates gold labels and near/far‑miss predictors, and evaluates Hard vs
Semantic F1 metrics under multiple similarity settings.

Outputs
- CSVs: per‑configuration summaries, Kendall tau vs radius, bootstrap CIs.
- Figures: metric vs radius (Fig A1) and Kendall tau bars (Fig A2).
- LaTeX table: bootstrap mean ± 95% CI for A vs B (Tab A1).

Example
  python scripts/semantic_f1/study_a1.py \
      --num-labels 24 --num-examples 20000 \
      --k 1 2 3 --p 0.0 0.25 0.5 0.75 1.0 \
      --near-radii 1 2 --far-radii 3 4 \
      --alpha 1.0 0.75 0.5 0.25 0.0 \
      --outdir logs/semantic_f1/study_a

Notes
- Hard F1 uses sklearn (micro/macro/samples).
- Semantic metrics use llm_subj.metrics.semantic_f1.*.
- No network access required; relies on numpy/pandas/sklearn/scipy/matplotlib.
- Supports both the original ring geometry and union-of-manifolds layouts via
  ``--geometry`` and ``--union-components`` (e.g., mixed rings and lines).
"""

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field
import logging
from typing import Iterable, Literal, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import f1_score
from semantic_f1_score import semantic_f1_score
from sklearn.preprocessing import MultiLabelBinarizer

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import (
    Axes3D,
)  # noqa: F401  # needed for 3D projection

# External plotting helpers with robust import when run as a script
try:
    import scripts.semantic_f1.study_a3_plots as study_a3_plots
except Exception:
    # Fallback: import sibling when executed directly via path
    sys.path.append(os.path.dirname(__file__))
    import study_a3_plots as study_a3_plots  # type: ignore


def make_labels(n: int) -> list[str]:
    """Create canonical string labels ``[c0, c1, ..., c{n-1}]``.

    Args:
        n: Number of labels to create.

    Returns:
        List of label names of length ``n``.
    """
    return [f"c{i}" for i in range(n)]


def make_similarity_ring(
    n: int, labels: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """Build ring-based similarity ``S`` using normalized cosine proximity.

    Labels lie on a unit circle at angles ``theta_i = 2π i / n`` and
    ``S[i,j] = (1 + cos(theta_i - theta_j)) / 2`` in ``[0,1]``. This avoids
    exponential kernels and directly matches the design note to use normalized
    cosine similarity as the ground-truth structure.

    Args:
        n: Number of labels arranged on a unit circle.

    Returns:
        A symmetric ``n x n`` DataFrame with values in ``[0,1]`` indexed by
        label names.
    """
    if labels is None:
        labels = make_labels(n)
    else:
        if len(labels) != n:
            raise ValueError(
                "Number of labels must match `n` when custom labels are provided."
            )
        labels = list(labels)
    thetas = np.linspace(0.0, 2 * math.pi, num=n, endpoint=False)
    mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            dtheta = thetas[i] - thetas[j]
            mat[i, j] = (1.0 + math.cos(dtheta)) / 2.0
    return pd.DataFrame(mat, index=labels, columns=labels)


def identity_similarity(labels: list[str]) -> pd.DataFrame:
    """Return the identity similarity matrix for the given labels.

    Args:
        labels: Ordered label names to use for both axes.

    Returns:
        ``len(labels) x len(labels)`` identity matrix as a DataFrame.
    """
    n = len(labels)
    mat = np.eye(n, dtype=float)
    return pd.DataFrame(mat, index=labels, columns=labels)


def permuted_rows_similarity(
    S: pd.DataFrame, seed: int | None = None
) -> pd.DataFrame:
    """Create a degraded ``S`` by permuting only the rows.

    Intentionally breaks row/column alignment while preserving column order,
    simulating an invalid similarity source (as in Study A.1 controls).

    Args:
        S: Original similarity matrix with matching index/columns.
        seed: RNG seed for reproducibility.

    Returns:
        A copy of ``S`` with rows permuted and index reset to the original
        label names (so labels appear correct but alignment is broken).
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(S))
    # Permute only rows (index) to intentionally break row/column alignment
    S_perm = S.copy().iloc[perm, :]
    S_perm.index = S.index  # keep original label names on rows
    return S_perm


def mix_similarity(
    S: pd.DataFrame,
    U: pd.DataFrame,
    alpha: float,
) -> pd.DataFrame:
    """Mix two similarity matrices as ``S_alpha = alpha*S + (1-alpha)*U``.

    Args:
        S: Primary similarity matrix.
        U: Secondary/noise similarity matrix (e.g., identity or uniform).
        alpha: Mixture weight in ``[0,1]``.

    Returns:
        Mixed similarity matrix (DataFrame) with the same indexing as ``S``.
    """
    assert 0.0 <= alpha <= 1.0
    mat = alpha * S.values + (1.0 - alpha) * U.values
    return pd.DataFrame(mat, index=S.index, columns=S.columns)


def make_uniform_noise_U(
    labels: list[str], offdiag: float = 0.5
) -> pd.DataFrame:
    """Create a uniform-noise matrix with unit diagonal and constant off-diagonal.

    Args:
        labels: Label names for axes.
        offdiag: Off-diagonal similarity value in ``[0,1]``.

    Returns:
        DataFrame where ``U[i,i]=1`` and ``U[i,j]=offdiag`` for ``i!=j``.
    """
    n = len(labels)
    mat = np.full((n, n), offdiag, dtype=float)
    np.fill_diagonal(mat, 1.0)
    return pd.DataFrame(mat, index=labels, columns=labels)


def make_deceptive_similarity(
    geometry: "BaseGeometry", gap: float
) -> pd.DataFrame:
    """Construct a deceptive Euclidean similarity ``1 / (1 + d)``.

    Embeds components in parallel layers separated by ``gap`` along a synthetic
    Z-axis (e.g., two rings stacked with vertical distance ``gap``) and computes
    Euclidean distances between label coordinates. Converts distance to
    similarity via ``1 / (1 + d)``.
    """

    embed = geometry.deceptive_embedding(gap)
    diff = embed[:, None, :] - embed[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    sim = 1.0 / (1.0 + dists)
    np.fill_diagonal(sim, 1.0)
    labels = geometry.labels
    return pd.DataFrame(sim, index=labels, columns=labels)


def plot_label_space(
    geometry: "BaseGeometry",
    outdir: str,
    gap: float,
    *,
    filename: str = "label_space.png",
) -> None:
    """Render a scatter plot of the label embedding to ``outdir``."""

    embed = geometry.deceptive_embedding(gap)
    comp_ids = geometry.component_ids()
    component_paths = list(geometry.component_paths())
    path_out = os.path.join(outdir, filename)

    fig = plt.figure(figsize=(7.5, 7.5))
    fig.patch.set_facecolor("white")

    palette = plt.get_cmap("tab10")
    unique_components = np.unique(comp_ids)
    colors = {
        comp: palette(i % palette.N) for i, comp in enumerate(unique_components)
    }
    point_colors = [colors.get(c, "#4c4c4c") for c in comp_ids]

    dims = min(embed.shape[1], 3)
    max_abs = float(np.max(np.abs(embed[:, :dims]))) if embed.size else 1.0
    if not np.isfinite(max_abs) or max_abs == 0.0:
        max_abs = 1.0
    axis_extent = max(1.0, max_abs)
    tick_positions = [-axis_extent, 0.0, axis_extent]
    tick_labels = ["-1", "0", "1"]
    line_style = (0, (5, 4))

    is_3d = embed.shape[1] >= 3 and not np.allclose(embed[:, 2], embed[0, 2])

    if is_3d:
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            embed[:, 0],
            embed[:, 1],
            embed[:, 2],
            c=point_colors,
            s=85,
            depthshade=False,
            edgecolor="black",
            linewidth=0.4,
        )

        if component_paths:
            paths_sorted = sorted(
                component_paths,
                key=lambda item: float(
                    np.mean(embed[np.asarray(item[0], dtype=int), 2])
                ),
            )
        else:
            paths_sorted = []

        for indices, closed, comp_idx in paths_sorted:
            idx_arr = np.asarray(indices, dtype=int)
            coords = embed[idx_arr]
            if coords.shape[0] < 2:
                continue
            color = colors.get(comp_idx, "#4c4c4c")
            if closed and coords.shape[0] >= 3:
                center = coords.mean(axis=0)
                radius = np.linalg.norm(
                    coords[:, :2] - center[:2], axis=1
                ).mean()
                angles = np.linspace(0.0, 2.0 * math.pi, 361)
                circle_x = center[0] + radius * np.cos(angles)
                circle_y = center[1] + radius * np.sin(angles)
                circle_z = np.full_like(circle_x, center[2])
                ax.plot(
                    circle_x,
                    circle_y,
                    circle_z,
                    linestyle=line_style,
                    linewidth=1.3,
                    color=color,
                    alpha=0.95,
                    zorder=5,
                )
            else:
                ax.plot(
                    coords[:, 0],
                    coords[:, 1],
                    coords[:, 2],
                    linestyle=line_style,
                    linewidth=1.1,
                    color=color,
                    alpha=0.9,
                    zorder=4,
                )

        ax.set_xlim(-axis_extent, axis_extent)
        ax.set_ylim(-axis_extent, axis_extent)
        ax.set_zlim(-axis_extent, axis_extent)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)
        ax.set_zticks(tick_positions)
        ax.set_zticklabels(tick_labels)

        ax.set_xlabel("x", fontsize=14, labelpad=12)
        ax.set_ylabel("y", fontsize=14, labelpad=12)
        ax.set_zlabel("z", fontsize=14, labelpad=12)
        ax.view_init(elev=25, azim=35)
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect([1, 1, 1])
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            if hasattr(axis, "pane"):
                axis.pane.fill = False
                axis.pane.set_edgecolor("white")
        ax.grid(False)
        ax.tick_params(axis="both", labelsize=11, pad=4)
        ax.zaxis.set_tick_params(labelsize=11, pad=6)
    else:
        ax = fig.add_subplot(111)
        scatter = ax.scatter(
            embed[:, 0],
            embed[:, 1],
            c=point_colors,
            s=110,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.95,
            zorder=5,
        )

        for indices, closed, comp_idx in component_paths:
            idx_arr = np.asarray(indices, dtype=int)
            coords = embed[idx_arr, :2]
            if coords.shape[0] < 2:
                continue
            color = colors.get(comp_idx, "#4c4c4c")
            if closed and coords.shape[0] >= 3:
                center = coords.mean(axis=0)
                radius = np.linalg.norm(coords - center, axis=1).mean()
                angles = np.linspace(0.0, 2.0 * math.pi, 361)
                circle_x = center[0] + radius * np.cos(angles)
                circle_y = center[1] + radius * np.sin(angles)
                ax.plot(
                    circle_x,
                    circle_y,
                    linestyle=line_style,
                    linewidth=1.5,
                    color=color,
                    alpha=0.95,
                    zorder=4,
                )
            else:
                ax.plot(
                    coords[:, 0],
                    coords[:, 1],
                    linestyle=line_style,
                    linewidth=1.2,
                    color=color,
                    alpha=0.9,
                    zorder=4,
                )

        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlim(-axis_extent, axis_extent)
        ax.set_ylim(-axis_extent, axis_extent)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)

        for spine_name in ("left", "bottom"):
            spine = ax.spines[spine_name]
            spine.set_position("zero")
            spine.set_linewidth(1.0)
            spine.set_color("#666666")
        ax.spines["right"].set_color("none")
        ax.spines["top"].set_color("none")
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
        ax.set_xlabel("x", fontsize=14, labelpad=10)
        ax.set_ylabel("y", fontsize=14, labelpad=10)
        ax.grid(False)
        ax.tick_params(axis="both", labelsize=11, pad=6)

    ax.set_title("Label Space", fontsize=18, weight="bold", y=0.92)
    ax.set_facecolor("white")

    if len(unique_components) > 1:
        from matplotlib.patches import Patch

        legend_handles = [
            Patch(
                facecolor=colors[c],
                edgecolor="black",
                linewidth=0.6,
                label=f"Component {c}",
            )
            for c in unique_components
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.85),
            frameon=False,
            fontsize=11,
            title="Components",
            title_fontsize=12,
        )

    fig.tight_layout()
    plt.subplots_adjust(top=0.99)
    png_path = path_out
    pdf_path = os.path.splitext(path_out)[0] + ".pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Label space plot written to {png_path}")
    print(f"Label space plot written to {pdf_path}")


@dataclass(frozen=True)
class UnionComponentSpec:
    """Specification for one component in a union-of-manifolds label layout."""

    name: str
    kind: Literal["ring", "line"]
    size: int
    radius: Optional[float] = None
    length: Optional[float] = None


@dataclass
class SynthConfig:
    """Configuration for Study A.1 synthetic experiment.

    Fields
    - ``num_labels``: Number of labels in the geometry.
    - ``num_examples``: Number of synthetic examples per configuration.
    - ``k_values``: Numbers of labels per example to sample for gold sets.
    - ``p_values``: Probabilities to perturb each gold label.
    - ``near_radii``: Radii for near-miss predictor A.
    - ``far_radii``: Radii for far-miss predictor B.
    - ``alphas``: Mix weights for ``S_alpha`` sensitivity sweeps.
    - ``outdir``: Output directory for CSVs/figures/tables.
    - ``seed``: RNG seed for reproducibility.
    - ``noise_mode``: Which ``U`` to use in the mix ("identity" or "uniform").
    - ``bootstrap``: Bootstrap iterations for CI estimation.
    - ``geometry_mode``: Geometry to use for labels (``"ring"`` or ``"union"``).
    - ``union_components``: Component specs when ``geometry_mode == "union"``.
    - ``union_component_weights``: Optional selection weights per component.
    - ``union_line_bandwidth``: Kernel bandwidth for line components (normalized).
    - ``union_cross_affinity``: Scaling for cross-component similarities.
    - ``union_cross_bandwidth``: Bandwidth for cross-component similarities.
    - ``union_ring_min_similarity``: Floor for ring-component similarities to
      guarantee same-manifold distances remain more similar than any
      cross-manifold pairing.
    - ``union_cross_jump_probs``: List of probabilities (conditional on a label being
      perturbed) of jumping to a different component during the perturbation
      step, enabling controllable cross-manifold confusions.
    - ``deceptive_ring_gap``: Vertical gap between parallel components when
      constructing the deceptive Euclidean similarity matrix.
    """

    num_labels: int
    num_examples: int
    k_values: list[int]
    p_values: list[float]
    near_radii: list[int]
    far_radii: list[int]
    alphas: list[float]
    outdir: str = "logs/semantic_f1/study_a"
    seed: int = 123
    noise_mode: Literal["identity", "uniform"] = "identity"
    bootstrap: int = 200
    geometry_mode: Literal["ring", "union"] = "ring"
    union_components: tuple[UnionComponentSpec, ...] = field(
        default_factory=tuple
    )
    union_component_weights: Optional[tuple[float, ...]] = None
    union_line_bandwidth: float = 0.35
    union_cross_affinity: float = 0.2
    union_cross_bandwidth: float = 1.5
    union_ring_min_similarity: float = 0.3
    union_cross_jump_probs: list[float] = field(default_factory=lambda: [0.0])
    deceptive_ring_gap: float = 1.0


class BaseGeometry:
    """Base class encapsulating label layout and perturbation mechanics.

    Concrete subclasses expose a common interface so the rest of Study A can
    remain agnostic to the actual label topology. Each geometry must provide
    methods for sampling gold label sets, perturbing them according to a
    near-/far-miss radius, and generating an "ideal" similarity matrix that
    matches the intended structure of the space.
    """

    def __init__(self, labels: Sequence[str], rng: np.random.Generator):
        self._labels = list(labels)
        self._rng = rng

    @property
    def labels(self) -> list[str]:
        return self._labels

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    @property
    def n(self) -> int:
        return len(self._labels)

    def sample_gold_labels(
        self, k: int
    ) -> list[int]:  # pragma: no cover - abstract
        """Sample a gold label set of size ``k`` under the geometry's prior."""
        raise NotImplementedError

    def perturb_labels(
        self, gold: Iterable[int], *, p: float, radius: int
    ) -> list[int]:  # pragma: no cover - abstract
        """Return a perturbed copy of ``gold`` according to ``p`` and ``radius``."""
        raise NotImplementedError

    def make_similarity(self) -> pd.DataFrame:  # pragma: no cover - abstract
        """Construct the similarity matrix that encodes the geometry."""
        raise NotImplementedError

    def component_ids(self) -> np.ndarray:  # pragma: no cover - abstract
        """Return component index per label (all zeros for single-component spaces)."""
        raise NotImplementedError

    def deceptive_embedding(
        self, gap: float
    ) -> np.ndarray:  # pragma: no cover - abstract
        """Return 3D embeddings used for the deceptive Euclidean similarity."""
        raise NotImplementedError

    def to_label_names(self, xs: list[list[int]]) -> list[list[str]]:
        return [[self._labels[i] for i in row] for row in xs]

    def component_paths(self) -> list[tuple[list[int], bool, int]]:
        """Return plotting paths as (indices, closed, component_id)."""
        raise NotImplementedError


class RingGeometry(BaseGeometry):
    """Original ring-based geometry used by Study A.

    Sampling draws contiguous labels around a latent angular center and
    perturbations move clockwise/counter-clockwise around the circle.
    """

    def __init__(self, labels: Sequence[str], rng: np.random.Generator):
        super().__init__(labels, rng)
        self._thetas = np.linspace(0.0, 2 * math.pi, num=self.n, endpoint=False)
        self._indices = np.arange(self.n, dtype=int)
        self._base_coords = np.column_stack(
            (np.cos(self._thetas), np.sin(self._thetas))
        )

    def sample_gold_labels(self, k: int) -> list[int]:
        """Sample ``k`` indices using cosine weights around a latent angle."""
        theta_c = self.rng.uniform(0.0, 2 * math.pi)
        dtheta = np.abs(self._thetas - theta_c)
        dtheta = np.minimum(dtheta, 2 * math.pi - dtheta)
        weights = (1.0 + np.cos(dtheta)) / 2.0
        total = float(weights.sum())
        probs = (weights / total) if total > 0 else None
        chosen = self.rng.choice(
            self._indices, size=min(k, self.n), replace=False, p=probs
        )
        return list(map(int, chosen))

    def perturb_labels(
        self, gold: Iterable[int], *, p: float, radius: int
    ) -> list[int]:
        """Apply a ring walk of ``radius`` steps with probability ``p`` per label."""
        out: set[int] = set()
        for g in gold:
            if self.rng.random() < p:
                direction = -1 if self.rng.random() < 0.5 else 1
                out.add((g + direction * radius) % self.n)
            else:
                out.add(g % self.n)
        return sorted(out)

    def make_similarity(self) -> pd.DataFrame:
        """Return the classic ring similarity with cosine-based affinities."""
        return make_similarity_ring(self.n, labels=self.labels)

    def component_ids(self) -> np.ndarray:
        return np.zeros(self.n, dtype=int)

    def deceptive_embedding(self, gap: float) -> np.ndarray:
        embed = np.zeros((self.n, 3), dtype=float)
        embed[:, :2] = self._base_coords
        # Single component → z remains 0
        return embed

    def component_paths(self) -> list[tuple[list[int], bool, int]]:
        return [(self._indices.tolist(), True, 0)]


@dataclass
class _UnionComponent:
    spec: UnionComponentSpec
    start: int
    indices: np.ndarray
    coords: np.ndarray
    base_coords: np.ndarray
    angles: Optional[np.ndarray]


class UnionManifoldGeometry(BaseGeometry):
    """Geometry where labels live on a union of manifolds/components.

    Each component can be a ring or a line. Within-component similarities use
    structure-specific kernels, while cross-component similarities use a radial
    basis function tuned by ``cross_affinity`` and ``cross_bandwidth``. A
    ``ring_min_similarity`` floor ensures ring components never assign lower
    similarity than any cross-component pair, preserving the intended ordering
    of "near miss" severities. When ``cross_jump_prob`` is positive, perturbation
    steps first map the label to its corresponding position in the target component,
    then apply radius-based perturbation within that component, enabling controllable
    cross-manifold confusions.
    """

    def __init__(
        self,
        labels: Sequence[str],
        rng: np.random.Generator,
        components: Sequence[UnionComponentSpec],
        *,
        component_weights: Optional[Sequence[float]] = None,
        line_bandwidth: float = 0.35,
        cross_affinity: float = 0.2,
        cross_bandwidth: float = 1.5,
        ring_min_similarity: float = 0.3,
        cross_jump_prob: float = 0.0,
    ) -> None:
        super().__init__(labels, rng)
        if not components:
            raise ValueError(
                "UnionManifoldGeometry requires at least one component"
            )
        total = sum(c.size for c in components)
        if total != self.n:
            raise ValueError(
                "Sum of union component sizes does not match number of labels"
            )
        if line_bandwidth <= 0:
            raise ValueError("line_bandwidth must be positive")
        if cross_affinity < 0:
            raise ValueError("cross_affinity must be non-negative")
        if cross_bandwidth <= 0:
            raise ValueError("cross_bandwidth must be positive")
        if not 0.0 <= ring_min_similarity < 1.0:
            raise ValueError("ring_min_similarity must lie in [0,1)")
        if cross_affinity >= ring_min_similarity:
            raise ValueError(
                "cross_affinity must be strictly lower than ring_min_similarity "
                "to ensure intra-ring minima exceed cross-component maxima"
            )
        if not 0.0 <= cross_jump_prob <= 1.0:
            raise ValueError("cross_jump_prob must lie in [0,1]")

        self._line_bandwidth = float(line_bandwidth)
        self._cross_affinity = float(cross_affinity)
        self._cross_bandwidth = float(cross_bandwidth)
        self._ring_min_similarity = float(ring_min_similarity)
        self._cross_jump_prob = float(cross_jump_prob)

        self._components: list[_UnionComponent] = []
        self._component_lookup = np.zeros((self.n, 2), dtype=int)
        coords_all = np.zeros((self.n, 2), dtype=float)

        offset = 0
        for idx, spec in enumerate(components):
            start = offset
            stop = start + spec.size
            indices = np.arange(start, stop, dtype=int)
            comp_coords, comp_base, comp_angles = self._make_component_coords(
                spec, idx
            )
            coords_all[start:stop, :] = comp_coords
            self._components.append(
                _UnionComponent(
                    spec=spec,
                    start=start,
                    indices=indices,
                    coords=comp_coords,
                    base_coords=comp_base,
                    angles=comp_angles,
                )
            )
            self._component_lookup[start:stop, 0] = idx
            self._component_lookup[start:stop, 1] = np.arange(
                spec.size, dtype=int
            )
            offset = stop

        self._coords = coords_all
        self._num_components = len(self._components)
        self._component_weights = self._normalize_component_weights(
            component_weights
        )
        self._component_centroids = np.vstack(
            [comp.coords.mean(axis=0) for comp in self._components]
        )
        self._cross_component_transition = self._build_cross_transition()

    def _normalize_component_weights(
        self, weights: Optional[Sequence[float]]
    ) -> np.ndarray:
        if weights is None:
            arr = np.array(
                [comp.spec.size for comp in self._components], dtype=float
            )
        else:
            arr = np.array(list(weights), dtype=float)
            if arr.shape[0] != len(self._components):
                raise ValueError(
                    "Component weights must match number of components"
                )
            if np.any(arr <= 0):
                raise ValueError("Component weights must be strictly positive")
        total = arr.sum()
        if total == 0:
            raise ValueError("Component weights sum to zero")
        return arr / total

    def _build_cross_transition(self) -> np.ndarray:
        """Construct per-component jump probabilities for cross-manifold moves."""

        m = len(self._components)
        trans = np.zeros((m, m), dtype=float)
        if m <= 1 or self._cross_jump_prob == 0.0:
            return trans

        sigma = self._cross_bandwidth
        for i in range(m):
            weights = np.zeros(m, dtype=float)
            for j in range(m):
                if i == j:
                    continue
                dist = float(
                    np.linalg.norm(
                        self._component_centroids[i]
                        - self._component_centroids[j]
                    )
                )
                weights[j] = math.exp(-(dist**2) / (2 * sigma**2))
            total = weights.sum()
            if total <= 0:
                weights[:] = 1.0
                weights[i] = 0.0
                total = weights.sum()
            trans[i, :] = weights / total
        return trans

    def _sample_cross_component(
        self, global_idx: int, source_comp_idx: int
    ) -> int:
        """Sample a label from a different component for cross-manifold jumps.

        DEPRECATED: This method is no longer used. Cross-component jumps now
        use correspondence mapping followed by radius-based perturbation.
        """

        if self._num_components <= 1:
            return global_idx

        weights = self._cross_component_transition[source_comp_idx]
        if weights.sum() <= 0:
            weights = np.ones(self._num_components, dtype=float)
            weights[source_comp_idx] = 0.0
            weights = weights / weights.sum()

        target_comp_idx = int(self.rng.choice(self._num_components, p=weights))
        target_comp = self._components[target_comp_idx]

        source_coord = self._coords[global_idx]
        diffs = target_comp.coords - source_coord
        d2 = np.sum(diffs * diffs, axis=1)
        sigma = self._cross_bandwidth
        local_weights = np.exp(-(d2) / (2 * sigma**2))
        if local_weights.sum() <= 0:
            local_weights = np.ones(len(target_comp.indices), dtype=float)
        local_weights = local_weights / local_weights.sum()

        local_choice = int(
            self.rng.choice(len(target_comp.indices), p=local_weights)
        )
        return int(target_comp.indices[local_choice])

    def _map_to_corresponding_label(
        self, global_idx: int, source_comp_idx: int, target_comp_idx: int
    ) -> int:
        """Map a label to its corresponding position in another component."""
        if source_comp_idx == target_comp_idx:
            return global_idx

        source_comp = self._components[source_comp_idx]
        target_comp = self._components[target_comp_idx]
        source_local_idx = int(self._component_lookup[global_idx, 1])

        # Map based on relative position within component
        if source_comp.spec.size == 0 or target_comp.spec.size == 0:
            return (
                int(target_comp.indices[0])
                if len(target_comp.indices) > 0
                else global_idx
            )

        # Scale the position proportionally
        relative_pos = source_local_idx / max(1, source_comp.spec.size - 1)
        target_local_idx = int(
            round(relative_pos * max(1, target_comp.spec.size - 1))
        )
        target_local_idx = min(target_local_idx, target_comp.spec.size - 1)

        return int(target_comp.indices[target_local_idx])

    def _make_component_coords(
        self, spec: UnionComponentSpec, idx: int
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if spec.size <= 0:
            raise ValueError("Component size must be positive")
        if spec.kind == "ring":
            radius = spec.radius if spec.radius is not None else 1.0
            angles = np.linspace(
                0.0, 2 * math.pi, num=spec.size, endpoint=False
            )
            base_coords = np.column_stack(
                (radius * np.cos(angles), radius * np.sin(angles))
            )
        elif spec.kind == "line":
            length = spec.length if spec.length is not None else 1.0
            if spec.size == 1:
                xs = np.array([0.0])
            else:
                xs = np.linspace(-length / 2.0, length / 2.0, num=spec.size)
            zeros = np.zeros_like(xs)
            base_coords = np.column_stack((xs, zeros))
            angles = None
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported component kind: {spec.kind}")

        # Stagger components in 2D space to avoid overlap
        layer = idx // 3
        col = idx % 3
        offset = np.array([col * 3.0, layer * 3.0], dtype=float)
        coords = base_coords + offset

        if spec.kind == "ring":
            return coords, base_coords, angles  # type: ignore[return-value]
        return coords, base_coords, None

    def sample_gold_labels(self, k: int) -> list[int]:
        """Sample ``k`` labels by first choosing a component and then a kernel."""
        comp_idx = int(
            self.rng.choice(len(self._components), p=self._component_weights)
        )
        comp = self._components[comp_idx]
        local_center = int(self.rng.integers(0, comp.spec.size))
        weights = self._component_kernel_weights(comp, local_center)
        local_indices = np.arange(comp.spec.size, dtype=int)
        chosen = self.rng.choice(
            local_indices,
            size=min(k, comp.spec.size),
            replace=False,
            p=weights,
        )
        return sorted(int(comp.indices[i]) for i in chosen)

    def _component_kernel_weights(
        self, comp: _UnionComponent, local_center: int
    ) -> np.ndarray:
        local_indices = np.arange(comp.spec.size, dtype=int)
        if comp.spec.kind == "ring":
            assert comp.angles is not None  # for type checker
            center_angle = comp.angles[local_center]
            diffs = np.abs(comp.angles - center_angle)
            diffs = np.minimum(diffs, 2 * math.pi - diffs)
            weights = (1.0 + np.cos(diffs)) / 2.0
        else:  # line
            diffs = np.abs(local_indices - local_center)
            if comp.spec.size <= 1:
                weights = np.ones(1, dtype=float)
            else:
                norm = diffs / max(1, comp.spec.size - 1)
                sigma = self._line_bandwidth
                weights = np.exp(-(norm**2) / (2 * sigma**2))
        total = weights.sum()
        if total == 0:
            weights = np.ones_like(weights)
            total = float(weights.sum())
        return weights / total

    def perturb_labels(
        self, gold: Iterable[int], *, p: float, radius: int
    ) -> list[int]:
        """Perturb labels with optional cross-component jumps.

        When cross_jump_prob triggers a cross-component jump, the label is first
        mapped to the corresponding position in the target component, then
        perturbed according to p and radius within that component.
        """
        out: set[int] = set()
        allow_cross = self._cross_jump_prob > 0 and self._num_components > 1
        for g in gold:
            comp_idx = int(self._component_lookup[g, 0])
            local_idx = int(self._component_lookup[g, 1])
            comp = self._components[comp_idx]
            new_global = int(comp.indices[local_idx])

            if p > 0 and self.rng.random() < p:
                cross_jump = (
                    allow_cross and self.rng.random() < self._cross_jump_prob
                )
                if cross_jump:
                    # First, choose target component
                    weights = self._cross_component_transition[comp_idx]
                    if weights.sum() <= 0:
                        weights = np.ones(self._num_components, dtype=float)
                        weights[comp_idx] = 0.0
                        weights = weights / weights.sum()

                    target_comp_idx = int(
                        self.rng.choice(self._num_components, p=weights)
                    )

                    # Map to corresponding label in target component
                    new_global = self._map_to_corresponding_label(
                        g, comp_idx, target_comp_idx
                    )

                    # Now apply radius-based perturbation within the target component
                    target_comp = self._components[target_comp_idx]
                    target_local_idx = int(
                        self._component_lookup[new_global, 1]
                    )

                    direction = -1 if self.rng.random() < 0.5 else 1
                    if target_comp.spec.kind == "ring":
                        if target_comp.spec.size > 0:
                            step = (
                                radius % target_comp.spec.size
                                if target_comp.spec.size
                                else 0
                            )
                            new_local = (
                                target_local_idx + direction * step
                            ) % target_comp.spec.size
                            new_global = int(target_comp.indices[new_local])
                    else:  # line
                        step = radius
                        new_local = target_local_idx + direction * step
                        new_local = min(
                            max(new_local, 0), target_comp.spec.size - 1
                        )
                        new_global = int(target_comp.indices[new_local])
                else:
                    # Regular within-component perturbation
                    direction = -1 if self.rng.random() < 0.5 else 1
                    if comp.spec.kind == "ring":
                        if comp.spec.size > 0:
                            step = (
                                radius % comp.spec.size if comp.spec.size else 0
                            )
                            new_local = (
                                local_idx + direction * step
                            ) % comp.spec.size
                            new_global = int(comp.indices[new_local])
                    else:  # line
                        step = radius
                        new_local = local_idx + direction * step
                        new_local = min(max(new_local, 0), comp.spec.size - 1)
                        new_global = int(comp.indices[new_local])
            out.add(new_global)
        return sorted(out)

    def make_similarity(self) -> pd.DataFrame:
        """Compose same- and cross-component similarities into one matrix."""
        mat = np.zeros((self.n, self.n), dtype=float)

        # Same-component similarities
        for comp_idx, comp in enumerate(self._components):
            for i_local, i_global in enumerate(comp.indices):
                for j_local, j_global in enumerate(comp.indices):
                    sim = self._same_component_similarity(
                        comp, i_local, j_local
                    )
                    mat[i_global, j_global] = sim

        # Cross-component similarities
        for i in range(self.n):
            comp_i = int(self._component_lookup[i, 0])
            for j in range(i + 1, self.n):
                comp_j = int(self._component_lookup[j, 0])
                if comp_i == comp_j:
                    continue
                dist = float(np.linalg.norm(self._coords[i] - self._coords[j]))
                sim = self._cross_affinity * math.exp(
                    -(dist**2) / (2 * self._cross_bandwidth**2)
                )
                mat[i, j] = sim
                mat[j, i] = sim

        np.fill_diagonal(mat, 1.0)
        return pd.DataFrame(mat, index=self.labels, columns=self.labels)

    def _same_component_similarity(
        self, comp: _UnionComponent, i_local: int, j_local: int
    ) -> float:
        """Return similarity for labels drawn from the same component."""
        if comp.spec.kind == "ring":
            assert comp.angles is not None
            theta_i = comp.angles[i_local]
            theta_j = comp.angles[j_local]
            base = (1.0 + math.cos(theta_i - theta_j)) / 2.0
            return float(
                self._ring_min_similarity
                + (1.0 - self._ring_min_similarity) * base
            )
        # line component
        if comp.spec.size <= 1:
            return 1.0
        dist = abs(i_local - j_local)
        norm = dist / (comp.spec.size - 1)
        sigma = self._line_bandwidth
        return float(math.exp(-(norm**2) / (2 * sigma**2)))

    def component_ids(self) -> np.ndarray:
        return self._component_lookup[:, 0].astype(int, copy=True)

    def deceptive_embedding(self, gap: float) -> np.ndarray:
        embed = np.zeros((self.n, 3), dtype=float)
        for comp_idx, comp in enumerate(self._components):
            base = comp.base_coords
            if base.shape[1] == 1:
                base_xy = np.column_stack((base[:, 0], np.zeros(base.shape[0])))
            else:
                base_xy = base[:, :2]
            embed[comp.indices, :2] = base_xy
            embed[comp.indices, 2] = comp_idx * gap
        return embed

    def component_paths(self) -> list[tuple[list[int], bool, int]]:
        paths: list[tuple[list[int], bool, int]] = []
        for comp_idx, comp in enumerate(self._components):
            closed = comp.spec.kind == "ring"
            paths.append((comp.indices.tolist(), closed, comp_idx))
        return paths


def build_geometry(
    cfg: SynthConfig, labels: list[str], cross_jump_prob: Optional[float] = None
) -> BaseGeometry:
    """Instantiate the geometry helper according to the configuration.

    The helper hides all geometry-specific logic (sampling, perturbations,
    similarities) from the rest of the Study A pipeline. When ``cfg.geometry``
    is ``"ring"`` the legacy setup is reproduced exactly. When it is
    ``"union"`` a :class:`UnionManifoldGeometry` is returned, parameterised by
    the provided component specs and similarity hyperparameters.
    """

    rng = np.random.default_rng(cfg.seed)
    if cfg.geometry_mode == "ring":
        return RingGeometry(labels=labels, rng=rng)
    if cfg.geometry_mode == "union":
        if not cfg.union_components:
            raise ValueError(
                "Union geometry selected but no union components were provided"
            )

        # Use provided cross_jump_prob or default to first value in list
        effective_cross_jump_prob = (
            cross_jump_prob
            if cross_jump_prob is not None
            else cfg.union_cross_jump_probs[0]
        )

        return UnionManifoldGeometry(
            labels=labels,
            rng=rng,
            components=cfg.union_components,
            component_weights=cfg.union_component_weights,
            line_bandwidth=cfg.union_line_bandwidth,
            cross_affinity=cfg.union_cross_affinity,
            cross_bandwidth=cfg.union_cross_bandwidth,
            ring_min_similarity=cfg.union_ring_min_similarity,
            cross_jump_prob=effective_cross_jump_prob,
        )
    raise ValueError(f"Unsupported geometry mode: {cfg.geometry_mode}")


def parse_union_components(
    raw: Sequence[str],
) -> tuple[UnionComponentSpec, ...]:
    """Parse CLI component specs of the form ``[name=]kind:size[@param=val,...]``.

    Examples
    --------
    ``ring_big=ring:12@radius=2``
        Named ring component with 12 labels and radius scaling of ``2``.
    ``line:4@length=3``
        Unnamed line component of length ``3`` (name auto-generated).

    Returns a tuple of :class:`UnionComponentSpec` instances ready for the
    geometry builder.
    """

    specs: list[UnionComponentSpec] = []
    for idx, token in enumerate(raw):
        piece = token.strip()
        if not piece:
            continue
        name: Optional[str] = None
        body = piece
        colon_pos = piece.find(":")
        if colon_pos == -1:
            raise ValueError(
                f"Union component spec '{piece}' must include kind:size"
            )
        eq_pos = piece.find("=")
        if eq_pos != -1 and eq_pos < colon_pos:
            name_part, body = piece.split("=", 1)
            name = name_part.strip()
            body = body.strip()

        params: dict[str, float] = {}
        if "@" in body:
            base, extras = body.split("@", 1)
            body = base.strip()
            for param_token in extras.split(","):
                if not param_token:
                    continue
                if "=" not in param_token:
                    raise ValueError(
                        f"Invalid union component parameter segment: {param_token}"
                    )
                key, value = param_token.split("=", 1)
                key = key.strip().lower()
                try:
                    params[key] = float(value.strip())
                except ValueError as exc:
                    raise ValueError(
                        f"Union component parameter '{param_token}' is not numeric"
                    ) from exc

        kind_part, size_part = body.split(":", 1)
        kind = kind_part.strip().lower()
        if kind not in {"ring", "line"}:
            raise ValueError(
                f"Unsupported component kind '{kind}' in '{piece}'"
            )
        try:
            size = int(size_part.strip())
        except ValueError as exc:
            raise ValueError(
                f"Union component size must be an integer in '{piece}'"
            ) from exc
        if size <= 0:
            raise ValueError("Union component size must be positive")

        if "radius" in params and kind != "ring":
            raise ValueError(
                "Radius parameter is only valid for ring components"
            )
        if "length" in params and kind != "line":
            raise ValueError(
                "Length parameter is only valid for line components"
            )

        if name is None:
            name = f"{kind}_{idx}"

        spec = UnionComponentSpec(
            name=name,
            kind=kind,  # type: ignore[arg-type]
            size=size,
            radius=params.get("radius"),
            length=params.get("length"),
        )
        specs.append(spec)
    return tuple(specs)


def to_label_names(xs: list[list[int]], labels: list[str]) -> list[list[str]]:
    """Map index-encoded label sets to string names.

    Args:
        xs: List of examples, each a list of integer indices.
        labels: Master list mapping index → name.

    Returns:
        Same structure as ``xs`` but with names.
    """
    return [[labels[i] for i in row] for row in xs]


def eval_hard_metrics(
    trues: list[list[str]], preds: list[list[str]], labels: list[str]
) -> dict[str, float]:
    """Compute sklearn Hard F1 variants for multilabel sets.

    Args:
        trues: Gold labels per example (strings).
        preds: Predicted labels per example (strings).
        labels: Global class order for binarization.

    Returns:
        Dict with keys ``{"hard_micro","hard_macro","hard_samples"}``.
    """
    mlb = MultiLabelBinarizer(classes=labels)
    Y_true = mlb.fit_transform(trues)
    Y_pred = mlb.transform(preds)
    micro = f1_score(Y_true, Y_pred, average="micro", zero_division=0)
    macro = f1_score(Y_true, Y_pred, average="macro", zero_division=0)
    samples = f1_score(Y_true, Y_pred, average="samples", zero_division=0)
    return {"hard_micro": micro, "hard_macro": macro, "hard_samples": samples}


def eval_semantic_metrics(
    trues: list[list[str]],
    preds: list[list[str]],
    S: pd.DataFrame,
) -> dict[str, float]:
    """Compute Semantic F1 variants under a given similarity matrix.

    Args:
        trues: Gold labels per example (strings).
        preds: Predicted labels per example (strings).
        S: Similarity matrix ``[0,1]`` indexed by label names.

    Returns:
        Dict with keys ``{"sem_micro","sem_macro","sem_samples"}``.
    """
    # sample-level mean pointwise semantic F1
    samples = float(semantic_f1_score(trues, preds, S, average="samples"))
    micro = float(semantic_f1_score(trues, preds, S, average="micro"))
    macro = float(semantic_f1_score(trues, preds, S, average="macro"))
    return {
        "sem_micro": micro,
        "sem_macro": macro,
        "sem_samples": samples,
    }


def bootstrap_ci(
    trues: list[list[str]],
    preds_a: list[list[str]],
    preds_b: list[list[str]],
    labels: list[str],
    S: pd.DataFrame,
    metric: Literal[
        "hard_micro",
        "hard_macro",
        "hard_samples",
        "sem_micro",
        "sem_macro",
        "sem_samples",
    ],
    B: int = 200,
    seed: int | None = 1234,
) -> tuple[float, float, float]:
    """Bootstrap the mean gap ``perf_a - perf_b`` for a selected metric with 95% CI.

    Resamples examples with replacement ``B`` times, recomputing the chosen
    metric for each bootstrap sample and returning the mean and percentile CI.

    Args:
        trues: Gold labels per example.
        preds_a: Predictions from system A.
        preds_b: Predictions from system B.
        labels: Class order for hard metrics.
        S: Similarity matrix for semantic metrics.
        metric: Which metric to evaluate (hard/semantic x micro/macro/samples).
        B: Number of bootstrap iterations.
        seed: RNG seed.

    Returns:
        Tuple ``(mean_gap, ci95_lo, ci95_hi)``.
    """
    rng = np.random.default_rng(seed)
    n = len(trues)

    # Prepare encoder for hard F1 only once
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.metrics import f1_score as sk_f1

    mlb = MultiLabelBinarizer(classes=labels)
    # set classes_ while respecting provided class order
    mlb.fit([[]])

    def compute(m: str, T_idx: np.ndarray, P: list[list[str]]):
        T = [trues[i] for i in T_idx]
        Psel = [P[i] for i in T_idx]
        if m.startswith("hard"):
            avg = m.split("_")[1]
            Y_true = mlb.transform(T)
            Y_pred = mlb.transform(Psel)
            return float(sk_f1(Y_true, Y_pred, average=avg, zero_division=0))
        else:
            # semantic_* -> map to sklearn-style average
            avg = m.split("_")[1]
            return float(semantic_f1_score(T, Psel, S, average=avg))

    diffs = []
    base_idx = np.arange(n)
    for _ in range(B):
        idx = rng.choice(base_idx, size=n, replace=True)
        fa = compute(metric, idx, preds_a)
        fb = compute(metric, idx, preds_b)
        diffs.append(fa - fb)
    diffs = np.array(diffs, dtype=float)
    mean = float(diffs.mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return mean, float(lo), float(hi)


def kendall_tau_vs_radius(
    geometry: BaseGeometry,
    k: int,
    p: float,
    radii: list[int],
    S_list: dict[str, pd.DataFrame],
    num_examples: int,
    labels: list[str],
) -> pd.DataFrame:
    """Compute Kendall's tau between radius and metric curves.

    Builds a fixed gold set and, for each radius, evaluates Hard and Semantic
    metrics. Returns one row per metric with tau and p-value vs the radius list.

    Args:
        geometry: Geometry helper for sampling and perturbation.
        k: Gold labels per example.
        p: Per-label perturbation probability.
        radii: Radii to evaluate.
        S_list: Mapping name → similarity matrix for semantic metrics.
        num_examples: Number of examples for the gold set.
        labels: Label names.

    Returns:
        DataFrame with columns ``{"metric","tau","pvalue"}``.
    """
    # Build a common set of trues, and then generate preds for each radius
    gold = [geometry.sample_gold_labels(k) for _ in range(num_examples)]
    gold_names = to_label_names([sorted(set(g)) for g in gold], labels)

    metrics_curve: dict[str, list[float]] = {}
    for r in radii:
        preds = [geometry.perturb_labels(g, p=p, radius=r) for g in gold]
        preds_names = to_label_names([sorted(set(x)) for x in preds], labels)

        # hard F1s
        hard_vals = eval_hard_metrics(gold_names, preds_names, labels)
        for k_ in hard_vals:
            metrics_curve.setdefault(k_, []).append(hard_vals[k_])

        # semantic for each S
        for name, S in S_list.items():
            sem_vals = eval_semantic_metrics(gold_names, preds_names, S)
            for k_, v in sem_vals.items():
                metrics_curve.setdefault(f"{k_}:{name}", []).append(v)

    # Now compute tau for each curve against radius list
    df_rows = []
    for mname, vals in metrics_curve.items():
        tau, pval = kendalltau(radii, vals)
        df_rows.append(
            {"metric": mname, "tau": float(tau), "pvalue": float(pval)}
        )

    return pd.DataFrame(df_rows)


def metrics_vs_radius(
    geometry: BaseGeometry,
    k: int,
    p: float,
    radii: list[int],
    S_list: dict[str, pd.DataFrame],
    num_examples: int,
    labels: list[str],
) -> pd.DataFrame:
    """Compute per-radius metric curves for Hard/Semantic F1.

    Produces rows with columns ``{"radius","metric","value"}``. Semantic
    metric names are namespaced as ``"sem_<avg>:<Sname>"``.

    Args:
        geometry: Geometry helper for sampling and perturbation.
        k: Gold labels per example.
        p: Per-label perturbation probability.
        radii: Radii to evaluate.
        S_list: Mapping name → similarity matrix for semantic metrics.
        num_examples: Number of examples for the gold set.
        labels: Label names.

    Returns:
        Long-form DataFrame of metric values per radius.
    """
    gold = [geometry.sample_gold_labels(k) for _ in range(num_examples)]
    gold_names = to_label_names([sorted(set(g)) for g in gold], labels)

    rows = []
    for r in radii:
        preds = [geometry.perturb_labels(g, p=p, radius=r) for g in gold]
        preds_names = to_label_names([sorted(set(x)) for x in preds], labels)

        hard_vals = eval_hard_metrics(gold_names, preds_names, labels)
        for mname, val in hard_vals.items():
            rows.append({"radius": r, "metric": mname, "value": float(val)})

        for name, S in S_list.items():
            sem_vals = eval_semantic_metrics(gold_names, preds_names, S)
            for mname, val in sem_vals.items():
                rows.append(
                    {
                        "radius": r,
                        "metric": f"{mname}:{name}",
                        "value": float(val),
                    }
                )
    return pd.DataFrame(rows)


def run_one_config(
    cfg: SynthConfig,
    outdir: str,
):
    """Run Study A.1 for a single configuration and write artifacts.

    Side effects
    - Writes CSVs: summary, Kendall tau, bootstrap results.
    - Writes figures: Fig A1/A2 (PNG/PDF).
    - Writes LaTeX table: Tab A1 with bootstrap CIs.
    - Logs to ``<outdir>/log.txt``.

    Args:
        cfg: Experiment configuration.
        outdir: Output directory (created if missing).
    """
    os.makedirs(outdir, exist_ok=True)

    # set up file logger
    logger = logging.getLogger("study_a")
    logger.setLevel(logging.INFO)
    # avoid duplicate handlers if re-run in interactive sessions
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        fh = logging.FileHandler(os.path.join(outdir, "log.txt"), mode="w")
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.info("Starting Study A with config: %s", cfg)
    labels = make_labels(cfg.num_labels)
    geometry = build_geometry(cfg, labels)
    labels = geometry.labels
    plot_label_space(geometry, outdir, cfg.deceptive_ring_gap)
    S_ideal = geometry.make_similarity()
    S_deceptive = make_deceptive_similarity(
        geometry, gap=cfg.deceptive_ring_gap
    )
    print("Initial ideal similarity matrix:\n", S_ideal)
    print("Initial deceptive similarity matrix:\n", S_deceptive)
    prev_ideal = S_ideal.copy()
    prev_deceptive = S_deceptive.copy()
    if cfg.noise_mode == "uniform":
        U = make_uniform_noise_U(labels, offdiag=0.5)
    else:
        U = identity_similarity(labels)
    logger.info(
        "Built geometry mode=%s with %d labels; similarity matrices: ideal=%s, deceptive=%s, noise_mode=%s",
        cfg.geometry_mode,
        len(labels),
        S_ideal.shape,
        S_deceptive.shape,
        cfg.noise_mode,
    )

    # Placeholders for summaries
    rows_summary = []
    rows_tau = []
    rows_boot = []

    # Iterate k, p, and cross_jump_prob; generate a gold set for each
    for k in cfg.k_values:
        for p in cfg.p_values:
            for cross_jump_prob in cfg.union_cross_jump_probs:
                logger.info(
                    "Begin (k=%s, p=%.2f, cross_jump_prob=%.2f): generating gold set of %s examples",
                    k,
                    p,
                    cross_jump_prob,
                    cfg.num_examples,
                )

                # Create geometry with specific cross_jump_prob
                geometry = build_geometry(
                    cfg, labels, cross_jump_prob=cross_jump_prob
                )

                # Pre-generate a gold dataset
                gold_idx = [
                    geometry.sample_gold_labels(k)
                    for _ in range(cfg.num_examples)
                ]
                gold_idx = [sorted(set(g)) for g in gold_idx]
                gold = geometry.to_label_names(gold_idx)

                # Create similarity matrices for this geometry
                S_ideal = geometry.make_similarity()
                S_deceptive = make_deceptive_similarity(
                    geometry, gap=cfg.deceptive_ring_gap
                )

                ideal_changed = not np.allclose(
                    S_ideal.values, prev_ideal.values
                )
                deceptive_changed = not np.allclose(
                    S_deceptive.values, prev_deceptive.values
                )

                if ideal_changed:
                    print(
                        f"Ideal similarity matrix (k={k}, p={p:.2f}, cross_jump_prob={cross_jump_prob:.2f}):\n",
                        S_ideal,
                    )
                    prev_ideal = S_ideal.copy()

                if deceptive_changed:
                    print(
                        f"Deceptive similarity matrix (k={k}, p={p:.2f}, cross_jump_prob={cross_jump_prob:.2f}):\n",
                        S_deceptive,
                    )
                    prev_deceptive = S_deceptive.copy()

                # Define S variants
                S_perm = permuted_rows_similarity(S_ideal, seed=cfg.seed)
                S_alpha_list = {
                    f"a={a:.2f}": mix_similarity(S_ideal, U, a)
                    for a in cfg.alphas
                }
                S_variants: dict[str, pd.DataFrame] = {
                    "ideal": S_ideal,
                    "permuted": S_perm,
                    "deceptive": S_deceptive,
                    **S_alpha_list,
                }
                logger.info(
                    "Prepared %d S variants for (k=%s, p=%.2f, cross_jump_prob=%.2f)",
                    len(S_variants),
                    k,
                    p,
                    cross_jump_prob,
                )

                # Evaluate kendall tau vs radius curves for this (k,p,cross_jump_prob)
                radii_union = sorted(set(cfg.near_radii + cfg.far_radii))
                df_tau = kendall_tau_vs_radius(
                    geometry,
                    k,
                    p,
                    radii_union,
                    {k_: v for k_, v in S_variants.items() if k_ != "identity"},
                    cfg.num_examples,
                    labels,
                )
                df_tau["k"] = k
                df_tau["p"] = p
                df_tau["cross_jump_prob"] = cross_jump_prob
                rows_tau.append(df_tau)
                logger.info(
                    "Computed Kendall tau for (k=%s, p=%.2f, cross_jump_prob=%.2f) across %d radii → %d rows",
                    k,
                    p,
                    cross_jump_prob,
                    len(radii_union),
                    len(df_tau),
                )

                # For tab A1: compare predictor A (near) vs B (far)
                for r_near in cfg.near_radii:
                    for r_far in cfg.far_radii:
                        logger.info(
                            "Evaluating near=%s vs far=%s (k=%s, p=%.2f, cross_jump_prob=%.2f) across %d S variants",
                            r_near,
                            r_far,
                            k,
                            p,
                            cross_jump_prob,
                            len(S_variants),
                        )
                        # Simulate A and B on the same gold
                        preds_a_idx = [
                            geometry.perturb_labels(g, p=p, radius=r_near)
                            for g in gold_idx
                        ]
                        preds_b_idx = [
                            geometry.perturb_labels(g, p=p, radius=r_far)
                            for g in gold_idx
                        ]
                        preds_a = geometry.to_label_names(
                            [sorted(set(x)) for x in preds_a_idx]
                        )
                        preds_b = geometry.to_label_names(
                            [sorted(set(x)) for x in preds_b_idx]
                        )

                        # Evaluate hard and semantic metrics under all S
                        hard_a = eval_hard_metrics(gold, preds_a, labels)
                        hard_b = eval_hard_metrics(gold, preds_b, labels)

                        for name_s, S in S_variants.items():
                            sem_a = eval_semantic_metrics(gold, preds_a, S)
                            sem_b = eval_semantic_metrics(gold, preds_b, S)

                            row = dict(
                                k=k,
                                p=p,
                                cross_jump_prob=cross_jump_prob,
                                r_near=r_near,
                                r_far=r_far,
                                S=name_s,
                            )
                            # merge metrics
                            for mname, aval in hard_a.items():
                                row[f"{mname}_A"] = aval
                                row[f"{mname}_B"] = hard_b[mname]
                                row[f"{mname}_gap"] = aval - hard_b[mname]
                            for mname, aval in sem_a.items():
                                row[f"{mname}_A"] = aval
                                row[f"{mname}_B"] = sem_b[mname]
                                row[f"{mname}_gap"] = aval - sem_b[mname]
                            rows_summary.append(row)

                            # Bootstrap CIs for gaps
                            for m in [
                                "hard_micro",
                                "hard_macro",
                                "hard_samples",
                                "sem_micro",
                                "sem_macro",
                                "sem_samples",
                            ]:
                                # note: B might be adjusted in future; keeping logged value
                                logger.info(
                                    "Bootstrapping CI for metric=%s (B=%d) [near=%s, far=%s, S=%s]",
                                    m,
                                    cfg.bootstrap,
                                    r_near,
                                    r_far,
                                    name_s,
                                )
                                mean, lo, hi = bootstrap_ci(
                                    gold,
                                    preds_a,
                                    preds_b,
                                    labels,
                                    S,
                                    metric=m,
                                    B=cfg.bootstrap,
                                    seed=cfg.seed,
                                )
                                if p != 0:
                                    rows_boot.append(
                                        dict(
                                            k=k,
                                            p=p,
                                            cross_jump_prob=cross_jump_prob,
                                            r_near=r_near,
                                            r_far=r_far,
                                            S=name_s,
                                            metric=m,
                                            mean_gap=mean,
                                            ci95_lo=lo,
                                            ci95_hi=hi,
                                        )
                                    )

    # Write outputs
    ts = int(time.time())
    df_summary = pd.DataFrame(rows_summary)
    df_tau_all = (
        pd.concat(rows_tau, ignore_index=True) if rows_tau else pd.DataFrame()
    )
    df_boot = pd.DataFrame(rows_boot)

    path_summary = os.path.join(outdir, f"studyA_summary_{ts}.csv")
    path_tau = os.path.join(outdir, f"studyA_kendall_tau_{ts}.csv")
    path_boot = os.path.join(outdir, f"studyA_bootstrap_{ts}.csv")
    df_summary.to_csv(path_summary, index=False)
    df_tau_all.to_csv(path_tau, index=False)
    df_boot.to_csv(path_boot, index=False)

    logger.info(
        "Output shapes — summary=%s, kendall_tau=%s, bootstrap=%s",
        df_summary.shape,
        df_tau_all.shape,
        df_boot.shape,
    )
    logger.info("Wrote: %s", path_summary)
    logger.info("Wrote: %s", path_tau)
    logger.info("Wrote: %s", path_boot)
    # keep prints for quick terminal visibility
    print(f"Wrote: {path_summary}")
    print(f"Wrote: {path_tau}")
    print(f"Wrote: {path_boot}")

    # --------------------------
    # Figures and Tables outputs
    # --------------------------
    try:
        # Use external plotting functions
        def _sample_gold_labels(_: int, k_val: int) -> list[int]:
            return geometry.sample_gold_labels(k_val)

        def _perturb_labels(
            gold_indices: Iterable[int],
            _: int,
            *,
            p: float,
            radius: int,
        ) -> list[int]:
            return geometry.perturb_labels(gold_indices, p=p, radius=radius)

        def _to_label_names(
            xs: list[list[int]],
            _: list[str],
        ) -> list[list[str]]:
            return geometry.to_label_names(xs)

        helpers = dict(
            sample_gold_labels=_sample_gold_labels,
            to_label_names=_to_label_names,
            perturb_labels=_perturb_labels,
            semantic_f1_score=semantic_f1_score,
            f1_score=f1_score,
            MultiLabelBinarizer=MultiLabelBinarizer,
        )
        study_a3_plots.plot_metric_vs_radius_grids(
            cfg=cfg,
            labels=labels,
            S_ideal=S_ideal,
            S_deceptive=S_deceptive,
            helpers=helpers,
            outdir=outdir,
            ts=ts,
            logger=logger,
        )
        # A9: Cross-jump-prob grids (averaging across p values)
        study_a3_plots.plot_cross_jump_prob_grids(
            df_summary=df_summary,
            outdir=outdir,
            ts=ts,
            logger=logger,
        )
        study_a3_plots.plot_kendall_tau_bars(
            df_tau_all=df_tau_all,
            outdir=outdir,
            ts=ts,
            logger=logger,
        )
        # Faceted grids: negative Kendall tau per (k,p) for each averaging metric
        study_a3_plots.plot_kendall_tau_grids(
            df_tau_all=df_tau_all,
            outdir=outdir,
            ts=ts,
            logger=logger,
        )
        # A2 (lines): per-k line panels over p
        study_a3_plots.plot_kendall_tau_linepanels(
            df_tau_all=df_tau_all,
            outdir=outdir,
            ts=ts,
            logger=logger,
        )
        # A2 Table: negative Kendall tau values by k×variant with p columns
        study_a3_plots.write_kendall_tau_table(
            df_tau_all=df_tau_all,
            outdir=outdir,
            ts=ts,
            logger=logger,
        )

        # Replace the massive table with useful visualizations and summary tables

        # Heatmap visualizations of bootstrap results
        study_a3_plots.plot_bootstrap_heatmaps(
            df_boot=df_boot,
            outdir=outdir,
            ts=ts,
            logger=logger,
        )

        # Summary charts showing key findings
        study_a3_plots.plot_bootstrap_summary_charts(
            df_boot=df_boot,
            outdir=outdir,
            ts=ts,
            logger=logger,
        )

        # Confidence interval plots for detailed analysis
        study_a3_plots.plot_bootstrap_confidence_intervals(
            df_boot=df_boot,
            outdir=outdir,
            ts=ts,
            logger=logger,
        )

        # Concise summary table (replaces the original massive table)
        study_a3_plots.write_bootstrap_table_summary(
            df_boot=df_boot,
            outdir=outdir,
            ts=ts,
            logger=logger,
        )

        # Optional: Still generate the original full table for appendix/reference
        # (Commented out by default due to excessive length)
        # study_a1_plots.write_bootstrap_table(
        #     df_boot=df_boot,
        #     outdir=outdir,
        #     ts=ts,
        #     logger=logger,
        # )
    except Exception as e:  # pragma: no cover - plotting is best-effort
        # Do not crash the run if plotting fails; log and proceed
        logging.getLogger("study_a").exception(
            "Non-fatal error while producing figures/tables: %s", e
        )


def parse_args() -> SynthConfig:
    """Parse CLI arguments into a ``SynthConfig``.

    Returns:
        A populated ``SynthConfig`` reflecting command-line options.
    """
    ap = argparse.ArgumentParser(
        description="Study A synthetic evaluation for Semantic F1"
    )
    ap.add_argument(
        "--num-labels",
        type=int,
        default=24,
        help="Number of labels to arrange on the ring (unit circle). Controls the label space size.",
    )
    ap.add_argument(
        "--num-examples",
        type=int,
        default=10000,
        help="Number of synthetic examples to generate per configuration.",
    )
    ap.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="List of numbers of labels per example to sample for gold sets.",
    )
    ap.add_argument(
        "--p",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="List of probabilities to perturb each gold label (controls prediction noise).",
    )
    ap.add_argument(
        "--near-radii",
        type=int,
        nargs="+",
        default=[1, 2],
        help="List of radii for near-miss predictor A (distance for label replacement).",
    )
    ap.add_argument(
        "--far-radii",
        type=int,
        nargs="+",
        default=[3, 4],
        help="List of radii for far-miss predictor B (distance for label replacement).",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        nargs="+",
        default=[1.0, 0.75, 0.5, 0.25, 0.0],
        help="List of mixture weights for similarity matrix sensitivity sweeps (alpha in S_alpha = alpha*S + (1-alpha)*U).",
    )
    ap.add_argument(
        "--seed", type=int, default=123, help="Random seed for reproducibility."
    )
    ap.add_argument(
        "--noise-mode",
        choices=["identity", "uniform"],
        default="identity",
        help="Type of noise matrix U to use in similarity mixing: 'identity' for identity matrix, 'uniform' for constant off-diagonal.",
    )
    ap.add_argument(
        "--geometry",
        choices=["ring", "union"],
        default="ring",
        help="Label-space geometry: 'ring' for the classic circle, 'union' for a union of manifolds.",
    )
    ap.add_argument(
        "--union-components",
        type=str,
        nargs="+",
        help="Union geometry components in the form [name=]kind:size[@param=val,...]",
    )
    ap.add_argument(
        "--union-component-weights",
        type=float,
        nargs="+",
        help="Optional sampling weights (will be normalized) per union component.",
    )
    ap.add_argument(
        "--union-line-bandwidth",
        type=float,
        default=0.35,
        help="Kernel bandwidth for line components (normalized distance).",
    )
    ap.add_argument(
        "--union-cross-affinity",
        type=float,
        default=0.2,
        help="Scaling factor for cross-component similarities in union geometry.",
    )
    ap.add_argument(
        "--union-cross-bandwidth",
        type=float,
        default=1.5,
        help="Bandwidth controlling cross-component similarity decay.",
    )
    ap.add_argument(
        "--union-cross-jump-prob",
        type=float,
        nargs="+",
        default=[0.0],
        help="Conditional probability that a perturbed label jumps to a different component. Multiple values allowed.",
    )
    ap.add_argument(
        "--union-ring-min-sim",
        type=float,
        default=0.3,
        help="Minimum similarity for any pair on the same ring component.",
    )
    ap.add_argument(
        "--deceptive-ring-gap",
        type=float,
        default=1.0,
        help="Vertical gap between parallel manifolds when building the deceptive similarity matrix.",
    )
    ap.add_argument(
        "--bootstrap",
        type=int,
        default=200,
        help="Number of bootstrap iterations for confidence interval estimation per metric.",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default="logs/semantic_f1/study_a",
        help="Output directory for CSVs, figures, and tables.",
    )
    args = ap.parse_args()

    num_labels = args.num_labels
    ring_min_sim = float(args.union_ring_min_sim)
    if not 0.0 <= ring_min_sim < 1.0:
        ap.error("--union-ring-min-sim must lie in [0,1)")
    cross_jump_probs = [float(x) for x in args.union_cross_jump_prob]
    for cross_jump_prob in cross_jump_probs:
        if not 0.0 <= cross_jump_prob <= 1.0:
            ap.error("--union-cross-jump-prob must lie in [0,1]")
    deceptive_gap = float(args.deceptive_ring_gap)
    if deceptive_gap <= 0.0:
        ap.error("--deceptive-ring-gap must be positive")
    union_components: tuple[UnionComponentSpec, ...] = tuple()
    union_component_weights: Optional[tuple[float, ...]] = None

    if args.geometry == "union":
        if not args.union_components:
            ap.error("--union-components is required when --geometry union")
        try:
            union_components = parse_union_components(args.union_components)
        except ValueError as exc:  # pragma: no cover - argument validation
            ap.error(str(exc))
        total = sum(spec.size for spec in union_components)
        if num_labels != total:
            ap.error(
                f"--num-labels must equal the total size of union components ({total}), got {num_labels}"
            )
        num_labels = total
        if args.union_component_weights:
            if len(args.union_component_weights) != len(union_components):
                ap.error(
                    "Number of --union-component-weights must match --union-components"
                )
            weights = tuple(float(w) for w in args.union_component_weights)
            if any(w <= 0 for w in weights):
                ap.error("Union component weights must be strictly positive")
            total_w = sum(weights)
            if total_w == 0:
                ap.error("Union component weights sum to zero")
            union_component_weights = tuple(w / total_w for w in weights)
        if args.union_cross_affinity >= ring_min_sim:
            ap.error(
                "--union-cross-affinity must be strictly lower than "
                "--union-ring-min-sim to keep intra-ring similarities higher"
            )
        if (
            any(cjp > 0 for cjp in cross_jump_probs)
            and len(union_components) <= 1
        ):
            ap.error("--union-cross-jump-prob requires at least two components")
    else:
        if args.union_components or args.union_component_weights:
            ap.error(
                "Union component arguments provided but --geometry is not set to 'union'"
            )
        if any(cjp > 0 for cjp in cross_jump_probs):
            ap.error(
                "--union-cross-jump-prob is only valid when --geometry union"
            )

    return SynthConfig(
        num_labels=num_labels,
        num_examples=args.num_examples,
        k_values=args.k,
        p_values=args.p,
        near_radii=args.near_radii,
        far_radii=args.far_radii,
        alphas=args.alpha,
        seed=args.seed,
        noise_mode=args.noise_mode,
        outdir=args.outdir,
        bootstrap=args.bootstrap,
        geometry_mode=args.geometry,
        union_components=union_components,
        union_component_weights=union_component_weights,
        union_line_bandwidth=args.union_line_bandwidth,
        union_cross_affinity=args.union_cross_affinity,
        union_cross_bandwidth=args.union_cross_bandwidth,
        union_ring_min_similarity=ring_min_sim,
        union_cross_jump_probs=cross_jump_probs,
        deceptive_ring_gap=deceptive_gap,
    )


def main():
    """Entry point: parse args and run Study A.1."""
    cfg = parse_args()
    os.makedirs(cfg.outdir, exist_ok=True)
    run_one_config(cfg, outdir=cfg.outdir)


if __name__ == "__main__":
    main()
