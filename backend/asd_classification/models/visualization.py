"""
Publication-Grade Visual Diagnostics Suite for BrainNetCNN Explainability.

Generates:
1. 3D Glass Brain ROI Plots: Visualizes salient brain regions in stereotaxic MNI space
   using Nilearn glass brain projection or multi-view orthographic fallback.
2. Yeo 7 Resting-State Network Enrichment Charts: Bar charts quantifying diagnostic biomarker
   concentration across canonical brain networks, highlighting the Default Mode Network (DMN)
   and Salience Network with respect to uniform expectation baselines.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Defensive Nilearn import
try:
    from nilearn import plotting  # type: ignore[import-untyped]
    HAS_NILEARN = True
except ImportError:
    HAS_NILEARN = False


def plot_glass_brain_diagnostics(
    roi_coords: np.ndarray,
    saliency_weights: np.ndarray,
    top_k: Optional[int] = 20,
    title: str = "BrainNetCNN Salient ROIs (CC200 MNI Space)",
    output_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Generate a publication-grade 3D glass brain ROI plot in MNI space.

    Args:
        roi_coords: Array of shape (N, 3) representing MNI centroids in mm.
        saliency_weights: Array of shape (N,) representing non-negative attribution weights.
        top_k: If specified, plots only the top k most salient regions.
        title: Plot header title.
        output_path: Optional destination file path (e.g. PNG/PDF).
        dpi: Resolution for saved raster figures.

    Returns:
        Matplotlib Figure object.

    Raises:
        ValueError: If length of roi_coords does not match length of saliency_weights.
    """
    coords = np.asarray(roi_coords, dtype=np.float32)
    weights = np.asarray(saliency_weights, dtype=np.float32).flatten()

    if len(coords) != len(weights):
        raise ValueError(f"Coordinate count {len(coords)} does not match weights count {len(weights)}.")

    # Filter to top_k if specified
    if top_k is not None and 0 < top_k < len(weights):
        top_indices = np.argsort(weights)[-top_k:]
        plot_coords = coords[top_indices]
        plot_weights = weights[top_indices]
    else:
        plot_coords = coords
        plot_weights = weights

    # Scale marker sizes proportionally to attribution weight
    min_w, max_w = float(np.min(plot_weights)), float(np.max(plot_weights))
    if max_w > min_w:
        normalized_weights = (plot_weights - min_w) / (max_w - min_w)
    else:
        normalized_weights = np.ones_like(plot_weights)

    node_sizes = 20.0 + normalized_weights * 100.0

    fig = None
    if HAS_NILEARN:
        try:
            fig, ax = plt.subplots(figsize=(10, 4), dpi=dpi)
            display = plotting.plot_markers(
                node_values=plot_weights,
                node_coords=plot_coords,
                node_size=node_sizes,
                node_cmap="hot",
                title=title,
                figure=fig,
                axes=ax,
                colorbar=True,
            )
        except Exception:
            # Fallback to orthographic multi-panel scatter if nilearn plotting raises
            fig = None

    if fig is None:
        # High-quality Matplotlib 3-panel orthographic projection (Axial, Coronal, Sagittal)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=dpi)
        fig.suptitle(title, fontsize=12, fontweight="bold")

        # 1. Axial View (X vs Y, looking top-down)
        scatter_ax = axes[0].scatter(
            plot_coords[:, 0],
            plot_coords[:, 1],
            c=plot_weights,
            s=node_sizes,
            cmap="hot",
            edgecolors="black",
            linewidth=0.5,
            alpha=0.85,
        )
        axes[0].set_title("Axial View (X-Y)")
        axes[0].set_xlabel("Left (-) / Right (+) [mm]")
        axes[0].set_ylabel("Posterior (-) / Anterior (+) [mm]")
        axes[0].grid(True, linestyle=":", alpha=0.5)

        # 2. Coronal View (X vs Z, looking from front)
        axes[1].scatter(
            plot_coords[:, 0],
            plot_coords[:, 2],
            c=plot_weights,
            s=node_sizes,
            cmap="hot",
            edgecolors="black",
            linewidth=0.5,
            alpha=0.85,
        )
        axes[1].set_title("Coronal View (X-Z)")
        axes[1].set_xlabel("Left (-) / Right (+) [mm]")
        axes[1].set_ylabel("Inferior (-) / Superior (+) [mm]")
        axes[1].grid(True, linestyle=":", alpha=0.5)

        # 3. Sagittal View (Y vs Z, looking from side)
        axes[2].scatter(
            plot_coords[:, 1],
            plot_coords[:, 2],
            c=plot_weights,
            s=node_sizes,
            cmap="hot",
            edgecolors="black",
            linewidth=0.5,
            alpha=0.85,
        )
        axes[2].set_title("Sagittal View (Y-Z)")
        axes[2].set_xlabel("Posterior (-) / Anterior (+) [mm]")
        axes[2].set_ylabel("Inferior (-) / Superior (+) [mm]")
        axes[2].grid(True, linestyle=":", alpha=0.5)

        # Colorbar
        cbar = fig.colorbar(scatter_ax, ax=axes, orientation="vertical", fraction=0.02, pad=0.03)
        cbar.set_label("Grad-CAM Attribution Weight", fontsize=10)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")

    return fig


def plot_network_enrichment(
    enrichment_scores: Dict[str, float],
    cohort_scores: Optional[Dict[str, float]] = None,
    title: str = "Canonical Yeo 7 Resting-State Network Enrichment",
    output_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Generate publication-grade bar chart for Yeo 7 resting-state network enrichment.

    Emphasizes Default Mode Network (DMN) and Salience / Ventral Attention Network,
    and includes reference benchmark for uniform expectation (1/7 = 14.3%).

    Args:
        enrichment_scores: Dictionary of subject scores mapping network name to float in [0, 1].
        cohort_scores: Optional cohort-level mean scores for visual comparison.
        title: Header text for chart.
        output_path: Optional file destination for saving figure.
        dpi: Output image resolution.

    Returns:
        Matplotlib Figure object.
    """
    # Ordered networks
    networks = list(enrichment_scores.keys())
    scores = [enrichment_scores[net] for net in networks]
    y_pos = np.arange(len(networks))

    fig, ax = plt.subplots(figsize=(9, 5), dpi=dpi)

    # Color palette highlighting DMN (deep coral) and Salience (teal)
    bar_colors = []
    for net in networks:
        if "Default" in net:
            bar_colors.append("#D95F02")  # Deep coral / amber for DMN
        elif "Salience" in net or "Ventral" in net:
            bar_colors.append("#1B9E77")  # Deep teal for Salience
        elif "Control" in net or "Frontoparietal" in net:
            bar_colors.append("#7570B3")  # Slate purple for FPCN
        else:
            bar_colors.append("#4A6572")  # Neutral slate for remaining systems

    if cohort_scores is not None:
        bar_height = 0.38
        cohort_vals = [cohort_scores.get(net, 0.0) for net in networks]

        bars_sub = ax.barh(y_pos + bar_height / 2, scores, height=bar_height, color=bar_colors, alpha=0.9, label="Subject")
        bars_coh = ax.barh(y_pos - bar_height / 2, cohort_vals, height=bar_height, color="#A0AAB2", alpha=0.7, label="Cohort Mean")

        # Value annotations
        for bar in bars_sub:
            w = bar.get_width()
            ax.text(w + 0.005, bar.get_y() + bar.get_height() / 2, f"{w*100:.1f}%", va="center", ha="left", fontsize=8)

        ax.legend(loc="lower right", frameon=True)
    else:
        bar_height = 0.6
        bars = ax.barh(y_pos, scores, height=bar_height, color=bar_colors, alpha=0.9)

        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.005, bar.get_y() + bar.get_height() / 2, f"{w*100:.1f}%", va="center", ha="left", fontsize=9, fontweight="semibold")

    # Reference uniform expectation line (1/7 ~= 14.3%)
    uniform_ref = 1.0 / len(networks)
    ax.axvline(
        uniform_ref,
        color="#E74C3C",
        linestyle="--",
        linewidth=1.2,
        label=f"Uniform Baseline ({uniform_ref*100:.1f}%)",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(networks, fontsize=9, fontweight="medium")
    ax.invert_yaxis()  # Top-down order

    ax.set_xlabel("Normalized Saliency Enrichment Score (E_net)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)

    # Format x-axis as percentage
    max_val = max(scores + ([max(cohort_scores.values())] if cohort_scores else [0.0]))
    ax.set_xlim(0.0, max(0.35, max_val * 1.25))
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0))

    ax.grid(axis="x", linestyle=":", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if cohort_scores is None:
        ax.legend(loc="lower right", frameon=True)

    plt.tight_layout()

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")

    return fig
