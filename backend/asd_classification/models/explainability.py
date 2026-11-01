"""
Neurobiological Explainability Engine and Quantitative Verification Suite for BrainNetCNN.

Ties together:
1. Grad-CAM on Edge-to-Node (E2N) layer.
2. Static CC200 Atlas mapping to canonical Yeo 7 resting-state functional networks.
3. Normalized Network Enrichment Score (E_net) quantification for DMN and Salience networks.
4. Quantitative graph perturbation fidelity tests (Fidelity+ and Fidelity-).
5. Publication-grade visual diagnostics (3D glass brain ROI projections and enrichment charts).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np

from asd_classification.models.atlas import CC200YeoAtlas, get_cc200_yeo_atlas
from asd_classification.models.fidelity import FidelityEvaluator
from asd_classification.models.gradcam import BrainNetGradCAM
from asd_classification.models.visualization import (
    plot_glass_brain_diagnostics,
    plot_network_enrichment,
)


class NeuroExplainabilitySuite:
    """
    Unified neurobiological explainability and fidelity verification suite.

    Enables clinicians and researchers to interpret BrainNetCNN decisions
    grounded in canonical neuroanatomy (Yeo 7 networks) backed by empirical
    attribution faithfulness benchmarks.
    """

    def __init__(
        self,
        model: Any,
        atlas: Optional[CC200YeoAtlas] = None,
        target_layer: str = "e2n",
    ) -> None:
        self.model = model
        self.atlas = atlas or get_cc200_yeo_atlas()
        self.target_layer = target_layer
        self.gradcam = BrainNetGradCAM(model, target_layer=target_layer)
        self.fidelity_evaluator = FidelityEvaluator(model)

    def explain_subject(
        self,
        fc_matrix: np.ndarray,
        top_k: int = 10,
        target_class: int = 1,
        compute_fidelity: bool = True,
        generate_plots: bool = False,
        plot_output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Generate structured explainability report for a single subject FC matrix.

        Args:
            fc_matrix: Functional connectivity matrix (200, 200).
            top_k: Number of most salient ROIs to return.
            target_class: Target diagnostic class index (1 = ASD, 0 = TD).
            compute_fidelity: Whether to execute quantitative Fidelity+ and Fidelity- perturbation tests.
            generate_plots: Whether to render publication-grade glass brain and enrichment figures.
            plot_output_dir: Optional filesystem directory to save generated plots.

        Returns:
            Structured dictionary matching Issue #29 and Issue #32 specifications:
            {
                "salient_rois": List[Dict[str, Any]],
                "network_enrichment": Dict[str, float],
                "fidelity": Dict[str, float],
                "figures": Optional[Dict[str, str]],
            }

        Raises:
            ValueError: If fc_matrix has incompatible dimensions.
        """
        # 1. Extract Grad-CAM ROI attribution weights
        saliency = self.gradcam.attribute(fc_matrix, target_class=target_class)
        saliency = np.asarray(saliency, dtype=np.float32).flatten()

        # 2. Rank ROIs in descending order of saliency
        sorted_indices = np.argsort(saliency)[::-1]
        top_indices = sorted_indices[:top_k]

        salient_rois: List[Dict[str, Any]] = []
        for idx in top_indices:
            roi_meta = self.atlas.get_roi(int(idx))
            salient_rois.append({
                "roi_id": int(idx),
                "saliency_weight": float(saliency[idx]),
                "yeo_network": roi_meta["yeo_network"],
                "mni_coords": roi_meta["mni_coords"],
                "label": roi_meta["label"],
            })

        # 3. Compute normalized Yeo 7 Network Enrichment Scores
        network_enrichment = self.atlas.compute_network_enrichment(saliency)

        # 4. Quantitative Fidelity perturbation verification
        fidelity_metrics: Dict[str, float] = {}
        if compute_fidelity:
            fidelity_metrics = self.fidelity_evaluator.evaluate(
                fc_matrix=fc_matrix,
                saliency_weights=saliency,
                target_class=target_class,
                ratios=(0.1, 0.2),
            )

        report: Dict[str, Any] = {
            "salient_rois": salient_rois,
            "network_enrichment": network_enrichment,
            "fidelity": fidelity_metrics,
        }

        # 5. Publication-grade Visual Diagnostics
        if generate_plots:
            figures_dict: Dict[str, str] = {}
            out_dir = Path(plot_output_dir) if plot_output_dir else Path("artifacts/explainability")
            out_dir.mkdir(parents=True, exist_ok=True)

            glass_brain_path = out_dir / "glass_brain_saliency.png"
            enrichment_path = out_dir / "network_enrichment.png"

            plot_glass_brain_diagnostics(
                roi_coords=self.atlas.get_coords(),
                saliency_weights=saliency,
                top_k=top_k,
                title=f"BrainNetCNN Salient ROIs (Class {target_class})",
                output_path=glass_brain_path,
            )
            figures_dict["glass_brain"] = str(glass_brain_path.resolve())

            plot_network_enrichment(
                enrichment_scores=network_enrichment,
                title=f"Yeo 7 Network Enrichment (Class {target_class})",
                output_path=enrichment_path,
            )
            figures_dict["network_enrichment"] = str(enrichment_path.resolve())

            report["figures"] = figures_dict

        return report

    def explain_cohort(
        self,
        cohort_fc: np.ndarray,
        top_k: int = 10,
        target_class: int = 1,
        compute_fidelity: bool = True,
    ) -> Dict[str, Any]:
        """
        Aggregate consensus neurobiological biomarkers across a cohort of subjects.

        Args:
            cohort_fc: Array of shape (M, 200, 200).
            top_k: Number of consensus ROIs to return.
            target_class: Target class for attribution.
            compute_fidelity: Whether to evaluate cohort-wide fidelity metrics.

        Returns:
            Dictionary containing 'cohort_salient_rois', 'cohort_network_enrichment',
            and 'cohort_fidelity'.

        Raises:
            ValueError: If cohort_fc array is empty.
        """
        n_subjects = len(cohort_fc)
        if n_subjects == 0:
            raise ValueError("Cohort array cannot be empty.")

        # Compute attributions across cohort
        cohort_saliency = np.zeros((n_subjects, 200), dtype=np.float32)
        for i in range(n_subjects):
            cohort_saliency[i] = self.gradcam.attribute(cohort_fc[i], target_class=target_class)

        # Average attribution per ROI
        mean_saliency = np.mean(cohort_saliency, axis=0)
        sorted_indices = np.argsort(mean_saliency)[::-1]
        top_indices = sorted_indices[:top_k]

        cohort_salient_rois: List[Dict[str, Any]] = []
        for idx in top_indices:
            roi_meta = self.atlas.get_roi(int(idx))
            cohort_salient_rois.append({
                "roi_id": int(idx),
                "mean_saliency_weight": float(mean_saliency[idx]),
                "yeo_network": roi_meta["yeo_network"],
                "mni_coords": roi_meta["mni_coords"],
                "label": roi_meta["label"],
            })

        # Cohort network enrichment
        cohort_enrichment = self.atlas.compute_cohort_enrichment(cohort_saliency)

        # Cohort fidelity
        cohort_fidelity: Dict[str, float] = {}
        if compute_fidelity:
            cohort_fidelity = self.fidelity_evaluator.evaluate_cohort(
                cohort_fc=cohort_fc,
                cohort_saliency=cohort_saliency,
                target_class=target_class,
                ratios=(0.1, 0.2),
            )

        return {
            "cohort_salient_rois": cohort_salient_rois,
            "cohort_network_enrichment": cohort_enrichment,
            "cohort_fidelity": cohort_fidelity,
        }
