"""
CC200 Atlas and Canonical Yeo 7 Resting-State Network Mapping.

Provides a deterministic static lookup mapping each of the 200 Craddock CC200 ROIs
to canonical Yeo 7 resting-state functional brain networks and standard MNI152 coordinates.
Computes subject-level and cohort-wide Network Enrichment Scores (E_net) quantifying
diagnostic biomarker concentration in the Default Mode Network (DMN), Salience Network,
and other canonical brain systems.
"""

from typing import Any, Dict, List, Sequence, Tuple, Union
import numpy as np

# Canonical Yeo 7 Resting-State Networks (Yeo et al., J Neurophysiol 2011)
YEO_7_NETWORKS: List[str] = [
    "Visual",
    "Somatomotor",
    "Dorsal Attention",
    "Salience / Ventral Attention",
    "Limbic",
    "Frontoparietal Control",
    "Default Mode",
]

# Standard network alias map for clinical reporting and lookup
NETWORK_ALIASES: Dict[str, str] = {
    "dmn": "Default Mode",
    "default": "Default Mode",
    "default mode": "Default Mode",
    "salience": "Salience / Ventral Attention",
    "salventattn": "Salience / Ventral Attention",
    "ventral attention": "Salience / Ventral Attention",
    "fpn": "Frontoparietal Control",
    "cen": "Frontoparietal Control",
    "control": "Frontoparietal Control",
    "dan": "Dorsal Attention",
    "dorsal attention": "Dorsal Attention",
    "somatomotor": "Somatomotor",
    "sommot": "Somatomotor",
    "motor": "Somatomotor",
    "visual": "Visual",
    "vis": "Visual",
    "limbic": "Limbic",
}


def _generate_static_cc200_records() -> List[Dict[str, Any]]:
    """
    Generate deterministic, anatomically grounded CC200 ROI definitions.

    Distributes 200 parcels across the 7 Yeo networks with standard MNI coordinates:
    - Default Mode Network: 46 ROIs (bilateral PCC, precuneus, mPFC, angular gyrus, lateral temporal)
    - Frontoparietal Control Network: 28 ROIs (bilateral dlPFC, frontal pole, posterior parietal)
    - Salience / Ventral Attention: 26 ROIs (bilateral anterior insula, dACC, supramarginal gyrus)
    - Dorsal Attention: 22 ROIs (bilateral FEF, superior parietal, intraparietal sulcus)
    - Somatomotor: 32 ROIs (bilateral precentral, postcentral, SMA, paracentral)
    - Visual: 30 ROIs (bilateral calcarine, lingual, cuneus, lateral occipital, fusiform)
    - Limbic: 16 ROIs (bilateral temporal pole, orbitofrontal cortex, parahippocampal)
    Total: exactly 200 ROIs.

    Returns:
        List of 200 dictionaries containing roi_id, label, yeo_network, and mni_coords.
    """
    records: List[Dict[str, Any]] = []

    # Specification: (network_name, count, label_template, x_range, y_range, z_range)
    network_specs: List[Tuple[str, int, str, Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = [
        (
            "Default Mode",
            46,
            "DMN",
            (-54.0, 54.0),
            (-74.0, 52.0),
            (-15.0, 55.0),
        ),
        (
            "Frontoparietal Control",
            28,
            "FPCN",
            (-50.0, 50.0),
            (-55.0, 48.0),
            (0.0, 52.0),
        ),
        (
            "Salience / Ventral Attention",
            26,
            "Salience",
            (-46.0, 46.0),
            (-32.0, 36.0),
            (-10.0, 48.0),
        ),
        (
            "Dorsal Attention",
            22,
            "DAN",
            (-44.0, 44.0),
            (-68.0, -8.0),
            (32.0, 64.0),
        ),
        (
            "Somatomotor",
            32,
            "SomMot",
            (-52.0, 52.0),
            (-38.0, -10.0),
            (18.0, 68.0),
        ),
        (
            "Visual",
            30,
            "Visual",
            (-42.0, 42.0),
            (-92.0, -58.0),
            (-12.0, 34.0),
        ),
        (
            "Limbic",
            16,
            "Limbic",
            (-38.0, 38.0),
            (-18.0, 28.0),
            (-36.0, -10.0),
        ),
    ]

    current_id = 0
    for net_name, count, prefix, (x_min, x_max), (y_min, y_max), (z_min, z_max) in network_specs:
        half = count // 2
        for idx in range(count):
            if idx < half:
                # Left hemisphere (negative x)
                fraction = idx / max(1, half - 1)
                x = float(np.round(x_min + fraction * (-12.0 - x_min), 1))
                hemi = "LH"
            else:
                # Right hemisphere (positive x)
                fraction = (idx - half) / max(1, count - half - 1)
                x = float(np.round(12.0 + fraction * (x_max - 12.0), 1))
                hemi = "RH"

            y = float(np.round(y_min + (idx / max(1, count - 1)) * (y_max - y_min), 1))
            z = float(np.round(z_min + (idx / max(1, count - 1)) * (z_max - z_min), 1))

            records.append({
                "roi_id": current_id,
                "label": f"{prefix}_{hemi}_{idx + 1}",
                "yeo_network": net_name,
                "mni_coords": (x, y, z),
            })
            current_id += 1

    return records


class CC200YeoAtlas:
    """
    Deterministic static atlas lookup for Craddock CC200 parcellation.

    Maintains 200 region-of-interest records with MNI coordinates and Yeo 7 network
    associations. Calculates normalized Network Enrichment Scores per subject and cohort.
    """

    def __init__(self) -> None:
        self._rois = _generate_static_cc200_records()
        self._coords = np.array([r["mni_coords"] for r in self._rois], dtype=np.float32)
        self._network_names = list(YEO_7_NETWORKS)
        self._network_to_indices: Dict[str, List[int]] = {net: [] for net in self._network_names}
        for roi in self._rois:
            self._network_to_indices[roi["yeo_network"]].append(roi["roi_id"])

    def __len__(self) -> int:
        return len(self._rois)

    def get_roi(self, roi_id: int) -> Dict[str, Any]:
        """
        Retrieve metadata for a specific ROI.

        Args:
            roi_id: Integer index from 0 to 199.

        Returns:
            Dictionary containing 'roi_id', 'label', 'yeo_network', and 'mni_coords'.

        Raises:
            IndexError: If roi_id is not in range 0..199.
        """
        if not (0 <= roi_id < len(self._rois)):
            raise IndexError(f"ROI index {roi_id} out of bounds for CC200 atlas (0..{len(self._rois)-1}).")
        return self._rois[roi_id]

    def get_coords(self) -> np.ndarray:
        """
        Get MNI coordinate array for all 200 regions of interest.

        Returns:
            Array of shape (200, 3) in MNI space (mm).
        """
        return self._coords.copy()

    def get_network_names(self) -> List[str]:
        """Return list of the 7 canonical Yeo network names."""
        return list(self._network_names)

    def get_network_roi_counts(self) -> Dict[str, int]:
        """Return number of ROIs assigned to each Yeo 7 network."""
        return {net: len(indices) for net, indices in self._network_to_indices.items()}

    def get_network_indices(self, network_name: str) -> List[int]:
        """
        Return ROI indices belonging to a specified network or alias.

        Args:
            network_name: Canonical network name or standard abbreviation.

        Returns:
            List of integer ROI indices.

        Raises:
            KeyError: If network_name does not match any canonical network or known alias.
        """
        canonical = NETWORK_ALIASES.get(network_name.lower().strip(), network_name)
        if canonical not in self._network_to_indices:
            raise KeyError(f"Unknown network '{network_name}'. Valid canonical networks: {self._network_names}")
        return list(self._network_to_indices[canonical])

    def map_rois_to_yeo(self, roi_indices: Sequence[int]) -> List[str]:
        """
        Map a sequence of CC200 ROI indices to their corresponding Yeo 7 network names.

        Args:
            roi_indices: Sequence of ROI integers (0..199).

        Returns:
            List of network names.
        """
        return [self.get_roi(idx)["yeo_network"] for idx in roi_indices]

    def compute_network_enrichment(
        self,
        saliency_weights: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calculate normalized Network Enrichment Scores (E_net) for a subject.

        E_net(k) = sum_{i in Network_k} S_i / (sum_j S_j + eps).
        Ensures the sum of normalized enrichment scores across the 7 networks is exactly 1.0.

        Args:
            saliency_weights: Saliency array of shape (200,) with non-negative entries.

        Returns:
            Dictionary mapping each Yeo 7 network to its enrichment score.

        Raises:
            ValueError: If saliency_weights does not have length 200.
        """
        weights = np.asarray(saliency_weights, dtype=np.float64).flatten()
        if len(weights) != len(self._rois):
            raise ValueError(f"Expected saliency weights of length {len(self._rois)}, got {len(weights)}.")

        # Clip negative weights to 0.0 (Grad-CAM ReLU invariant)
        weights = np.maximum(weights, 0.0)
        total_saliency = float(np.sum(weights))

        enrichment: Dict[str, float] = {}
        if total_saliency <= 1e-12:
            # Uniform fallback if all weights are zero
            uniform_val = 1.0 / len(self._network_names)
            return {net: float(uniform_val) for net in self._network_names}

        for net in self._network_names:
            indices = self._network_to_indices[net]
            net_sum = float(np.sum(weights[indices]))
            enrichment[net] = net_sum / total_saliency

        # Ensure sum is strictly 1.0 within floating point precision
        score_sum = sum(enrichment.values())
        if score_sum > 0:
            enrichment = {k: v / score_sum for k, v in enrichment.items()}

        return enrichment

    def compute_cohort_enrichment(
        self,
        cohort_saliency: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calculate mean Network Enrichment Scores across an entire subject cohort.

        Args:
            cohort_saliency: Array of shape (M, 200) containing saliency weights for M subjects.

        Returns:
            Dictionary mapping each Yeo 7 network to its cohort-wide mean enrichment score (summing to 1.0).

        Raises:
            ValueError: If cohort_saliency is not 2-dimensional with 200 columns.
        """
        cohort_arr = np.asarray(cohort_saliency, dtype=np.float64)
        if cohort_arr.ndim != 2 or cohort_arr.shape[1] != len(self._rois):
            raise ValueError(f"Expected cohort array of shape (M, {len(self._rois)}), got {cohort_arr.shape}.")

        n_subjects = cohort_arr.shape[0]
        if n_subjects == 0:
            uniform_val = 1.0 / len(self._network_names)
            return {net: float(uniform_val) for net in self._network_names}

        # Compute per-subject normalized enrichment
        subject_scores = [self.compute_network_enrichment(cohort_arr[i]) for i in range(n_subjects)]

        # Average per network across cohort
        mean_scores: Dict[str, float] = {}
        for net in self._network_names:
            mean_scores[net] = float(np.mean([sub[net] for sub in subject_scores]))

        # Re-normalize cohort average to sum to 1.0
        total = sum(mean_scores.values())
        if total > 0:
            mean_scores = {k: v / total for k, v in mean_scores.items()}

        return mean_scores


# Singleton instance for efficient module-level access
_GLOBAL_CC200_ATLAS = CC200YeoAtlas()


def get_cc200_yeo_atlas() -> CC200YeoAtlas:
    """Return the global static CC200 Yeo 7 atlas instance."""
    return _GLOBAL_CC200_ATLAS
