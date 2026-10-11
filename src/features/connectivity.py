"""Functional Connectivity (FC) matrix extraction and feature vectorization."""

from typing import Optional
import numpy as np


def compute_fc(time_series: np.ndarray, metric: str = "correlation") -> np.ndarray:
    """
    Computes Functional Connectivity matrix from BOLD time-series.
    
    Args:
        time_series: Array of shape (n_timepoints, n_rois).
        metric: Connectivity metric ('correlation', 'partial correlation', 'tangent').
        
    Returns:
        Square symmetric matrix of shape (n_rois, n_rois).
    """
    raise NotImplementedError("Implementation will be added in implementation phase.")


def vectorize_fc(fc_matrix: np.ndarray, discard_diagonal: bool = True) -> np.ndarray:
    """
    Extracts the upper triangle of the FC matrix as a 1D feature vector.
    
    Args:
        fc_matrix: Array of shape (n_rois, n_rois).
        discard_diagonal: Whether to exclude diagonal entries (self-connections).
        
    Returns:
        1D feature vector of shape (n_rois * (n_rois - 1) / 2,).
    """
    raise NotImplementedError("Implementation will be added in implementation phase.")


def threshold_fc(fc_matrix: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """
    Applies absolute value thresholding to retain salient connections.
    
    Args:
        fc_matrix: Array of shape (n_rois, n_rois).
        threshold: Absolute threshold for connectivity strength.
        
    Returns:
        Thresholded connectivity matrix with sub-threshold weights set to 0.
    """
    raise NotImplementedError("Implementation will be added in implementation phase.")
