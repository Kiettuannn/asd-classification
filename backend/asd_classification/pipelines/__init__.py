"""Pipeline modules for training, evaluation, cross-validation, and metrics."""

from .evaluate import (
    aggregate_soft_voting_probabilities,
    compute_classification_metrics,
    evaluate_multiview_ensemble,
    find_optimal_threshold,
    format_benchmark_comparison,
    format_dual_level_report,
    plot_roc_curve,
    run_evaluation_pipeline,
)
from .train import (
    EarlyStopping,
    evaluate_epoch,
    run_stratified_kfold_cv,
    run_training_pipeline,
    train_3dcnn_model,
    train_epoch,
)

__all__ = [
    "run_training_pipeline",
    "run_evaluation_pipeline",
    "compute_classification_metrics",
    "find_optimal_threshold",
    "aggregate_soft_voting_probabilities",
    "evaluate_multiview_ensemble",
    "format_dual_level_report",
    "format_benchmark_comparison",
    "plot_roc_curve",
    "EarlyStopping",
    "train_epoch",
    "evaluate_epoch",
    "train_3dcnn_model",
    "run_stratified_kfold_cv",
]
