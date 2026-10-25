"""Pipeline modules for training, evaluation, cross-validation, and metrics."""

from .evaluate import (
    aggregate_soft_voting_probabilities,
    compute_classification_metrics,
    evaluate_mil_model,
    evaluate_multiview_ensemble,
    find_optimal_threshold,
    format_benchmark_comparison,
    format_dual_level_report,
    plot_roc_curve,
    run_evaluation_pipeline,
)
from .train import (
    EarlyStopping,
    compute_differential_cosine_warmup_lr,
    configure_differential_optimizer,
    evaluate_epoch,
    run_stratified_kfold_cv,
    run_training_pipeline,
    train_3dcnn_model,
    train_epoch,
    train_mil_epoch,
    train_mil_model,
    update_learning_rates,
)

__all__ = [
    "run_training_pipeline",
    "run_evaluation_pipeline",
    "compute_classification_metrics",
    "find_optimal_threshold",
    "aggregate_soft_voting_probabilities",
    "evaluate_multiview_ensemble",
    "evaluate_mil_model",
    "format_dual_level_report",
    "format_benchmark_comparison",
    "plot_roc_curve",
    "EarlyStopping",
    "train_epoch",
    "evaluate_epoch",
    "train_3dcnn_model",
    "train_mil_epoch",
    "train_mil_model",
    "configure_differential_optimizer",
    "compute_differential_cosine_warmup_lr",
    "update_learning_rates",
    "run_stratified_kfold_cv",
]
