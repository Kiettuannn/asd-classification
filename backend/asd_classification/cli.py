"""Command-line interface for training, evaluation, and serving."""

from pathlib import Path
from typing import Optional

try:
    import typer
    app = typer.Typer(
        name="asd",
        help="ASD Classification and Brain Connectivity CLI",
        add_completion=False,
    )
except ImportError:
    app = None  # type: ignore

if app is not None:
    @app.command()
    def train(
        model: str = typer.Option("svm", "--model", "-m", help="Model type: svm, gcn, or gat"),
        epochs: int = typer.Option(100, "--epochs", "-e", help="Number of training epochs (for GNN)"),
        threshold: float = typer.Option(0.3, "--threshold", "-t", help="FC edge threshold"),
    ):
        """Train a baseline or GNN model on the ABIDE dataset."""
        typer.echo(f"Starting training pipeline for model: {model} (epochs={epochs}, threshold={threshold})...")
        typer.echo("Note: Full training implementation will execute in implementation phase.")

    @app.command()
    def evaluate(
        model_path: Path = typer.Option(..., "--model-path", "-p", help="Path to saved model checkpoint"),
    ):
        """Evaluate a trained model checkpoint on held-out test data."""
        typer.echo(f"Evaluating model checkpoint at: {model_path}...")
        typer.echo("Note: Full evaluation implementation will execute in implementation phase.")

    @app.command()
    def serve(
        host: str = typer.Option("0.0.0.0", "--host", "-h", help="API host interface"),
        port: int = typer.Option(8000, "--port", "-p", help="API port"),
    ):
        """Start the FastAPI inference and explainability server."""
        typer.echo(f"Starting FastAPI server on {host}:{port}...")
        try:
            import uvicorn
            uvicorn.run("asd_classification.api.app:create_app", factory=True, host=host, port=port, reload=True)
        except ImportError:
            typer.echo("Uvicorn/FastAPI not installed. Please run: pip install '.[api]'")


def main():
    if app is not None:
        app()
    else:
        print("Typer is not installed. Please install dependencies: pip install typer")


if __name__ == "__main__":
    main()
