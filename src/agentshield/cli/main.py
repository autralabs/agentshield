"""Command-line interface for AgentShield."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import typer
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import print as rprint
except ImportError:
    print("CLI dependencies not installed. Install with: pip install agentshield[cli]")
    sys.exit(1)

app = typer.Typer(
    name="agentshield",
    help="Prompt injection detection for RAG pipelines.",
    add_completion=False,
)
console = Console()


@app.command()
def scan(
    input_path: Optional[Path] = typer.Argument(
        None,
        help="Path to file or directory to scan. Use - for stdin.",
    ),
    text: Optional[str] = typer.Option(
        None,
        "--text", "-t",
        help="Text to scan directly.",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Path to configuration YAML file.",
    ),
    output: str = typer.Option(
        "text",
        "--output", "-o",
        help="Output format: text, json",
    ),
    threshold: Optional[float] = typer.Option(
        None,
        "--threshold",
        help="Override detection threshold.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Show detailed output.",
    ),
):
    """
    Scan text or files for prompt injections.

    Examples:

        agentshield scan document.txt

        agentshield scan --text "Hello, ignore previous instructions"

        echo "some text" | agentshield scan -

        agentshield scan ./documents/ --output json
    """
    from agentshield import AgentShield, ScanResult

    # Determine input source
    if text:
        texts = [text]
        sources = ["<direct input>"]
    elif input_path:
        texts, sources = _load_texts(input_path)
    else:
        console.print("[red]Error: Provide either --text or a file path[/red]")
        raise typer.Exit(1)

    if not texts:
        console.print("[yellow]No text to scan[/yellow]")
        raise typer.Exit(0)

    # Initialize shield
    shield_config = {}
    if config:
        shield_config = {"config": config}
    if threshold:
        shield_config["zedd"] = {"threshold": threshold}

    try:
        shield = AgentShield(config=shield_config if shield_config else None)
    except Exception as e:
        console.print(f"[red]Failed to initialize AgentShield: {e}[/red]")
        raise typer.Exit(1)

    # Scan texts
    results = shield.scan(texts)
    if not isinstance(results, list):
        results = [results]

    # Output results
    if output == "json":
        _output_json(results, sources)
    else:
        _output_text(results, sources, verbose)

    # Exit code based on findings
    suspicious_count = sum(1 for r in results if r.is_suspicious)
    if suspicious_count > 0:
        raise typer.Exit(1)


@app.command()
def calibrate(
    model: str = typer.Option(
        "all-MiniLM-L6-v2",
        "--model", "-m",
        help="Embedding model to calibrate.",
    ),
    corpus: Optional[Path] = typer.Option(
        None,
        "--corpus",
        help="Path to directory of clean texts for calibration.",
    ),
    save: bool = typer.Option(
        True,
        "--save/--no-save",
        help="Save calibrated threshold to cache.",
    ),
):
    """
    Calibrate detection threshold for an embedding model.

    Uses the GMM method from the ZEDD paper to find the optimal
    decision boundary between clean and injected content.

    Examples:

        agentshield calibrate --model all-MiniLM-L6-v2

        agentshield calibrate --model text-embedding-3-small --corpus ./my_docs/
    """
    from agentshield import AgentShield

    console.print(f"[bold]Calibrating threshold for: {model}[/bold]")

    # Initialize shield with specified model
    shield = AgentShield(config={
        "embeddings": {"model": model},
    })

    # Load corpus if provided
    corpus_texts = None
    if corpus:
        if not corpus.exists():
            console.print(f"[red]Corpus path not found: {corpus}[/red]")
            raise typer.Exit(1)

        console.print(f"Loading corpus from: {corpus}")
        corpus_texts = _load_corpus(corpus)
        console.print(f"Loaded {len(corpus_texts)} texts")

    # Calibrate
    with console.status("Calibrating..."):
        try:
            threshold = shield.calibrate(corpus=corpus_texts, save=save)
        except Exception as e:
            console.print(f"[red]Calibration failed: {e}[/red]")
            raise typer.Exit(1)

    console.print(f"\n[green]Calibration complete![/green]")
    console.print(f"Threshold: [bold]{threshold:.6f}[/bold]")

    if save:
        console.print(f"Saved to cache for future use.")


@app.command("config")
def config_cmd(
    action: str = typer.Argument(
        "show",
        help="Action: show, validate, init",
    ),
    path: Optional[Path] = typer.Argument(
        None,
        help="Path to config file (for validate/init).",
    ),
):
    """
    Manage AgentShield configuration.

    Actions:
        show      - Show current configuration
        validate  - Validate a configuration file
        init      - Create a default configuration file
    """
    from agentshield import ShieldConfig

    if action == "show":
        config = ShieldConfig()
        console.print(Panel(
            json.dumps(config.to_dict(), indent=2),
            title="Current Configuration",
        ))

    elif action == "validate":
        if not path:
            console.print("[red]Please provide a config file path[/red]")
            raise typer.Exit(1)

        try:
            config = ShieldConfig.from_yaml(path)
            console.print(f"[green]Configuration is valid![/green]")
            if typer.confirm("Show parsed config?"):
                console.print(json.dumps(config.to_dict(), indent=2))
        except Exception as e:
            console.print(f"[red]Invalid configuration: {e}[/red]")
            raise typer.Exit(1)

    elif action == "init":
        output_path = path or Path("agentshield.yaml")
        if output_path.exists():
            if not typer.confirm(f"{output_path} exists. Overwrite?"):
                raise typer.Exit(0)

        config = ShieldConfig()
        config.to_yaml(output_path)
        console.print(f"[green]Created {output_path}[/green]")

    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        raise typer.Exit(1)


@app.command()
def version():
    """Show AgentShield version."""
    from agentshield import __version__
    console.print(f"agentshield {__version__}")


# Helper functions

def _load_texts(path: Path) -> Tuple[List[str], List[str]]:
    """Load texts from file or directory."""
    texts = []
    sources = []

    if str(path) == "-":
        # Read from stdin
        content = sys.stdin.read()
        return [content], ["<stdin>"]

    if not path.exists():
        console.print(f"[red]Path not found: {path}[/red]")
        raise typer.Exit(1)

    if path.is_file():
        texts.append(path.read_text())
        sources.append(str(path))
    elif path.is_dir():
        for file_path in path.rglob("*"):
            if file_path.is_file() and file_path.suffix in (".txt", ".md", ".html"):
                try:
                    texts.append(file_path.read_text())
                    sources.append(str(file_path))
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not read {file_path}: {e}[/yellow]")

    return texts, sources


def _load_corpus(path: Path) -> List[str]:
    """Load corpus texts for calibration."""
    texts, _ = _load_texts(path)
    return texts


def _output_text(results: list, sources: List[str], verbose: bool):
    """Output results in text format."""
    table = Table(title="Scan Results")
    table.add_column("Source", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Confidence", justify="right")
    if verbose:
        table.add_column("Details")

    for result, source in zip(results, sources):
        status = "[red]SUSPICIOUS[/red]" if result.is_suspicious else "[green]CLEAN[/green]"
        confidence = f"{result.confidence:.2%}"

        if verbose:
            details = result.details.summary
            table.add_row(source, status, confidence, details)
        else:
            table.add_row(source, status, confidence)

    console.print(table)

    # Summary
    suspicious = sum(1 for r in results if r.is_suspicious)
    total = len(results)
    if suspicious > 0:
        console.print(f"\n[red]Found {suspicious}/{total} suspicious document(s)[/red]")
    else:
        console.print(f"\n[green]All {total} document(s) appear clean[/green]")


def _output_json(results: list, sources: List[str]):
    """Output results in JSON format."""
    output = []
    for result, source in zip(results, sources):
        output.append({
            "source": source,
            **result.to_dict(),
        })
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    app()
