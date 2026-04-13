"""Hippo CLI — command-line interface."""

import os
import sys
import json
import logging

import typer
import requests

app = typer.Typer(help="Hippo 🦛 — Lightweight local LLM manager")

BASE_URL = os.environ.get("HIPPO_URL", os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"))


def _api_url(path: str) -> str:
    return f"{BASE_URL}{path}"


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-H", help="Bind host"),
    port: int = typer.Option(11434, "--port", "-p", help="Bind port"),
    config_path: str = typer.Option(None, "--config", "-c", help="Config file path"),
):
    """Start the Hippo server."""
    import uvicorn
    from hippo.config import load_config
    from hippo.model_manager import ModelManager
    from hippo import api

    cfg = load_config(config_path)
    cfg.server.host = host
    cfg.server.port = port

    # Wire up via app.state instead of globals
    api.app.state.config = cfg
    api.app.state.manager = ModelManager(cfg)
    api.app.state.manager.start_cleanup_thread()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    uvicorn.run(api.app, host=cfg.server.host, port=cfg.server.port, log_level="info")


@app.command(name="list")
def list_models():
    """List available models."""
    try:
        resp = requests.get(_api_url("/api/tags"), timeout=5)
        resp.raise_for_status()
        data = resp.json()

        if not data["models"]:
            typer.echo("No models found. Run: hippo pull <model>")
            return

        for m in data["models"]:
            size_gb = m.get("size", 0) / (1024**3)
            typer.echo(f"  {m['name']:30s} {size_gb:.2f} GB")
    except requests.ConnectionError:
        typer.echo("Error: Hippo server not running. Start with: hippo serve", err=True)
        sys.exit(1)


@app.command()
def run(
    model: str = typer.Argument(..., help="Model name"),
    prompt: str = typer.Argument(None, help="Prompt text"),
):
    """Run a model with a prompt."""
    if not prompt:
        typer.echo("Error: prompt required. Usage: hippo run <model> <prompt>", err=True)
        sys.exit(1)

    try:
        resp = requests.post(
            _api_url("/api/generate"),
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
        typer.echo(data.get("response", ""))
    except requests.ConnectionError:
        typer.echo("Error: Hippo server not running. Start with: hippo serve", err=True)
        sys.exit(1)
    except requests.HTTPError as e:
        detail = e.response.json() if e.response else {}
        typer.echo(f"Error: {detail.get('error', str(e))}", err=True)
        sys.exit(1)


@app.command()
def info(
    model: str = typer.Argument(..., help="Model name"),
):
    """Show model details."""
    try:
        resp = requests.post(
            _api_url("/api/show"),
            json={"name": model},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        details = data.get("details", {})
        model_info = data.get("model_info", {})
        size_gb = data.get("size", 0) / (1024**3)

        typer.echo(f"  Model:      {model}")
        typer.echo(f"  Format:     {details.get('format', 'gguf')}")
        typer.echo(f"  Family:     {details.get('family', 'unknown')}")
        typer.echo(f"  Quant:      {details.get('quantization_level', 'unknown')}")
        typer.echo(f"  Size:       {size_gb:.2f} GB")
        ctx = model_info.get("llama.context_length", "unknown")
        typer.echo(f"  Context:    {ctx}")
    except requests.ConnectionError:
        typer.echo("Error: Hippo server not running. Start with: hippo serve", err=True)
        sys.exit(1)
    except requests.HTTPError as e:
        detail = e.response.json() if e.response else {}
        typer.echo(f"Error: {detail.get('error', str(e))}", err=True)
        sys.exit(1)


@app.command()
def pull(
    name: str = typer.Argument(..., help="Model name or HuggingFace repo"),
):
    """Download a model from HuggingFace."""
    try:
        headers = {}
        api_key = os.environ.get("HIPPO_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        resp = requests.post(
            _api_url("/api/pull"),
            json={"name": name},
            timeout=600,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()
        typer.echo(f"✅ Downloaded: {data.get('path', name)}")
    except requests.ConnectionError:
        typer.echo("Error: Hippo server not running. Start with: hippo serve", err=True)
        sys.exit(1)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-n", help="Max results"),
):
    """Search for GGUF models on HuggingFace."""
    try:
        resp = requests.get(
            _api_url("/api/search"),
            params={"q": query, "limit": limit},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        models = data.get("models", [])
        if not models:
            typer.echo("No models found.")
            return
        for m in models:
            typer.echo(f"  {m['id']:50s} ↓{m.get('downloads', 0):>8d}")
    except requests.ConnectionError:
        typer.echo("Error: Hippo server not running. Start with: hippo serve", err=True)
        sys.exit(1)


@app.command()
def remove(
    name: str = typer.Argument(..., help="Model name to remove"),
):
    """Remove a model from disk."""
    try:
        headers = {}
        api_key = os.environ.get("HIPPO_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        resp = requests.delete(
            _api_url("/api/delete"),
            json={"name": name},
            timeout=10,
            headers=headers,
        )
        resp.raise_for_status()
        typer.echo(f"✅ Removed: {name}")
    except requests.ConnectionError:
        typer.echo("Error: Hippo server not running. Start with: hippo serve", err=True)
        sys.exit(1)


if __name__ == "__main__":
    app()
