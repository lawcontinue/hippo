"""Hippo CLI — command-line interface."""

import os
import sys
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
    ssl: bool = typer.Option(False, "--ssl", help="Enable HTTPS (requires --cert and --key)"),
    cert: str = typer.Option(None, "--cert", help="SSL certificate file path"),
    key: str = typer.Option(None, "--key", help="SSL private key file path"),
    audit_log: str = typer.Option(None, "--audit-log", help="Enable audit logging (JSONL file path)"),
):
    """Start the Hippo server."""
    import uvicorn
    from hippo.config import load_config
    from hippo.model_manager import ModelManager
    from hippo import api

    cfg = load_config(config_path)
    cfg.server.host = host
    cfg.server.port = port

    # SSL/TLS 配置
    if ssl or cert or key:
        if not cert or not key:
            typer.echo("Error: --ssl requires both --cert and --key", err=True)
            sys.exit(1)
        cfg.server.ssl_enabled = True
        cfg.server.ssl_cert_path = cert
        cfg.server.ssl_key_path = key

    # Wire up via app.state instead of globals
    api.app.state.config = cfg
    api.app.state.manager = ModelManager(cfg)
    api.app.state.manager.start_cleanup_thread()
    api.app.state._audit_log_path = audit_log

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # SSL 配置
    if cfg.server.ssl_enabled:
        logging.info(f"HTTPS enabled: cert={cfg.server.ssl_cert_path}, key={cfg.server.ssl_key_path}")
        uvicorn.run(
            api.app,
            host=cfg.server.host,
            port=cfg.server.port,
            log_level="info",
            ssl_keyfile=cfg.server.ssl_key_path,
            ssl_certfile=cfg.server.ssl_cert_path
        )
    else:
        uvicorn.run(
            api.app,
            host=cfg.server.host,
            port=cfg.server.port,
            log_level="info"
        )


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
def tui(
    refresh: float = typer.Option(2.0, "--refresh", "-r", help="Refresh interval in seconds"),
):
    """Live terminal dashboard for Hippo."""
    from hippo.tui_v2 import run_tui
    run_tui(refresh_interval=refresh)


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


@app.command()
def quantize(
    input_path: str = typer.Argument(None, help="Input GGUF file path"),
    output_path: str = typer.Argument(None, help="Output GGUF file path"),
    format: str = typer.Option("q4_k_m", "--format", "-f", help="Quantization format"),
    list_formats_flag: bool = typer.Option(False, "--list", "-l", help="List available formats"),
):
    """Quantize a GGUF model to a different format."""
    from hippo.quantize import list_formats, quantize_model

    if list_formats_flag:
        list_formats()
        return

    if not input_path or not output_path:
        typer.echo("Error: provide input and output paths, or use --list", err=True)
        raise typer.Exit(1)

    quantize_model(input_path, output_path, format)


if __name__ == "__main__":
    app()
