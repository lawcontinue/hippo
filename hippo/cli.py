"""Hippo CLI — command-line interface."""

import os
import sys
import logging

import typer
import requests

from hippo.config import load_config
from hippo.model_manager import ModelManager

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


@app.command()
def prewarm(
    models: list[str] = typer.Argument(None, help="Model names to pre-warm"),
    all_models: bool = typer.Option(False, "--all", "-a", help="Pre-warm ALL available models (not just loaded)"),
):
    """Pre-warm models by running a short inference to fill KV cache.

    By default, only pre-warms currently loaded models.
    Use --all to load and pre-warm all available models.
    """
    config = load_config()
    manager = ModelManager(config)

    if models:
        # User specified explicit models
        target_names = list(models)
    elif all_models:
        # --all: load and pre-warm everything
        available = manager.list_available()
        target_names = [m["name"] for m in available]
    else:
        # Default: only pre-warm already-loaded models
        target_names = [m["name"] for m in manager.list_loaded()]

    if not target_names:
        typer.echo("No models to pre-warm. Load models first or use --all.")
        return

    # Load models that aren't already loaded
    for name in target_names:
        try:
            manager.get(name)
            typer.echo(f"  ✅ Ready: {name}")
        except FileNotFoundError:
            typer.echo(f"  ❌ Not found: {name}")

    typer.echo(f"Pre-warming {len(target_names)} model(s)...")
    manager.prewarm(target_names)
    typer.echo("✅ Pre-warm complete")


@app.command()
def routes(
    reload_flag: bool = typer.Option(False, "--reload", "-r", help="Reload routes from disk"),
):
    """Show or reload route configuration."""
    from hippo.routes import RouteConfig

    rc = RouteConfig()
    if reload_flag:
        rc.reload()
        typer.echo("✅ Routes reloaded")
    else:
        route_list = rc.list_routes()
        if not route_list:
            typer.echo("No routes configured. Create ~/.hippo/routes.json")
            return
        for r in route_list:
            params = f" params={r['params']}" if r.get("params") else ""
            typer.echo(f"  {r['intent']:>10} → {r['model']}{params}")


@app.command()
def gateway(
    host: str = typer.Option("0.0.0.0", "--host", "-H", help="Bind host"),
    port: int = typer.Option(11434, "--port", "-p", help="Bind port"),
    name: str = typer.Option("hippo-gateway", "--name", "-n", help="Gateway name"),
    no_discovery: bool = typer.Option(False, "--no-mdns", help="Disable mDNS discovery"),
):
    """Start as a cluster Gateway (coordinator node).

    The Gateway:
    - Accepts Worker registrations via mDNS or HTTP
    - Routes inference requests to the best Worker
    - Handles failover on Worker failures
    """
    import asyncio
    import uvicorn
    from hippo.cluster.gateway import GatewayService
    from hippo import api

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    gw = GatewayService(port=port)

    # Create a fresh app instance
    from hippo.config import HippoConfig as _HC
    from hippo.model_manager import ModelManager as _MM
    from hippo.api import create_app

    app = create_app(config=_HC(), manager=_MM(_HC()))
    app.state._cluster_gateway = gw
    app.include_router(gw.router)

    typer.echo(f"🦛 Hippo Gateway starting on {host}:{port}")
    typer.echo(f"   mDNS discovery: {'enabled' if not no_discovery else 'disabled'}")
    typer.echo(f"   Workers can register at http://{host}:{port}/cluster/register")

    uvicorn.run(app, host=host, port=port, log_level="info")


@app.command()
def worker(
    name: str = typer.Option("", "--name", "-n", help="Worker name (auto-generated if empty)"),
    port: int = typer.Option(11435, "--port", "-p", help="Worker listen port"),
    gateway_url: str = typer.Option("", "--gateway", "-g", help="Gateway URL (overrides mDNS)"),
    memory: float = typer.Option(13.0, "--memory", "-m", help="Available memory in GB"),
    models_dir: str = typer.Option("", "--models-dir", help="Models directory"),
    no_discovery: bool = typer.Option(False, "--no-mdns", help="Disable mDNS"),
):
    """Start as a cluster Worker node.

    The Worker:
    - Registers with the Gateway via mDNS or HTTP
    - Sends periodic heartbeats
    - Serves inference requests on its local port
    """
    import asyncio
    import uvicorn
    from hippo.cluster.worker import WorkerService, WorkerConfig
    from hippo import api
    from hippo.config import load_config

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    cfg = load_config()
    cfg.server.host = "0.0.0.0"
    cfg.server.port = port

    worker_config = WorkerConfig(
        gateway_host=gateway_url.split("//")[1].split(":")[0] if gateway_url and "//" in gateway_url else (gateway_url or None),
        gateway_port=11434,
        worker_port=port,
        gpu_memory_gb=memory,
        models_dir=models_dir or str(cfg.models_dir),
    )

    w = WorkerService(worker_config)

    # Create a fresh app instance (not lazy default) to avoid __getattr__ issues
    from hippo.config import HippoConfig as _HC
    from hippo.model_manager import ModelManager as _MM
    from hippo.api import create_app

    app = create_app(config=cfg, manager=_MM(cfg))
    app.state._cluster_worker = w

    typer.echo(f"🦛 Hippo Worker starting on 0.0.0.0:{port}")
    typer.echo(f"   Memory: {memory}GB | Discovery: {'on' if not no_discovery else 'off'}")
    if gateway_url:
        typer.echo(f"   Gateway: {gateway_url}")

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    app()
