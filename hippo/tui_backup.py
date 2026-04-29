"""Rich TUI for Hippo — live dashboard."""

import os
import time

import requests
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

BASE_URL = os.environ.get("HIPPO_URL", os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"))


def _api_get(path: str):
    try:
        resp = requests.get(f"{BASE_URL}{path}", timeout=3)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        return None


def _build_dashboard() -> Panel:
    """Build the TUI dashboard panel."""
    data = _api_get("/api/tags")
    version = _api_get("/api/version")

    ver = version.get("version", "?") if version else "offline"

    if data is None:
        return Panel(
            Text("❌ Hippo server not reachable", style="bold red"),
            title=f"Hippo 🦛 v{ver}",
            border_style="red",
        )

    models = data.get("models", [])

    table = Table(show_header=True, header_style="bold cyan", expand=True)
    table.add_column("Model", style="white")
    table.add_column("Size", justify="right", style="green")
    table.add_column("Family", style="yellow")
    table.add_column("Quant", style="magenta")

    total_size = 0.0
    for m in models:
        size_gb = m.get("size", 0) / (1024**3)
        total_size += size_gb
        details = m.get("details", {})
        table.add_row(
            m.get("name", "?"),
            f"{size_gb:.2f} GB",
            details.get("family", "?"),
            details.get("quantization_level", "?"),
        )

    if not models:
        table.add_row("(no models)", "", "", "")

    summary = f"📊 {len(models)} models · {total_size:.2f} GB total · Server v{ver}"
    return Panel(
        table,
        title="Hippo 🦛 Dashboard",
        subtitle=summary,
        border_style="blue",
    )


def run_tui(refresh_interval: float = 2.0):
    """Run the live TUI dashboard."""
    console = Console()
    with Live(console=console, refresh_per_second=1.0 / refresh_interval) as live:
        live.update(_build_dashboard())
        try:
            while True:
                time.sleep(refresh_interval)
                live.update(_build_dashboard())
        except KeyboardInterrupt:
            live.update(Panel("👋 Hippo TUI stopped", border_style="yellow"))
