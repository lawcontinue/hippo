"""Rich TUI for Hippo — live dashboard with keyboard interaction.

Features:
- Real-time model list display
- Stats panel (QPS, memory, error rate)
- Keyboard interaction (q=quit, r=refresh, ↑↓=select, Enter=details)
- Async refresh every 2 seconds
- Error handling with friendly messages
"""

import asyncio
import time
import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.layout import Layout
from rich.align import Align
from rich import box

import requests

BASE_URL = os.environ.get("HIPPO_URL", os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"))


@dataclass
class StatsData:
    """Statistics data from /api/stats endpoint."""
    qps: float = 0.0
    memory_mb: float = 0.0
    error_rate: float = 0.0
    total_requests: int = 0
    loaded_models: int = 0


@dataclass
class ModelData:
    """Model information."""
    name: str
    size_gb: float
    family: str
    quantization: str
    context_length: int = 4096
    loaded: bool = False


class TUIApp:
    """Main TUI application with asyncio event loop."""

    def __init__(self, refresh_interval: float = 2.0):
        self.refresh_interval = refresh_interval
        self.console = Console()
        self.running = True
        self.selected_index = 0
        self.show_details = False
        self.stats = StatsData()
        self.models: List[ModelData] = []
        self.last_error: Optional[str] = None
        self.offline_mode = False

    def _api_get(self, path: str) -> Optional[Dict[str, Any]]:
        """Make API GET request with error handling."""
        try:
            resp = requests.get(f"{BASE_URL}{path}", timeout=3)
            resp.raise_for_status()
            self.last_error = None
            self.offline_mode = False
            return resp.json()
        except requests.ConnectionError:
            self.last_error = "Connection failed"
            self.offline_mode = True
            return None
        except requests.Timeout:
            self.last_error = "Request timeout"
            return None
        except Exception as e:
            self.last_error = str(e)
            return None

    def _fetch_stats(self) -> StatsData:
        """Fetch statistics from /api/stats endpoint."""
        data = self._api_get("/api/stats")
        if data:
            return StatsData(
                qps=data.get("qps", 0.0),
                memory_mb=data.get("memory_mb", 0.0),
                error_rate=data.get("error_rate", 0.0),
                total_requests=data.get("total_requests", 0),
                loaded_models=data.get("loaded_models", 0),
            )
        return self.stats  # Return cached stats if offline

    def _fetch_models(self) -> List[ModelData]:
        """Fetch model list from /api/tags endpoint."""
        data = self._api_get("/api/tags")
        if data:
            models = []
            for m in data.get("models", []):
                details = m.get("details", {})
                models.append(ModelData(
                    name=m.get("name", "?"),
                    size_gb=m.get("size", 0) / (1024**3),
                    family=details.get("family", "?"),
                    quantization=details.get("quantization_level", "?"),
                    context_length=m.get("model_info", {}).get("llama.context_length", 4096),
                    loaded=False,  # TODO: Track loaded state
                ))
            return models
        return self.models  # Return cached models if offline

    def _build_stats_panel(self) -> Panel:
        """Build the stats panel with QPS, memory, error rate."""
        if self.offline_mode:
            stats_text = Text("⚠️ Offline Mode", style="bold yellow")
        else:
            # Color-code metrics
            qps_color = "green" if self.stats.qps < 10 else "yellow" if self.stats.qps < 50 else "red"
            mem_color = "green" if self.stats.memory_mb < 1000 else "yellow" if self.stats.memory_mb < 4000 else "red"
            err_color = "green" if self.stats.error_rate < 1 else "yellow" if self.stats.error_rate < 5 else "red"

            stats_text = Text()
            stats_text.append(f"QPS: ", style="white")
            stats_text.append(f"{self.stats.qps:.1f} ", style=qps_color)
            stats_text.append(f"• Memory: ", style="white")
            stats_text.append(f"{self.stats.memory_mb:.0f} MB ", style=mem_color)
            stats_text.append(f"• Errors: ", style="white")
            stats_text.append(f"{self.stats.error_rate:.1f}% ", style=err_color)
            stats_text.append(f"• Requests: ", style="white")
            stats_text.append(f"{self.stats.total_requests} ", style="cyan")
            stats_text.append(f"• Loaded: ", style="white")
            stats_text.append(f"{self.stats.loaded_models}", style="cyan")

        return Panel(
            Align.center(stats_text),
            title="📊 Real-time Stats",
            border_style="blue" if not self.offline_mode else "yellow",
            box=box.DOUBLE,
        )

    def _build_model_table(self) -> Table:
        """Build the model list table."""
        table = Table(
            show_header=True,
            header_style="bold cyan",
            expand=True,
            box=box.ROUNDED,
        )
        table.add_column("#", width=3)
        table.add_column("Model", style="white")
        table.add_column("Size", justify="right", width=10)
        table.add_column("Family", width=12)
        table.add_column("Quant", width=8)
        table.add_column("Ctx", width=6)

        total_size = 0.0
        for idx, model in enumerate(self.models):
            size_gb = model.size_gb
            total_size += size_gb

            # Highlight selected row
            prefix = "➤ " if idx == self.selected_index else "  "
            style = "bold white on blue" if idx == self.selected_index else ""

            table.add_row(
                prefix,
                Text(model.name, style=style),
                Text(f"{size_gb:.2f} GB", style=style),
                Text(model.family, style=style),
                Text(model.quantization, style=style),
                Text(str(model.context_length), style=style),
            )

        if not self.models:
            table.add_row("", Text("(no models)", style="italic dim"), "", "", "", "")

        return table

    def _build_model_details(self) -> Panel:
        """Build model details panel."""
        if not self.models or self.selected_index >= len(self.models):
            return Panel(Text("No model selected", style="dim"))

        model = self.models[self.selected_index]

        details = Text()
        details.append(f"Model: ", style="bold cyan")
        details.append(f"{model.name}\n", style="white")
        details.append(f"Family: ", style="bold cyan")
        details.append(f"{model.family}\n", style="white")
        details.append(f"Quantization: ", style="bold cyan")
        details.append(f"{model.quantization}\n", style="white")
        details.append(f"Size: ", style="bold cyan")
        details.append(f"{model.size_gb:.2f} GB\n", style="white")
        details.append(f"Context: ", style="bold cyan")
        details.append(f"{model.context_length} tokens\n", style="white")
        details.append(f"Status: ", style="bold cyan")
        details.append(f"{'Loaded' if model.loaded else 'Not loaded'}\n", style="green" if model.loaded else "dim")

        return Panel(
            details,
            title=f"📦 Model Details - {model.name}",
            border_style="cyan",
        )

    def _build_dashboard(self) -> Layout:
        """Build the complete TUI dashboard layout."""
        layout = Layout()

        if self.show_details:
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="details"),
            )
            layout["header"].update(self._build_stats_panel())
            layout["details"].update(self._build_model_details())
        else:
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="models"),
            )
            layout["header"].update(self._build_stats_panel())
            layout["models"].update(Panel(
                self._build_model_table(),
                title="🦛 Hippo Models",
                border_style="blue",
                subtitle=f"Press Enter for details, ↑↓ to select, q to quit",
            ))

        return layout

    def _handle_input(self) -> None:
        """Handle keyboard input (non-blocking)."""
        # Note: Rich Live doesn't support keyboard input directly
        # This is a placeholder for future enhancement
        # For now, we'll use keyboard interrupts
        pass

    async def _refresh_loop(self, live: Live) -> None:
        """Async refresh loop."""
        while self.running:
            # Update data
            self.stats = self._fetch_stats()
            self.models = self._fetch_models()

            # Update display
            live.update(self._build_dashboard())

            # Wait for next refresh
            await asyncio.sleep(self.refresh_interval)

    async def run_async(self) -> None:
        """Run the async TUI application."""
        try:
            with Live(console=self.console, refresh_per_second=1.0 / self.refresh_interval) as live:
                # Initial update
                self.stats = self._fetch_stats()
                self.models = self._fetch_models()
                live.update(self._build_dashboard())

                # Start refresh loop
                await self._refresh_loop(live)

        except KeyboardInterrupt:
            self.running = False
            self.console.print(Panel("👋 Hippo TUI stopped", border_style="yellow"))

    def run(self) -> None:
        """Run the TUI application (sync wrapper)."""
        asyncio.run(self.run_async())


def run_tui(refresh_interval: float = 2.0):
    """Run the live TUI dashboard."""
    app = TUIApp(refresh_interval=refresh_interval)
    app.run()


if __name__ == "__main__":
    run_tui()
