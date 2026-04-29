"""Model quantization support for Hippo."""

import sys
from pathlib import Path

QUANT_FORMATS = {
    "q2_k": "Q2_K — smallest, lowest quality",
    "q3_k_s": "Q3_K_S — small, low quality",
    "q3_k_m": "Q3_K_M — small, moderate quality",
    "q3_k_l": "Q3_K_L — small-medium, moderate quality",
    "q4_0": "Q4_0 — legacy, small",
    "q4_k_s": "Q4_K_S — medium, good quality",
    "q4_k_m": "Q4_K_M — medium, recommended default",
    "q5_0": "Q5_0 — medium-large, good quality",
    "q5_k_s": "Q5_K_S — large, high quality",
    "q5_k_m": "Q5_K_M — large, high quality",
    "q6_k": "Q6_K — very large, very high quality",
    "q8_0": "Q8_0 — largest, near-original quality",
    "f16": "F16 — no quantization (full precision)",
    "f32": "F32 — no quantization (32-bit float)",
}


def list_formats():
    """Print available quantization formats."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Available Quantization Formats")
    table.add_column("Format", style="cyan")
    table.add_column("Description", style="white")

    for fmt, desc in QUANT_FORMATS.items():
        table.add_row(fmt, desc)

    console.print(table)


def quantize_model(input_path: str, output_path: str, fmt: str = "q4_k_m"):
    """Quantize a GGUF model to a different format.

    Uses llama-cpp-python's convert functionality.
    Falls back to subprocess llama-quantize if available.
    """
    fmt = fmt.lower().replace("-", "_")

    if fmt not in QUANT_FORMATS:
        print(f"❌ Unknown format: {fmt}")
        print("   Run 'hippo quantize --list' to see available formats")
        sys.exit(1)

    input_p = Path(input_path)
    if not input_p.exists():
        print(f"❌ Input file not found: {input_path}")
        sys.exit(1)

    from rich.console import Console

    console = Console()
    console.print(f"🔄 Quantizing [cyan]{input_path}[/] → [green]{output_path}[/] ({fmt})")

    # Try llama-quantize CLI first (most reliable)
    import shutil
    import subprocess

    llama_quantize = shutil.which("llama-quantize")
    if llama_quantize:
        console.print("   Using llama-quantize CLI")
        result = subprocess.run(
            [llama_quantize, str(input_p), output_path, fmt],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"❌ Quantization failed: {result.stderr}")
            sys.exit(1)
        console.print(f"✅ Done: {output_path}")
        return

    # Try Python API
    try:
        from llama_cpp import llama_model_quantize  # noqa

        # Direct C API call
        from llama_cpp import _llama_lib
        import ctypes

        input_c = ctypes.c_char_p(str(input_p).encode())
        output_c = ctypes.c_char_p(output_path.encode())
        fmt_c = ctypes.c_char_p(fmt.encode())

        result = _llama_lib.llama_model_quantize(input_c, output_c, fmt_c)
        if result != 0:
            print(f"❌ Quantization failed with code {result}")
            sys.exit(1)

        console.print(f"✅ Done: {output_path}")
    except (ImportError, AttributeError):
        console.print("❌ No quantization backend available.")
        console.print("   Install llama.cpp tools: [bold]brew install llama.cpp[/]")
        console.print("   Or: [bold]pip install llama-cpp-python[server][/]")
        sys.exit(1)
