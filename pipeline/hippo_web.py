#!/usr/bin/env python3
"""
hippo_web.py — Gradio Web UI for Hippo Pipeline.

A simple chat interface that connects to the Hippo API server.
Shows real-time performance metrics (tok/s, AR, memory).

Usage:
  # Start API server first
  python3 hippo_api.py --port 8080 --mode dflash --model qwen3-4b

  # Then start Web UI
  python3 hippo_web.py --api http://localhost:8080
  python3 hippo_web.py --api http://localhost:8080 --key sk-hippo-secret
"""

from __future__ import annotations

import argparse
import os

import requests

try:
    import gradio as gr
except ImportError:
    print("❌ gradio not installed. Run: pip install gradio")
    raise

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def call_hippo(api_url: str, api_key: str | None, messages: list[dict],
               model: str, max_tokens: int, temperature: float) -> tuple[str, dict]:
    """Call Hippo API and return (response_text, metrics)."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    resp = requests.post(
        f"{api_url}/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()

    text = data["choices"][0]["message"]["content"]
    metrics = data.get("_hippo", {})
    return text, metrics


def get_models(api_url: str, api_key: str | None) -> list[str]:
    """Fetch available models from Hippo API."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        resp = requests.get(f"{api_url}/v1/models", headers=headers, timeout=5)
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", [])]
    except Exception:
        return ["qwen3-4b", "gemma-3-12b", "qwen3-8b"]


def create_ui(api_url: str, api_key: str | None):
    """Create Gradio UI."""

    def chat_fn(message, history, model, max_tokens, temperature):
        messages = []
        for user_msg, assistant_msg in history:
            messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": message})

        text, metrics = call_hippo(api_url, api_key, messages, model, max_tokens, temperature)

        # Format metrics as footer
        tok_s = metrics.get("tok_s", 0)
        ar = metrics.get("ar")
        time_s = metrics.get("time_s", 0)
        mode = metrics.get("mode", "?")

        footer = f"\n\n---\n🦛 **{tok_s:.1f} tok/s** | AR: {ar:.1%} | {time_s:.1f}s | mode: {mode}" if tok_s else ""
        return text + footer

    def health_fn():
        try:
            resp = requests.get(f"{api_url}/health", timeout=5)
            data = resp.json()
            return f"✅ {data['status']} | model: {data.get('model', '?')} | mode: {data.get('mode', '?')}"
        except Exception as e:
            return f"❌ {e}"

    models = get_models(api_url, api_key)

    with gr.Blocks(
        title="🦛 Hippo Chat",
        theme=gr.themes.Soft(),
        css="""
        .contain { max-width: 900px; margin: auto; }
        footer { display: none !important; }
        """
    ) as demo:
        gr.Markdown(
            "# 🦛 Hippo Chat\n"
            "Local LLM inference powered by Hippo Pipeline. "
            "Select a model and start chatting."
        )

        with gr.Row():
            health_btn = gr.Button("🔄 Health Check", size="sm")
            health_output = gr.Textbox(label="Status", interactive=False, scale=3)
        health_btn.click(fn=health_fn, outputs=health_output)

        with gr.Row():
            model_dd = gr.Dropdown(choices=models, value=models[0] if models else "qwen3-4b", label="Model")
            max_tokens_sl = gr.Slider(64, 4096, value=512, step=64, label="Max Tokens")
            temp_sl = gr.Slider(0.0, 1.5, value=0.0, step=0.1, label="Temperature")

        gr.ChatInterface(
            fn=chat_fn,
            additional_inputs=[model_dd, max_tokens_sl, temp_sl],
            type="messages",
        )

        gr.Markdown(
            f"**API**: `{api_url}` | "
            f"[Hippo Pipeline](https://github.com/lawcontinue/hippo-pipeline)"
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Hippo Web UI")
    parser.add_argument("--api", default="http://localhost:8080", help="Hippo API URL")
    parser.add_argument("--key", default=None, help="API key")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public URL")
    args = parser.parse_args()

    print("🦛 Hippo Web UI")
    print(f"   API: {args.api}")
    print(f"   URL: http://localhost:{args.port}")
    print()

    demo = create_ui(args.api, args.key)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
