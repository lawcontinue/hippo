"""Model management routes: /api/tags, /api/show, /api/delete, /v1/models"""

import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from hippo.dependencies import _get_manager, _check_auth
from hippo.gguf import _model_info, _read_gguf_metadata_fast

router = APIRouter()


@router.get("/api/tags")
async def list_models(request: Request):
    """List available models (Ollama compatible)."""
    manager = _get_manager(request)
    models = manager.list_available()
    model_list = []
    for m in models:
        info = _model_info(m["name"], manager)
        info["size"] = int(m["size_gb"] * 1024**3)
        model_list.append(info)

    loaded_names = set(m["name"] for m in models)
    for entry in manager.list_loaded():
        name = entry["name"]
        if name not in loaded_names:
            model_list.append(_model_info(name, manager))

    return {"models": model_list}


@router.post("/api/show")
async def show_model(request: Request):
    """Show model details (Ollama compatible)."""
    manager = _get_manager(request)
    body = await request.json()
    name = body.get("name", "") or body.get("model", "")

    model_path = manager._resolve_model_path(name)
    if not model_path:
        return JSONResponse({"error": f"Model '{name}' not found"}, status_code=404)

    family = manager._detect_family(model_path)
    size_bytes = model_path.stat().st_size

    # Extract quantization from filename (e.g., Q4_K_M)
    quant = "unknown"
    fname = model_path.name.upper()
    for q in ["Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q5_0", "Q4_K_M", "Q4_K_S", "Q4_0", "Q3_K_M", "Q3_K_S", "Q2_K", "F16", "F32"]:
        if q in fname:
            quant = q
            break

    # Read context length from GGUF metadata directly (fast, <1ms)
    # No need to load the model with llama_cpp
    n_ctx = 4096
    param_size = ""
    meta = _read_gguf_metadata_fast(model_path)
    if meta:
        n_ctx = int(meta.get("llama.context_length",
                   meta.get("default.context_length",
                   meta.get("general.context_length", 4096))))
        # Also try to get parameter count for parameter_size
        n_params = meta.get("llama.parameter_count",
                   meta.get("general.parameter_count", 0))
        if n_params:
            param_size = f"{n_params / 1e9:.1f}B"

    return {
        "modelfile": "",
        "parameters": "",
        "template": "",
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": family,
            "families": [family],
            "parameter_size": param_size,
            "quantization_level": quant,
        },
        "model_info": {
            "general.architecture": family,
            "general.file_type": quant,
            "llama.context_length": n_ctx,
        },
        "modified_at": "",
        "type": "model",
        "size": size_bytes,
    }


@router.delete("/api/delete")
async def delete(request: Request):
    """Delete a model from disk."""
    auth_result = await _check_auth(request)
    if auth_result:
        return auth_result

    manager = _get_manager(request)
    body = await request.json()
    name = body.get("name", "")

    manager.unload(name)

    model_path = manager._resolve_model_path(name)
    if model_path and model_path.exists():
        model_path.unlink()
        return {"status": "success"}
    return JSONResponse({"error": f"Model '{name}' not found"}, status_code=404)


@router.get("/v1/models")
async def list_models_openai(request: Request):
    """List models in OpenAI-compatible format."""
    manager = _get_manager(request)
    models = manager.list_available()
    data = []
    for m in models:
        data.append({
            "id": m["name"],
            "object": "model",
            "created": int(time.time()),
            "owned_by": "hippo",
        })
    for entry in manager.list_loaded():
        name = entry["name"]
        if not any(d["id"] == name for d in data):
            data.append({
                "id": name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "hippo",
            })
    return {"object": "list", "data": data}
