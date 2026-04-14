"""Model management routes: /api/pull, /api/search"""

import os
import json
import urllib.parse
import urllib.request

from fastapi import APIRouter, Request, Query
from fastapi.responses import JSONResponse

from hippo.dependencies import _get_config, _get_manager, _check_auth
from hippo.config import HippoConfig
from hippo.downloader import pull_model
from hippo import __version__

router = APIRouter()


@router.post("/api/pull")
async def pull(request: Request):
    """Pull (download) a model from HuggingFace."""
    auth_result = await _check_auth(request)
    if auth_result:
        return auth_result

    config: HippoConfig = request.app.state.config
    body = await request.json()
    name = body.get("name", "")

    try:
        path = pull_model(name, config.models_dir)
        return {"status": "success", "path": str(path)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/api/search")
async def search_models(q: str = Query("", alias="q"), limit: int = Query(10)):
    """Search GGUF models on HuggingFace."""
    if not q:
        return {"models": []}

    hf_endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    url = f"{hf_endpoint}/api/models?search={urllib.parse.quote(q + ' gguf')}&limit={limit}&sort=downloads&direction=-1&filter=gguf"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": f"hippo/{__version__}"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            items = json.loads(resp.read().decode())

        results = []
        for item in items[:limit]:
            results.append({
                "id": item.get("id", ""),
                "author": item.get("author", ""),
                "downloads": item.get("downloads", 0),
                "tags": item.get("tags", []),
                "url": f"{hf_endpoint}/{item.get('id', '')}",
            })
        return {"models": results}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
