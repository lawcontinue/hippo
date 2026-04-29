"""Transport layer — HTTP-based communication between cluster nodes.

For Phase 0 we use simple HTTP/JSON. Future: MessagePack + TCP for lower latency.
"""

import logging
from typing import Optional

import aiohttp

logger = logging.getLogger("hippo.cluster.transport")

DEFAULT_TIMEOUT = 120  # seconds for inference


class Transport:
    """HTTP transport for inter-node communication.

    Wraps aiohttp for request/response between gateway and workers.
    """

    def __init__(self, timeout: int = DEFAULT_TIMEOUT):
        self._session: Optional[aiohttp.ClientSession] = None
        self._timeout = timeout

    async def start(self):
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._timeout)
        )

    async def stop(self):
        if self._session:
            await self._session.close()
            self._session = None

    async def post(self, url: str, payload: dict) -> dict:
        """Send a POST request and return JSON response."""
        if not self._session:
            raise RuntimeError("Transport not started")

        async with self._session.post(url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def post_stream(self, url: str, payload: dict):
        """Send a POST request and yield streamed response chunks."""
        if not self._session:
            raise RuntimeError("Transport not started")

        async with self._session.post(url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.content:
                yield line

    async def get(self, url: str) -> dict:
        """Send a GET request and return JSON response."""
        if not self._session:
            raise RuntimeError("Transport not started")

        async with self._session.get(url) as resp:
            resp.raise_for_status()
            return await resp.json()
