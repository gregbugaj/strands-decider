"""HTTP server exposing the System One API.

The path and the request and response shapes follow the public Jev API documentation.
JevBench's typesafe adapter runs against this server unchanged (evaluation/jevbench.md).
Compatibility with the Jev API itself is not verified.
"""

from __future__ import annotations

import os
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .infer import EngineConfig, SystemOneEngine
from .modeling import StrandsDeciderModel
from .schema import SystemOneRequest, SystemOneResponse
from .vision import ImageLimits, ImagePayloadTooLarge


class RequestSizeLimit:
    """Bound the full body before FastAPI allocates parsed JSON/base64 strings."""

    def __init__(self, app: ASGIApp, maximum: int):
        self.app = app
        self.maximum = maximum

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] != "/v1/systemone":
            await self.app(scope, receive, send)
            return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > self.maximum:
                response = JSONResponse(
                    status_code=413,
                    content={"detail": f"request bytes exceed allowed {self.maximum}"},
                )
                await response(scope, receive, send)
                return
            if not message.get("more_body", False):
                break
        pending = True

        async def replay() -> Message:
            nonlocal pending
            if pending:
                pending = False
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


_engine: SystemOneEngine | None = None


def get_engine() -> SystemOneEngine:
    if _engine is None:  # pragma: no cover - guarded by create_app
        raise RuntimeError("engine not initialised")
    return _engine


def create_app(
    checkpoint: str,
    *,
    device: str = "cuda",
    use_prefix_cache: bool = True,
    model_name: str | None = None,
    attn_implementation: str | None = None,
    image_limits: ImageLimits | None = None,
) -> FastAPI:
    global _engine

    app = FastAPI(
        title="strands-decider System One",
        version="0.1.0",
        description="Typed, calibrated answers. Choice, Score and Noul over one state.",
    )

    # Identify the response's `model` by the checkpoint served, so multiple servers
    # on the same host cannot be confused. HF repo ids ("org/name") collapse to `name`.
    resolved_name = model_name or os.path.basename(checkpoint.rstrip("/")) or checkpoint

    model = StrandsDeciderModel.load(checkpoint, attn_implementation=attn_implementation)
    _engine = SystemOneEngine(
        model,
        EngineConfig(
            device=device,
            use_prefix_cache=use_prefix_cache,
            model_name=resolved_name,
            image_limits=image_limits or ImageLimits(),
        ),
    )

    app.add_middleware(RequestSizeLimit, maximum=(image_limits or ImageLimits()).max_request_bytes)
    engine = _engine

    @app.get("/health")
    def health() -> dict:
        eng = engine
        return {
            "status": "ok",
            "model": eng.cfg.model_name,
            "checkpoint": checkpoint,
            "base_model": eng.model.config.base_model,
            "num_slots": eng.model.config.num_slots,
            "max_length": eng.model.config.max_length,
            "temperature": eng.model.config.temperature,
            "device": eng.cfg.device,
            "prefix_cache": eng.cfg.use_prefix_cache
            and getattr(eng.model.config, "input_mode", "text") == "text",
            "input_mode": getattr(eng.model.config, "input_mode", "text"),
            "vision_budgets": {
                key: getattr(eng.model.config, key, None)
                for key in ("max_images", "max_visual_tokens_per_image", "max_total_visual_tokens")
            },
            "image_limits": vars(image_limits or ImageLimits()),
        }

    @app.post("/v1/systemone", response_model=SystemOneResponse)
    def systemone(request: SystemOneRequest) -> JSONResponse:
        eng = engine
        started = time.perf_counter()
        try:
            response = eng.evaluate(request)
        except ImagePayloadTooLarge as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except ValueError as exc:
            # Option count over num_slots, malformed permutation, etc. -- caller error.
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        elapsed_ms = (time.perf_counter() - started) * 1000
        payload = response.model_dump()
        payload["latency_ms"] = round(elapsed_ms, 2)
        return JSONResponse(content=payload)

    return app


def serve(
    checkpoint: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    device: str = "cuda",
    use_prefix_cache: bool = True,
    model_name: str | None = None,
    image_limits: ImageLimits | None = None,
) -> None:
    import uvicorn

    app = create_app(
        checkpoint,
        device=device,
        use_prefix_cache=use_prefix_cache,
        model_name=model_name,
        image_limits=image_limits,
    )
    # Single worker: the model owns the GPU, and forking more would just duplicate it.
    uvicorn.run(app, host=host, port=port, workers=1)
