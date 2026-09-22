"""FastAPI application - the ECG Intelligence inference API.

This is a from-scratch FastAPI rewrite of the project's former Flask
backend (see git history for the previous app/main.py). The migration
only replaces the web framework: every ECG-specific behavior -
preprocessing, aggregation, calibration, explainability, upload
security, model bootstrap - is unchanged and still lives in ecg/ (ML
logic) and app/model_state.py + app/uploads.py (HTTP-layer wiring), not
in this file or in app/routes/.

This service's only client is the static frontend (public/), deployed
separately on Vercel - this API serves no HTML, only JSON.
"""
from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse

from app import model_state
from app.routes import health, model_info, predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ecg.app")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # No-op if the model is already on disk (the normal case for local
    # dev); downloads it from ECG_MODEL_URL first if it's missing and
    # that env var is set (the deployment case - see ecg/model_bootstrap.py).
    model_state.load()
    yield


app = FastAPI(
    title="ECG Intelligence API",
    description=(
        "RCNN (1D CNN + LSTM) ECG arrhythmia classification, trained on PTB-XL. "
        "Research/portfolio project - not a certified medical device."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: the Vercel static frontend calls this API cross-origin. Only the
# exact origin in ECG_FRONTEND_ORIGIN is allowed - deliberately not "*".
# Unset means no allowed origins, so a cross-origin browser call simply
# gets no CORS headers and fails client-side (same-origin / non-browser
# calls, e.g. curl or the container health check, are unaffected either
# way).
_FRONTEND_ORIGIN = os.environ.get("ECG_FRONTEND_ORIGIN")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[_FRONTEND_ORIGIN] if _FRONTEND_ORIGIN else [],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
    # Keeps the API's error shape ({"error": "..."}) that the frontend
    # (public/script.js) already expects, instead of FastAPI's default
    # {"detail": "..."}.
    logger.info("Rejecting request: %s", exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    logger.info("Rejecting request: invalid request (%s)", exc)
    return JSONResponse(status_code=422, content={"error": "Invalid request"})


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    # Belt-and-braces: no code path - anticipated or not - should ever
    # return a raw traceback or filesystem path to the client.
    logger.exception("Unhandled exception")
    return JSONResponse(status_code=500, content={"error": "An unexpected error occurred"})


@app.get("/", include_in_schema=False)
def root() -> dict:
    # This API's only real client is the separately-deployed static
    # frontend (public/, on Vercel); this just avoids a bare 404 for
    # anyone who hits the backend URL directly.
    return {"name": "ECG Intelligence API", "docs": "/docs"}


app.include_router(health.router)
app.include_router(model_info.router)
app.include_router(predict.router)


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("ECG_APP_HOST", "127.0.0.1")
    port = int(os.environ.get("ECG_APP_PORT", 5000))
    uvicorn.run(app, host=host, port=port)
