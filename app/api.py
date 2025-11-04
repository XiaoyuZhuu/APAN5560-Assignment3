# app/api.py
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, Response
from starlette.status import HTTP_503_SERVICE_UNAVAILABLE

from .inference_gan import GanSampler

app = FastAPI(title="Assignment 3 - GAN on MNIST")

_gan_sampler: GanSampler | None = None

def get_sampler() -> GanSampler:
    global _gan_sampler
    if _gan_sampler is None:
        _gan_sampler = GanSampler("models/gan_G.pt")
    return _gan_sampler

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/gan/generate.png", summary="Return a PNG grid of generated digits")
def generate_png(n: int = Query(16, ge=1, le=64), seed: int | None = None, nrow: int = 4):
    try:
        sampler = get_sampler()
    except Exception as e:
        return JSONResponse(
            {"error": f"Model not ready: {e}"},
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
        )
    png = sampler.sample_png_bytes(n=n, seed=seed, nrow=nrow)
    return Response(content=png, media_type="image/png")

@app.get("/gan/generate.json", summary="Return base64 PNG of generated digits")
def generate_json(n: int = Query(16, ge=1, le=64), seed: int | None = None, nrow: int = 4):
    try:
        sampler = get_sampler()
    except Exception as e:
        return JSONResponse(
            {"error": f"Model not ready: {e}"},
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
        )
    b64 = sampler.sample_base64(n=n, seed=seed, nrow=nrow)
    return JSONResponse({"image_base64": b64, "n": n, "seed": seed, "nrow": nrow})
