"""Local image review and road damage detection API."""

import io
import time
from pathlib import Path
from threading import Lock
from typing import Optional

import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import ImageOps, JpegImagePlugin, PngImagePlugin

from ..severity.severity import compute_severity, get_severity_band
from .detector import Detector
from .schemas import DetectResponse, HealthResponse

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).with_name("static")
EXAMPLES = ROOT / "data" / "demo"
MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 24_000_000


def get_detector(request: Request) -> Detector:
    state = request.app.state
    with state.model_lock:
        if state.detector is None:
            for name in ("best.pt", "best.onnx"):
                path = ROOT / "models" / name
                if path.is_file():
                    state.detector = Detector(str(path), state.conf_threshold)
                    break
            else:
                raise HTTPException(503, "Place the trained best.pt or best.onnx in models/.")
    return state.detector


def create_app(model_path: Optional[str] = None, conf_threshold: float = 0.25) -> FastAPI:
    app = FastAPI(title="RoadScan", description="Road image review", version="0.2.0")
    app.state.detector = Detector(model_path, conf_threshold) if model_path else None
    app.state.model_lock = Lock()
    app.state.inference_lock = Lock()
    app.state.conf_threshold = conf_threshold
    started = time.monotonic()
    app.mount("/assets", StaticFiles(directory=STATIC), name="assets")

    @app.get("/", include_in_schema=False)
    def review():
        return FileResponse(STATIC / "index.html")

    @app.get("/examples")
    def examples():
        return sorted(p.name for p in EXAMPLES.glob("*") if p.is_file()
                      and not p.is_symlink() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    @app.get("/examples/{name}", include_in_schema=False)
    def example(name: str):
        if name not in examples():
            raise HTTPException(404, "Example unavailable")
        return FileResponse(EXAMPLES / name)

    @app.get("/health", response_model=HealthResponse)
    def health(detector: Detector = Depends(get_detector)):
        return HealthResponse(status="healthy", model_version=detector.model_version,
                              uptime_seconds=time.monotonic() - started)

    @app.post("/detect", response_model=DetectResponse)
    def detect(file: UploadFile = File(...),
               conf: float = Query(default=conf_threshold, gt=0, le=1),
               detector: Detector = Depends(get_detector)):
        if file.content_type not in {"image/jpeg", "image/jpg", "image/png"}:
            raise HTTPException(400, "Choose a JPG or PNG image.")
        contents = file.file.read(MAX_BYTES + 1)
        if len(contents) > MAX_BYTES:
            raise HTTPException(413, "Choose a file smaller than 10 MiB.")
        # Use only the two accepted decoders. Ultralytics patches Image.open to
        # install a HEIF plugin when an upload cannot be identified.
        decoder = PngImagePlugin.PngImageFile if contents.startswith(b"\x89PNG") else JpegImagePlugin.JpegImageFile
        try:
            with decoder(io.BytesIO(contents)) as source:
                if source.width * source.height > MAX_PIXELS:
                    raise HTTPException(413, "Choose an image of 24 megapixels or less.")
                image = ImageOps.exif_transpose(source).convert("RGB")
        except (OSError, ValueError, SyntaxError) as exc:
            raise HTTPException(400, "The image could not be decoded.") from exc

        with app.state.inference_lock:
            detections, latency = detector.predict(np.asarray(image), conf=conf)
        severity = compute_severity(detections, *image.size)
        return DetectResponse(image_w=image.width, image_h=image.height, detections=detections,
                              severity=severity, band=get_severity_band(severity),
                              model_version=detector.model_version, latency_ms=latency)

    return app


app = create_app()
