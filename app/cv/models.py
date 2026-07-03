"""InsightFace model loading, staged for a 6 GB VRAM budget.

``buffalo_l`` bundles SCRFD detection + ArcFace-R50 recognition (plus extras we
don't need). We load only ``detection`` and ``recognition`` to keep VRAM low, and
expose ``release_face_app()`` so heavier later-phase models can be swapped in on
the same GPU (Constitution: models run staged, load-and-release).
"""

import gc
import threading

from insightface.app import FaceAnalysis

from app.config import get_settings

_lock = threading.Lock()
_face_app: FaceAnalysis | None = None


def get_face_app() -> FaceAnalysis:
    """Return a prepared FaceAnalysis (detection + recognition), loading once."""
    global _face_app
    with _lock:
        if _face_app is None:
            settings = get_settings()
            app = FaceAnalysis(
                name=settings.insightface_model,
                allowed_modules=["detection", "recognition"],
                providers=settings.onnx_provider_list,
            )
            # ctx_id=0 → first CUDA device; onnxruntime falls back to CPU if the
            # CUDA provider is unavailable.
            app.prepare(ctx_id=0, det_size=(640, 640))
            _face_app = app
        return _face_app


def release_face_app() -> None:
    """Release the loaded model so VRAM can be reclaimed for another stage."""
    global _face_app
    with _lock:
        _face_app = None
        gc.collect()
