"""InsightFace model loading, staged for a 6 GB VRAM budget.

``buffalo_l`` bundles SCRFD detection + ArcFace-R50 recognition (plus extras we
don't need). We load only ``detection`` and ``recognition`` to keep VRAM low, and
expose ``release_face_app()`` so heavier later-phase models can be swapped in on
the same GPU (Constitution: models run staged, load-and-release).
"""

import gc
import logging
import threading

from insightface.app import FaceAnalysis

from app.config import get_settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_face_app: FaceAnalysis | None = None


def _preload_cuda_dlls(requested: list[str]) -> None:
    """Make the CUDA/cuDNN runtime DLLs loadable before any ONNX session builds.

    onnxruntime-gpu is a CUDA-12 + cuDNN-9 build: it ships the CUDA execution-
    provider DLL but not the runtime it links against. On Windows those live in the
    ``nvidia-*-cu12`` pip wheels (``cudnn64_9.dll``, ``cublasLt64_12.dll``, …) under
    ``site-packages/nvidia/*/bin`` — NOT on the default DLL search path. Without
    them the provider fails to load (or errors at the first matmul) and the whole
    face pipeline silently drops to CPU.

    ``onnxruntime.preload_dlls()`` guesses a CUDA version and misses the cu12
    cuBLAS here, so we instead register every ``nvidia/*/bin`` directory explicitly
    with the Windows loader. Best-effort: never let a preload hiccup stop import.
    """
    if "CUDAExecutionProvider" not in requested:
        return
    import glob
    import os

    try:
        import nvidia
    except Exception:  # nvidia wheels not installed → nothing to add (CPU path)
        return

    import ctypes

    bin_dirs: list[str] = []
    for pkg_root in nvidia.__path__:
        try:
            entries = os.listdir(pkg_root)
        except OSError:
            continue
        for entry in entries:
            bin_dir = os.path.join(pkg_root, entry, "bin")
            if os.path.isdir(bin_dir):
                bin_dirs.append(bin_dir)
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(bin_dir)
                    except OSError:  # pragma: no cover - environment dependent
                        pass

    # Some CUDA libraries pull in siblings (e.g. cuBLAS → cublasLt64_12.dll) by
    # bare name via LoadLibrary, which searches PATH — NOT add_dll_directory dirs.
    # So also put the wheel bin dirs on PATH.
    if bin_dirs:
        os.environ["PATH"] = os.pathsep.join(bin_dirs) + os.pathsep + os.environ.get("PATH", "")

    # Eagerly load the DLLs whose bare-name lookups fail at the first matmul
    # (cublasLt before cublas, which depends on it), by full path, so the module is
    # already resident when onnxruntime asks for it by name.
    for pattern in ("cublasLt64_*.dll", "cublas64_*.dll", "cudnn64_*.dll"):
        for bin_dir in bin_dirs:
            for dll in glob.glob(os.path.join(bin_dir, pattern)):
                try:
                    ctypes.WinDLL(dll)
                except OSError:  # pragma: no cover - environment dependent
                    pass

    logger.debug("Registered %d nvidia CUDA DLL directories for onnxruntime.", len(bin_dirs))
    # NB: we deliberately do NOT call onnxruntime.preload_dlls() — the explicit
    # WinDLL loads above are sufficient for the provider to load, and preload_dlls
    # re-CDLLs the large CUDA DLLs, which can stall for a long time behind on-access
    # AV scanning on this box.


def _report_active_providers(app: FaceAnalysis, requested: list[str]) -> None:
    """Log the providers the models ACTUALLY loaded on, flagging a CPU fallback.

    ``ort.get_available_providers()`` can list CUDA even when a session fails to
    initialize it (missing cuDNN, etc.) and quietly falls back. The ground truth is
    the provider each prepared session reports, so we inspect that instead.
    """
    active = {
        p
        for model in app.models.values()
        for p in getattr(getattr(model, "session", None), "get_providers", list)()
    }
    if "CUDAExecutionProvider" in requested and "CUDAExecutionProvider" not in active:
        logger.warning(
            "CUDAExecutionProvider was requested (ONNX_PROVIDERS=%s) but the face "
            "pipeline is running on CPU (active: %s). Fix: install onnxruntime-gpu "
            "(not plain onnxruntime) and the matching CUDA + cuDNN 9 runtime.",
            requested,
            sorted(active),
        )
    elif "CUDAExecutionProvider" in active:
        logger.info("Face pipeline using CUDAExecutionProvider (GPU).")


def get_face_app() -> FaceAnalysis:
    """Return a prepared FaceAnalysis (detection + recognition), loading once."""
    global _face_app
    with _lock:
        if _face_app is None:
            settings = get_settings()
            _preload_cuda_dlls(settings.onnx_provider_list)
            app = FaceAnalysis(
                name=settings.insightface_model,
                allowed_modules=["detection", "recognition"],
                providers=settings.onnx_provider_list,
            )
            # ctx_id=0 → first CUDA device; onnxruntime falls back to CPU if the
            # CUDA provider is unavailable.
            app.prepare(ctx_id=0, det_size=(640, 640))
            _report_active_providers(app, settings.onnx_provider_list)
            _face_app = app
        return _face_app


def release_face_app() -> None:
    """Release the loaded model so VRAM can be reclaimed for another stage."""
    global _face_app
    with _lock:
        _face_app = None
        gc.collect()
