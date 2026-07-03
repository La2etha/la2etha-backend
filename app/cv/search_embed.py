"""SigLIP-2-Base image + text embeddings for natural-language search (F5, FR-015).

SigLIP-2 maps images and text into one shared space, so a free-text query can be
matched to photos by cosine similarity — and it's multilingual, so Arabic queries
work (research-cv-ai.md). Embeddings are L2-normalized so cosine == dot product,
matching how pgvector's cosine index is used.

The model is optional and lazy-loaded: torch + transformers are the heavy `search`
extra, not core deps. If they're absent, ``search_available()`` returns False and
the app runs normally with search disabled — nothing here is imported at module
load, so importing this file never fails.
"""

from __future__ import annotations

import threading

import numpy as np

from app.config import get_settings
from app.cv.vectors import l2_normalize

_lock = threading.Lock()
_model = None
_processor = None


def search_available() -> bool:
    """True if the SigLIP dependencies are importable (so search can run)."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except Exception:
        return False


def _load():
    """Lazily load SigLIP-2 (model + processor) once. ponytail: load-and-keep is
    fine — one model at a time on the 6 GB GPU; the face pipeline releases first."""
    global _model, _processor
    if _model is not None:
        return _model, _processor
    with _lock:
        if _model is None:
            import torch
            from transformers import AutoModel, AutoProcessor

            name = get_settings().search_model
            model = AutoModel.from_pretrained(name)
            model.eval()
            if torch.cuda.is_available():
                model = model.to("cuda")
            _processor = AutoProcessor.from_pretrained(name)
            _model = model
    return _model, _processor


def _to_numpy(tensor) -> np.ndarray:
    return tensor.detach().cpu().float().numpy()[0]


def embed_image(image_rgb: np.ndarray) -> np.ndarray:
    """Embed an RGB image array → L2-normalized SigLIP image vector."""
    import torch
    from PIL import Image

    model, processor = _load()
    pil = Image.fromarray(image_rgb)
    inputs = processor(images=pil, return_tensors="pt")
    if next(model.parameters()).is_cuda:
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
    return l2_normalize(_to_numpy(feats))


def embed_text(query: str) -> np.ndarray:
    """Embed a free-text query → L2-normalized SigLIP text vector."""
    import torch

    model, processor = _load()
    inputs = processor(
        text=[query], return_tensors="pt", padding="max_length", truncation=True
    )
    if next(model.parameters()).is_cuda:
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
    return l2_normalize(_to_numpy(feats))


if __name__ == "__main__":  # runnable self-check (Constitution V)
    # Works with or without the heavy deps installed: verify the graceful contract.
    if search_available():
        v = embed_text("a person near a birthday cake")
        assert v.shape[0] == 768 and abs(float(np.linalg.norm(v)) - 1.0) < 1e-3
        print("search_embed self-check OK (model loaded)")
    else:
        raised = False
        try:
            embed_text("x")
        except Exception:
            raised = True
        assert raised, "embed_text must raise when deps are missing"
        print("search_embed self-check OK (deps absent -- search disabled, no crash)")
