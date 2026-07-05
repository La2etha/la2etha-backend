"""Privacy-preserving export endpoint (F6, FR-016).

Access-guarded (only someone verified in the photo, or the host, may export it).
``remove_strangers`` is opt-in and off by default; when off, the original bytes
are returned unchanged. When on, unclaimed background people are inpainted out.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.access.guard import require_photo_read
from app.auth.users import current_active_user
from app.cv.edit_gemini import edit_image, gemini_available
from app.cv.image import load_image
from app.cv.removal import lama_available
from app.config import get_settings
from app.db.base import get_async_session
from app.db.models import Account, Event
from app.services.export import faces_to_remove, is_solo_editable
from app.storage.base import get_storage

router = APIRouter(tags=["export"])


class ExportRequest(BaseModel):
    remove_strangers: bool = False


class EditRequest(BaseModel):
    prompt: str
    consent: bool = False


def _encode_jpeg(image_rgb) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(image_rgb).save(buf, format="JPEG", quality=92)
    return buf.getvalue()


@router.post("/photos/{photo_id}/export")
async def export_photo(
    photo_id: uuid.UUID,
    payload: ExportRequest,
    user: Account = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> Response:
    """Export a photo, optionally with unclaimed background people removed."""
    photo = await require_photo_read(session, user.id, photo_id)
    data = get_storage().get(photo.storage_key)

    if not payload.remove_strangers:
        # Default path: original bytes, no model needed (SC — off by default).
        return Response(content=data, media_type="image/jpeg")

    if not lama_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Background removal is unavailable — the LaMa model isn't installed.",
        )

    bboxes = await faces_to_remove(session, photo_id)
    if not bboxes:
        # Nothing to remove — return the original rather than a pointless re-encode.
        return Response(content=data, media_type="image/jpeg")

    # Imported lazily so the endpoint module doesn't require torch at import time.
    from app.cv.removal import remove_regions

    rgb, _, _, _ = load_image(data)
    edited = remove_regions(rgb, bboxes)
    return Response(content=_encode_jpeg(edited), media_type="image/jpeg")


@router.post("/photos/{photo_id}/edit")
async def edit_photo(
    photo_id: uuid.UUID,
    payload: EditRequest,
    user: Account = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> Response:
    """AI-edit a SOLO photo of the caller (F7 stretch). Opt-in, consented, and
    sent to the cloud only when the photo contains no one but the caller. The
    original is untouched — the edited copy is returned as new bytes."""
    if not payload.consent:
        raise HTTPException(status_code=422, detail="Explicit consent is required to AI-edit.")
    if not payload.prompt.strip():
        raise HTTPException(status_code=422, detail="An edit instruction is required.")

    photo = await require_photo_read(session, user.id, photo_id)

    # Solo guard: the global edit_solo_only flag is the secure default and the
    # only way to disable it app-wide (dev escape hatch, never in a deploy). An
    # event's ai_edit_scope=any_photo (spec 005 US5) is a per-event host opt-in
    # that relaxes it for that event only; consent is still required either way.
    event = await session.get(Event, photo.event_id)
    solo_required = get_settings().edit_solo_only and (
        event is None or event.ai_edit_scope != "any_photo"
    )
    if solo_required and not await is_solo_editable(session, photo_id, user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="AI editing is only allowed on a photo of just you (no other people).",
        )
    if not gemini_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI editing is unavailable — no Gemini API key is configured.",
        )

    data = get_storage().get(photo.storage_key)
    edited = edit_image(data, payload.prompt.strip())
    return Response(content=edited, media_type="image/jpeg")
