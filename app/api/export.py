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
from app.cv.image import load_image
from app.cv.removal import lama_available
from app.db.base import get_async_session
from app.db.models import Account
from app.services.export import faces_to_remove
from app.storage.base import get_storage

router = APIRouter(tags=["export"])


class ExportRequest(BaseModel):
    remove_strangers: bool = False


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
