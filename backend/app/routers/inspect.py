from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile

from app.errors import ApiError
from app.schemas import InspectResponse
from app.services.inference import MAX_BYTES, decode, inspect_image
from app.services.model_registry import registry

router = APIRouter()


@router.post("/inspect", response_model=InspectResponse)
async def inspect(
    image: Annotated[UploadFile, File()], model_id: Annotated[str, Form()]
) -> InspectResponse:
    try:
        meta, detector = registry.get(model_id)
    except KeyError:
        raise ApiError(404, "model_not_found", f"No model with id '{model_id}'.") from None
    data = await image.read(MAX_BYTES + 1)  # never read more than the limit plus one byte
    img = decode(data, image.filename or "", image.content_type or "")
    return inspect_image(img, meta, detector)
