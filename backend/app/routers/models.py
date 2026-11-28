from fastapi import APIRouter

from app.schemas import ModelInfo
from app.services.model_registry import registry

router = APIRouter()


@router.get("/models", response_model=list[ModelInfo])
def list_models() -> list[ModelInfo]:
    return registry.list()
