from fastapi import APIRouter

from app import config
from app.schemas import ResultTable
from app.services.results_store import load_tables

router = APIRouter()


@router.get("/results", response_model=list[ResultTable])
def results() -> list[ResultTable]:
    return load_tables(config.ARTIFACTS_DIR)
