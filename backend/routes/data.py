"""Data endpoints — load, query, and inspect datasets."""

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from backend.services.data_service import DataService

router = APIRouter()
data_svc = DataService()


@router.get("/datasets")
async def list_datasets():
    """List available datasets in data/raw and data/processed."""
    return data_svc.list_datasets()


@router.post("/load/{filename}")
async def load_dataset(filename: str, subdir: str = Query("raw")):
    """Load a dataset into memory."""
    data_svc.load(filename, subdir=subdir)
    return {"loaded": filename, "shape": list(data_svc.current_shape())}


@router.get("/info")
async def dataset_info():
    """Return column info for the loaded dataset."""
    return JSONResponse(content=data_svc.info())


@router.get("/head")
async def head(n: int = Query(20, ge=1, le=500)):
    """Return first n rows as JSON records."""
    return JSONResponse(content=data_svc.head(n))


@router.get("/columns")
async def columns():
    """Return column names and dtypes."""
    return data_svc.columns()


@router.get("/stats")
async def summary_stats():
    """Return extended summary statistics."""
    return JSONResponse(content=data_svc.stats())
