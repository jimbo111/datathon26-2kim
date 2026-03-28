"""Chart endpoints — return Plotly JSON for frontend rendering."""

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from backend.services.chart_service import ChartService

router = APIRouter()
charts = ChartService()


@router.get("/")
async def list_charts():
    """List all available chart endpoints."""
    return {
        "available": [
            {"name": "distribution", "params": ["column", "bins"]},
            {"name": "correlation", "params": ["method"]},
            {"name": "bar", "params": ["column", "top_n"]},
            {"name": "scatter", "params": ["x", "y", "hue"]},
            {"name": "timeseries", "params": ["date_col", "value_col", "freq"]},
        ]
    }


@router.get("/distribution")
async def distribution(column: str, bins: int = Query(30, ge=5, le=200)):
    fig_json = charts.distribution(column, bins=bins)
    return JSONResponse(content=fig_json)


@router.get("/correlation")
async def correlation(method: str = Query("pearson", pattern="^(pearson|spearman|kendall)$")):
    fig_json = charts.correlation_matrix(method=method)
    return JSONResponse(content=fig_json)


@router.get("/bar")
async def bar_chart(column: str, top_n: int = Query(20, ge=1, le=100)):
    fig_json = charts.bar(column, top_n=top_n)
    return JSONResponse(content=fig_json)


@router.get("/scatter")
async def scatter(x: str, y: str, hue: str | None = None):
    fig_json = charts.scatter(x, y, hue=hue)
    return JSONResponse(content=fig_json)


@router.get("/timeseries")
async def timeseries(date_col: str, value_col: str, freq: str | None = None):
    fig_json = charts.time_series(date_col, value_col, freq=freq)
    return JSONResponse(content=fig_json)
