"""Chart service — generates Plotly JSON figures from loaded data."""

import json

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from backend.services.data_service import DataService

# Shared data service instance
_data_svc = DataService()


def _get_df():
    return _data_svc.df


def _fig_to_dict(fig: go.Figure) -> dict:
    return json.loads(pio.to_json(fig))


class ChartService:
    def __init__(self):
        self._data = _data_svc

    @property
    def data_svc(self) -> DataService:
        return self._data

    def distribution(self, column: str, bins: int = 30) -> dict:
        df = _get_df()
        fig = px.histogram(df, x=column, nbins=bins, title=f"Distribution of {column}")
        fig.update_layout(bargap=0.05)
        return _fig_to_dict(fig)

    def correlation_matrix(self, method: str = "pearson") -> dict:
        df = _get_df()
        corr = df.select_dtypes(include="number").corr(method=method)
        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title=f"Correlation Matrix ({method})",
        )
        return _fig_to_dict(fig)

    def bar(self, column: str, top_n: int = 20) -> dict:
        df = _get_df()
        counts = df[column].value_counts().head(top_n).reset_index()
        counts.columns = [column, "count"]
        fig = px.bar(counts, x="count", y=column, orientation="h", title=f"{column} — Top {top_n}")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        return _fig_to_dict(fig)

    def scatter(self, x: str, y: str, hue: str | None = None) -> dict:
        df = _get_df()
        fig = px.scatter(df, x=x, y=y, color=hue, opacity=0.6, title=f"{y} vs {x}")
        return _fig_to_dict(fig)

    def time_series(self, date_col: str, value_col: str, freq: str | None = None) -> dict:
        df = _get_df()
        ts = df.copy()
        ts[date_col] = ts[date_col].astype("datetime64[ns]")
        if freq:
            ts = ts.set_index(date_col).resample(freq)[[value_col]].mean().reset_index()
        fig = px.line(ts, x=date_col, y=value_col, title=f"{value_col} over time")
        return _fig_to_dict(fig)
