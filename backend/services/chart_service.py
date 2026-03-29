"""Chart service — generates Plotly JSON figures from loaded data."""

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from backend.services.data_service import DataService
from backend.services.sample_data import generate_health_data

# Shared data service instance
_data_svc = DataService()

# Health data (sample until Jimmy delivers real pipeline)
_health_df = generate_health_data()


def _get_df():
    return _data_svc.df


def _decode_bdata(obj):
    """Recursively decode Plotly 6.x binary-encoded arrays to plain lists."""
    import base64
    import struct

    DTYPE_MAP = {
        "i1": ("b", 1), "u1": ("B", 1),
        "i2": ("<h", 2), "u2": ("<H", 2),
        "i4": ("<i", 4), "u4": ("<I", 4),
        "f4": ("<f", 4), "f8": ("<d", 8),
    }

    if isinstance(obj, dict):
        if "bdata" in obj and "dtype" in obj:
            raw = base64.b64decode(obj["bdata"])
            fmt, size = DTYPE_MAP.get(obj["dtype"], ("<d", 8))
            count = len(raw) // size
            values = list(struct.unpack(f"<{count}{fmt[-1]}", raw))
            return values
        return {k: _decode_bdata(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_bdata(item) for item in obj]
    return obj


def _fig_to_dict(fig: go.Figure) -> dict:
    raw = json.loads(pio.to_json(fig))
    return _decode_bdata(raw)


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

    # --- Health Analysis Choropleths ---

    def choropleth_food_access(self) -> dict:
        df = _health_df
        fig = px.choropleth(
            df,
            locations="state",
            locationmode="USA-states",
            color="food_access_score",
            color_continuous_scale="RdYlGn",
            range_color=[20, 95],
            scope="usa",
            hover_name="state_name",
            hover_data=["food_access_score", "pct_food_desert", "median_income"],
            title="Food Access Score by State",
        )
        fig.update_layout(
            geo=dict(bgcolor="rgba(0,0,0,0)", lakecolor="rgba(0,0,0,0)"),
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            font_color="#e1e4e8",
            coloraxis_colorbar_title="Score",
            margin=dict(t=50, b=10, l=10, r=10),
        )
        return _fig_to_dict(fig)

    def choropleth_diabetes(self) -> dict:
        df = _health_df
        fig = px.choropleth(
            df,
            locations="state",
            locationmode="USA-states",
            color="diabetes_rate",
            color_continuous_scale="YlOrRd",
            range_color=[5, 18],
            scope="usa",
            hover_name="state_name",
            hover_data=["diabetes_rate", "obesity_rate", "life_expectancy"],
            title="Diabetes Rate (%) by State",
        )
        fig.update_layout(
            geo=dict(bgcolor="rgba(0,0,0,0)", lakecolor="rgba(0,0,0,0)"),
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            font_color="#e1e4e8",
            coloraxis_colorbar_title="%",
            margin=dict(t=50, b=10, l=10, r=10),
        )
        return _fig_to_dict(fig)

    def choropleth_obesity(self) -> dict:
        df = _health_df
        fig = px.choropleth(
            df,
            locations="state",
            locationmode="USA-states",
            color="obesity_rate",
            color_continuous_scale="YlOrRd",
            range_color=[20, 45],
            scope="usa",
            hover_name="state_name",
            hover_data=["obesity_rate", "diabetes_rate", "food_access_score"],
            title="Obesity Rate (%) by State",
        )
        fig.update_layout(
            geo=dict(bgcolor="rgba(0,0,0,0)", lakecolor="rgba(0,0,0,0)"),
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            font_color="#e1e4e8",
            coloraxis_colorbar_title="%",
            margin=dict(t=50, b=10, l=10, r=10),
        )
        return _fig_to_dict(fig)

    def choropleth_life_expectancy(self) -> dict:
        df = _health_df
        fig = px.choropleth(
            df,
            locations="state",
            locationmode="USA-states",
            color="life_expectancy",
            color_continuous_scale="RdYlGn",
            range_color=[72, 83],
            scope="usa",
            hover_name="state_name",
            hover_data=["life_expectancy", "median_income", "pct_uninsured"],
            title="Life Expectancy at Birth by State",
        )
        fig.update_layout(
            geo=dict(bgcolor="rgba(0,0,0,0)", lakecolor="rgba(0,0,0,0)"),
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            font_color="#e1e4e8",
            coloraxis_colorbar_title="Years",
            margin=dict(t=50, b=10, l=10, r=10),
        )
        return _fig_to_dict(fig)

    def scatter_food_vs_diabetes(self) -> dict:
        df = _health_df
        fig = px.scatter(
            df,
            x="food_access_score",
            y="diabetes_rate",
            color="pct_minority",
            size="median_income",
            size_max=20,
            color_continuous_scale="Viridis",
            hover_name="state_name",
            hover_data=["obesity_rate", "life_expectancy", "pct_food_desert"],
            title="Food Access vs Diabetes Rate (size = income, color = % minority)",
            trendline="ols",
        )
        fig.update_layout(
            paper_bgcolor="#0d1117",
            plot_bgcolor="#161b22",
            font_color="#e1e4e8",
            xaxis_title="Food Access Score",
            yaxis_title="Diabetes Rate (%)",
            margin=dict(t=50, b=50, l=60, r=20),
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d"),
        )
        return _fig_to_dict(fig)

    def scatter_food_vs_obesity(self) -> dict:
        df = _health_df
        fig = px.scatter(
            df,
            x="food_access_score",
            y="obesity_rate",
            color="pct_minority",
            size="median_income",
            size_max=20,
            color_continuous_scale="Viridis",
            hover_name="state_name",
            hover_data=["diabetes_rate", "life_expectancy"],
            title="Food Access vs Obesity Rate (size = income, color = % minority)",
            trendline="ols",
        )
        fig.update_layout(
            paper_bgcolor="#0d1117",
            plot_bgcolor="#161b22",
            font_color="#e1e4e8",
            xaxis_title="Food Access Score",
            yaxis_title="Obesity Rate (%)",
            margin=dict(t=50, b=50, l=60, r=20),
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d"),
        )
        return _fig_to_dict(fig)

    def bar_life_expectancy_by_income(self) -> dict:
        df = _health_df.copy()
        df["income_quintile"] = pd.qcut(df["median_income"], 5, labels=[
            "Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"
        ])
        grouped = df.groupby("income_quintile", observed=True)["life_expectancy"].mean().reset_index()
        fig = px.bar(
            grouped,
            x="income_quintile",
            y="life_expectancy",
            color="life_expectancy",
            color_continuous_scale="RdYlGn",
            title="Life Expectancy by Income Quintile",
        )
        fig.update_layout(
            paper_bgcolor="#0d1117",
            plot_bgcolor="#161b22",
            font_color="#e1e4e8",
            xaxis_title="Income Quintile",
            yaxis_title="Life Expectancy (years)",
            yaxis_range=[70, 85],
            margin=dict(t=50, b=50, l=60, r=20),
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d"),
            showlegend=False,
        )
        return _fig_to_dict(fig)

    def health_summary_table(self) -> list[dict]:
        df = _health_df
        return df.sort_values("food_access_score").head(10).to_dict(orient="records")
