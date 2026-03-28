"""Data service — manages loaded datasets in memory."""

from pathlib import Path

import pandas as pd
import numpy as np

from src.loaders.loader import load_data

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


class DataService:
    def __init__(self):
        self._df: pd.DataFrame | None = None
        self._name: str | None = None

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            raise ValueError("No dataset loaded. POST /api/data/load/{filename} first.")
        return self._df

    def load(self, filename: str, subdir: str = "raw") -> None:
        self._df = load_data(filename, subdir=subdir)
        self._name = filename

    def current_shape(self) -> tuple[int, int]:
        return self.df.shape

    def list_datasets(self) -> dict:
        result = {}
        for sub in ["raw", "processed", "external"]:
            d = DATA_DIR / sub
            if d.exists():
                files = [f.name for f in d.iterdir() if f.is_file() and f.name != ".gitkeep"]
                if files:
                    result[sub] = files
        return result

    def info(self) -> dict:
        df = self.df
        info = []
        for col in df.columns:
            s = df[col]
            info.append({
                "column": col,
                "dtype": str(s.dtype),
                "non_null": int(s.notna().sum()),
                "null": int(s.isna().sum()),
                "unique": int(s.nunique()),
                "sample": _safe_sample(s),
            })
        return {"name": self._name, "shape": list(df.shape), "columns": info}

    def head(self, n: int = 20) -> list[dict]:
        return self.df.head(n).replace({np.nan: None}).to_dict(orient="records")

    def columns(self) -> dict:
        return {
            "numeric": self.df.select_dtypes(include="number").columns.tolist(),
            "categorical": self.df.select_dtypes(include=["object", "category"]).columns.tolist(),
            "datetime": self.df.select_dtypes(include="datetime").columns.tolist(),
        }

    def stats(self) -> list[dict]:
        desc = self.df.describe(include="all").T
        desc = desc.where(desc.notna(), None)
        return desc.reset_index().rename(columns={"index": "column"}).to_dict(orient="records")


def _safe_sample(s: pd.Series) -> str | None:
    if s.notna().any():
        val = s.dropna().iloc[0]
        text = str(val)
        return text[:80] if len(text) > 80 else text
    return None
