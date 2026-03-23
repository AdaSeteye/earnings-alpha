"""
FRED macro data pipeline: fetch configured macro series.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_config() -> dict[str, Any]:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "data_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def main() -> None:
    config = load_config()
    paths = config.get("paths", {})
    raw_macro_dir = Path(paths.get("raw_macro", "data/raw/macro"))
    raw_macro_dir.mkdir(parents=True, exist_ok=True)
    series_list = config.get("ingestion", {}).get("macro_series", ["DGS10", "DGS2", "T10Y2Y", "FEDFUNDS"])
    date_range = config.get("date_range", {})
    start = date_range.get("start", "2013-01-01")
    end = date_range.get("end", "2024-12-31")

    try:
        from pandas_datareader import data as pdr
        fred_key = os.environ.get("FRED_API_KEY")
        if not fred_key:
            print("FRED_API_KEY not set. Using pandas-datareader without key (may hit rate limits).")
        for series_id in series_list:
            try:
                df = pdr.DataReader(series_id, "fred", start=start, end=end)
                if df.empty:
                    continue
                df = df.reset_index()
                df.columns = ["date", "value"]
                df["series_id"] = series_id
                out_path = raw_macro_dir / f"{series_id}.parquet"
                df.to_parquet(out_path, index=False)
            except Exception as e:
                print(f"Skip {series_id}: {e}")
        print(f"Macro data saved to {raw_macro_dir}")
    except ImportError:
        # Placeholder series if pandas-datareader not available
        dates = pd.date_range(start=start, end=end, freq="D")
        for series_id in series_list:
            df = pd.DataFrame({"date": dates, "value": 0.0, "series_id": series_id})
            raw_macro_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(raw_macro_dir / f"{series_id}.parquet", index=False)
        print("pandas_datareader not installed; wrote placeholder macro files.")


if __name__ == "__main__":
    main()
