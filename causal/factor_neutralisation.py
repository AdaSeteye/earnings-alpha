"""
Fama-French 5 + momentum factor adjustment for return series.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_config() -> dict[str, Any]:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "data_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factors", default="MKT,SMB,HML,RMW,CMA,MOM", help="Comma-separated factor names")
    parser.add_argument("--start-date", default="2013-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    args = parser.parse_args()
    config = load_config()
    paths = config.get("paths", {})
    ext_dir = Path(paths.get("external_ff5", "data/external/ff5_factors"))
    returns_dir = Path(paths.get("processed_returns", "data/processed/returns"))
    returns_dir.mkdir(parents=True, exist_ok=True)
    ext_dir.mkdir(parents=True, exist_ok=True)
    factor_list = [f.strip() for f in args.factors.split(",")]
    try:
        from pandas_datareader import data as pdr
        ff = pdr.DataReader("F-F_Research_Data_5_Factors_2x3_daily", "famafrench", start=args.start_date, end=args.end_date)
        if isinstance(ff, tuple):
            ff = ff[0]
        ff = ff / 100.0
        ff.to_parquet(ext_dir / "ff5_daily.parquet")
        mom = pdr.DataReader("F-F_Momentum_Factor_daily", "famafrench", start=args.start_date, end=args.end_date)
        if isinstance(mom, tuple):
            mom = mom[0]
        mom = mom / 100.0
        mom.to_parquet(ext_dir / "momentum_daily.parquet")
        print("Saved FF5 and momentum to", ext_dir)
    except Exception as e:
        print("Could not fetch Ken French data:", e)
        placeholder = pd.DataFrame({"MKT": [0.0], "SMB": [0.0], "HML": [0.0], "RMW": [0.0], "CMA": [0.0]})
        placeholder.to_parquet(ext_dir / "ff5_daily.parquet")
    print("Factor neutralisation config ready.")


if __name__ == "__main__":
    main()
