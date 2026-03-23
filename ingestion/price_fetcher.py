"""
yfinance price pipeline: fetch daily OHLCV and aligned returns for configured universe.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml
import yfinance as yf
from tqdm import tqdm


def load_config() -> dict[str, Any]:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "data_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_sp500_tickers() -> list[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    return tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()


def main() -> None:
    config = load_config()
    paths = config.get("paths", {})
    raw_prices_dir = Path(paths.get("raw_prices", "data/raw/prices"))
    raw_prices_dir.mkdir(parents=True, exist_ok=True)
    date_range = config.get("date_range", {})
    start = date_range.get("start", "2013-01-01")
    end = date_range.get("end", "2024-12-31")
    lookback = config.get("ingestion", {}).get("price_lookback_days", 252)

    universe = config.get("universe", {})
    if universe.get("ticker_source") == "sp500":
        tickers = get_sp500_tickers()
    else:
        tickers = universe.get("tickers", ["AAPL", "MSFT", "GOOGL"])

    all_prices = []
    for ticker in tqdm(tickers, desc="Fetching prices"):
        try:
            t = yf.Ticker(ticker)
            hist = t.history(start=start, end=end, auto_adjust=True)
            if hist.empty or len(hist) < 10:
                continue
            hist = hist.reset_index()
            hist["ticker"] = ticker
            all_prices.append(hist)
        except Exception:
            continue

    if not all_prices:
        print("No price data fetched.")
        return
    df = pd.concat(all_prices, ignore_index=True)
    out_path = raw_prices_dir / "daily_prices.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved prices to {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
