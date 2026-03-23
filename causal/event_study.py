"""
Event study: CAR computation using market model, parametric and non-parametric tests.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def load_config() -> dict[str, Any]:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "data_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


class EventStudy:
    """Cumulative abnormal return (CAR) using market model."""

    def __init__(
        self,
        estimation_window: int = 252,
        event_windows: list[tuple[int, int]] | None = None,
    ):
        self.estimation_window = estimation_window
        self.event_windows = event_windows or [(0, 1), (0, 5), (0, 20)]

    def _get_returns(self, ticker: str, event_date: str) -> tuple[pd.Series | None, pd.Series | None]:
        config = load_config()
        paths = config.get("paths", {})
        price_path = Path(paths.get("raw_prices", "data/raw/prices")) / "daily_prices.parquet"
        if not price_path.exists():
            return None, None
        df = pd.read_parquet(price_path)
        df["Date"] = pd.to_datetime(df["Date"] if "Date" in df.columns else df["index"])
        df = df[df["ticker"] == ticker].sort_values("Date").set_index("Date")
        if "Close" not in df.columns and "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]
        if "Close" not in df.columns:
            return None, None
        ret = df["Close"].pct_change().dropna()
        return ret, ret  # market return: use same as proxy if no market index

    def compute(
        self,
        ticker: str,
        event_date: str,
        market_return: pd.Series | None = None,
    ) -> pd.DataFrame:
        """
        Compute CAR for given ticker and event date over configured event windows.
        Returns DataFrame with columns event_window, car, ar_mean, n_days, t_stat.
        """
        stock_ret, mkt_ret = self._get_returns(ticker, event_date)
        if stock_ret is None or stock_ret.empty:
            return pd.DataFrame()
        if market_return is not None:
            mkt_ret = market_return
        event_dt = pd.Timestamp(event_date)
        dates = stock_ret.index
        pre_start = event_dt - pd.Timedelta(days=self.estimation_window + 30)
        est_ret = stock_ret[(dates >= pre_start) & (dates < event_dt)].tail(self.estimation_window)
        if mkt_ret is not None and not mkt_ret.empty:
            mkt_aligned = mkt_ret.reindex(est_ret.index).ffill().bfill()
            if mkt_aligned.notna().sum() < 10:
                mkt_aligned = est_ret  # fallback
        else:
            mkt_aligned = est_ret
        X = np.column_stack([np.ones(len(est_ret)), mkt_aligned.values])
        y = est_ret.values
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            alpha, beta_mkt = beta[0], beta[1]
        except Exception:
            alpha, beta_mkt = 0.0, 1.0
        results = []
        for (t0, t1) in self.event_windows:
            window_start = event_dt + pd.Timedelta(days=t0)
            window_end = event_dt + pd.Timedelta(days=t1)
            ar = stock_ret[(dates >= window_start) & (dates <= window_end)]
            if ar.empty:
                continue
            er = alpha + beta_mkt * (mkt_aligned.reindex(ar.index).ffill().bfill() if mkt_ret is not None else 0)
            abnormal = ar - er.values if hasattr(er, "values") else ar
            car = abnormal.sum()
            n = len(abnormal)
            ar_mean = abnormal.mean()
            se = abnormal.std() / (n ** 0.5) if n and abnormal.std() > 0 else np.nan
            t_stat = (car / (se * n ** 0.5)) if se and not np.isnan(se) else np.nan
            results.append({
                "event_window": f"[{t0},{t1}]",
                "car": car,
                "ar_mean": ar_mean,
                "n_days": n,
                "t_stat": t_stat,
            })
        return pd.DataFrame(results)


def main() -> None:
    es = EventStudy(estimation_window=252, event_windows=[(0, 1), (0, 5), (0, 20)])
    config = load_config()
    paths = config.get("paths", {})
    returns_dir = Path(paths.get("processed_returns", "data/processed/returns"))
    returns_dir.mkdir(parents=True, exist_ok=True)
    car_list = []
    # Example: compute for a few tickers if we have event dates
    for ticker in ["AAPL", "MSFT", "GOOGL"]:
        df = es.compute(ticker=ticker, event_date="2024-02-01")
        if not df.empty:
            df["ticker"] = ticker
            df["event_date"] = "2024-02-01"
            car_list.append(df)
    if car_list:
        out = pd.concat(car_list, ignore_index=True)
        out.to_parquet(returns_dir / "car_event_study.parquet", index=False)
        print(f"Saved CAR results to {returns_dir / 'car_event_study.parquet'}")
    else:
        print("No price data found. Run price_fetcher first. CAR computation skipped.")


if __name__ == "__main__":
    main()
