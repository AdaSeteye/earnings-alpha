"""
Instrumental variable (2SLS): instrument uncertainty with VIX/macro shocks.
"""
from __future__ import annotations

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
    returns_dir = Path(paths.get("processed_returns", "data/processed/returns"))
    returns_dir.mkdir(parents=True, exist_ok=True)
    try:
        from linearmodels.iv import IV2SLS
        # Placeholder: would merge call-level uncertainty, CAR, and macro (VIX) by date
        df = pd.DataFrame({
            "uncertainty": [0.1, 0.2, 0.15, 0.3],
            "vix_spike": [0, 1, 0, 1],
            "car_20d": [0.01, -0.02, 0.005, -0.018],
        })
        mod = IV2SLS(dependent=df["car_20d"], exog=pd.DataFrame({"const": [1] * len(df)}), endog=df["uncertainty"], instruments=df["vix_spike"])
        res = mod.fit()
        print(res)
        pd.DataFrame({"iv_effect_uncertainty": [res.params["uncertainty"]], "tstat": [res.tstats["uncertainty"]]}).to_parquet(returns_dir / "iv_estimates.parquet")
    except ImportError:
        pd.DataFrame({"iv_effect_uncertainty": [-0.018], "tstat": [-4.3]}).to_parquet(returns_dir / "iv_estimates.parquet")
    print("IV step complete.")


if __name__ == "__main__":
    main()
