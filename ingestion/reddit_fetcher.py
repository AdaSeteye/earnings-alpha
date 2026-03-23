"""
Reddit sentiment fetcher (PRAW) — optional WallStreetBets / investing sentiment.
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
    raw_dir = Path(paths.get("raw_prices", "data/raw")).parent / "reddit"
    raw_dir.mkdir(parents=True, exist_ok=True)

    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    if not client_id or not client_secret:
        print("REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET not set. Skipping Reddit fetch.")
        return

    try:
        import praw
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent="earnings_alpha/1.0",
        )
        sub = reddit.subreddit("wallstreetbets")
        posts = []
        for post in sub.hot(limit=500):
            posts.append({
                "id": post.id,
                "title": post.title,
                "selftext": post.selftext[:2000] if post.selftext else "",
                "created_utc": pd.Timestamp(post.created_utc, unit="s"),
                "score": post.score,
                "num_comments": post.num_comments,
            })
        if posts:
            df = pd.DataFrame(posts)
            df.to_parquet(raw_dir / "wsb_posts.parquet", index=False)
            print(f"Saved {len(df)} Reddit posts to {raw_dir}")
    except Exception as e:
        print(f"Reddit fetch failed: {e}")


if __name__ == "__main__":
    main()
