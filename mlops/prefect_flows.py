"""
Prefect pipeline DAGs: full pipeline, NLP-only, etc.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def full_pipeline_flow(start_date: str, end_date: str) -> None:
    """Define full pipeline as Prefect flow (stub)."""
    try:
        from prefect import flow, task
        @task
        def ingest():
            from ingestion import edgar_scraper, price_fetcher, macro_fetcher
            edgar_scraper.main()
            price_fetcher.main()
            macro_fetcher.main()

        @task
        def nlp():
            from nlp.sentiment import finbert_inference
            from nlp.uncertainty import uncertainty_scorer
            finbert_inference.main()
            uncertainty_scorer.main()

        @flow
        def pipeline():
            ingest()
            nlp()

        pipeline()
    except ImportError:
        print("Install prefect. Running steps manually.")
        from ingestion import edgar_scraper, price_fetcher, macro_fetcher
        from nlp.sentiment import finbert_inference
        from nlp.uncertainty import uncertainty_scorer
        edgar_scraper.main()
        price_fetcher.main()
        macro_fetcher.main()
        finbert_inference.main()
        uncertainty_scorer.main()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow", default="full-pipeline")
    parser.add_argument("--start-date", default="2013-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    args = parser.parse_args()
    if args.flow == "full-pipeline":
        full_pipeline_flow(args.start_date, args.end_date)
    print("Flow complete.")


if __name__ == "__main__":
    main()
