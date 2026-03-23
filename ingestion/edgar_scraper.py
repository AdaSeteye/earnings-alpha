"""
SEC EDGAR transcript downloader.
Fetches 8-K exhibits containing earnings call transcripts for configured tickers and date range.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import yaml


def load_config() -> dict[str, Any]:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "data_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_sp500_tickers() -> list[str]:
    """Fetch current S&P 500 tickers from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    return df["Symbol"].str.replace(".", "-", regex=False).tolist()


def get_cik_map() -> dict[str, str]:
    """Download SEC company tickers JSON and build ticker -> CIK map."""
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {"User-Agent": "Research research@example.com"}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    return {v["ticker"]: str(v["cik_str"]).zfill(10) for v in data.values()}


def search_edgar_8k(cik: str, user_agent: str) -> list[dict[str, Any]]:
    """Search EDGAR for 8-K filings for a given CIK."""
    base = "https://data.sec.gov/submissions"
    headers = {"User-Agent": user_agent}
    url = f"{base}/CIK{cik}.json"
    r = requests.get(url, headers=headers, timeout=30)
    if r.status_code != 200:
        return []
    data = r.json()
    filings = data.get("filings", {}).get("recent", {})
    if not filings:
        return []
    forms = filings.get("form", [])
    accession_numbers = filings.get("accessionNumber", [])
    primary_docs = filings.get("primaryDocument", [])
    filing_dates = filings.get("filingDate", [])
    results = []
    for i, form in enumerate(forms):
        if form and form.upper() == "8-K":
            acc = accession_numbers[i] if i < len(accession_numbers) else ""
            doc = primary_docs[i] if i < len(primary_docs) else ""
            fd = filing_dates[i] if i < len(filing_dates) else ""
            results.append({
                "accession_number": acc,
                "primary_document": doc,
                "filing_date": fd,
            })
    return results


def fetch_filing_documents(cik: str, accession_number: str, user_agent: str) -> list[dict[str, str]]:
    """Get list of documents in a filing (to find transcript exhibits)."""
    acc_clean = accession_number.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/index.json"
    headers = {"User-Agent": user_agent}
    r = requests.get(url, headers=headers, timeout=30)
    if r.status_code != 200:
        return []
    data = r.json()
    items = data.get("directory", {}).get("item", [])
    if isinstance(items, dict):
        items = [items]
    return items


def download_transcript_html(base_url: str, user_agent: str) -> str | None:
    """Download one document and return raw HTML/text."""
    headers = {"User-Agent": user_agent}
    r = requests.get(base_url, headers=headers, timeout=30)
    if r.status_code != 200:
        return None
    return r.text


def is_likely_transcript(name: str) -> bool:
    name_lower = name.lower()
    if "transcript" in name_lower:
        return True
    if "earnings" in name_lower and ("call" in name_lower or "ex" in name_lower):
        return True
    return False


def main() -> None:
    config = load_config()
    paths = config.get("paths", {})
    raw_dir = Path(paths.get("raw_transcripts", "data/raw/transcripts"))
    raw_dir.mkdir(parents=True, exist_ok=True)
    edgar_cfg = config.get("edgar", {})
    user_agent = edgar_cfg.get("user_agent", "Research research@example.com")
    delay = edgar_cfg.get("rate_limit_delay_seconds", 0.2)
    date_start = config.get("date_range", {}).get("start", "2013-01-01")
    date_end = config.get("date_range", {}).get("end", "2024-12-31")
    ingestion_range = config.get("ingestion", {})
    start = ingestion_range.get("transcript_start", date_start)
    end = ingestion_range.get("transcript_end", date_end)

    universe = config.get("universe", {})
    if universe.get("ticker_source") == "sp500":
        tickers = get_sp500_tickers()
    else:
        tickers = universe.get("tickers", ["AAPL", "MSFT", "GOOGL"])[:50]

    cik_map = get_cik_map()
    metadata_rows = []

    for ticker in tqdm(tickers[:100], desc="Tickers"):  # Cap for demo; remove [:100] for full run
        cik = cik_map.get(ticker)
        if not cik:
            continue
        time.sleep(delay)
        filings = search_edgar_8k(cik, user_agent)
        for fil in filings:
            fd = fil.get("filing_date", "")
            if fd < start or fd > end:
                continue
            acc = fil.get("accession_number", "")
            docs = fetch_filing_documents(cik, acc, user_agent)
            time.sleep(delay)
            for doc in docs:
                doc_name = doc.get("name", "")
                if not is_likely_transcript(doc_name):
                    continue
                acc_clean = acc.replace("-", "")
                doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/{doc_name}"
                content = download_transcript_html(doc_url, user_agent)
                time.sleep(delay)
                if not content:
                    continue
                out_name = f"{ticker}_{fd}_{acc_clean}_{doc_name}"
                out_path = raw_dir / out_name
                out_path.write_text(content, encoding="utf-8", errors="replace")
                metadata_rows.append({
                    "ticker": ticker,
                    "cik": cik,
                    "filing_date": fd,
                    "accession_number": acc,
                    "document": doc_name,
                    "local_path": str(out_path),
                })

    if metadata_rows:
        meta_path = raw_dir / "_metadata.csv"
        pd.DataFrame(metadata_rows).to_csv(meta_path, index=False)
    print(f"Downloaded {len(metadata_rows)} transcripts to {raw_dir}")


if __name__ == "__main__":
    main()
