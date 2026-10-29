"""Data loading: public market + financial-statement data via yfinance (Yahoo Finance)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class CompanyData:
    ticker: str
    name: str
    sector: str
    currency: str
    prices: pd.DataFrame                      # Open High Low Close Volume, DatetimeIndex
    info: dict
    income: Optional[pd.DataFrame] = None     # yfinance orientation: rows=items, cols=dates
    balance: Optional[pd.DataFrame] = None
    cashflow: Optional[pd.DataFrame] = None
    benchmark: Optional[pd.Series] = None     # benchmark index close prices
    benchmark_name: str = ""
    news: list = field(default_factory=list)
    source: str = "yahoo"


def default_benchmark(ticker: str) -> str:
    """Pick a sensible market index for the ticker's exchange suffix."""
    t = ticker.upper()
    suffix_map = {
        ".NS": "^NSEI", ".BO": "^BSESN", ".L": "^FTSE", ".T": "^N225",
        ".HK": "^HSI", ".TO": "^GSPTSE", ".DE": "^GDAXI", ".PA": "^FCHI",
        ".AX": "^AXJO", ".SI": "^STI",
    }
    for suf, idx in suffix_map.items():
        if t.endswith(suf):
            return idx
    return "^GSPC"


def _clean_index(df):
    if df is None or len(df) == 0:
        return df
    df = df.copy()
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    return df


def row(df: Optional[pd.DataFrame], *names: str) -> Optional[pd.Series]:
    """First matching statement row as a numeric Series sorted oldest -> newest."""
    if df is None or len(df) == 0:
        return None
    for n in names:
        if n in df.index:
            s = pd.to_numeric(df.loc[n], errors="coerce").dropna()
            if len(s):
                s.index = pd.to_datetime(s.index)
                return s.sort_index()
    return None


def extract_headlines(news: list, k: int = 8) -> list:
    out = []
    for item in news or []:
        if not isinstance(item, dict):
            continue
        content = item.get("content") if isinstance(item.get("content"), dict) else {}
        title = content.get("title") or item.get("title")
        if title:
            out.append(str(title))
    return out[:k]


def load_company(ticker: str, period: str = "5y", benchmark: Optional[str] = None) -> CompanyData:
    """Fetch prices, fundamentals, statements, benchmark and headlines for a ticker.

    Indian stocks need an exchange suffix, e.g. RELIANCE.NS or TCS.BO.
    """
    import yfinance as yf  # imported lazily so the engine can be tested offline

    ticker = ticker.strip().upper()
    t = yf.Ticker(ticker)
    prices = _clean_index(t.history(period=period, auto_adjust=True))
    if prices is None or prices.empty or "Close" not in prices:
        raise ValueError(
            f"No price data found for '{ticker}'. Check the symbol "
            "(Indian stocks need .NS or .BO, e.g. RELIANCE.NS)."
        )
    prices = prices[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])

    def safe(fn):
        try:
            return fn()
        except Exception:
            return None

    info = safe(lambda: t.info) or {}
    income = _clean_index_cols(safe(lambda: t.financials))
    balance = _clean_index_cols(safe(lambda: t.balance_sheet))
    cashflow = _clean_index_cols(safe(lambda: t.cashflow))
    news = safe(lambda: t.news) or []

    bench_ticker = benchmark or default_benchmark(ticker)
    bench = None
    try:
        b = _clean_index(yf.Ticker(bench_ticker).history(period=period, auto_adjust=True))
        if b is not None and not b.empty:
            bench = b["Close"].astype(float)
    except Exception:
        pass

    return CompanyData(
        ticker=ticker,
        name=info.get("longName") or info.get("shortName") or ticker,
        sector=info.get("sector") or info.get("industry") or "n/a",
        currency=info.get("currency") or "",
        prices=prices,
        info=info,
        income=income,
        balance=balance,
        cashflow=cashflow,
        benchmark=bench,
        benchmark_name=bench_ticker,
        news=news,
        source="yahoo",
    )


def _clean_index_cols(df):
    if df is None or len(df) == 0:
        return None
    return df


def synthetic_company(ticker: str = "DEMO", seed: int = 7, drift: float = 0.12,
                      vol: float = 0.28, years: int = 5) -> CompanyData:
    """Offline demo data (clearly fake) so the app and tests run without internet."""
    rng = np.random.default_rng(seed)
    n = int(252 * years)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    r = rng.normal(drift / 252 - 0.5 * vol ** 2 / 252, vol / np.sqrt(252), n)
    close = 100 * np.exp(np.cumsum(r))
    prices = pd.DataFrame({
        "Open": close * (1 + rng.normal(0, 0.003, n)),
        "High": close * 1.01,
        "Low": close * 0.99,
        "Close": close,
        "Volume": rng.integers(1_000_000, 5_000_000, n),
    }, index=idx)
    br = rng.normal(0.08 / 252, 0.15 / np.sqrt(252), n)
    bench = pd.Series(1000 * np.exp(np.cumsum(br)), index=idx)

    d = pd.to_datetime(["2025-12-31", "2024-12-31", "2023-12-31", "2022-12-31"])
    income = pd.DataFrame({
        d[0]: {"Total Revenue": 1400e6, "Net Income": 190e6, "Operating Income": 260e6},
        d[1]: {"Total Revenue": 1260e6, "Net Income": 165e6, "Operating Income": 230e6},
        d[2]: {"Total Revenue": 1120e6, "Net Income": 140e6, "Operating Income": 200e6},
        d[3]: {"Total Revenue": 1000e6, "Net Income": 120e6, "Operating Income": 175e6},
    })
    balance = pd.DataFrame({
        d[0]: {"Total Debt": 300e6, "Stockholders Equity": 900e6, "Cash And Cash Equivalents": 250e6},
        d[1]: {"Total Debt": 320e6, "Stockholders Equity": 780e6, "Cash And Cash Equivalents": 210e6},
        d[2]: {"Total Debt": 340e6, "Stockholders Equity": 680e6, "Cash And Cash Equivalents": 180e6},
        d[3]: {"Total Debt": 360e6, "Stockholders Equity": 600e6, "Cash And Cash Equivalents": 150e6},
    })
    cashflow = pd.DataFrame({
        d[0]: {"Free Cash Flow": 170e6, "Operating Cash Flow": 240e6},
        d[1]: {"Free Cash Flow": 150e6, "Operating Cash Flow": 215e6},
        d[2]: {"Free Cash Flow": 125e6, "Operating Cash Flow": 185e6},
        d[3]: {"Free Cash Flow": 100e6, "Operating Cash Flow": 160e6},
    })
    last = float(close[-1])
    info = {
        "trailingPE": 24.0, "forwardPE": 21.0, "priceToBook": 4.2, "trailingPegRatio": 1.6,
        "profitMargins": 0.136, "operatingMargins": 0.186, "returnOnEquity": 0.21,
        "debtToEquity": 33.0, "currentRatio": 1.8, "revenueGrowth": 0.11, "earningsGrowth": 0.15,
        "freeCashflow": 170e6, "marketCap": 6.5e9, "beta": 1.05,
        "targetMeanPrice": last * 1.12, "numberOfAnalystOpinions": 14,
        "dividendRate": last * 0.01,
        "longName": "Demo Industries Ltd (synthetic data)", "sector": "Demo", "currency": "USD",
    }
    return CompanyData(
        ticker=ticker, name=info["longName"], sector="Demo", currency="USD",
        prices=prices, info=info, income=income, balance=balance, cashflow=cashflow,
        benchmark=bench, benchmark_name="SYNTH-INDEX",
        news=[{"title": "Synthetic headline: demo data, not real news"}],
        source="synthetic",
    )
