"""Analysis engine: fundamentals, valuation, trend/risk, a probabilistic forecast, and a verdict.

Everything here is a transparent rule-based heuristic. It is decision support, not
financial advice, and it cannot predict prices.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .data import CompanyData, row, extract_headlines

CATEGORIES = ["Growth", "Profitability", "Financial health", "Valuation", "Momentum", "Risk"]

# Category weights per investor profile (sum to 1).
WEIGHTS = {
    "conservative": {"Growth": .15, "Profitability": .15, "Financial health": .20,
                     "Valuation": .20, "Momentum": .10, "Risk": .20},
    "balanced":     {"Growth": .20, "Profitability": .15, "Financial health": .15,
                     "Valuation": .20, "Momentum": .15, "Risk": .15},
    "aggressive":   {"Growth": .25, "Profitability": .15, "Financial health": .10,
                     "Valuation": .15, "Momentum": .20, "Risk": .15},
}
# (score needed for INVEST, score below which it is DO NOT INVEST)
THRESHOLDS = {"conservative": (70, 50), "balanced": (65, 45), "aggressive": (60, 40)}
PROFILES = list(WEIGHTS)


# ----------------------------------------------------------------------------- helpers
def _num(x) -> Optional[float]:
    try:
        if x is None:
            return None
        x = float(x)
        return None if (np.isnan(x) or np.isinf(x)) else x
    except (TypeError, ValueError):
        return None


def scale(x: Optional[float], bad: float, good: float) -> Optional[float]:
    """Linear map: `bad` -> 0, `good` -> 100 (either direction), clipped."""
    if x is None:
        return None
    if good == bad:
        return 50.0
    return float(np.clip((x - bad) / (good - bad) * 100.0, 0, 100))


def fmt_big(v: float) -> str:
    a = abs(v)
    for div, s in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= div:
            return f"{v / div:.2f}{s}"
    return f"{v:.0f}"


def fmt(v: Optional[float], kind: str) -> str:
    if v is None:
        return "n/a"
    return {
        "pct": f"{v * 100:.1f}%",
        "spct": f"{v * 100:+.1f}%",
        "pp": f"{v * 100:+.1f} pp",
        "pct100": f"{v:.0f}%",
        "x": f"{v:.1f}x",
        "big": fmt_big(v),
    }.get(kind, f"{v:.2f}")


@dataclass
class Metric:
    category: str
    label: str
    value: float
    score: float
    kind: str = "num"

    @property
    def display(self) -> str:
        return fmt(self.value, self.kind)


class _Scorer:
    def __init__(self):
        self.items: list[Metric] = []
        self.total = 0

    def add(self, cat, label, value, score, kind="num"):
        self.total += 1
        if value is None or score is None:
            return
        self.items.append(Metric(cat, label, float(value), float(score), kind))


# ----------------------------------------------------------------------------- technicals
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def compute_technicals(prices: pd.DataFrame, benchmark: Optional[pd.Series] = None,
                       risk_free: float = 0.04) -> dict:
    close = prices["Close"].astype(float).dropna()
    n = len(close)
    logret = np.log(close).diff().dropna()
    last = float(close.iloc[-1])

    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    rsi_s = rsi(close)
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    macd_hist = macd - macd.ewm(span=9, adjust=False).mean()

    def ret_n(k):
        return float(close.iloc[-1] / close.iloc[-1 - k] - 1) if n > k else None

    tail = close.tail(252)
    hi, lo = float(tail.max()), float(tail.min())
    vol = float(logret.tail(252).std() * np.sqrt(252)) if len(logret) > 30 else None
    years = n / 252
    cagr = float((close.iloc[-1] / close.iloc[0]) ** (1 / years) - 1) if years >= 0.5 else None
    sharpe = (cagr - risk_free) / vol if (cagr is not None and vol) else None
    mdd = float((close / close.cummax() - 1).min())

    beta, excess12 = None, None
    if benchmark is not None and len(benchmark.dropna()) > 60:
        b = benchmark.astype(float).dropna()
        bl = np.log(b).diff()
        j = pd.concat([logret.rename("s"), bl.rename("b")], axis=1, join="inner").dropna()
        if len(j) > 60 and j["b"].var() > 0:
            beta = float(j["s"].cov(j["b"]) / j["b"].var())
        if len(b) > 253 and n > 252:
            excess12 = ret_n(252) - float(b.iloc[-1] / b.iloc[-253] - 1)

    def lastval(s):
        v = s.iloc[-1]
        return None if pd.isna(v) else float(v)

    return {
        "last": last, "as_of": close.index[-1],
        "sma50": lastval(sma50), "sma200": lastval(sma200),
        "rsi": lastval(rsi_s), "macd_hist": lastval(macd_hist),
        "ret_1m": ret_n(21), "ret_3m": ret_n(63), "ret_6m": ret_n(126), "ret_12m": ret_n(252),
        "hi52": hi, "lo52": lo,
        "pos52": (last - lo) / (hi - lo) if hi > lo else None,
        "vol": vol, "cagr": cagr, "sharpe": sharpe, "max_drawdown": mdd,
        "beta": beta, "excess_12m": excess12, "years": years,
        "series": {"close": close, "sma50": sma50, "sma200": sma200, "rsi": rsi_s},
    }


# ----------------------------------------------------------------------------- fundamentals
def _cagr(s: Optional[pd.Series]) -> Optional[float]:
    if s is None or len(s) < 2:
        return None
    a, b = float(s.iloc[0]), float(s.iloc[-1])
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    if a <= 0 or b <= 0 or yrs <= 0:
        return None
    return (b / a) ** (1 / yrs) - 1


def _yoy(s: Optional[pd.Series]) -> Optional[float]:
    if s is None or len(s) < 2:
        return None
    a, b = float(s.iloc[-2]), float(s.iloc[-1])
    return b / a - 1 if a > 0 else None


def _ratio_last(a: Optional[pd.Series], b: Optional[pd.Series]) -> Optional[float]:
    if a is None or b is None:
        return None
    j = pd.concat([a, b], axis=1, join="inner").dropna()
    if j.empty or j.iloc[-1, 1] == 0:
        return None
    return float(j.iloc[-1, 0] / j.iloc[-1, 1])


def compute_fundamentals(data: CompanyData, last_price: float) -> tuple[dict, dict]:
    info = data.info or {}
    rev = row(data.income, "Total Revenue", "Operating Revenue")
    ni = row(data.income, "Net Income", "Net Income Common Stockholders")
    opinc = row(data.income, "Operating Income")
    fcf_s = row(data.cashflow, "Free Cash Flow")
    ocf_s = row(data.cashflow, "Operating Cash Flow")
    debt = row(data.balance, "Total Debt")
    equity = row(data.balance, "Stockholders Equity", "Total Equity Gross Minority Interest")
    cash = row(data.balance, "Cash And Cash Equivalents")

    margin_series = None
    if rev is not None and ni is not None:
        j = pd.concat([ni, rev], axis=1, join="inner").dropna()
        j = j[j.iloc[:, 1] != 0]
        if len(j) >= 2:
            margin_series = j.iloc[:, 0] / j.iloc[:, 1]

    de = _num(info.get("debtToEquity"))
    if de is None:
        r = _ratio_last(debt, equity)
        de = r * 100 if r is not None else None

    fcf = _num(info.get("freeCashflow"))
    if fcf is None and fcf_s is not None:
        fcf = float(fcf_s.iloc[-1])

    peg = _num(info.get("trailingPegRatio")) or _num(info.get("pegRatio"))
    div_rate = _num(info.get("dividendRate"))

    f = {
        "pe": _num(info.get("trailingPE")), "fwd_pe": _num(info.get("forwardPE")),
        "pb": _num(info.get("priceToBook")), "peg": peg,
        "net_margin": _num(info.get("profitMargins")) if _num(info.get("profitMargins")) is not None
        else _ratio_last(ni, rev),
        "op_margin": _num(info.get("operatingMargins")) if _num(info.get("operatingMargins")) is not None
        else _ratio_last(opinc, rev),
        "roe": _num(info.get("returnOnEquity")) if _num(info.get("returnOnEquity")) is not None
        else _ratio_last(ni, equity),
        "de": de, "current_ratio": _num(info.get("currentRatio")),
        "cash_cover": _ratio_last(cash, debt),
        "rev_growth": _num(info.get("revenueGrowth")) if _num(info.get("revenueGrowth")) is not None
        else _yoy(rev),
        "earn_growth": _num(info.get("earningsGrowth")) if _num(info.get("earningsGrowth")) is not None
        else _yoy(ni),
        "rev_cagr": _cagr(rev), "ni_cagr": _cagr(ni),
        "margin_trend": float(margin_series.iloc[-1] - margin_series.iloc[0]) if margin_series is not None else None,
        "fcf": fcf, "market_cap": _num(info.get("marketCap")),
        "div_yield": (div_rate / last_price) if (div_rate and last_price) else None,
        "target": _num(info.get("targetMeanPrice")),
        "n_analysts": _num(info.get("numberOfAnalystOpinions")),
        "beta_info": _num(info.get("beta")),
        "loss_making": (_num(info.get("profitMargins")) is not None and _num(info.get("profitMargins")) < 0)
        or (ni is not None and float(ni.iloc[-1]) < 0),
    }
    series = {k: v for k, v in {"Revenue": rev, "Net income": ni, "Free cash flow": fcf_s,
                                "Operating cash flow": ocf_s}.items() if v is not None}
    return f, series


# ----------------------------------------------------------------------------- forecast
def monte_carlo(close: pd.Series, composite: float, horizon: int = 252, n_paths: int = 4000,
                seed: int = 42) -> dict:
    """Scenario simulation (fat-tailed GBM), NOT a price prediction.

    Expected return = 25% own recent history + 75% long-run market assumption (8%),
    tilted by +/-4% according to the composite score. Volatility blends long-run and
    last-3-month realised volatility.
    """
    close = close.astype(float).dropna()
    logret = np.log(close).diff().dropna()
    vol_long = float(logret.std() * np.sqrt(252))
    vol_recent = float(logret.tail(63).std() * np.sqrt(252)) if len(logret) > 63 else vol_long
    vol = 0.5 * vol_long + 0.5 * vol_recent
    hist_arith = float(logret.tail(756).mean() * 252) + 0.5 * vol ** 2
    mu = float(np.clip(0.25 * hist_arith + 0.75 * 0.08 + (composite - 50) / 50 * 0.04, -0.15, 0.30))

    rng = np.random.default_rng(seed)
    df = 5
    z = rng.standard_t(df, size=(n_paths, horizon)) / np.sqrt(df / (df - 2))
    daily = (mu - 0.5 * vol ** 2) / 252 + vol / np.sqrt(252) * z
    last = float(close.iloc[-1])
    paths = last * np.exp(np.cumsum(daily, axis=1))

    out = {"mu": mu, "vol": vol, "horizons": {}}
    for h, label in ((63, "3 months"), (126, "6 months"), (252, "12 months")):
        if h > horizon:
            continue
        v = paths[:, h - 1]
        q = np.percentile(v, [10, 25, 50, 75, 90])
        out["horizons"][label] = {
            "p10": float(q[0]), "p25": float(q[1]), "p50": float(q[2]), "p75": float(q[3]), "p90": float(q[4]),
            "prob_loss": float((v < last).mean()),
            "prob_gain_20": float((v > last * 1.2).mean()),
            "prob_loss_20": float((v < last * 0.8).mean()),
            "expected_return": float(v.mean() / last - 1),
        }
    dates = pd.bdate_range(close.index[-1] + pd.Timedelta(days=1), periods=horizon)
    bands = pd.DataFrame(np.percentile(paths, [10, 25, 50, 75, 90], axis=0).T, index=dates,
                         columns=["p10", "p25", "p50", "p75", "p90"])
    out["bands"] = bands
    return out


# ----------------------------------------------------------------------------- report
@dataclass
class Report:
    ticker: str
    name: str
    sector: str
    currency: str
    last_price: float
    as_of: pd.Timestamp
    profile: str
    composite: Optional[float]
    category_scores: dict
    verdict: str
    action: str
    confidence: str
    coverage: float
    strengths: list = field(default_factory=list)
    concerns: list = field(default_factory=list)
    red_flags: list = field(default_factory=list)
    metrics: list = field(default_factory=list)
    technicals: dict = field(default_factory=dict)
    fundamentals: dict = field(default_factory=dict)
    forecast: dict = field(default_factory=dict)
    fin_series: dict = field(default_factory=dict)
    headlines: list = field(default_factory=list)
    source: str = "yahoo"


def _rsi_score(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    if v > 75:
        return 30
    if v > 65:
        return 60
    if v >= 50:
        return 80
    if v >= 40:
        return 60
    if v >= 30:
        return 45
    return 40


def analyze(data: CompanyData, profile: str = "balanced", risk_free: float = 0.04) -> Report:
    if profile not in WEIGHTS:
        raise ValueError(f"profile must be one of {PROFILES}")
    tech = compute_technicals(data.prices, data.benchmark, risk_free)
    f, fin_series = compute_fundamentals(data, tech["last"])
    last = tech["last"]
    S = _Scorer()

    # Growth
    S.add("Growth", "Revenue growth (latest YoY)", f["rev_growth"], scale(f["rev_growth"], -0.05, 0.20), "pct")
    S.add("Growth", "Revenue CAGR (statements)", f["rev_cagr"], scale(f["rev_cagr"], 0.0, 0.15), "pct")
    S.add("Growth", "Earnings growth (latest YoY)", f["earn_growth"], scale(f["earn_growth"], -0.10, 0.30), "pct")
    S.add("Growth", "Net income CAGR", f["ni_cagr"], scale(f["ni_cagr"], 0.0, 0.20), "pct")

    # Profitability
    S.add("Profitability", "Net margin", f["net_margin"], scale(f["net_margin"], 0.0, 0.20), "pct")
    S.add("Profitability", "Operating margin", f["op_margin"], scale(f["op_margin"], 0.0, 0.25), "pct")
    S.add("Profitability", "Return on equity", f["roe"], scale(f["roe"], 0.05, 0.25), "pct")
    S.add("Profitability", "Net margin trend", f["margin_trend"], scale(f["margin_trend"], -0.03, 0.03), "pp")

    # Financial health
    S.add("Financial health", "Debt / equity", f["de"], scale(f["de"], 250, 30), "pct100")
    S.add("Financial health", "Current ratio", f["current_ratio"], scale(f["current_ratio"], 0.7, 2.0), "num")
    S.add("Financial health", "Cash / debt", f["cash_cover"], scale(f["cash_cover"], 0.1, 1.0), "num")
    S.add("Financial health", "Free cash flow", f["fcf"],
          None if f["fcf"] is None else (100 if f["fcf"] > 0 else 10), "big")

    # Valuation
    pe_eff = f["fwd_pe"] if (f["fwd_pe"] and f["fwd_pe"] > 0) else f["pe"]
    if f["loss_making"] and (pe_eff is None or pe_eff <= 0):
        S.add("Valuation", "P/E (company is loss-making)", 0.0, 10, "x")
    elif pe_eff and pe_eff > 0:
        S.add("Valuation", "P/E (forward if available)", pe_eff, scale(pe_eff, 40, 12), "x")
    else:
        S.add("Valuation", "P/E (forward if available)", None, None, "x")
    S.add("Valuation", "PEG ratio", f["peg"] if (f["peg"] and f["peg"] > 0) else None,
          scale(f["peg"], 2.5, 1.0) if (f["peg"] and f["peg"] > 0) else None, "num")
    S.add("Valuation", "Price / book", f["pb"] if (f["pb"] and f["pb"] > 0) else None,
          scale(f["pb"], 8, 1.5) if (f["pb"] and f["pb"] > 0) else None, "x")
    upside = (f["target"] / last - 1) if (f["target"] and (f["n_analysts"] or 0) >= 3) else None
    S.add("Valuation", "Analyst target upside", upside, scale(upside, -0.10, 0.25), "spct")

    # Momentum
    vs200 = (last / tech["sma200"] - 1) if tech["sma200"] else None
    s50_200 = (tech["sma50"] / tech["sma200"] - 1) if (tech["sma50"] and tech["sma200"]) else None
    S.add("Momentum", "Price vs 200-day average", vs200, scale(vs200, -0.15, 0.15), "spct")
    S.add("Momentum", "50-day vs 200-day average", s50_200, scale(s50_200, -0.05, 0.05), "spct")
    S.add("Momentum", "12-month return", tech["ret_12m"], scale(tech["ret_12m"], -0.20, 0.40), "spct")
    S.add("Momentum", "6-month return", tech["ret_6m"], scale(tech["ret_6m"], -0.15, 0.25), "spct")
    S.add("Momentum", "RSI (14)", tech["rsi"], _rsi_score(tech["rsi"]), "num")
    S.add("Momentum", "MACD histogram", tech["macd_hist"],
          None if tech["macd_hist"] is None else (70 if tech["macd_hist"] > 0 else 35), "num")
    S.add("Momentum", "12m return vs market", tech["excess_12m"], scale(tech["excess_12m"], -0.20, 0.20), "spct")

    # Risk
    S.add("Risk", "Volatility (annualised)", tech["vol"], scale(tech["vol"], 0.60, 0.15), "pct")
    S.add("Risk", "Max drawdown", tech["max_drawdown"], scale(tech["max_drawdown"], -0.70, -0.15), "pct")
    S.add("Risk", "Sharpe ratio", tech["sharpe"], scale(tech["sharpe"], -0.2, 1.2), "num")
    S.add("Risk", "Beta vs market", tech["beta"], scale(tech["beta"], 1.8, 0.7), "num")

    cat_scores = {}
    for c in CATEGORIES:
        vals = [m.score for m in S.items if m.category == c]
        cat_scores[c] = float(np.mean(vals)) if vals else None

    w = WEIGHTS[profile]
    avail = {c: s for c, s in cat_scores.items() if s is not None}
    composite = (sum(w[c] * s for c, s in avail.items()) / sum(w[c] for c in avail)) if avail else None
    coverage = len(S.items) / S.total if S.total else 0.0

    if coverage >= 0.8 and tech["years"] >= 3:
        confidence = "High"
    elif coverage >= 0.6 and tech["years"] >= 1.5:
        confidence = "Medium"
    else:
        confidence = "Low"

    # Red flags
    flags = []
    if f["loss_making"]:
        flags.append("Company is loss-making")
    if f["fcf"] is not None and f["fcf"] < 0:
        flags.append("Negative free cash flow")
    if f["de"] is not None and f["de"] > 200:
        flags.append(f"Very high leverage (debt/equity {f['de']:.0f}%)")
    if f["current_ratio"] is not None and f["current_ratio"] < 1:
        flags.append(f"Current ratio below 1 ({f['current_ratio']:.2f}): short-term liquidity risk")
    if f["rev_growth"] is not None and f["rev_growth"] < -0.05:
        flags.append("Revenue is shrinking")
    if tech["max_drawdown"] < -0.50:
        flags.append(f"Deep historical drawdown ({tech['max_drawdown'] * 100:.0f}%)")
    if vs200 is not None and tech["ret_6m"] is not None and vs200 < 0 and tech["ret_6m"] < -0.15:
        flags.append("Downtrend: below 200-day average and down >15% in 6 months")
    if pe_eff is not None and pe_eff > 60:
        flags.append(f"Very expensive on earnings (P/E {pe_eff:.0f})")

    forecast = monte_carlo(data.prices["Close"], composite if composite is not None else 50.0)
    h12 = forecast["horizons"].get("12 months", {})

    hi_t, lo_t = THRESHOLDS[profile]
    if composite is None or coverage < 0.30:
        verdict = "INSUFFICIENT DATA"
    elif composite >= hi_t:
        verdict = "INVEST"
    elif composite < lo_t:
        verdict = "DO NOT INVEST"
    else:
        verdict = "HOLD / WAIT"

    notes = []
    if verdict == "INVEST":
        if len(flags) >= 2:
            verdict = "HOLD / WAIT"
            notes.append("Downgraded from INVEST: two or more red flags present.")
        elif confidence == "Low":
            verdict = "HOLD / WAIT"
            notes.append("Downgraded from INVEST: data confidence is low.")
        elif profile == "conservative" and h12.get("prob_loss", 0) > 0.40:
            verdict = "HOLD / WAIT"
            notes.append("Downgraded from INVEST: simulated 12-month loss probability above 40%.")

    if verdict == "HOLD / WAIT" and len(flags) >= 3:
        verdict = "DO NOT INVEST"
        notes.append("Escalated to DO NOT INVEST: three or more red flags present.")

    strengths = [f"{m.label}: {m.display}" for m in sorted(S.items, key=lambda m: -m.score) if m.score >= 75][:6]
    concerns = [f"{m.label}: {m.display}" for m in sorted(S.items, key=lambda m: m.score) if m.score <= 30][:6]

    action = _action_text(verdict, concerns, tech, notes)

    return Report(
        ticker=data.ticker, name=data.name, sector=data.sector, currency=data.currency,
        last_price=last, as_of=tech["as_of"], profile=profile, composite=composite,
        category_scores=cat_scores, verdict=verdict, action=action, confidence=confidence,
        coverage=coverage, strengths=strengths, concerns=concerns, red_flags=flags,
        metrics=S.items, technicals=tech, fundamentals=f, forecast=forecast,
        fin_series=fin_series, headlines=extract_headlines(data.news), source=data.source,
    )


def _action_text(verdict: str, concerns: list, tech: dict, notes: list) -> str:
    extra = (" " + " ".join(notes)) if notes else ""
    hot = tech.get("rsi") is not None and tech["rsi"] > 75
    if verdict == "INVEST":
        base = "Fundamentals, valuation and trend are broadly supportive. Consider building a position in stages (e.g. 3 tranches over a few weeks) and size it so a 30% drop would not hurt you."
        if hot:
            base += " Short-term RSI is very high, so waiting for a pullback may give a better entry."
        return base + extra
    if verdict == "HOLD / WAIT":
        base = "Mixed picture. If you already own it, holding is reasonable; if not, wait for improvement."
        if concerns:
            base += " Watch: " + "; ".join(concerns[:3]) + "."
        return base + extra
    if verdict == "DO NOT INVEST":
        return "Risk/reward looks unfavourable on the available data. Avoid for now and revisit after new quarterly results." + extra
    return "Not enough public data was available to form a reliable view." + extra


def render_text(r: Report) -> str:
    L = []
    L.append(f"{r.name} ({r.ticker})  |  {r.sector}  |  as of {r.as_of:%Y-%m-%d}")
    L.append(f"Price: {r.last_price:,.2f} {r.currency}   Profile: {r.profile}   Data source: {r.source}")
    L.append("=" * 72)
    score = "n/a" if r.composite is None else f"{r.composite:.0f}/100"
    L.append(f"VERDICT: {r.verdict}   (score {score}, confidence {r.confidence}, data coverage {r.coverage * 100:.0f}%)")
    L.append(r.action)
    L.append("")
    L.append("Category scores:")
    for c, s in r.category_scores.items():
        L.append(f"  {c:<18} {'n/a' if s is None else f'{s:.0f}'}")
    if r.strengths:
        L.append("\nStrengths:")
        L += [f"  + {s}" for s in r.strengths]
    if r.concerns:
        L.append("\nConcerns:")
        L += [f"  - {s}" for s in r.concerns]
    if r.red_flags:
        L.append("\nRed flags:")
        L += [f"  ! {s}" for s in r.red_flags]
    L.append("\nSimulated price range (scenario, not a prediction):")
    for label, h in r.forecast.get("horizons", {}).items():
        L.append(f"  {label:<10} 10th-90th pct: {h['p10']:,.2f} to {h['p90']:,.2f} | median {h['p50']:,.2f} | chance of loss {h['prob_loss'] * 100:.0f}%")
    L.append("\nNot financial advice. Public data can be wrong or stale; do your own research.")
    return "\n".join(L)
