# Stock Advisor

Enter a stock ticker. The app pulls free public data (Yahoo Finance via `yfinance`), analyzes the
company, and shows an **INVEST / HOLD-WAIT / DO NOT INVEST** view with reasons, a price-range
simulation, and performance over time.

## Setup (Python 3.10+)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py               # web app at http://localhost:8501
```

Command line: `python cli.py AAPL` or `python cli.py RELIANCE.NS --profile conservative --period 10y`
Offline demo (fake data): `python cli.py DEMO --demo` or tick "offline demo data" in the app.
Tests: `python -m pytest`

Ticker examples: US `AAPL`, India NSE `RELIANCE.NS` / BSE `TCS.BO`, UK `HSBA.L`, Japan `7203.T`.
The market index used for beta and relative strength is picked automatically from the suffix
(e.g. `^NSEI` for `.NS`) or can be set manually.

## How the verdict is made

Six categories, each scored 0-100 from several metrics, then weighted by your risk profile:

| Category | Looks at |
|---|---|
| Growth | revenue and earnings growth, multi-year CAGR |
| Profitability | net/operating margin, ROE, margin trend |
| Financial health | debt/equity, current ratio, cash vs debt, free cash flow |
| Valuation | P/E (forward if available), PEG, P/B, analyst target upside |
| Momentum | price vs 50/200-day averages, 6/12-month return, RSI, MACD, return vs market |
| Risk | volatility, max drawdown, Sharpe ratio, beta |

Score >= 65 -> INVEST, 45-65 -> HOLD / WAIT, < 45 -> DO NOT INVEST (balanced profile; conservative is
stricter, aggressive is looser). Safety rules override the score: two or more red flags (loss-making,
negative free cash flow, very high debt, shrinking revenue, deep drawdown, downtrend, extreme P/E)
block an INVEST, three or more force DO NOT INVEST, and low data confidence never yields INVEST.

**Forecast:** a 4,000-path fat-tailed simulation giving 10th/25th/50th/75th/90th percentile prices at
3, 6 and 12 months plus the chance of loss. Expected return blends the stock's own history (25%)
with an 8% long-run market assumption (75%), tilted +/-4% by the composite score.

## Limitations (please read)

- It is **decision support, not financial advice**, and no model can reliably predict stock prices.
- Free Yahoo data can be delayed, missing or wrong; the app reports a data-confidence level.
- Thresholds are generic and not sector-aware (banks, utilities and software firms differ a lot).
- News headlines are shown but not scored. It does not read annual reports, management quality,
  competition, regulation or macro conditions.
- Risk-free rate for Sharpe defaults to 4%; change `risk_free` in `analyze()` for your market.
- Yahoo's terms apply to the data; this tool is for personal/educational use.

## Ideas to extend

Sector-relative valuation, news sentiment with an LLM, earnings-call text analysis, portfolio/watchlist
view, backtesting the scoring rules, alerts when a verdict changes.

## Layout

```
app.py                     Streamlit UI
cli.py                     terminal version
stock_advisor/data.py      data loading (yfinance) + synthetic demo data
stock_advisor/analysis.py  scoring, red flags, forecast, verdict
tests/test_engine.py       offline tests
```
