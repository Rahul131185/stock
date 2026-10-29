"""Streamlit UI.  Run:  streamlit run app.py"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from stock_advisor import PROFILES, analyze, load_company, synthetic_company

st.set_page_config(page_title="Stock Advisor", page_icon="📈", layout="wide")

VERDICT_COLORS = {
    "INVEST": "#1a7f4b",
    "HOLD / WAIT": "#b7791f",
    "DO NOT INVEST": "#c0392b",
    "INSUFFICIENT DATA": "#6b7280",
}


@st.cache_data(ttl=3600, show_spinner=False)
def _load(ticker: str, period: str, benchmark: str):
    return load_company(ticker, period, benchmark or None)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Analyze a company")
    ticker = st.text_input("Ticker symbol", value="AAPL",
                           help="US: AAPL, MSFT. India: RELIANCE.NS, TCS.NS (NSE) or .BO (BSE). "
                                "UK: .L, Germany: .DE, Japan: .T")
    period = st.selectbox("Price history", ["3y", "5y", "10y"], index=1)
    profile = st.radio("Your risk profile", PROFILES, index=1, horizontal=False,
                       help="Changes score weights and how strict the Invest / Avoid cut-offs are.")
    benchmark = st.text_input("Benchmark index (optional)", value="",
                              help="Leave empty to auto-pick, e.g. ^NSEI for .NS stocks, ^GSPC otherwise.")
    demo = st.checkbox("Use offline demo data (synthetic)", value=False)
    go_btn = st.button("Analyze", type="primary", use_container_width=True)

st.title("📈 Stock Advisor")
st.caption("Analyzes public financials and price history, then gives an Invest / Hold / Do-not-invest view. "
           "Decision support only, not financial advice.")

if go_btn:
    try:
        with st.spinner("Fetching public data..."):
            st.session_state["data"] = synthetic_company() if demo else _load(ticker, period, benchmark.strip())
    except Exception as e:  # network errors, bad ticker, etc.
        st.session_state.pop("data", None)
        st.error(f"Could not load data: {e}")

data = st.session_state.get("data")
if data is None:
    st.info("Enter a ticker in the sidebar and press **Analyze**. "
            "Tip: tick *offline demo data* to see how the app works without internet.")
    st.stop()

r = analyze(data, profile)
if r.source == "synthetic":
    st.warning("Showing synthetic demo data, not a real company.")

# ------------------------------------------------------------------ verdict banner
color = VERDICT_COLORS.get(r.verdict, "#6b7280")
score_txt = "n/a" if r.composite is None else f"{r.composite:.0f}/100"
st.markdown(
    f"""<div style="padding:18px 22px;border-radius:12px;background:{color};color:white">
    <div style="font-size:14px;opacity:.9">{r.name} · {r.ticker} · {r.sector}</div>
    <div style="font-size:34px;font-weight:700;line-height:1.2">{r.verdict}</div>
    <div style="font-size:15px;margin-top:6px">{r.action}</div></div>""",
    unsafe_allow_html=True,
)
st.write("")

h12 = r.forecast["horizons"].get("12 months", {})
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Price", f"{r.last_price:,.2f} {r.currency}")
c2.metric("Composite score", score_txt)
c3.metric("Data confidence", r.confidence, f"{r.coverage * 100:.0f}% coverage", delta_color="off")
c4.metric("12m median scenario", f"{h12['p50']:,.2f}", f"{(h12['p50'] / r.last_price - 1) * 100:+.1f}%")
c5.metric("12m chance of loss", f"{h12['prob_loss'] * 100:.0f}%")

tab_over, tab_price, tab_fc, tab_fund, tab_news = st.tabs(
    ["Overview", "Price & trend", "Forecast", "Fundamentals", "Headlines"])

# ------------------------------------------------------------------ overview
with tab_over:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Category scores")
        cats = [(c, s) for c, s in r.category_scores.items() if s is not None]
        fig = go.Figure(go.Bar(
            x=[s for _, s in cats], y=[c for c, _ in cats], orientation="h",
            marker_color=["#1a7f4b" if s >= 65 else "#b7791f" if s >= 45 else "#c0392b" for _, s in cats],
            text=[f"{s:.0f}" for _, s in cats], textposition="outside"))
        fig.update_layout(xaxis=dict(range=[0, 110], title="Score (0-100)"), yaxis=dict(autorange="reversed"),
                          height=320, margin=dict(l=0, r=10, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        if r.strengths:
            st.subheader("Strengths")
            for s in r.strengths:
                st.markdown(f"- ✅ {s}")
        if r.concerns:
            st.subheader("Concerns")
            for s in r.concerns:
                st.markdown(f"- ⚠️ {s}")
        if r.red_flags:
            st.subheader("Red flags")
            for s in r.red_flags:
                st.markdown(f"- 🚩 {s}")

# ------------------------------------------------------------------ price & trend
with tab_price:
    t = r.technicals
    ser = t["series"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=ser["close"].index, y=ser["close"], name="Close", line=dict(width=1.6)), 1, 1)
    fig.add_trace(go.Scatter(x=ser["sma50"].index, y=ser["sma50"], name="50-day avg", line=dict(width=1)), 1, 1)
    fig.add_trace(go.Scatter(x=ser["sma200"].index, y=ser["sma200"], name="200-day avg", line=dict(width=1)), 1, 1)
    fig.add_trace(go.Scatter(x=ser["rsi"].index, y=ser["rsi"], name="RSI(14)", line=dict(width=1)), 2, 1)
    fig.add_hline(y=70, line_dash="dot", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", row=2, col=1)
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Performance over time")
    perf = pd.DataFrame({
        "Period": ["1 month", "3 months", "6 months", "12 months"],
        "Return": [t["ret_1m"], t["ret_3m"], t["ret_6m"], t["ret_12m"]],
    })
    perf["Return"] = perf["Return"].map(lambda v: "n/a" if v is None else f"{v * 100:+.1f}%")
    a, b = st.columns(2)
    a.dataframe(perf, hide_index=True, use_container_width=True)
    risk = pd.DataFrame({
        "Measure": ["Volatility (annualised)", "Max drawdown", "Sharpe ratio", "Beta vs " + (data.benchmark_name or "market"),
                    "52-week range position", "RSI(14)"],
        "Value": [f"{t['vol'] * 100:.1f}%" if t["vol"] else "n/a", f"{t['max_drawdown'] * 100:.1f}%",
                  f"{t['sharpe']:.2f}" if t["sharpe"] is not None else "n/a",
                  f"{t['beta']:.2f}" if t["beta"] is not None else "n/a",
                  f"{t['pos52'] * 100:.0f}% (low {t['lo52']:,.2f} / high {t['hi52']:,.2f})" if t["pos52"] is not None else "n/a",
                  f"{t['rsi']:.0f}" if t["rsi"] is not None else "n/a"],
    })
    b.dataframe(risk, hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ forecast
with tab_fc:
    st.caption("A simulation of 4,000 possible paths using the stock's own volatility, a blended return assumption "
               "and fat-tailed shocks. It shows a **range of outcomes**, not a prediction.")
    bands = r.forecast["bands"]
    hist = r.technicals["series"]["close"].tail(252)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist, name="History", line=dict(width=1.6)))
    fig.add_trace(go.Scatter(x=bands.index, y=bands["p90"], line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=bands.index, y=bands["p10"], fill="tonexty", name="10th-90th pct",
                             line=dict(width=0), fillcolor="rgba(100,140,220,0.18)"))
    fig.add_trace(go.Scatter(x=bands.index, y=bands["p75"], line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=bands.index, y=bands["p25"], fill="tonexty", name="25th-75th pct",
                             line=dict(width=0), fillcolor="rgba(100,140,220,0.30)"))
    fig.add_trace(go.Scatter(x=bands.index, y=bands["p50"], name="Median", line=dict(width=1.6, dash="dash")))
    fig.update_layout(height=460, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    rows = []
    for label, h in r.forecast["horizons"].items():
        rows.append({"Horizon": label, "Bad case (10th pct)": f"{h['p10']:,.2f}", "Median": f"{h['p50']:,.2f}",
                     "Good case (90th pct)": f"{h['p90']:,.2f}", "Chance of loss": f"{h['prob_loss'] * 100:.0f}%",
                     "Chance of -20% or worse": f"{h['prob_loss_20'] * 100:.0f}%",
                     "Chance of +20% or better": f"{h['prob_gain_20'] * 100:.0f}%"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(f"Assumptions: expected return {r.forecast['mu'] * 100:.1f}%/yr, volatility {r.forecast['vol'] * 100:.1f}%/yr.")
    tgt = r.fundamentals.get("target")
    if tgt:
        st.caption(f"Cross-check: average analyst target {tgt:,.2f} ({(tgt / r.last_price - 1) * 100:+.1f}% vs price, "
                   f"{int(r.fundamentals.get('n_analysts') or 0)} analysts). Analyst targets are often optimistic.")

# ------------------------------------------------------------------ fundamentals
with tab_fund:
    df = pd.DataFrame([{"Category": m.category, "Metric": m.label, "Value": m.display, "Score": m.score}
                       for m in r.metrics])
    st.dataframe(df, hide_index=True, use_container_width=True,
                 column_config={"Score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f")})
    if r.fin_series:
        st.subheader("Annual financials")
        fig = go.Figure()
        for name, s in r.fin_series.items():
            fig.add_trace(go.Bar(x=[d.year for d in s.index], y=s.values, name=name))
        fig.update_layout(barmode="group", height=380, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Annual statements were not available for this ticker.")

# ------------------------------------------------------------------ headlines
with tab_news:
    if r.headlines:
        st.caption("Shown for context only; headlines are not scored.")
        for h in r.headlines:
            st.markdown(f"- {h}")
    else:
        st.write("No recent headlines available.")

st.divider()
st.caption("⚠️ Educational tool, not investment advice. Scores are rule-based heuristics on free public data "
           "(Yahoo Finance) that can be delayed, incomplete or wrong, and they ignore things like sector norms, "
           "taxes and your personal situation. Past performance does not predict future returns. "
           "Consider a SEBI/SEC-registered adviser before investing.")
