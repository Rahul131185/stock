"""Command-line version.  Usage:  python cli.py AAPL --profile balanced
                                 python cli.py RELIANCE.NS --period 10y
                                 python cli.py DEMO --demo
"""
import argparse

from stock_advisor import PROFILES, analyze, load_company, render_text, synthetic_company


def main():
    p = argparse.ArgumentParser(description="Analyze a listed company from public data.")
    p.add_argument("ticker")
    p.add_argument("--period", default="5y", choices=["3y", "5y", "10y"])
    p.add_argument("--profile", default="balanced", choices=PROFILES)
    p.add_argument("--benchmark", default=None, help="index ticker, e.g. ^NSEI")
    p.add_argument("--demo", action="store_true", help="use synthetic offline data")
    a = p.parse_args()

    data = synthetic_company(a.ticker) if a.demo else load_company(a.ticker, a.period, a.benchmark)
    print(render_text(analyze(data, a.profile)))


if __name__ == "__main__":
    main()
