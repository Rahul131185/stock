"""Stock Advisor: analyze a company from public data and give an invest / hold / avoid view."""
from .data import CompanyData, load_company, synthetic_company, default_benchmark
from .analysis import analyze, render_text, Report, Metric, PROFILES

__all__ = [
    "CompanyData", "load_company", "synthetic_company", "default_benchmark",
    "analyze", "render_text", "Report", "Metric", "PROFILES",
]
