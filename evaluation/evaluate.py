"""
PayBack AI - Evaluation orchestrator.

Runs the full baseline-vs-PayBack-AI comparison and saves it as both a
machine-readable JSON report and a human-readable Markdown report.
Reproducible: same database + same CostAssumptions -> same results.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from app.database import get_connection
from evaluation.costs import CostAssumptions
from evaluation.metrics import (
    compute_full_baseline_metrics,
    compute_full_payback_ai_metrics,
    measure_decision_latency,
)

RESULTS_JSON_PATH = Path("evaluation/results.json")
RESULTS_MD_PATH = Path("evaluation/results.md")

DATA_NOTE = (
    "All figures are SIMULATED outcomes computed on SYNTHETIC data. "
    "Not representative of real-world Razorpay performance."
)

# (display label, metrics dict key, format string)
REPORT_METRIC_ROWS = [
    ("Recovery Rate", "recovery_rate", "{:.2%}"),
    ("Recovered Revenue (INR)", "recovered_revenue", "{:,.2f}"),
    ("Average Recovery Time (min)", "average_recovery_time_minutes", "{:.1f}"),
    ("Unnecessary Retry Rate", "unnecessary_retry_rate", "{:.2%}"),
    ("Messages per Recovered Payment", "messages_per_recovered_payment", "{:.3f}"),
    ("Cost per Recovery (INR)", "cost_per_recovery_inr", "{:.4f}"),
    ("Human Escalation Rate", "human_escalation_rate", "{:.2%}"),
    ("False-Positive Decision Rate", "false_positive_decision_rate", "{:.2%}"),
    ("Policy Violations Caught", "policy_violation_count", "{}"),
]


def _fmt(value, fmt: str) -> str:
    if value is None:
        return "N/A"
    try:
        return fmt.format(value)
    except (ValueError, TypeError):
        return str(value)


def run_evaluation(
    db_path: str = "payback.db",
    cost_assumptions: CostAssumptions | None = None,
    latency_sample_size: int = 200,
) -> dict:
    cost_assumptions = cost_assumptions or CostAssumptions()
    conn = get_connection(db_path)
    try:
        baseline = compute_full_baseline_metrics(conn, cost_assumptions)
        payback_ai = compute_full_payback_ai_metrics(conn, cost_assumptions)
        baseline["decision_latency"] = measure_decision_latency(conn, "baseline", latency_sample_size)
        payback_ai["decision_latency"] = measure_decision_latency(conn, "payback_ai", latency_sample_size)
    finally:
        conn.close()

    return {
        "baseline": baseline,
        "payback_ai": payback_ai,
        "cost_assumptions": asdict(cost_assumptions),
        "data_note": DATA_NOTE,
    }


def write_json_report(results: dict, path: Path = RESULTS_JSON_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))


def write_markdown_report(results: dict, path: Path = RESULTS_MD_PATH) -> None:
    lines = [
        "# PayBack AI — Evaluation Report",
        "",
        f"**{DATA_NOTE}**",
        "",
        "| Metric | Baseline (fixed-retry) | PayBack AI |",
        "|---|---|---|",
    ]
    for label, key, fmt in REPORT_METRIC_ROWS:
        b_val = _fmt(results["baseline"].get(key), fmt)
        a_val = _fmt(results["payback_ai"].get(key), fmt)
        lines.append(f"| {label} | {b_val} | {a_val} |")

    b_lat = results["baseline"]["decision_latency"]
    a_lat = results["payback_ai"]["decision_latency"]
    lines.append(
        f"| Decision Latency (avg ms, n={b_lat['sample_size']}) "
        f"| {b_lat['avg_decision_latency_ms']} | {a_lat['avg_decision_latency_ms']} |"
    )
    lines.append("")
    lines.append(
        "> Note: baseline decision latency is a pure in-memory lookup with no database "
        "access; PayBack AI's latency includes real SQL reads (classifier output, recovery "
        "probability, action history). These are intentionally not apples-to-apples - the "
        "asymmetry itself reflects the two systems' designs."
    )
    lines.append("")
    lines.append("## Cost Assumptions (illustrative only, NOT Razorpay's real costs)")
    for key, value in results["cost_assumptions"].items():
        lines.append(f"- {key}: {value}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    results = run_evaluation()
    write_json_report(results)
    write_markdown_report(results)

    print("[EVALUATION] Baseline vs PayBack AI (SIMULATED outcomes, synthetic data):")
    for label, key, fmt in REPORT_METRIC_ROWS:
        print(
            f"  {label}: baseline={_fmt(results['baseline'].get(key), fmt)} "
            f"| payback_ai={_fmt(results['payback_ai'].get(key), fmt)}"
        )
    b_lat = results["baseline"]["decision_latency"]
    a_lat = results["payback_ai"]["decision_latency"]
    print(
        f"  Decision Latency (avg ms): baseline={b_lat['avg_decision_latency_ms']} "
        f"| payback_ai={a_lat['avg_decision_latency_ms']}"
    )
    print(f"[EVALUATION] Reports saved to {RESULTS_JSON_PATH} and {RESULTS_MD_PATH}")
