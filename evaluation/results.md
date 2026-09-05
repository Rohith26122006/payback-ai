# PayBack AI — Evaluation Report

**All figures are SIMULATED outcomes computed on SYNTHETIC data. Not representative of real-world Razorpay performance.**

| Metric | Baseline (fixed-retry) | PayBack AI |
|---|---|---|
| Recovery Rate | 34.58% | 29.58% |
| Recovered Revenue (INR) | 2,084,060.68 | 1,834,544.83 |
| Average Recovery Time (min) | 723.2 | 720.3 |
| Unnecessary Retry Rate | 45.75% | 20.42% |
| Messages per Recovered Payment | 0.749 | 0.594 |
| Cost per Recovery (INR) | 3.5217 | 18.0831 |
| Human Escalation Rate | 0.00% | 9.17% |
| False-Positive Decision Rate | 4.83% | 6.50% |
| Policy Violations Caught | 0 | 4 |
| Decision Latency (avg ms, n=200) | 0.0003 | 0.2576 |

> Note: baseline decision latency is a pure in-memory lookup with no database access; PayBack AI's latency includes real SQL reads (classifier output, recovery probability, action history). These are intentionally not apples-to-apples - the asymmetry itself reflects the two systems' designs.

## Cost Assumptions (illustrative only, NOT Razorpay's real costs)
- retry_cost_inr: 2.0
- message_cost_inr: 0.5
- human_review_cost_inr: 50.0
