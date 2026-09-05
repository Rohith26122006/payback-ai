# PayBack AI

**Synthetic revenue-recovery decision system** — built for the Razorpay AI Buildathon (AI Revenue Recovery track).

> ⚠️ This project uses **synthetic data** and **simulated payment outcomes** only.
> It is **not connected** to Razorpay production systems, real customer data, or real payment credentials.

## What this is

PayBack AI decides the safest next action for a failed payment (retry, alternate payment link, reminder,
human escalation, stop, or no action), governed by a deterministic **policy engine** that neither the ML model
nor the LLM layer can override. All outcomes are simulated and logged for audit, and compared honestly against
a deliberately "dumb" fixed-retry baseline — see [Results](#results) below.

## Status

**Development version. All 17 build stages complete, 167/167 tests passing.** See `PROJECT_STATE.md` for the
detailed build log and `docs/limitations.md` for an honest account of what this prototype does and doesn't do
(including a real bug found and fixed during development).

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Run the full pipeline

```bash
python -m data.generate_synthetic_data --n 1200 --seed 42 --output data/synthetic_payments.csv
python -m data.seed_database --csv data/synthetic_payments.csv --db payback.db --reset
python -m core.baseline
python -m core.classifier
python -m core.recovery_model
python -m core.decision_engine
python -m core.simulator
python -m evaluation.evaluate
```

## Run the API

```bash
uvicorn app.main:app --reload
```
Swagger UI: `http://127.0.0.1:8000/docs`. Reference: `docs/api.md`.

## Run the dashboard

```bash
streamlit run dashboard/streamlit_app.py
```
Overview, Transactions (with drill-down), Human Review Queue, What-If Simulator, Audit Log.

## Run tests

```bash
pytest -v
```
167 tests, including a dedicated `tests/test_reliability.py` mapping 1:1 to all 13 required
reliability scenarios from the spec.

## Results

Baseline (fixed-retry) vs PayBack AI, on the same 1200 synthetic transactions (seed=42):

| Metric | Baseline | PayBack AI |
|---|---|---|
| Recovery Rate | 34.58% | 29.58% |
| Unnecessary Retry Rate | 45.75% | **20.42%** |
| Cost per Recovery (INR) | 3.52 | 18.08 |
| Human Escalation Rate | 0.00% | 9.17% |
| Policy Violations Caught | 0 | 4 |

**This is not a one-sided win for PayBack AI, on purpose.** It trades some raw recovery rate
and cost-per-recovery for dramatically fewer wasted automated retries and enforced human
oversight on risky/high-value transactions. Full methodology, metric definitions, and the
complete table: `docs/evaluation.md`.

## Documentation

- `docs/architecture.md` — pipeline, database schema, module design
- `docs/api.md` — endpoint reference
- `docs/evaluation.md` — methodology, metric definitions, full results
- `docs/safety.md` — every safeguard mapped to the code that enforces it
- `docs/demo_script.md` — ordered live-demo walkthrough
- `docs/pitch_script.md` — 5-minute pitch script
- `docs/limitations.md` — honest limitations and future work

## Folder structure

```
payback-ai/
  app/          FastAPI app bootstrap, database connection, shared schemas
  api/          API route modules (transactions, decisions, metrics)
  core/         classifier, recovery model, policy engine, decision engine,
                baseline, simulator, audit logger, review queue, what-if simulator
  llm/          LLM provider abstraction + mock/Claude providers + message generator
  data/         synthetic data generation + seeding
  evaluation/   cost assumptions, extended metrics, full comparison report
  dashboard/    Streamlit UI
  tests/        167 pytest tests across 17 modules
  docs/         architecture, API, evaluation, safety, demo/pitch scripts, limitations
```
