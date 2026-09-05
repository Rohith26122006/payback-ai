# Architecture

PayBack AI is a decision-support pipeline for failed-payment recovery. It never touches
real money — every payment, decision, and outcome in this system is synthetic or simulated.

## Pipeline

```
Synthetic Data Generator (data/generate_synthetic_data.py)
        |
SQLite (app/database.py: merchants, payments, model_predictions,
        recovery_decisions, action_events, audit_logs, review_queue)
        |
Failure Classifier (core/classifier.py)          -- deterministic, rule-based
        |
Recovery Probability Model (core/recovery_model.py) -- scikit-learn logistic regression
        |
Policy & Safety Engine (core/policy_engine.py)    -- AUTHORITATIVE, can override anything above
        |
Decision Engine (core/decision_engine.py)          -- category + probability + policy -> action
        |
LLM Message Generator (llm/message_generator.py)   -- optional, drafts customer_message only
        |
Payment Simulator (core/simulator.py)              -- simulates the outcome, never real
        |
Audit Logger (core/audit_logger.py)                -- append-only trail of every state change
        |
FastAPI (api/) + Streamlit Dashboard (dashboard/)  -- expose everything above
```

A parallel, deliberately "dumb" **fixed-retry baseline** (`core/baseline.py`) runs on the
same data through its own code path (never touching the policy engine at all) so the
comparison in `evaluation/` is credible — it isn't a strawman built to lose.

## Why the Policy Engine sits where it does

The Decision Engine *proposes* an action from a deterministic table (failure category +
recovery probability + attempt number). The Policy Engine then has the final word — it can
downgrade, block, or replace that proposal, but it can never be talked into something riskier.
Neither the recovery model nor the LLM can bypass it. This ordering is the direct answer to
"how do you stop the AI from doing something reckless," and it's exercised for real in the
test suite (e.g. a `suspected_risk` transaction proposing `delayed_retry` gets forced to
`human_escalation` — see `tests/test_policy_engine.py`).

## Database Entities

| Table | Purpose |
|---|---|
| `merchants` | merchant_id → merchant_category |
| `payments` | the raw synthetic failed-payment event (source of truth, immutable) |
| `model_predictions` | classifier output (category, confidence, reason) + recovery_probability |
| `recovery_decisions` | decision engine's final action, reason, safety_checks, review flag |
| `action_events` | what actually happened, per source (`baseline` or `payback_ai`) |
| `audit_logs` | append-only log of every state transition, system-wide |
| `review_queue` | human-escalation items, open/resolved |

Splitting `payments` (raw event) from `model_predictions` (derived) keeps the ML leakage
boundary explicit: training features only ever come from columns available *before* the
outcome is known (see the leakage note in `core/recovery_model.py`).

## API Modules

- `api/routes_transactions.py` — list/filter/detail failed payments
- `api/routes_decisions.py` — live decision compute (`GET`) and run-and-persist (`POST .../run`)
- `api/routes_metrics.py` — baseline/PayBack AI/comparison/recovery-model metrics

See `docs/api.md` for the full endpoint reference.

## What's genuinely different from the Stage 0 plan

Two components were added beyond the original folder structure because they turned out to
be necessary, not optional:
- `core/review_queue.py` and `core/whatif.py` (Stage 13) — the review queue table existed
  from Stage 3 but nothing wrote to it until Stage 13; the what-if simulator needed its own
  `ignore_history` code path in `make_decision()` after a real bug was found during testing
  (see `docs/limitations.md`).
- `evaluation/costs.py` — cost assumptions were pulled into their own file rather than
  buried in `evaluation/metrics.py`, so they're easy to find and change.
