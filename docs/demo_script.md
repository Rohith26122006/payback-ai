# Demo Script

A ~5-8 minute ordered walkthrough for a live demo. Assumes the pipeline has already been
run once (see README "Quick Start") so the dashboard has data to show.

## Setup (before the demo starts)

```bash
uvicorn app.main:app --reload &
streamlit run dashboard/streamlit_app.py
```

## 1. Overview page (30s)

Open the dashboard. Point out the three badges: **Synthetic Data**, **Simulated Outcomes**,
**Development Version** - say plainly that nothing here touches real money.

Show the 4 KPI cards and the baseline-vs-PayBack-AI bar chart. Say: *"PayBack AI's raw
recovery rate is actually a bit lower than the dumb baseline - that's not a bug, it's a
trade-off, and we'll see why."*

## 2. Transaction drill-down: the safety override (90s) - the key moment

Go to **Transactions**, filter by `failure_category = suspected_risk`, pick one.

Point at the **Recommended action & explanation** panel: action is `human_escalation`.
Open the **Safety-rule results** table and find `no_auto_retry_suspected_risk`. Say:
*"This is the policy engine, not the AI, making this call. Even if the decision engine or
an LLM proposed a retry here, this rule would override it - that's enforced in code and
tested directly (`tests/test_policy_engine.py`), not just a prompt instruction."*

Scroll to **Audit history** - show the timestamped trail of exactly what happened.

## 3. Human Review Queue (45s)

Go to **Human Review Queue**. Show the list, open one entry, click **Approve & Resolve**.
Say: *"Every escalation the policy engine forces lands here - a human always sees it before
anything real would happen."*

## 4. What-If Simulator (90s)

Go to **What-If Simulator**. Drag `max_auto_retries` down to 0 and `high_value_threshold`
down to 0. Watch the metrics recompute live. Say: *"Nothing here touches the real database -
this is a merchant exploring policy trade-offs before committing to them."*

Move the sliders back to defaults and show the recovery rate lands on **exactly 29.58%** -
matching the real persisted run. Say: *"That's not a coincidence - it's how we validated
this tool was actually correct, and it caught a real bug during development"* (see
`docs/limitations.md` if asked).

## 5. Evaluation report (60s)

Open `evaluation/results.md` (or run `python -m evaluation.evaluate` live). Walk through:
recovery rate trade-off, the **20.42% vs 45.75% unnecessary retry rate** (the headline win),
and cost-per-recovery being higher for PayBack AI (honest about the trade-off).

## 6. (Optional) API / tests (30s)

`http://127.0.0.1:8000/docs` for the Swagger UI, or `pytest -v` to show **167 passing tests**
covering every safety rule individually.

## Closing line

*"PayBack AI doesn't try to win on every metric - it trades some raw recovery rate for
dramatically fewer wasted retries and enforced human oversight on risky, high-value cases.
The numbers are here to let a merchant decide if that trade-off is worth it, not to hide it."*
