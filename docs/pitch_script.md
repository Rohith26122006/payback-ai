# 5-Minute Pitch Script

## The problem (45s)

Online merchants lose real revenue when legitimate payments fail — a card expired, a bank
timed out, a customer's balance was momentarily low. Most systems treat every failure the
same way: retry blindly, send a generic reminder, hope for the best. That wastes money on
retries that were never going to succeed, annoys customers who didn't need a reminder, and
worse — it can retry suspicious transactions without a second thought.

## The idea (60s)

PayBack AI picks the safest next action for *each* failed payment individually — retry,
alternate payment link, reminder, human escalation, or stop — based on why it failed and how
likely it is to recover. But the part we actually care about proving isn't the AI part.
It's the part that stops the AI from doing something reckless: a policy engine that sits
between every recommendation and every action, with the final word. It enforces nine
concrete safeguards — max 2 retries, a cooldown, one message per day, never auto-retry a
suspected-risk transaction, always escalate what the model is uncertain about or what's
high-value — and neither the machine-learning model nor the LLM can override any of them.

## The demo (2 min)

*(Live: pull up a suspected-risk transaction, show `human_escalation` and exactly which
safety rule forced it. Then the What-If Simulator, dragging policy sliders and watching
recovery rate and human-review count recompute live, with nothing touching the real
database.)*

## The honest result (60s)

Here's the part I want to be upfront about: PayBack AI does **not** win on every metric
against a dumb fixed-retry baseline. Its raw recovery rate is actually a bit lower — 29.6%
versus 34.6% — and its cost per recovery is higher, because it escalates risky and
high-value cases to a human instead of blindly retrying them. What it does dramatically
better is cut unnecessary retries **in half** — 20.4% versus 45.8% — while catching real
policy violations before they'd have executed. That's the trade-off: fewer wasted, riskier
automated actions, in exchange for some raw recovery percentage. We think that's the right
trade for a payments system to make, and the evaluation report is built to let you check
that math yourself, not take our word for it.

## Close (15s)

Everything here runs on synthetic data with simulated outcomes, is fully tested — 167 tests,
one for essentially every safety rule — and runs completely without an API key. It's a
prototype built the way we'd want a real revenue-recovery system to be built: safety-first,
and honest about its own trade-offs.
