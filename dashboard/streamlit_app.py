"""
PayBack AI - Streamlit dashboard.

Reads directly from the SQLite database built by the core/ pipeline
(no FastAPI dependency needed to run this - simpler for a prototype).
Everything shown here is SYNTHETIC DATA and SIMULATED OUTCOMES.
"""
from __future__ import annotations

import json
import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st

from app.database import get_connection, resolve_db_path
from core.audit_logger import get_audit_trail
from core.baseline import compute_baseline_metrics
from core.policy_engine import PolicyConfig
from core.review_queue import list_open_reviews, resolve_review_item
from core.simulator import compute_payback_ai_metrics
from core.whatif import run_what_if_from_db

st.set_page_config(page_title="PayBack AI", layout="wide")


# --- data access helpers ---------------------------------------------------


def _get_conn(db_path: str) -> sqlite3.Connection:
    # NOT cached: Streamlit can run script reruns on different threads,
    # and SQLite connections are not safe to share across threads by
    # default. Opening a fresh connection per rerun is cheap for a
    # dataset this size and avoids "SQLite objects created in a thread
    # can only be used in that same thread" errors.
    return get_connection(db_path)


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def _has_rows(conn: sqlite3.Connection, table: str, where: str = "1=1") -> bool:
    try:
        row = conn.execute(f"SELECT COUNT(*) AS c FROM {table} WHERE {where}").fetchone()
        return bool(row and row["c"] > 0)
    except sqlite3.OperationalError:
        return False


def load_payments(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM payments", conn)


def load_decisions(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM recovery_decisions", conn)


def get_transaction_detail(conn: sqlite3.Connection, transaction_id: str) -> dict:
    payment = conn.execute(
        "SELECT * FROM payments WHERE transaction_id = ?", (transaction_id,)
    ).fetchone()
    prediction = conn.execute(
        "SELECT * FROM model_predictions WHERE transaction_id = ? ORDER BY id DESC LIMIT 1",
        (transaction_id,),
    ).fetchone()
    decision = conn.execute(
        "SELECT * FROM recovery_decisions WHERE transaction_id = ? ORDER BY id DESC LIMIT 1",
        (transaction_id,),
    ).fetchone()
    baseline_event = conn.execute(
        "SELECT * FROM action_events WHERE transaction_id = ? AND source = 'baseline' "
        "ORDER BY id DESC LIMIT 1",
        (transaction_id,),
    ).fetchone()
    ai_event = conn.execute(
        "SELECT * FROM action_events WHERE transaction_id = ? AND source = 'payback_ai' "
        "ORDER BY id DESC LIMIT 1",
        (transaction_id,),
    ).fetchone()
    audit_trail = get_audit_trail(conn, transaction_id)
    return {
        "payment": dict(payment) if payment else None,
        "prediction": dict(prediction) if prediction else None,
        "decision": dict(decision) if decision else None,
        "baseline_event": dict(baseline_event) if baseline_event else None,
        "ai_event": dict(ai_event) if ai_event else None,
        "audit_trail": audit_trail,
    }


# --- sidebar -----------------------------------------------------------


st.sidebar.title("PayBack AI")
db_path = st.sidebar.text_input("Database path", value=resolve_db_path())
page = st.sidebar.radio(
    "View",
    ["Overview", "Transactions", "Human Review Queue", "What-If Simulator", "Audit Log"],
)

conn = _get_conn(db_path)

st.title("PayBack AI — Synthetic Revenue Recovery Simulator")
badge_col1, badge_col2, badge_col3 = st.columns(3)
with badge_col1:
    st.info("Synthetic Data")
with badge_col2:
    st.info("Simulated Outcomes")
with badge_col3:
    st.warning("Development Version")

if not _table_exists(conn, "payments") or not _has_rows(conn, "payments"):
    st.error(
        "No payment data found. Run the pipeline first:\n\n"
        "```\npython -m data.seed_database --csv data/synthetic_payments.csv --db payback.db --reset\n"
        "python -m core.baseline\npython -m core.classifier\npython -m core.recovery_model\n"
        "python -m core.decision_engine\npython -m core.simulator\n```"
    )
    st.stop()


# --- Overview page -------------------------------------------------------

if page == "Overview":
    payments = load_payments(conn)

    total_failed = len(payments)
    total_at_risk = payments["amount_inr"].sum()

    if _has_rows(conn, "action_events", "source = 'payback_ai'"):
        ai_metrics = compute_payback_ai_metrics(conn)
        recovered_revenue = ai_metrics["recovered_revenue"]
        recovery_rate = ai_metrics["recovery_rate"]
    else:
        ai_metrics = None
        recovered_revenue = 0.0
        recovery_rate = 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Failed Transactions", f"{total_failed:,}")
    k2.metric("Total At-Risk Revenue (INR)", f"{total_at_risk:,.0f}")
    k3.metric("Recovered Revenue (INR, PayBack AI)", f"{recovered_revenue:,.0f}")
    k4.metric("Recovery Rate (PayBack AI)", f"{recovery_rate:.1%}")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Baseline vs PayBack AI")
        if _has_rows(conn, "action_events", "source = 'baseline'") and ai_metrics:
            baseline_metrics = compute_baseline_metrics(conn)
            comparison_df = pd.DataFrame(
                {
                    "System": ["Baseline (fixed-retry)", "PayBack AI"],
                    "Recovery Rate": [
                        baseline_metrics["recovery_rate"],
                        ai_metrics["recovery_rate"],
                    ],
                    "Recovered Revenue (INR)": [
                        baseline_metrics["recovered_revenue"],
                        ai_metrics["recovered_revenue"],
                    ],
                    "Unnecessary Retries": [
                        baseline_metrics["unnecessary_retries"],
                        ai_metrics["unnecessary_retries"],
                    ],
                }
            )
            fig = px.bar(
                comparison_df,
                x="System",
                y="Recovery Rate",
                text="Recovery Rate",
                title="Recovery Rate: Baseline vs PayBack AI (SIMULATED)",
            )
            fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
            fig.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        else:
            st.info("Run `python -m core.baseline` and `python -m core.simulator` to see this comparison.")

    with col_right:
        st.subheader("Failure Category Distribution")
        category_counts = payments["failure_category"].value_counts().reset_index()
        category_counts.columns = ["failure_category", "count"]
        fig2 = px.bar(
            category_counts,
            x="failure_category",
            y="count",
            title="Failed Transactions by Category",
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Recommended Action Distribution")
    if _has_rows(conn, "recovery_decisions"):
        decisions = load_decisions(conn)
        action_counts = decisions["action"].value_counts().reset_index()
        action_counts.columns = ["action", "count"]
        fig3 = px.bar(action_counts, x="action", y="count", title="PayBack AI Recommended Actions")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Run `python -m core.decision_engine` to see recommended actions.")


# --- Transactions page ----------------------------------------------------

elif page == "Transactions":
    st.subheader("Transaction-Level Decision View")
    payments = load_payments(conn)

    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        category_options = ["All"] + sorted(payments["failure_category"].unique().tolist())
        selected_category = st.selectbox("Filter by failure category", category_options)
    with filter_col2:
        merchant_options = ["All"] + sorted(payments["merchant_id"].unique().tolist())
        selected_merchant = st.selectbox("Filter by merchant", merchant_options)

    filtered = payments.copy()
    if selected_category != "All":
        filtered = filtered[filtered["failure_category"] == selected_category]
    if selected_merchant != "All":
        filtered = filtered[filtered["merchant_id"] == selected_merchant]

    st.dataframe(
        filtered[
            [
                "transaction_id",
                "merchant_id",
                "amount_inr",
                "payment_method",
                "failure_category",
                "attempt_number",
                "event_timestamp",
            ]
        ],
        use_container_width=True,
        hide_index=True,
        height=300,
    )

    st.divider()
    st.subheader("Transaction Detail")
    txn_options = filtered["transaction_id"].tolist()
    if not txn_options:
        st.warning("No transactions match the current filters.")
    else:
        selected_txn = st.selectbox("Select a transaction to inspect", txn_options)
        detail = get_transaction_detail(conn, selected_txn)

        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Failure details**")
            st.json(
                {
                    k: detail["payment"][k]
                    for k in (
                        "failure_code",
                        "failure_category",
                        "attempt_number",
                        "amount_inr",
                        "payment_method",
                    )
                }
            )
            if detail["prediction"]:
                st.markdown("**Recovery probability & classification**")
                st.json(
                    {
                        "classified_category": detail["prediction"]["classified_category"],
                        "classification_confidence": detail["prediction"]["classification_confidence"],
                        "recovery_probability": detail["prediction"]["recovery_probability"],
                    }
                )
            else:
                st.info("Not yet classified. Run `python -m core.classifier`.")

        with d2:
            if detail["decision"]:
                st.markdown("**Recommended action & explanation**")
                st.write(f"Action: `{detail['decision']['action']}`")
                st.write(f"Requires human review: `{bool(detail['decision']['requires_human_review'])}`")
                st.write(detail["decision"]["reason"])
                st.markdown("**Safety-rule results**")
                safety_checks = json.loads(detail["decision"]["safety_checks"])
                st.dataframe(pd.DataFrame(safety_checks), use_container_width=True, hide_index=True)
            else:
                st.info("No decision yet. Run `python -m core.decision_engine`.")

            st.markdown("**Simulated outcome**")
            if detail["ai_event"]:
                st.write(f"PayBack AI: `{detail['ai_event']['status']}` (SIMULATED)")
            else:
                st.info("Not yet simulated. Run `python -m core.simulator`.")
            if detail["baseline_event"]:
                st.write(f"Baseline: `{detail['baseline_event']['status']}` (SIMULATED)")

        st.markdown("**Audit history**")
        if detail["audit_trail"]:
            st.dataframe(pd.DataFrame(detail["audit_trail"]), use_container_width=True, hide_index=True)
        else:
            st.info("No audit events recorded for this transaction yet.")


# --- Human Review Queue page ----------------------------------------------

elif page == "Human Review Queue":
    st.subheader("Human Review Queue")
    open_reviews = list_open_reviews(conn)

    if not open_reviews:
        st.info("No transactions currently require human review.")
    else:
        st.metric("Transactions Awaiting Review", len(open_reviews))
        for item in open_reviews:
            with st.expander(
                f"{item['transaction_id']} — {item['failure_category']} — "
                f"INR {item['amount_inr']:,.0f} — proposed: {item['action']}"
            ):
                st.write(f"**Escalation reason:** {item['escalation_reason']}")
                st.write(f"**Added:** {item['added_at']}")
                b1, b2 = st.columns(2)
                with b1:
                    if st.button("Approve & Resolve", key=f"approve_{item['id']}"):
                        resolve_review_item(
                            conn, item["id"], resolved_by="merchant", note="Approved via dashboard."
                        )
                        st.rerun()
                with b2:
                    if st.button("Reject & Resolve", key=f"reject_{item['id']}"):
                        resolve_review_item(
                            conn, item["id"], resolved_by="merchant", note="Rejected via dashboard."
                        )
                        st.rerun()


# --- What-If Simulator page ------------------------------------------------

elif page == "What-If Simulator":
    st.subheader("What-If Policy Simulator")
    st.caption(
        "Adjust policy thresholds and see recomputed metrics instantly. "
        "Nothing here is saved — this is a hypothetical exploration only."
    )

    default_policy = PolicyConfig()
    c1, c2 = st.columns(2)
    with c1:
        max_auto_retries = st.slider("Max automatic retries", 0, 5, default_policy.max_auto_retries)
        retry_cooldown_minutes = st.slider(
            "Retry cooldown (minutes)", 0, 120, default_policy.retry_cooldown_minutes, step=5
        )
    with c2:
        max_messages_per_24h = st.slider("Max messages per 24h", 0, 5, default_policy.max_messages_per_24h)
        high_value_threshold_inr = st.slider(
            "High-value threshold (INR)", 0, 200000, int(default_policy.high_value_threshold_inr), step=5000
        )

    what_if_policy = PolicyConfig(
        max_auto_retries=max_auto_retries,
        retry_cooldown_minutes=retry_cooldown_minutes,
        max_messages_per_24h=max_messages_per_24h,
        high_value_threshold_inr=float(high_value_threshold_inr),
    )

    with st.spinner("Recomputing decisions under this policy..."):
        result = run_what_if_from_db(db_path, what_if_policy)

    st.caption(result["note"])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recovery Rate", f"{result['recovery_rate']:.1%}")
    m2.metric("Recovered Revenue (INR)", f"{result['recovered_revenue']:,.0f}")
    m3.metric("Human Review Count", result["human_review_count"])
    m4.metric("Total At-Risk Revenue (INR)", f"{result['total_at_risk_revenue']:,.0f}")

    wc1, wc2 = st.columns(2)
    with wc1:
        st.markdown("**Action distribution under this policy**")
        action_df = pd.DataFrame(
            {"action": list(result["action_distribution"].keys()), "count": list(result["action_distribution"].values())}
        )
        st.plotly_chart(px.bar(action_df, x="action", y="count"), use_container_width=True)
    with wc2:
        st.markdown("**Simulated status distribution under this policy**")
        status_df = pd.DataFrame(
            {"status": list(result["status_distribution"].keys()), "count": list(result["status_distribution"].values())}
        )
        st.plotly_chart(px.bar(status_df, x="status", y="count"), use_container_width=True)

    st.markdown("**Compare to the actual, currently-persisted PayBack AI run:**")
    if _has_rows(conn, "action_events", "source = 'payback_ai'"):
        actual = compute_payback_ai_metrics(conn)
        compare_df = pd.DataFrame(
            {
                "Scenario": ["Current (persisted)", "What-If"],
                "Recovery Rate": [actual["recovery_rate"], result["recovery_rate"]],
                "Recovered Revenue (INR)": [actual["recovered_revenue"], result["recovered_revenue"]],
            }
        )
        st.dataframe(compare_df, use_container_width=True, hide_index=True)
    else:
        st.info("Run `python -m core.simulator` to compare against the actual persisted run.")


# --- Audit Log page --------------------------------------------------------

elif page == "Audit Log":
    st.subheader("Audit Log")
    txn_filter = st.text_input("Filter by transaction ID (optional)")
    if txn_filter:
        audit_df = pd.read_sql_query(
            "SELECT * FROM audit_logs WHERE transaction_id = ? ORDER BY created_at DESC",
            conn,
            params=(txn_filter,),
        )
    else:
        audit_df = pd.read_sql_query(
            "SELECT * FROM audit_logs ORDER BY created_at DESC LIMIT 200", conn
        )
    if audit_df.empty:
        st.info("No audit log entries found.")
    else:
        st.caption(f"Showing {len(audit_df)} most recent entries" + (f" for {txn_filter}" if txn_filter else ""))
        st.dataframe(audit_df, use_container_width=True, hide_index=True, height=600)