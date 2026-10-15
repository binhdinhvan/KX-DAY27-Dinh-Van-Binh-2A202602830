"""Exercise 5: run with `uv run streamlit run app.py`."""
from __future__ import annotations

import asyncio
import re
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_conn, db_path, replay_events
from exercises.exercise_4_audit import build_graph


async def recent_sessions():
    async with db_conn() as conn:
        async with conn.execute("""
            SELECT thread_id, pr_url, MAX(timestamp) AS last_event, COUNT(*) AS events,
                   CASE MAX(CASE risk_level WHEN 'high' THEN 3 WHEN 'med' THEN 2 ELSE 1 END)
                       WHEN 3 THEN 'high' WHEN 2 THEN 'med' ELSE 'low' END AS worst_risk
            FROM audit_events GROUP BY thread_id, pr_url
            ORDER BY MAX(timestamp) DESC LIMIT 25
        """) as cur:
            return [dict(row) for row in await cur.fetchall()]


async def run_graph(pr_url: str, thread_id: str, resume_value=None, *, restore_only=False):
    """Restore checkpoints without executing nodes; resume only on user action."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        graph = build_graph(cp, confirm_escalation=True)
        cfg = {"configurable": {"thread_id": thread_id}}
        snapshot = await graph.aget_state(cfg)
        if snapshot.values:
            if pr_url and snapshot.values.get("pr_url") != pr_url:
                raise ValueError("PR URL does not match this saved session.")
            if restore_only or not snapshot.next:
                result = dict(snapshot.values)
                if snapshot.interrupts:
                    result["__interrupt__"] = snapshot.interrupts
                result["_pending"] = bool(snapshot.next and not snapshot.interrupts)
                return result
            if snapshot.interrupts:
                if resume_value is None:
                    return {**snapshot.values, "__interrupt__": snapshot.interrupts}
                return await graph.ainvoke(Command(resume=resume_value), cfg)
            return await graph.ainvoke(None, cfg)
        if restore_only or resume_value is not None:
            raise ValueError("Saved session not found.")
        if not re.fullmatch(r"https://github\.com/[^/\s]+/[^/\s]+/pull/\d+/?", pr_url):
            raise ValueError("Enter a GitHub PR URL, for example https://github.com/owner/repo/pull/1")
        return await graph.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)


def accept_result(result):
    st.session_state.pr_url = result.get("pr_url", st.session_state.pr_url)
    st.session_state.interrupt_payload = (
        result["__interrupt__"][0].value if result.get("__interrupt__") else None
    )
    st.session_state.pending = result.get("_pending", False)
    st.session_state.final = None if result.get("__interrupt__") or st.session_state.pending else result


def open_session(thread_id):
    result = asyncio.run(run_graph("", thread_id, restore_only=True))
    st.session_state.thread_id = thread_id
    st.query_params["thread"] = thread_id
    accept_result(result)


def render_approval_card(payload):
    refined = bool(payload.get("escalation_answers"))
    title = "Confirm refined review" if refined else "Approval requested"
    st.subheader(f"{title} — confidence {payload['confidence']:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])
    for c in payload.get("comments", []):
        st.markdown(f"- **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")
    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")
    thread = st.session_state.thread_id
    feedback = st.text_area("Feedback for Approve / Reject (optional)", key=f"feedback_{thread}")
    replacement = st.text_area(
        "Complete replacement comment for Edit", key=f"replacement_{thread}",
        help="Edit posts this text as the entire review comment.",
    )
    col1, col2, col3 = st.columns(3)
    if col1.button("Approve", type="primary"):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("Reject"):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("Edit and post"):
        if replacement.strip():
            return {"choice": "edit", "feedback": replacement.strip()}
        st.error("Enter the replacement comment before choosing Edit.")
    return None


def render_escalation_card(payload):
    st.subheader(f"Strong escalation — confidence {payload['confidence']:.0%}")
    st.caption(payload["confidence_reasoning"])
    for risk in payload.get("risk_factors", []):
        st.error(risk)
    st.markdown(payload["summary"])
    st.info("Answer each question. Use 'unknown' if unsure. You will confirm the refined review before posting.")
    with st.form(f"escalation_{st.session_state.thread_id}"):
        answers = {
            q: st.text_area(q, key=f"qa_{st.session_state.thread_id}_{i}")
            for i, q in enumerate(payload["questions"])
        }
        if st.form_submit_button("Submit answers"):
            if all(a.strip() for a in answers.values()):
                return {q: a.strip() for q, a in answers.items()}
            st.error("Answer every question; use 'unknown' when needed.")
    return None


def main():
    load_dotenv()
    st.set_page_config(page_title="HITL PR Review", page_icon="🔎", layout="wide")
    st.title("HITL PR Review Agent")
    st.caption("Analyze a GitHub PR, review risks and keep a complete decision trail.")
    for key, value in {"thread_id": None, "pr_url": "", "interrupt_payload": None,
                       "final": None, "pending": False}.items():
        st.session_state.setdefault(key, value)

    requested = st.query_params.get("thread")
    if requested and requested != st.session_state.thread_id:
        try:
            open_session(requested)
        except Exception as error:
            st.error(f"Could not restore session: {error}")

    with st.sidebar:
        st.header("Recent sessions")
        try:
            sessions = asyncio.run(recent_sessions())
            if sessions:
                st.dataframe([
                    {k: row[k] for k in ("thread_id", "pr_url", "worst_risk", "last_event", "events")}
                    for row in sessions
                ], hide_index=True)
                selected = st.selectbox("Saved session", sessions,
                                        format_func=lambda row: f"{row['thread_id'][:8]} · {row['pr_url'].split('/')[-1]} · {row['worst_risk']}")
                if st.button("Open session"):
                    open_session(selected["thread_id"])
                    st.rerun()
            else:
                st.caption("No sessions yet.")
        except Exception as error:
            st.error(f"Could not load sessions: {error}")

    with st.form("start"):
        pr_url = st.text_input("PR URL", value=st.session_state.pr_url,
                               placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1")
        st.caption("Confidence ≥73% posts automatically. Other reviews require your input.")
        submitted = st.form_submit_button("Run review", type="primary")
    if submitted:
        if not re.fullmatch(r"https://github\.com/[^/\s]+/[^/\s]+/pull/\d+/?", pr_url.strip()):
            st.error("Enter a valid GitHub PR URL.")
        else:
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.pr_url = pr_url.strip()
            st.session_state.interrupt_payload = None
            st.session_state.final = None
            st.session_state.pending = False
            st.query_params["thread"] = st.session_state.thread_id
            try:
                with st.spinner("Fetching PR and analyzing changes..."):
                    accept_result(asyncio.run(run_graph(st.session_state.pr_url, st.session_state.thread_id)))
                st.rerun()
            except Exception as error:
                st.session_state.pending = True
                st.error(f"Review failed: {error}")

    if st.session_state.pending:
        st.info("This session has unfinished work. Continue it or start a new review.")
        if st.button("Continue saved session"):
            try:
                with st.spinner("Continuing..."):
                    accept_result(asyncio.run(run_graph(st.session_state.pr_url, st.session_state.thread_id)))
                st.rerun()
            except Exception as error:
                st.error(f"Could not continue: {error}")

    payload = st.session_state.interrupt_payload
    if payload:
        answer = render_approval_card(payload) if payload["kind"] == "approval_request" else render_escalation_card(payload)
        if answer is not None:
            try:
                with st.spinner("Resuming review..."):
                    accept_result(asyncio.run(run_graph(st.session_state.pr_url, st.session_state.thread_id, answer)))
                st.rerun()
            except Exception as error:
                # Restore the actual checkpoint if a downstream step failed.
                try:
                    open_session(st.session_state.thread_id)
                except Exception:
                    pass
                st.error(f"Could not resume: {error}")

    final = st.session_state.final
    if final is not None:
        action = final.get("final_action", "unknown")
        if action in {"committed", "auto_committed"}:
            st.success("Review comment posted to GitHub.")
            if final.get("posted_comment_url"):
                st.link_button("View comment on GitHub", final["posted_comment_url"])
            else:
                st.link_button("View PR on GitHub", st.session_state.pr_url)
        elif action == "rejected":
            st.warning("Rejected — no comment posted.")
        elif "failed" in action:
            st.error("Posting failed. Check GitHub access and the terminal output.")
        else:
            st.info(f"Result: {action}")
        if final.get("analysis"):
            a = final["analysis"]
            st.metric("Confidence", f"{a.confidence:.0%}")
            st.markdown(a.summary)
            st.caption(a.confidence_reasoning)
        if final.get("posted_comment_body"):
            with st.expander("Posted review"):
                st.markdown(final["posted_comment_body"])

    if st.session_state.thread_id:
        st.caption(f"Session: {st.session_state.thread_id}")
        with st.expander("Audit trail"):
            events = asyncio.run(replay_events(st.session_state.thread_id))
            st.dataframe(events, hide_index=True)
        st.code(f"uv run python -m audit.replay --thread {st.session_state.thread_id}", language="powershell")


if __name__ == "__main__":
    main()
