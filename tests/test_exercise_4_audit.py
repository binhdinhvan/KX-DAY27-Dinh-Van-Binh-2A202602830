"""Exercise 4 checks use temporary SQLite databases and never call external APIs."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import replay_events
from common.schemas import PRAnalysis
from exercises import exercise_4_audit as ex


class AuditTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmp.name) / "audit.db")
        self.env = patch.dict(os.environ, {"HITL_DB_PATH": self.path, "GITHUB_USER": "tester"})
        self.env.start()
        self.pr = SimpleNamespace(title="Test PR", diff="test diff", files_changed=["auth.py"], head_sha="abcdef123")
        self.url = "https://github.com/test/repo/pull/1"

    async def asyncTearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def analysis(self, confidence=0.65):
        return PRAnalysis(summary="Review", confidence=confidence,
                          confidence_reasoning="Needs context",
                          escalation_questions=["auth.py: Is this deployed?"])

    async def test_approval_choices_and_post_failure(self):
        for choice, fail in [("approve", False), ("reject", False), ("edit", False), ("approve", True)]:
            thread = f"{choice}-{fail}"
            with patch.object(ex, "fetch_pr", return_value=self.pr), patch.object(ex, "get_llm") as factory, patch.object(ex, "post_review_comment") as post:
                factory.return_value.with_structured_output.return_value.ainvoke = AsyncMock(return_value=self.analysis())
                if fail:
                    post.side_effect = RuntimeError("Simulated GitHub failure")
                async with AsyncSqliteSaver.from_conn_string(self.path) as cp:
                    await cp.setup()
                    app = ex.build_graph(cp)
                    cfg = {"configurable": {"thread_id": thread}}
                    result = await app.ainvoke({"pr_url": self.url, "thread_id": thread}, cfg)
                    self.assertIn("__interrupt__", result)
                    post.assert_not_called()
                    result = await app.ainvoke(Command(resume={"choice": choice, "feedback": "Replacement"}), cfg)
                expected = "commit_failed" if fail else "rejected" if choice == "reject" else "committed"
                self.assertEqual(result["final_action"], expected)
                self.assertEqual(post.call_count, 0 if choice == "reject" else 1)
                if choice == "edit":
                    self.assertEqual(post.call_args.args[1], "Replacement")
            events = await replay_events(thread)
            self.assertEqual([e["action"] for e in events],
                             ["fetch_pr", "analyze", "route", "human_approval", "human_approval", "commit"])
            self.assertEqual(events[4]["reviewer_id"], "tester")
            self.assertEqual(events[4]["decision"], choice)
            self.assertIn(expected, events[-1]["reason"])
            self.assertTrue(all(e["execution_time_ms"] >= 0 for e in events))

    async def test_escalation_reopens_checkpoint_and_completed_session(self):
        with patch.object(ex, "fetch_pr", return_value=self.pr) as fetch, patch.object(ex, "get_llm") as factory, patch.object(ex, "post_review_comment") as post:
            llm = factory.return_value.with_structured_output.return_value
            llm.ainvoke = AsyncMock(side_effect=[self.analysis(0.5), self.analysis(0.45)])
            async with AsyncSqliteSaver.from_conn_string(self.path) as cp:
                await cp.setup()
                result = await ex.build_graph(cp).ainvoke(
                    {"pr_url": self.url, "thread_id": "restart"},
                    {"configurable": {"thread_id": "restart"}},
                )
                self.assertEqual(result["__interrupt__"][0].value["kind"], "escalation")
            # run() opens a new connection and resumes the persisted interrupt.
            with patch.object(ex, "handle_interrupt", return_value={"auth.py: Is this deployed?": "unknown"}):
                await ex.run(self.url, "restart")
            fetch.assert_called_once()
            self.assertEqual(llm.ainvoke.await_count, 2)
            post.assert_called_once()
            self.assertIn("unknown", post.call_args.args[1])
            self.assertIn("Initial analysis", llm.ainvoke.call_args.args[0][1]["content"])
            events = await replay_events("restart")
            self.assertEqual([e["action"] for e in events],
                             ["fetch_pr", "analyze", "route", "escalate", "escalate", "synthesize", "commit"])
            self.assertEqual(events[-2]["confidence"], 0.45)
            self.assertEqual(events[-2]["risk_level"], "high")
            await ex.run(self.url, "restart")
            post.assert_called_once()
            self.assertEqual(len(await replay_events("restart")), len(events))
            with self.assertRaises(ValueError):
                await ex.run("https://github.com/test/repo/pull/99", "restart")
            with self.assertRaises(ValueError):
                await ex.run(self.url, "missing-thread")

    async def test_auto_approval_audit(self):
        with patch.object(ex, "fetch_pr", return_value=self.pr), patch.object(ex, "get_llm") as factory, patch.object(ex, "post_review_comment") as post:
            factory.return_value.with_structured_output.return_value.ainvoke = AsyncMock(return_value=self.analysis(0.9))
            await ex.run(self.url, None)
            post.assert_called_once()
        import sqlite3
        conn = sqlite3.connect(self.path)
        try:
            rows = conn.execute("SELECT action, decision, risk_level, reviewer_id FROM audit_events ORDER BY id").fetchall()
        finally:
            conn.close()
        self.assertEqual(rows[-1], ("auto_approve", "auto", "low", None))
        self.assertEqual(len(rows), 4)


if __name__ == "__main__":
    unittest.main()
