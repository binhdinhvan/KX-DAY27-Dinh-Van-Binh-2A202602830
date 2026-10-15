"""Streamlit interaction tests; external APIs are mocked."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest
from common.schemas import PRAnalysis
from exercises import exercise_4_audit as ex


class AppTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {"HITL_DB_PATH": str(Path(self.tmp.name) / "ui.db"), "GITHUB_USER": "tester"})
        self.env.start()
        self.fetch = patch.object(ex, "fetch_pr", return_value=SimpleNamespace(
            title="Test PR", diff="diff --git auth.py", files_changed=["auth.py"], head_sha="abcdef123"))
        self.fetch.start()
        self.factory = patch.object(ex, "get_llm")
        self.llm = self.factory.start().return_value.with_structured_output.return_value
        self.post_patch = patch.object(ex, "post_review_comment", return_value="https://github.com/test/repo/pull/1#issuecomment-123")
        self.post = self.post_patch.start()

    def tearDown(self):
        self.post_patch.stop()
        self.factory.stop()
        self.fetch.stop()
        self.env.stop()
        self.tmp.cleanup()

    def analysis(self, c):
        return PRAnalysis(summary="Review summary", confidence=c, confidence_reasoning="Needs context",
                          risk_factors=["Weak hashing"], escalation_questions=["auth.py: Is this deployed?"])

    def button(self, at, label):
        return next(b for b in at.button if b.label == label)

    def start(self):
        at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20).run()
        self.assertEqual(len(at.exception), 0)
        at.text_input[0].set_value("https://github.com/test/repo/pull/1")
        self.button(at, "Run review").click().run()
        self.assertEqual(len(at.exception), 0)
        return at

    def test_approval_restores_after_reload_and_rejects(self):
        self.llm.ainvoke = AsyncMock(return_value=self.analysis(0.65))
        at = self.start()
        self.post.assert_not_called()
        thread = at.session_state["thread_id"]
        restored = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20)
        restored.query_params["thread"] = thread
        restored.run()
        self.assertEqual(len(restored.exception), 0)
        self.assertEqual(restored.session_state["interrupt_payload"]["kind"], "approval_request")
        self.button(restored, "Reject").click().run()
        self.assertEqual(restored.session_state["final"]["final_action"], "rejected")
        self.post.assert_not_called()
        self.assertEqual(self.llm.ainvoke.await_count, 1)

    def test_escalation_requires_final_confirmation_and_edit(self):
        self.llm.ainvoke = AsyncMock(side_effect=[self.analysis(0.5), self.analysis(0.55)])
        at = self.start()
        self.assertEqual(at.session_state["interrupt_payload"]["kind"], "escalation")
        self.button(at, "Submit answers").click().run()
        self.assertTrue(any("Answer every question" in e.value for e in at.error))
        at.text_area[0].set_value("unknown")
        self.button(at, "Submit answers").click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.session_state["interrupt_payload"]["kind"], "approval_request")
        self.post.assert_not_called()
        self.assertTrue(any("Confirm refined review" in h.value for h in at.subheader))
        self.button(at, "Edit and post").click().run()
        self.post.assert_not_called()
        at.text_area[1].set_value("My final edited review")
        self.button(at, "Edit and post").click().run()
        self.assertEqual(len(at.exception), 0)
        self.post.assert_called_once()
        self.assertEqual(self.post.call_args.args[1], "My final edited review")
        self.assertEqual(at.session_state["final"]["posted_comment_url"], self.post.return_value)

    def test_reject_after_escalation_never_posts(self):
        self.llm.ainvoke = AsyncMock(side_effect=[self.analysis(0.5), self.analysis(0.55)])
        at = self.start()
        at.text_area[0].set_value("unknown")
        self.button(at, "Submit answers").click().run()
        self.button(at, "Reject").click().run()
        self.assertEqual(at.session_state["final"]["final_action"], "rejected")
        self.post.assert_not_called()

    def test_auto_approval_and_completed_reload(self):
        self.llm.ainvoke = AsyncMock(return_value=self.analysis(0.9))
        at = self.start()
        self.post.assert_called_once()
        self.assertEqual(at.session_state["final"]["final_action"], "auto_committed")
        thread = at.session_state["thread_id"]
        restored = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20)
        restored.query_params["thread"] = thread
        restored.run()
        self.assertEqual(len(restored.exception), 0)
        self.post.assert_called_once()
        self.assertEqual(restored.session_state["final"]["posted_comment_url"], self.post.return_value)

    def test_invalid_pr_does_not_call_llm(self):
        self.llm.ainvoke = AsyncMock()
        at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20).run()
        at.text_input[0].set_value("bad-url")
        self.button(at, "Run review").click().run()
        self.assertTrue(at.error)
        self.llm.ainvoke.assert_not_called()

    def test_approve_and_failed_post_status(self):
        self.llm.ainvoke = AsyncMock(return_value=self.analysis(0.65))
        at = self.start()
        self.button(at, "Approve").click().run()
        self.assertEqual(at.session_state["final"]["final_action"], "committed")
        self.post.assert_called_once()
        self.post.side_effect = RuntimeError("Simulated failure")
        self.button(at, "Run review").click().run()
        self.button(at, "Approve").click().run()
        self.assertEqual(at.session_state["final"]["final_action"], "commit_failed")
        self.assertTrue(any("Posting failed" in e.value for e in at.error))
        self.assertFalse(at.success)


if __name__ == "__main__":
    unittest.main()
