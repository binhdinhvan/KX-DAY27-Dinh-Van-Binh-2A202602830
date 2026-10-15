"""Verify the UI comment adapter without sending any network requests."""
import os
import unittest
from unittest.mock import patch

import httpx

from exercises.github_review import post_review_comment


class GitHubReviewTests(unittest.TestCase):
    def test_returns_comment_url(self):
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(201, json={"html_url": "https://github.com/test/repo/pull/1#issuecomment-123"})

        client = httpx.Client(transport=httpx.MockTransport(respond))
        with patch.dict(os.environ, {"GITHUB_TOKEN": "test-only-token"}), patch("exercises.github_review.httpx.Client", return_value=client):
            url = post_review_comment("https://github.com/test/repo/pull/1", "Review")
        self.assertEqual(url, "https://github.com/test/repo/pull/1#issuecomment-123")
        self.assertEqual(len(requests), 1)
        self.assertEqual(str(requests[0].url), "https://api.github.com/repos/test/repo/issues/1/comments")
        self.assertEqual(requests[0].method, "POST")
        self.assertEqual(requests[0].read(), b'{"body":"Review"}')

    def test_propagates_github_access_error(self):
        client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(403)))
        with patch.dict(os.environ, {"GITHUB_TOKEN": "test-only-token"}), patch("exercises.github_review.httpx.Client", return_value=client):
            with self.assertRaises(httpx.HTTPStatusError):
                post_review_comment("https://github.com/test/repo/pull/1", "Review")
