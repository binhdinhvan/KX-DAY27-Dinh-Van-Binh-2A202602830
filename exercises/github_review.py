"""UI extension: return the posted comment URL without changing common/."""

import os

import httpx

from common.github import parse_pr_url


def post_review_comment(pr_url: str, body: str) -> str:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN is not set")
    owner, repo, number = parse_pr_url(pr_url)
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            f"https://api.github.com/repos/{owner}/{repo}/issues/{number}/comments",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "Day27-HITL-Lab",
            },
            json={"body": body},
        )
        response.raise_for_status()
        return response.json()["html_url"]
