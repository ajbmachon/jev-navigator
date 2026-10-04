"""Thin LLM connectors for LlmStep: a command-line tool or an OpenAI-compatible HTTP endpoint.

The presets' flags come from each CLI's ``--help``. Any other tool works through ``CommandConnector``
with the argv from its own help.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import urllib.request
from collections.abc import Callable, Sequence

from .errors import JvnRefusal

PROMPT_PLACEHOLDER = "{prompt}"
DEFAULT_TIMEOUT_SECONDS = 180
HERMES = "hermes"


class ConnectorError(JvnRefusal, RuntimeError):
    pass


class CommandConnector:
    """Runs ``argv`` in a fresh empty directory. The prompt replaces ``{prompt}`` in an argument, or
    goes on stdin when no argument holds the placeholder. stdout is the reply."""

    def __init__(
        self, argv: Sequence[str], *, name: str, model: str = "", timeout: float = DEFAULT_TIMEOUT_SECONDS
    ):
        self.argv = list(argv)
        self.name = name
        self.model = model
        self.timeout = timeout

    def complete(self, prompt: str) -> str:
        on_stdin = not any(PROMPT_PLACEHOLDER in argument for argument in self.argv)
        argv = [argument.replace(PROMPT_PLACEHOLDER, prompt) for argument in self.argv]
        with tempfile.TemporaryDirectory(prefix="jev-llm-step-") as empty_directory:
            completed = subprocess.run(
                argv,
                input=prompt if on_stdin else None,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=empty_directory,
            )
        if completed.returncode != 0:
            raise ConnectorError(f"{self.name} exited {completed.returncode}: {completed.stderr[:300]}")
        return completed.stdout


def hermes(model: str = "", reasoning: str = "low") -> CommandConnector:
    """The Hermes agent CLI (found on PATH) in one-shot mode (``-z``). Hermes loads its tools with
    approvals bypassed in that mode, so it runs in an empty temporary directory with rules skipped."""
    argv = [HERMES, "--ignore-rules", "--reasoning", reasoning]
    if model:
        argv += ["-m", model]
    return CommandConnector([*argv, "-z", PROMPT_PLACEHOLDER], name="hermes", model=model or "hermes-default")


def pi(model: str, provider: str = "", thinking: str = "low") -> CommandConnector:
    """The pi CLI in print mode with every tool disabled."""
    argv = ["pi", "--print", "--no-tools", "--mode", "text", "--thinking", thinking, "--model", model]
    if provider:
        argv += ["--provider", provider]
    return CommandConnector([*argv, PROMPT_PLACEHOLDER], name="pi", model=model)


def claude(model: str = "") -> CommandConnector:
    """Claude in print mode with no tools.

    Warning: every headless run spends from your Claude plan or API budget, and an automation can
    start many of them. Enable it deliberately, with a call budget in the ``LlmGuard``.
    """
    argv = ["claude", "--print", "--tools", "", "--output-format", "text"]
    if model:
        argv += ["--model", model]
    return CommandConnector(argv, name="claude", model=model or "claude-default")


class OpenAICompatibleConnector:
    """POSTs the prompt to ``{base_url}/chat/completions``; the key comes from ``api_key_env``."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str = "",
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        post: Callable[[str, dict, dict, float], dict] | None = None,
    ) -> None:
        self.name = "openai-compatible"
        self.model = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key_env = api_key_env
        self.timeout = timeout
        self._post = post or _post_json

    def complete(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key_env:
            headers["Authorization"] = f"Bearer {os.environ[self.api_key_env]}"
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0}
        return self._post(self.url, body, headers, self.timeout)["choices"][0]["message"]["content"]


def _post_json(url: str, body: dict, headers: dict, timeout: float) -> dict:
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())
