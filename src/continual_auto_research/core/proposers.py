"""Proposer backends — turn an LLM into a ``(context) -> proposal`` callable.

CAR's equivalent of auto-scientist's ``llm=local|api|claude`` switch. The fork
routes at executor level (vessel/employee machinery CAR doesn't have); CAR only
needs a factory returning the callable the :class:`HillClimber` already accepts.

Verified from the fork's box-local env profiles: ``api`` and ``local``/``ollama``
are the SAME OpenAI-compatible client — they differ ONLY in ``base_url`` + model
(api → a hosted LiteLLM/OpenRouter proxy; local → the on-box Ollama proxy at
:11435). So there are just two implementations:

* :class:`OpenAICompatProposer` — one ``ChatOpenAI(base_url, model)`` covering
  both api and local/ollama.
* :class:`ClaudeCliProposer` — the ``claude -p`` subscription CLI (a subprocess,
  the one genuinely-different backend; uses the CLI's own OAuth, no API key).

A proposer is given the controller's ``proposer_context()`` (incumbent best +
recent attempts + plateau guidance) and must return the next candidate's proposal
text. Keep the system prompt focused: "read the context, propose ONE improved
candidate; if a run command/SCORE= contract applies, follow it."
"""
from __future__ import annotations

import json
import os
import subprocess
from typing import Callable, Optional

from loguru import logger

_DEFAULT_SYSTEM = (
    "You are an optimisation proposer in a hill-climbing loop. You are given the "
    "best candidate so far and recent attempts with their measured scores. Propose "
    "exactly ONE new candidate that should beat the best. Be concrete and concise. "
    "If the task specifies a run command or a SCORE= contract, follow it exactly. "
    "Output ONLY the candidate — no preamble, no explanation."
)

# The on-box Ollama wake-proxy (see ucl-infra/start-ollama-proxy.sh). Local mode
# points here; it speaks the OpenAI API at /v1.
OLLAMA_PROXY_URL = os.environ.get("OLLAMA_PROXY_URL", "http://127.0.0.1:11435/v1")


class OpenAICompatProposer:
    """Propose via any OpenAI-compatible chat endpoint. Covers both the hosted
    ``api`` proxy and the on-box ``ollama``/``local`` proxy — switched by
    ``base_url`` (and model), exactly like the fork's llm.api / llm.local
    profiles."""

    def __init__(self, model: str, base_url: str, api_key: str = "x",
                 system: Optional[str] = None, temperature: float = 0.7,
                 wake_url: Optional[str] = None):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.system = system or _DEFAULT_SYSTEM
        self.temperature = temperature
        # For the Ollama wake-proxy: hitting /_proxy/health wakes the GPU and
        # tolerates cold start. Set wake_url to the proxy's health endpoint.
        self.wake_url = wake_url
        self._client = None  # lazy: don't require openai at import time

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI  # imported lazily so the lib import is optional
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    def _wake(self) -> None:
        if not self.wake_url:
            return
        try:
            import httpx
            httpx.get(self.wake_url, timeout=5.0)
        except Exception as exc:  # noqa: BLE001 — best-effort wake
            logger.debug("ollama wake ping failed ({}); proceeding anyway", exc)

    def __call__(self, context: str) -> str:
        self._wake()
        client = self._ensure_client()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system},
                {"role": "user", "content": context},
            ],
            temperature=self.temperature,
            timeout=300,
        )
        return (resp.choices[0].message.content or "").strip()


class ClaudeCliProposer:
    """Propose via the ``claude -p`` subscription CLI (no API key — uses the CLI's
    own OAuth). A subprocess call; modeled on the fork's claude_session.py."""

    def __init__(self, model: Optional[str] = None, system: Optional[str] = None,
                 binary: str = "claude", timeout_s: float = 300.0):
        self.model = model
        self.system = system or _DEFAULT_SYSTEM
        self.binary = binary
        self.timeout_s = timeout_s

    def __call__(self, context: str) -> str:
        cmd = [self.binary, "-p", "--output-format", "json"]
        if self.model:
            cmd += ["--model", self.model]
        if self.system:
            cmd += ["--append-system-prompt", self.system]
        try:
            r = subprocess.run(
                cmd, input=context, capture_output=True, text=True, timeout=self.timeout_s,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"claude CLI not found ({self.binary}): {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"claude CLI timed out after {self.timeout_s}s") from exc
        out = (r.stdout or "").strip()
        if r.returncode != 0:
            raise RuntimeError(f"claude CLI failed (rc={r.returncode}): {(r.stderr or out)[:300]}")
        # --output-format json wraps the text in a result envelope; fall back to
        # raw stdout if it isn't JSON (older CLI / plain text).
        try:
            obj = json.loads(out)
            return str(obj.get("result", obj.get("text", out))).strip()
        except (json.JSONDecodeError, AttributeError):
            return out


def build_proposer(cfg: dict) -> Callable[[str], str]:
    """Construct a proposer from a config dict:

        {"kind": "api"|"ollama"|"local"|"claude",
         "model": "...", "base_url": "...", "api_key": "...", "system": "..."}

    api/ollama/local → OpenAICompatProposer; claude → ClaudeCliProposer.
    """
    kind = (cfg.get("kind") or "").strip().lower()
    system = cfg.get("system")
    if kind == "claude":
        return ClaudeCliProposer(model=cfg.get("model"), system=system)
    if kind in ("ollama", "local"):
        base_url = cfg.get("base_url") or OLLAMA_PROXY_URL
        wake = cfg.get("wake_url")
        if wake is None and "11435" in base_url:
            wake = base_url.replace("/v1", "/_proxy/health")
        return OpenAICompatProposer(
            model=cfg["model"], base_url=base_url, api_key=cfg.get("api_key", "ollama"),
            system=system, wake_url=wake,
        )
    if kind in ("api", "openai", "openrouter", "custom"):
        return OpenAICompatProposer(
            model=cfg["model"],
            base_url=cfg.get("base_url") or os.environ.get("DEFAULT_API_BASE_URL", ""),
            api_key=cfg.get("api_key") or os.environ.get("CUSTOM_API_KEY")
                    or os.environ.get("OPENROUTER_API_KEY", "x"),
            system=system,
        )
    raise ValueError(f"unknown proposer kind: {kind!r}")


# ── backend readiness (for the UI's status lights) ───────────────────────────

def backend_status() -> dict:
    """Report which proposer backends are usable right now — for status lights.
    Cheap, never raises: a probe failure just reads as not-ready."""
    out = {}

    # claude CLI present?
    try:
        r = subprocess.run(["claude", "--version"], capture_output=True, text=True, timeout=5)
        out["claude"] = {"ready": r.returncode == 0, "detail": (r.stdout or "").strip()[:40]}
    except Exception as exc:  # noqa: BLE001
        out["claude"] = {"ready": False, "detail": str(exc)[:60]}

    # api proxy: do we have a base_url + key?
    api_url = os.environ.get("DEFAULT_API_BASE_URL", "")
    api_key = os.environ.get("CUSTOM_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")
    out["api"] = {"ready": bool(api_url and api_key),
                  "detail": api_url or "set DEFAULT_API_BASE_URL + key"}

    # ollama proxy reachable?
    try:
        import httpx
        health = OLLAMA_PROXY_URL.replace("/v1", "/_proxy/health")
        resp = httpx.get(health, timeout=3.0)
        out["ollama"] = {"ready": resp.status_code == 200, "detail": OLLAMA_PROXY_URL}
    except Exception as exc:  # noqa: BLE001
        out["ollama"] = {"ready": False, "detail": f"{OLLAMA_PROXY_URL} unreachable"}

    return out
