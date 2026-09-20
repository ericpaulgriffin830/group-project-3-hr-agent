"""Shared LLM client. Every model call in this project goes through here.

Why one module: the evaluation reruns have to be comparable across the week, and
that only holds if temperature, seed and model are set in exactly one place. A
second call site with its own defaults would quietly break the ablation.

Determinism (rubric item 1, "set fixed seeds where applicable"):
  temperature=0, top_p=1, a fixed seed. Groq's seed is best-effort, not a hard
  guarantee -- it narrows variance, it does not remove it. Anything that must be
  byte-stable across runs (ids, tokens, chunk boundaries) is computed from a
  digest elsewhere, never from the model.

Key rotation: the free tier is rate-limited per key, so all three of us supply
one. On a 429 the client moves to the next key and retries rather than failing
the request. With one key configured it still works; it just has nowhere to go.
"""

from __future__ import annotations

import itertools
import os
import random
import time
from dataclasses import dataclass
from typing import Iterable

from groq import Groq
from groq import APIConnectionError, APIStatusError, RateLimitError

DEFAULT_MODEL = "llama-3.3-70b-versatile"
SEED = 20260920

MAX_ATTEMPTS = 6
BASE_BACKOFF_S = 0.5
MAX_BACKOFF_S = 8.0


class LLMError(RuntimeError):
    """Raised when every key and every retry has been exhausted."""


class LLMNotConfigured(LLMError):
    """No API key present. Distinct from a failure, so setup reads differently."""


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    attempts: int


def _keys() -> list[str]:
    """Every configured key, in order, de-duplicated.

    Accepts GROQ_API_KEY / _2 / _3, or a comma-separated GROQ_API_KEYS.
    """
    raw: list[str] = []
    bulk = os.getenv("GROQ_API_KEYS", "")
    if bulk:
        raw.extend(bulk.split(","))
    raw.extend(os.getenv(name, "") for name in ("GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3"))

    seen: set[str] = set()
    keys: list[str] = []
    for key in (k.strip() for k in raw):
        if key and key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def _model() -> str:
    return os.getenv("GROQ_MODEL") or DEFAULT_MODEL


def _backoff(attempt: int) -> float:
    """Exponential with full jitter -- three clients retrying in lockstep is a stampede."""
    ceiling = min(MAX_BACKOFF_S, BASE_BACKOFF_S * (2**attempt))
    return random.uniform(0, ceiling)


def complete(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> Completion:
    """One deterministic chat completion, retrying across keys on rate limits.

    Raises LLMNotConfigured if no key is set, LLMError if everything is exhausted.
    Never returns a fabricated answer on failure -- a caller that cannot tell a
    real answer from a fallback string will ship the fallback to a grader.
    """
    keys = _keys()
    if not keys:
        raise LLMNotConfigured(
            "No Groq API key. Copy .env.example to .env and set GROQ_API_KEY "
            "(free key at https://console.groq.com -> API Keys)."
        )

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    chosen = model or _model()
    started = time.monotonic()
    last_error: Exception | None = None

    # Walk keys in order, cycling, so a single-key setup still gets its retries.
    key_cycle: Iterable[str] = itertools.islice(itertools.cycle(keys), MAX_ATTEMPTS)

    for attempt, key in enumerate(key_cycle):
        try:
            response = Groq(api_key=key).chat.completions.create(
                model=chosen,
                messages=messages,
                temperature=temperature,
                top_p=1,
                seed=SEED,
                max_tokens=max_tokens,
            )
        except RateLimitError as exc:
            # Expected on the free tier. Next key, and back off in case it shares a quota.
            last_error = exc
            time.sleep(_backoff(attempt))
            continue
        except (APIConnectionError, APIStatusError) as exc:
            status = getattr(exc, "status_code", None)
            # 4xx other than 429 is our bug -- bad model id, malformed request.
            # Retrying it just burns the clock before the same failure.
            if status is not None and 400 <= status < 500 and status != 429:
                raise LLMError(f"Groq rejected the request ({status}): {exc}") from exc
            last_error = exc
            time.sleep(_backoff(attempt))
            continue

        usage = response.usage
        return Completion(
            text=response.choices[0].message.content or "",
            model=chosen,
            latency_ms=int((time.monotonic() - started) * 1000),
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
            attempts=attempt + 1,
        )

    raise LLMError(
        f"All {len(keys)} key(s) exhausted after {MAX_ATTEMPTS} attempts. Last error: {last_error}"
    )


def health() -> dict:
    """Configuration check for /health. Never sends a completion, never logs a key."""
    keys = _keys()
    return {
        "provider": "groq",
        "configured": bool(keys),
        "keys_available": len(keys),
        "model": _model(),
        "seed": SEED,
    }


def list_models() -> list[str]:
    """Model ids this key can actually reach.

    Worth running before pinning GROQ_MODEL -- provider lineups change, and a
    stale id fails as a 404 at the worst moment rather than at setup.
    """
    keys = _keys()
    if not keys:
        raise LLMNotConfigured("No Groq API key configured.")
    return sorted(m.id for m in Groq(api_key=keys[0]).models.list().data)


if __name__ == "__main__":
    import json
    import sys

    if "--models" in sys.argv:
        for model_id in list_models():
            print(model_id)
    else:
        print(json.dumps(health(), indent=2))
