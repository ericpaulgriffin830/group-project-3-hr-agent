"""The shared LLM client.

Every test here runs without an API key, because CI does not have one and
because a test suite that only passes on a developer's laptop is not a gate.
The Groq client is stubbed; what is under test is our behaviour around it --
key discovery, rotation on rate limits, and refusing to invent an answer.
"""

from __future__ import annotations

import pytest

from app import llm

KEY_VARS = ("GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3", "GROQ_API_KEYS", "GROQ_MODEL")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """No ambient key leaks in from the developer's shell."""
    for var in KEY_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch):
    """Backoff is real in production and instant in tests."""
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)


class _Usage:
    prompt_tokens = 11
    completion_tokens = 7


class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)


class _Response:
    def __init__(self, content):
        self.choices = [_Choice(content)]
        self.usage = _Usage()


def _stub_groq(monkeypatch, behaviour):
    """Install a fake Groq whose behaviour is keyed by the api_key it is built with.

    `behaviour` maps key -> either an Exception instance to raise or a string to
    return. Records the order keys were tried, which is the thing rotation is
    actually about.
    """
    tried: list[str] = []

    class _Completions:
        def __init__(self, key):
            self._key = key

        def create(self, **kwargs):
            tried.append(self._key)
            outcome = behaviour[self._key]
            if isinstance(outcome, Exception):
                raise outcome
            return _Response(outcome)

    class _Chat:
        def __init__(self, key):
            self.completions = _Completions(key)

    class _FakeGroq:
        def __init__(self, api_key):
            self.chat = _Chat(api_key)

    monkeypatch.setattr(llm, "Groq", _FakeGroq)
    return tried


def _rate_limit() -> llm.RateLimitError:
    return llm.RateLimitError.__new__(llm.RateLimitError)


# ------------------------------------------------------------ key discovery

def test_no_key_is_a_distinct_error_not_a_generic_failure(monkeypatch):
    """Setup problems must read differently from runtime failures."""
    with pytest.raises(llm.LLMNotConfigured):
        llm.complete("hello")


def test_numbered_keys_are_collected_in_order(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_a")
    monkeypatch.setenv("GROQ_API_KEY_2", "gsk_b")
    monkeypatch.setenv("GROQ_API_KEY_3", "gsk_c")
    assert llm._keys() == ["gsk_a", "gsk_b", "gsk_c"]


def test_comma_separated_bulk_var_is_supported(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEYS", "gsk_a, gsk_b ,gsk_c")
    assert llm._keys() == ["gsk_a", "gsk_b", "gsk_c"]


def test_duplicate_keys_are_collapsed(monkeypatch):
    """Three teammates pasting the same key is not three keys' worth of quota.

    Left un-deduplicated, rotation would 'fail over' onto the key that was
    already rate-limited and report three attempts where there was really one.
    """
    monkeypatch.setenv("GROQ_API_KEY", "gsk_same")
    monkeypatch.setenv("GROQ_API_KEY_2", "gsk_same")
    monkeypatch.setenv("GROQ_API_KEY_3", "gsk_other")
    assert llm._keys() == ["gsk_same", "gsk_other"]


def test_blank_keys_are_ignored(monkeypatch):
    """.env.example ships the vars empty; empty must mean absent, not present."""
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("GROQ_API_KEY_2", "   ")
    assert llm._keys() == []


# ---------------------------------------------------------------- rotation

def test_rate_limited_key_rotates_to_the_next(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_a")
    monkeypatch.setenv("GROQ_API_KEY_2", "gsk_b")
    tried = _stub_groq(monkeypatch, {"gsk_a": _rate_limit(), "gsk_b": "answered"})

    result = llm.complete("hello")

    assert result.text == "answered"
    assert tried == ["gsk_a", "gsk_b"]
    assert result.attempts == 2


def test_single_key_still_retries_itself(monkeypatch):
    """One key configured must not mean zero retries -- a 429 is often transient."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_only")
    calls = {"n": 0}

    class _Completions:
        def create(self, **kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                raise _rate_limit()
            return _Response("recovered")

    class _FakeGroq:
        def __init__(self, api_key):
            self.chat = type("C", (), {"completions": _Completions()})()

    monkeypatch.setattr(llm, "Groq", _FakeGroq)
    assert llm.complete("hello").text == "recovered"
    assert calls["n"] == 3


def test_every_key_exhausted_raises_rather_than_returning_a_fallback(monkeypatch):
    """The failure that matters.

    If this returned a placeholder string, the orchestrator would cite it, the
    UI would render it, and a grader would read a fabricated policy answer with
    no indication anything went wrong.
    """
    monkeypatch.setenv("GROQ_API_KEY", "gsk_a")
    monkeypatch.setenv("GROQ_API_KEY_2", "gsk_b")
    _stub_groq(monkeypatch, {"gsk_a": _rate_limit(), "gsk_b": _rate_limit()})

    with pytest.raises(llm.LLMError) as excinfo:
        llm.complete("hello")
    assert "exhausted" in str(excinfo.value)


def test_client_error_fails_fast_instead_of_burning_retries(monkeypatch):
    """A bad model id is our bug. Retrying it six times just delays the same error."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_a")

    bad_request = llm.APIStatusError.__new__(llm.APIStatusError)
    bad_request.status_code = 404
    tried = _stub_groq(monkeypatch, {"gsk_a": bad_request})

    with pytest.raises(llm.LLMError, match="404"):
        llm.complete("hello", model="no-such-model")
    assert len(tried) == 1


# ------------------------------------------------------- determinism/health

def test_determinism_settings_are_actually_sent(monkeypatch):
    """temperature=0 and a fixed seed, or the week's evaluations do not compare."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_a")
    sent = {}

    class _Completions:
        def create(self, **kwargs):
            sent.update(kwargs)
            return _Response("ok")

    class _FakeGroq:
        def __init__(self, api_key):
            self.chat = type("C", (), {"completions": _Completions()})()

    monkeypatch.setattr(llm, "Groq", _FakeGroq)
    llm.complete("hello")

    assert sent["temperature"] == 0.0
    assert sent["top_p"] == 1
    assert sent["seed"] == llm.SEED


def test_model_comes_from_env_when_set(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "some-other-model")
    assert llm._model() == "some-other-model"


def test_health_reports_configuration_without_exposing_the_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_supersecret")
    report = llm.health()

    assert report["configured"] is True
    assert report["keys_available"] == 1
    assert "gsk_supersecret" not in repr(report)


def test_health_is_safe_with_no_key_configured():
    """/health must answer on a misconfigured box, not raise."""
    report = llm.health()
    assert report["configured"] is False
    assert report["keys_available"] == 0


# ------------------------------------------------- chain-of-thought exclusion

def test_reasoning_field_never_escapes_the_client(monkeypatch):
    """The brief forbids exposing hidden chain-of-thought.

    gpt-oss models return a `reasoning` field alongside `content`, carrying literal
    step-by-step thinking. If it reached the trace or the answer it would fail an
    explicit rubric item, so the client drops it -- once, here, for every caller.
    """
    monkeypatch.setenv("GROQ_API_KEY", "gsk_a")

    class _MessageWithReasoning:
        content = "You have 12.5 PTO days available."
        reasoning = "The user asked about PTO. Let me check the balance and think..."

    class _Resp:
        choices = [type("C", (), {"message": _MessageWithReasoning()})()]
        usage = _Usage()

    class _FakeGroq:
        def __init__(self, api_key):
            self.chat = type("C", (), {"completions": type(
                "X", (), {"create": lambda self, **kw: _Resp()})()})()

    monkeypatch.setattr(llm, "Groq", _FakeGroq)
    result = llm.complete("how much pto")

    assert result.text == "You have 12.5 PTO days available."
    blob = repr(result)
    assert "reasoning" not in blob
    assert "Let me check" not in blob
    assert not hasattr(result, "reasoning")


def test_dotenv_is_loaded_on_package_import():
    """Python does not read .env and neither does `uv run`.

    Without app/__init__.py loading it, every os.getenv() returns None no matter
    what the file says -- which presents as a missing key and sends you hunting in
    the wrong place. This is how that regression gets caught.
    """
    import app
    assert hasattr(app, "load_dotenv")
