"""Tests for the locally served model client and the plan that pays for it.

Covers the two things that make a local run safe rather than merely cheap: the client can only
address the loopback interface and declares that to the seal, and a local plan commits $0.00
while naming the model it will actually call, so the operator's approval binds to the models as
well as to the money. Also covers the failure mapping -- a local server that errors, truncates
or emits output the schema rejects must cost the run one slot's claims, never the whole run.

No network and no model server are required: the transport is stubbed at ``urlopen``.
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from sealed_window.claims.schema import ClaimBatch
from sealed_window.governance import policy
from sealed_window.governance.errors import SealViolation
from sealed_window.governance.llm_gateway import ModelRequest
from sealed_window.governance.local_client import (
    MIN_EVALUATED_PROMPT_RATIO,
    REASONING_EFFORT,
    LocalOpenAIModelClient,
)
from sealed_window.governance.spend import (
    DEFAULT_LOCAL_MODEL,
    SlotClass,
    compile_plan,
    price_of,
    specs_for_mode,
)

SNAP, CFG = "a" * 64, "b" * 64


def _request(max_tokens: int = 1_000) -> ModelRequest:
    """A claim request of the shape the gateway builds for a local slot (no thinking, no effort)."""
    return ModelRequest(model=DEFAULT_LOCAL_MODEL, max_tokens=max_tokens, system="s", user="u",
                        output_type=ClaimBatch, thinking=None, effort=None)


class _FakeResponse:
    """Context-manager stand-in for ``urlopen``'s response object."""

    def __init__(self, body: str) -> None:
        """Hold the body the fake server returns."""
        self._body = body

    def __enter__(self) -> "_FakeResponse":
        """Enter the context, as ``urlopen`` does."""
        return self

    def __exit__(self, *exc: object) -> bool:
        """Leave the context without suppressing anything."""
        return False

    def read(self) -> bytes:
        """Return the body as bytes, matching the real response."""
        return self._body.encode("utf-8")


def _serve(monkeypatch, body: str | Exception, captured: list | None = None):
    """Point the client's ``urlopen`` at a canned body, or make it raise ``body``."""

    def fake_urlopen(request, timeout=None):
        """Record the outgoing request and return the canned response."""
        if captured is not None:
            captured.append(json.loads(request.data.decode("utf-8")))
        if isinstance(body, Exception):
            raise body
        return _FakeResponse(body)

    monkeypatch.setattr("sealed_window.governance.local_client.urllib.request.urlopen", fake_urlopen)


def _completion(content: str, finish: str = "stop", prompt_tokens: int = 120, completion_tokens: int = 30) -> str:
    """Build an OpenAI-shaped chat completion response body."""
    return json.dumps({
        "choices": [{"message": {"content": content}, "finish_reason": finish}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    })


# ---- the seal ------------------------------------------------------------------------------


def test_client_declares_loopback_and_refuses_any_other_host():
    """The client opens only the loopback set, and a non-loopback base URL fails closed."""
    assert LocalOpenAIModelClient().permitted_hosts == policy.LOCAL_MODEL_HOSTS
    assert "api.anthropic.com" not in LocalOpenAIModelClient().permitted_hosts
    for hostile in ("https://evil.example/v1", "http://10.0.0.5:11434/v1", "https://api.anthropic.com/v1"):
        with pytest.raises(SealViolation):
            LocalOpenAIModelClient(base_url=hostile)


def test_gateway_would_open_the_loopback_window_not_anthropic():
    """The gateway reads hosts off the client, so a local call never opens the provider endpoint."""
    hosts = getattr(LocalOpenAIModelClient(), "permitted_hosts", None)
    assert hosts is not None and hosts & policy.LOCAL_MODEL_HOSTS and not (hosts & policy.MODEL_PROVIDER_HOSTS)


# ---- the request ---------------------------------------------------------------------------


def test_request_constrains_output_to_the_schema(monkeypatch):
    """Output is constrained by json_schema, not merely requested as JSON."""
    sent: list = []
    _serve(monkeypatch, _completion('{"claims": []}'), captured=sent)
    LocalOpenAIModelClient().complete(_request())
    body = sent[0]
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["name"] == "ClaimBatch"
    assert body["response_format"]["json_schema"]["schema"] == ClaimBatch.model_json_schema()
    assert body["max_tokens"] == 1_000 and body["model"] == DEFAULT_LOCAL_MODEL


def test_reasoning_preamble_is_suppressed(monkeypatch):
    """Every request asks for no reasoning preamble; without it qwen3 truncates its own JSON.

    Pinned deliberately: dropping this field is silent and expensive. Reasoning tokens count
    against ``max_tokens``, so a long preamble costs the slot its claims and can turn a short
    run into hours. Only ``reasoning_effort`` works on this route -- ``enable_thinking``,
    ``/no_think`` and ``think: false`` were all measured against Ollama 0.34.0 and ignored.
    """
    sent: list = []
    _serve(monkeypatch, _completion('{"claims": []}'), captured=sent)
    LocalOpenAIModelClient().complete(_request())
    assert sent[0]["reasoning_effort"] == REASONING_EFFORT == "none"


def test_input_tokens_are_estimated_without_calling_the_server():
    """Counting never dispatches: a real count would need a full prefill costing minutes."""
    client = LocalOpenAIModelClient()
    small, large = _request(), _request()
    assert client.count_input_tokens(small) > 0
    long_request = ModelRequest(model=DEFAULT_LOCAL_MODEL, max_tokens=10, system="s" * 350, user="u" * 350,
                                output_type=ClaimBatch, thinking=None, effort=None)
    assert client.count_input_tokens(long_request) == 200
    assert client.count_input_tokens(large) == client.count_input_tokens(small)


# ---- failure mapping -----------------------------------------------------------------------


def test_truncated_prompt_is_refused_not_answered(monkeypatch):
    """A server that evaluated far fewer tokens than we sent cut the prompt; the answer is refused.

    Regression for the qwen3:8b runs of 17 Sep 2026: the default context evaluated 2,050 tokens of a
    6,290-token veto prompt, so the auditor never saw the claims it was asked to audit and returned an
    empty refutation list that read as a considered finding. An answer about evidence the model never
    received is worse than no answer, which is why ``PromptOverBudget`` refuses the mirror case.
    """
    request = _request()
    estimated = LocalOpenAIModelClient().count_input_tokens(request)
    truncated = int(estimated * MIN_EVALUATED_PROMPT_RATIO) - 1
    _serve(monkeypatch, _completion('{"claims": []}', prompt_tokens=truncated))
    result = LocalOpenAIModelClient().complete(request)
    assert result.parsed is None and result.stop_reason == "context_truncated"
    assert str(truncated) in (result.detail or "")


def test_full_prompt_evaluation_is_accepted(monkeypatch):
    """A server that evaluated about what we sent is not flagged; the check targets real truncation."""
    request = _request()
    estimated = LocalOpenAIModelClient().count_input_tokens(request)
    _serve(monkeypatch, _completion('{"claims": []}', prompt_tokens=estimated))
    assert LocalOpenAIModelClient().complete(request).parsed is not None
    # Denser tokenization legitimately reports above the estimate, and must never be flagged.
    _serve(monkeypatch, _completion('{"claims": []}', prompt_tokens=estimated * 2))
    assert LocalOpenAIModelClient().complete(request).parsed is not None


def test_valid_output_is_parsed_and_usage_reported(monkeypatch):
    """A well-formed response yields a parsed batch and the server's own token counts."""
    _serve(monkeypatch, _completion('{"claims": []}', prompt_tokens=222, completion_tokens=33))
    result = LocalOpenAIModelClient().complete(_request())
    assert isinstance(result.parsed, ClaimBatch)
    assert (result.stop_reason, result.input_tokens, result.output_tokens) == ("end_turn", 222, 33)


@pytest.mark.parametrize(
    "body, expected",
    [
        (_completion("not json at all"), "schema_error"),
        (_completion('{"wrong_field": 1}'), "schema_error"),
        (_completion('{"claims": []}', finish="length"), "max_tokens"),
        ('{"choices": []}', "malformed_response"),
        ("this is not json", "malformed_response"),
    ],
)
def test_bad_responses_cost_one_slot_not_the_run(monkeypatch, body, expected):
    """Every malformed or truncated response returns parsed=None with a reason, and never raises."""
    _serve(monkeypatch, body)
    result = LocalOpenAIModelClient().complete(_request())
    assert result.parsed is None and result.stop_reason == expected


@pytest.mark.parametrize(
    "error, expected",
    [
        (urllib.error.HTTPError("http://127.0.0.1:11434", 500, "boom", {}, None), "api_error"),
        (urllib.error.URLError("server not running"), "connection_error"),
        (TimeoutError("too slow"), "connection_error"),
    ],
)
def test_transport_failures_are_returned_not_raised(monkeypatch, error, expected):
    """A server that is down or slow ends one call, not the run; nothing propagates to the gateway."""
    _serve(monkeypatch, error)
    result = LocalOpenAIModelClient().complete(_request())
    assert result.parsed is None and result.stop_reason == expected
    assert result.usage_estimated is False, "a call that never reached the model is not charged"


# ---- the plan ------------------------------------------------------------------------------


def test_local_plan_commits_nothing_and_names_the_model_it_will_call():
    """A local plan is $0.00 against the model actually served, not Anthropic prices."""
    specs = specs_for_mode("local", "llama3.2:latest")
    assert {spec.model for spec in specs} == {"llama3.2:latest"}
    assert all(spec.effort is None for spec in specs), "no adaptive thinking on a local model"
    plan = compile_plan(snapshot_hash=SNAP, screen_config_hash=CFG, candidate_count=10, specs=specs)
    assert plan.committed_total_microusd == 0 and plan.hard_stop_microusd == 0
    assert price_of("llama3.2:latest") == (0, 0)


def test_mode_is_part_of_what_the_operator_approves():
    """Plan hashes differ by mode, so an Anthropic approval cannot start a local run or the reverse."""
    def plan_for(mode: str):
        """Compile a plan for one mode over the same snapshot and screen."""
        return compile_plan(snapshot_hash=SNAP, screen_config_hash=CFG, candidate_count=10,
                            specs=specs_for_mode(mode))

    anthropic_plan, local_plan = plan_for("anthropic"), plan_for("local")
    assert anthropic_plan.plan_hash != local_plan.plan_hash
    assert anthropic_plan.committed_total_microusd > 0 and local_plan.committed_total_microusd == 0
    assert local_plan.row(SlotClass.VETO).model == DEFAULT_LOCAL_MODEL
    # Same shape either way: a local run is not a smaller run, only a free one.
    assert [r.calls for r in local_plan.rows] == [r.calls for r in anthropic_plan.rows]
