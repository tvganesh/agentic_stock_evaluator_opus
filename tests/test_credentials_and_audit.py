"""Tests for least-privilege credential handling and the hash-chained audit log."""

from __future__ import annotations

import json
import pickle

import pytest

from sealed_window.governance.audit import AuditLog
from sealed_window.governance.credentials import (
    AnalyticsToken,
    forbidden_credential_names,
    load_analytics_token,
)
from sealed_window.governance.errors import AuditChainBroken, CredentialViolation

TOKEN = "analytics-token-value-0123456789"


def test_trading_token_presence_blocks_the_etl():
    """A trading/OAuth token next to the analytics token aborts, naming the variable but not its value."""
    env = {"UPSTOX_ANALYTICS_TOKEN": TOKEN, "UPSTOX_ACCESS_TOKEN": "secret-trading-token"}
    assert forbidden_credential_names(env) == ["UPSTOX_ACCESS_TOKEN"]
    with pytest.raises(CredentialViolation) as info:
        load_analytics_token(env)
    assert "UPSTOX_ACCESS_TOKEN" in str(info.value)
    assert "secret-trading-token" not in str(info.value)


def test_token_is_scrubbed_redacted_and_unpicklable():
    """Loading removes the token from the environment; the holder never reveals or serialises it."""
    env = {"UPSTOX_ANALYTICS_TOKEN": TOKEN}
    token = load_analytics_token(env)
    assert "UPSTOX_ANALYTICS_TOKEN" not in env
    assert TOKEN not in repr(token) and TOKEN not in str(token)
    with pytest.raises(TypeError):
        pickle.dumps(token)


def test_missing_token_and_dropped_token_fail_closed():
    """No token means no acquisition; a dropped token cannot produce a header."""
    with pytest.raises(CredentialViolation):
        load_analytics_token({})
    token = AnalyticsToken(TOKEN)
    token.drop()
    with pytest.raises(CredentialViolation):
        token.bearer_header()


def test_audit_chain_verifies_and_resumes(tmp_path):
    """A written log verifies, and reopening continues the same chain."""
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.record("a", {"x": 1})
    log.record("b", {"input_tokens": 5})
    AuditLog(path).record("c")
    assert AuditLog.verify(path) == 3


def test_audit_tampering_is_detected(tmp_path):
    """Editing any recorded entry breaks the hash chain."""
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.record("egress.deny", {"reason": "portfolio"})
    log.record("run.start")
    lines = path.read_text().splitlines()
    entry = json.loads(lines[0])
    entry["event"] = "egress.allow"
    lines[0] = json.dumps(entry)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(AuditChainBroken):
        AuditLog.verify(path)


def test_audit_refuses_secret_looking_keys():
    """The log refuses detail keys that look like credentials."""
    with pytest.raises(ValueError):
        AuditLog().record("oops", {"authorization": "Bearer x"})
