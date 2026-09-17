"""Process-role isolation tests, each in a fresh subprocess so import guards start clean.

Proves that the ETL process cannot load an LLM client or the orchestrator, that the analysis
process cannot load the Upstox adapter, credentials or a generic HTTP client, that Upstox
variables are scrubbed from the analysis process, and that the CLI refuses to acquire when a
trading token is present.
"""

from __future__ import annotations

import os
import subprocess
import sys

from conftest import PROJECT_ROOT


def _clean_env(**extra: str) -> dict[str, str]:
    """Environment without any Upstox variables or model endpoint overrides, plus ``extra``."""
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("UPSTOX_") and k != "ANTHROPIC_BASE_URL"}
    env.update(extra)
    return env


def _run(code: str, **env: str) -> subprocess.CompletedProcess:
    """Run Python ``code`` in a subprocess at the project root and capture output."""
    return subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, env=_clean_env(**env),
                          capture_output=True, text=True, timeout=120)


def _import_attempt(role_call: str, module: str) -> str:
    """Code that enters a role then tries to import ``module``, printing the outcome class name."""
    return (
        "from sealed_window.governance import process_roles as r\n"
        f"r.{role_call}\n"
        "try:\n"
        f"    import {module}\n"
        "    print('IMPORTED')\n"
        "except Exception as exc:\n"
        "    print(type(exc).__name__)\n"
    )


def test_acquire_role_cannot_load_models_or_orchestrator():
    """No model can exist in the process that holds the network."""
    for module in ("anthropic", "sealed_window.agents.claim_agents", "sealed_window.orchestrator",
                   "sealed_window.governance.llm_gateway"):
        result = _run(_import_attempt("enter_acquire_role()", module))
        assert result.stdout.strip() == "ProcessRoleViolation", (module, result.stdout, result.stderr)


def test_sealed_role_cannot_load_upstox_adapter_credentials_or_http_client():
    """The analysis process has no path to Upstox code, credentials or a generic HTTP client."""
    for module in ("sealed_window.acquire.upstox_adapter", "sealed_window.governance.egress",
                   "sealed_window.governance.credentials", "httpx"):
        result = _run(_import_attempt("enter_sealed_role()", module))
        assert result.stdout.strip() == "ProcessRoleViolation", (module, result.stdout, result.stderr)


def test_sealed_role_loads_the_analysis_stack():
    """The orchestrator, app and Anthropic SDK all import under the sealed role's guards."""
    code = (
        "from sealed_window.governance import process_roles as r\n"
        "r.enter_sealed_role()\n"
        "import anthropic, sealed_window.orchestrator, sealed_window.app.server\n"
        "print('OK')\n"
    )
    result = _run(code)
    assert result.stdout.strip() == "OK", result.stderr


def test_sealed_role_scrubs_upstox_env_and_blocks_sockets():
    """Upstox variables disappear and any outbound connection raises in the analysis process."""
    code = (
        "import os, socket\n"
        "from sealed_window.governance import process_roles as r\n"
        "print(r.enter_sealed_role())\n"
        "print('UPSTOX_ANALYTICS_TOKEN' in os.environ)\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 443), timeout=1)\n"
        "except Exception as exc:\n"
        "    print(type(exc).__name__)\n"
    )
    result = _run(code, UPSTOX_ANALYTICS_TOKEN="t" * 40)
    assert result.stdout.splitlines() == ["['UPSTOX_ANALYTICS_TOKEN']", "False", "SealViolation"], result.stderr


def test_sealed_role_rejects_redirected_model_endpoint():
    """ANTHROPIC_BASE_URL pointing elsewhere is refused as an exfiltration channel."""
    code = (
        "from sealed_window.governance import process_roles as r\n"
        "try:\n"
        "    r.enter_sealed_role()\n"
        "except Exception as exc:\n"
        "    print(type(exc).__name__)\n"
    )
    assert _run(code, ANTHROPIC_BASE_URL="https://evil.example").stdout.strip() == "CredentialViolation"


def test_cli_acquire_refuses_when_trading_token_present(tmp_path):
    """The real CLI exits with a governance denial before any request if a trading token is set."""
    result = subprocess.run(
        [sys.executable, "-m", "sealed_window", "--snapshots", str(tmp_path), "acquire", "--source", "upstox",
         "--as-of", "2026-09-11"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120,
        env=_clean_env(UPSTOX_ANALYTICS_TOKEN="t" * 40, UPSTOX_ACCESS_TOKEN="trading"),
    )
    assert result.returncode == 3
    assert "CredentialViolation" in result.stderr and "trading" not in result.stderr
