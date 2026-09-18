"""Least-privilege access policy: the single reviewed source of what may be touched.

This module is pure data. It holds no network client and imports nothing that can open
a socket. Every enforcing control -- the egress gate, the credential loader, the socket
seal and the process-role import guards -- reads its rules from here, so the question
"what can this system reach?" is answered by reading one file, and changing the answer
is a pull request against it (ARCHITECTURE_OPUS.md, "Queries: a fixed, code-reviewed set").

Policy summary
--------------
* Credentials: ``UPSTOX_ANALYTICS_TOKEN`` (read-only by issue) is the only Upstox
  credential, and it lives only in the ACQUIRE process. The model provider credential is
  resolved by the Anthropic SDK in the SEALED process. Any other ``UPSTOX_*`` variable
  (OAuth/trading access tokens, API keys, secrets) aborts the ETL.
* Method: ``GET`` only. There is no code path that builds any other verb.
* Hosts: ``api.upstox.com`` (market data; token attached) and ``assets.upstox.com``
  (public instrument master; token NOT attached -- least privilege per request).
* Paths: a deny-by-default allowlist of exact path templates with typed segments and
  typed query parameters. Absence from the allowlist is denial.
* The allowlist is derived from what the pipeline consumes, never from what the vendor
  offers. Option chains, market quotes, IPO data and the WebSocket feed are not listed
  because no phase consumes them.

Why query parameters are part of the policy
-------------------------------------------
Upstox now lets the Analytics token read account APIs (holdings, positions, funds,
orders, P&L) when a static IP is configured, so "it's the analytics token" no longer
implies "market data only". ``GET /v2/news`` also accepts ``category=holdings`` and
``category=positions``, which return news keyed to the account's portfolio -- account
data through a market-data path. The allowlist therefore pins ``category`` to
``instrument_keys``, and every account path family is absent (and tested as denied).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

# --------------------------------------------------------------------------------------
# Transport rules
# --------------------------------------------------------------------------------------

ALLOWED_METHOD = "GET"
ALLOWED_SCHEME = "https"

UPSTOX_API_HOST = "api.upstox.com"
UPSTOX_ASSETS_HOST = "assets.upstox.com"
UPSTOX_HOSTS: frozenset[str] = frozenset({UPSTOX_API_HOST, UPSTOX_ASSETS_HOST})

REQUEST_TIMEOUT_S = 20.0
"""Per-request timeout for the ETL's HTTP client."""

MAX_RESPONSE_BYTES = 32 * 1024 * 1024
"""Hard cap on any single response body; larger bodies are denied mid-stream."""

MIN_REQUEST_INTERVAL_S = 0.2
"""Minimum spacing between Upstox requests (smooths bursts; the rolling limits below do the rest)."""

UPSTOX_RATE_LIMITS: tuple[tuple[float, int], ...] = ((1.0, 45), (60.0, 450), (1800.0, 1900))
"""Client-side rolling limits as (window seconds, max requests), set below Upstox's published
50/second, 500/minute and 2,000/30 minutes so we never risk a suspension."""

MAX_REQUESTS_PER_ACQUISITION = 4_000
"""Upper bound on requests one ETL run may issue; a runaway loop fails closed.
Sized for Nifty 500: 500 x 5 per-instrument reads + news pages + master ~= 2,600."""

# --------------------------------------------------------------------------------------
# Credentials
# --------------------------------------------------------------------------------------

UPSTOX_TOKEN_ENV = "UPSTOX_ANALYTICS_TOKEN"
FORBIDDEN_ENV_PREFIXES: tuple[str, ...] = ("UPSTOX_",)
"""Any variable with these prefixes other than ``UPSTOX_TOKEN_ENV`` is a forbidden credential."""

FORBIDDEN_MODEL_ENV_NAMES: tuple[str, ...] = ("ANTHROPIC_BASE_URL",)
"""Redirecting model traffic to another host would open an exfiltration channel."""

MODEL_PROVIDER_BASE_URL = "https://api.anthropic.com"
MODEL_PROVIDER_HOSTS: frozenset[str] = frozenset({"api.anthropic.com"})
"""The only hosts reachable from the sealed process, and only inside a model window."""

LOCAL_MODEL_BASE_URL = "http://127.0.0.1:11434/v1"
LOCAL_MODEL_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})
"""Loopback only, for a model served on this machine (Ollama and similar).

A deliberate, narrow entry rather than a general opening: a local model window reaches the loopback
interface and nothing else, so an injected instruction still has nowhere off the machine to send to.
Serving a model from another host would be a further reviewed change, not covered by this."""


def model_hosts_for(mode: str) -> frozenset[str]:
    """Hosts a model window may reach for ``mode`` ('anthropic' or 'local'); unknown modes get none."""
    return {"anthropic": MODEL_PROVIDER_HOSTS, "local": LOCAL_MODEL_HOSTS}.get(mode, frozenset())

# --------------------------------------------------------------------------------------
# Process-role import boundaries
# --------------------------------------------------------------------------------------

ACQUIRE_ROLE_FORBIDDEN_MODULES: tuple[str, ...] = (
    "anthropic",
    "sealed_window.agents",
    "sealed_window.governance.llm_gateway",
    "sealed_window.governance.local_client",
    "sealed_window.orchestrator",
    "sealed_window.app",
)
"""Modules the ETL process may never import: no model can exist while the network is live."""

SEALED_ROLE_FORBIDDEN_MODULES: tuple[str, ...] = (
    "sealed_window.acquire",
    "sealed_window.governance.egress",
    "sealed_window.governance.credentials",
    "httpx",
)
"""Modules the analysis process may never import: it has no Upstox adapter and no generic HTTP client."""

# --------------------------------------------------------------------------------------
# Capability allowlist
# --------------------------------------------------------------------------------------

ISIN_RE = r"[A-Z]{2}[A-Z0-9]{9}[0-9]"
NSE_EQ_KEY_RE = rf"NSE_EQ\|{ISIN_RE}"
DATE_RE = r"\d{4}-\d{2}-\d{2}"
NEWS_KEYS_RE = rf"{NSE_EQ_KEY_RE}(?:,{NSE_EQ_KEY_RE}){{0,29}}"
PAGE_RE = r"(?:100|[1-9][0-9]?)"


class CapabilityName(str, Enum):
    """Names of the only Upstox reads the ETL may perform; callers ask by name, never by URL."""

    INSTRUMENT_MASTER = "instrument_master"
    DAILY_CANDLES = "daily_candles"
    INTRADAY_DAILY_CANDLE = "intraday_daily_candle"
    KEY_RATIOS = "key_ratios"
    INCOME_STATEMENT = "income_statement"
    BALANCE_SHEET = "balance_sheet"
    NEWS_BY_INSTRUMENT = "news_by_instrument"


@dataclass(frozen=True)
class ParamRule:
    """A typed rule for one path segment or query parameter: a full-match regex, optionally required."""

    pattern: str
    required: bool = True

    def matches(self, value: str) -> bool:
        """Return True if ``value`` fully matches this rule's pattern (partial matches are denied)."""
        return re.fullmatch(self.pattern, value) is not None


@dataclass(frozen=True)
class Capability:
    """One allowlisted Upstox read: host, exact path template, typed params and token rule.

    ``consumed_by`` records which pipeline stage needs the capability; an entry nobody
    consumes should be deleted, not kept "in case".
    """

    name: CapabilityName
    host: str
    path_template: str
    path_params: Mapping[str, ParamRule]
    query_params: Mapping[str, ParamRule]
    attach_token: bool
    consumed_by: str


def _frozen(mapping: dict) -> Mapping:
    """Wrap a dict in a read-only proxy so policy tables cannot be mutated at runtime."""
    return MappingProxyType(mapping)


CAPABILITIES: Mapping[CapabilityName, Capability] = _frozen(
    {
        CapabilityName.INSTRUMENT_MASTER: Capability(
            name=CapabilityName.INSTRUMENT_MASTER,
            host=UPSTOX_ASSETS_HOST,
            path_template="/market-quote/instruments/exchange/NSE.json.gz",
            path_params=_frozen({}),
            query_params=_frozen({}),
            attach_token=False,
            consumed_by="acquire: resolve universe symbols to instrument keys and ISINs",
        ),
        CapabilityName.DAILY_CANDLES: Capability(
            name=CapabilityName.DAILY_CANDLES,
            host=UPSTOX_API_HOST,
            # unit and interval are literals: only daily candles are permitted.
            path_template="/v3/historical-candle/{instrument_key}/days/1/{to_date}/{from_date}",
            path_params=_frozen(
                {
                    "instrument_key": ParamRule(NSE_EQ_KEY_RE),
                    "to_date": ParamRule(DATE_RE),
                    "from_date": ParamRule(DATE_RE),
                }
            ),
            query_params=_frozen({}),
            attach_token=True,
            consumed_by="acquire: technical indicators (RSI, MACD, ATR, MAs, returns)",
        ),
        CapabilityName.INTRADAY_DAILY_CANDLE: Capability(
            name=CapabilityName.INTRADAY_DAILY_CANDLE,
            host=UPSTOX_API_HOST,
            # unit and interval are literals: one daily candle for the current session, never minute data.
            path_template="/v3/historical-candle/intraday/{instrument_key}/days/1",
            path_params=_frozen({"instrument_key": ParamRule(NSE_EQ_KEY_RE)}),
            query_params=_frozen({}),
            attach_token=True,
            consumed_by="acquire: today's session candle when as_of is today "
                        "(the historical candle API excludes the current day)",
        ),
        CapabilityName.KEY_RATIOS: Capability(
            name=CapabilityName.KEY_RATIOS,
            host=UPSTOX_API_HOST,
            path_template="/v2/fundamentals/{isin}/key-ratios",
            path_params=_frozen({"isin": ParamRule(ISIN_RE)}),
            query_params=_frozen({}),
            attach_token=True,
            consumed_by="acquire: P/E, P/B, ROE, ROA, ROCE, EV/EBITDA vs sector",
        ),
        CapabilityName.INCOME_STATEMENT: Capability(
            name=CapabilityName.INCOME_STATEMENT,
            host=UPSTOX_API_HOST,
            path_template="/v2/fundamentals/{isin}/income-statement",
            path_params=_frozen({"isin": ParamRule(ISIN_RE)}),
            query_params=_frozen(
                {
                    "type": ParamRule(r"consolidated"),
                    "time_period": ParamRule(r"yearly|quarterly"),
                }
            ),
            attach_token=True,
            consumed_by="acquire: revenue/profit growth and operating margin",
        ),
        CapabilityName.BALANCE_SHEET: Capability(
            name=CapabilityName.BALANCE_SHEET,
            host=UPSTOX_API_HOST,
            path_template="/v2/fundamentals/{isin}/balance-sheet",
            path_params=_frozen({"isin": ParamRule(ISIN_RE)}),
            query_params=_frozen({"type": ParamRule(r"consolidated")}),
            attach_token=True,
            consumed_by="acquire: liabilities-to-equity leverage",
        ),
        CapabilityName.NEWS_BY_INSTRUMENT: Capability(
            name=CapabilityName.NEWS_BY_INSTRUMENT,
            host=UPSTOX_API_HOST,
            path_template="/v2/news",
            path_params=_frozen({}),
            query_params=_frozen(
                {
                    # Pinned: 'holdings' and 'positions' would leak account data.
                    "category": ParamRule(r"instrument_keys"),
                    "instrument_keys": ParamRule(NEWS_KEYS_RE),
                    "page_number": ParamRule(PAGE_RE, required=False),
                    "page_size": ParamRule(PAGE_RE, required=False),
                }
            ),
            attach_token=True,
            consumed_by="acquire: 30-day headline window per instrument",
        ),
    }
)
"""The complete allowlist. Anything not described here is denied."""
