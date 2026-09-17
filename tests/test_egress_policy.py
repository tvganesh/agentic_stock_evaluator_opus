"""Denied-path suite for the egress gate (build-order gate P1).

"A governance rule you have never tried to violate in a test is a rule you do not have."
Every account, trading and write family is attempted here and must be refused before any
byte leaves the process. Allowed requests are also pinned so the allowlist cannot silently
widen or break.
"""

from __future__ import annotations

import httpx
import pytest

from sealed_window.governance import policy
from sealed_window.governance.audit import AuditLog
from sealed_window.governance.credentials import AnalyticsToken
from sealed_window.governance.egress import EgressGate, EgressHTTPError, authorise, build_url
from sealed_window.governance.errors import EgressDenied, SealViolation
from sealed_window.governance.policy import CapabilityName as C
from sealed_window.governance.seal import SEAL

ISIN = "INE002A01018"
KEY = f"NSE_EQ%7C{ISIN}"

ALLOWED = [
    ("https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz", C.INSTRUMENT_MASTER),
    (f"https://api.upstox.com/v3/historical-candle/{KEY}/days/1/2026-09-11/2024-08-12", C.DAILY_CANDLES),
    (f"https://api.upstox.com/v3/historical-candle/intraday/{KEY}/days/1", C.INTRADAY_DAILY_CANDLE),
    (f"https://api.upstox.com/v2/fundamentals/{ISIN}/key-ratios", C.KEY_RATIOS),
    (f"https://api.upstox.com/v2/fundamentals/{ISIN}/income-statement?time_period=quarterly&type=consolidated",
     C.INCOME_STATEMENT),
    (f"https://api.upstox.com/v2/fundamentals/{ISIN}/balance-sheet?type=consolidated", C.BALANCE_SHEET),
    (f"https://api.upstox.com/v2/news?category=instrument_keys&instrument_keys={KEY}&page_number=1&page_size=100",
     C.NEWS_BY_INSTRUMENT),
]

DENIED = [
    # write verbs, even on allowed paths
    ("POST", f"https://api.upstox.com/v2/fundamentals/{ISIN}/key-ratios"),
    ("PUT", f"https://api.upstox.com/v2/fundamentals/{ISIN}/key-ratios"),
    ("DELETE", f"https://api.upstox.com/v2/fundamentals/{ISIN}/key-ratios"),
    ("get", f"https://api.upstox.com/v2/fundamentals/{ISIN}/key-ratios"),
    # orders and trading
    ("POST", "https://api.upstox.com/v2/order/place"),
    ("GET", "https://api.upstox.com/v2/order/retrieve-all"),
    ("GET", "https://api.upstox.com/v2/order/trades/get-trades-for-day"),
    ("GET", "https://api.upstox.com/v3/order/gtt/order-details"),
    # portfolio, funds, profile, P&L, charges, mutual funds
    ("GET", "https://api.upstox.com/v2/portfolio/long-term-holdings"),
    ("GET", "https://api.upstox.com/v2/portfolio/short-term-positions"),
    ("GET", "https://api.upstox.com/v2/user/get-funds-and-margin?segment=SEC"),
    ("GET", "https://api.upstox.com/v2/user/profile"),
    ("GET", "https://api.upstox.com/v2/trade/profit-loss/data?segment=EQ"),
    ("GET", "https://api.upstox.com/v2/charges/brokerage"),
    ("GET", "https://api.upstox.com/v2/mf/holdings"),
    # market data not consumed by any phase
    ("GET", f"https://api.upstox.com/v2/market-quote/ohlc?instrument_key={KEY}&interval=1d"),
    ("GET", "https://api.upstox.com/v2/option/chain?instrument_key=NSE_INDEX%7CNifty%2050&expiry_date=2026-09-25"),
    (f"GET", f"https://api.upstox.com/v3/historical-candle/{KEY}/minutes/1/2026-09-11/2026-09-01"),
    ("GET", f"https://api.upstox.com/v3/historical-candle/intraday/{KEY}/minutes/1"),
    ("GET", f"https://api.upstox.com/v3/historical-candle/intraday/{KEY}/days/1?extra=1"),
    ("GET", "https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz"),
    # account data through the news path
    ("GET", "https://api.upstox.com/v2/news?category=holdings"),
    ("GET", "https://api.upstox.com/v2/news?category=positions"),
    ("GET", f"https://api.upstox.com/v2/news?category=instrument_keys&instrument_keys={KEY}&category=holdings"),
    ("GET", "https://api.upstox.com/v2/news?category=instrument_keys"),
    # query tampering
    ("GET", f"https://api.upstox.com/v2/fundamentals/{ISIN}/key-ratios?extra=1"),
    ("GET", f"https://api.upstox.com/v2/fundamentals/{ISIN}/income-statement?type=standalone&time_period=yearly"),
    # URL tricks
    ("GET", f"http://api.upstox.com/v2/fundamentals/{ISIN}/key-ratios"),
    ("GET", f"https://api.upstox.com:8443/v2/fundamentals/{ISIN}/key-ratios"),
    ("GET", f"https://user@api.upstox.com/v2/fundamentals/{ISIN}/key-ratios"),
    ("GET", f"https://api.upstox.com.evil.example/v2/fundamentals/{ISIN}/key-ratios"),
    ("GET", f"https://evil.example/v2/fundamentals/{ISIN}/key-ratios"),
    ("GET", f"https://API.UPSTOX.COM/v2/fundamentals/{ISIN}/key-ratios"),
    ("GET", f"https://api.upstox.com/v2/fundamentals/{ISIN}/../../portfolio/long-term-holdings"),
    ("GET", "https://api.upstox.com/v2/fundamentals/%2e%2e/key-ratios"),
    ("GET", f"https://api.upstox.com/v2/fundamentals/{ISIN}%2Fx/key-ratios"),
    ("GET", f"https://api.upstox.com//v2/fundamentals/{ISIN}/key-ratios"),
    ("GET", f"https://api.upstox.com/v2/fundamentals/{ISIN}/key-ratios#fragment"),
]


@pytest.mark.parametrize("url,capability", ALLOWED)
def test_allowlisted_requests_are_authorised(url, capability):
    """Each consumed capability's canonical URL is authorised and maps to the right capability."""
    assert authorise("GET", url).capability.name is capability


@pytest.mark.parametrize("method,url", DENIED)
def test_denied_requests_raise(method, url):
    """Every trading, account, unconsumed or malformed request is refused by the reference check."""
    with pytest.raises(EgressDenied):
        authorise(method, url)


def test_build_url_rejects_path_injection():
    """A parameter value cannot smuggle extra path segments."""
    with pytest.raises(EgressDenied):
        build_url(policy.CAPABILITIES[C.KEY_RATIOS], {"isin": f"{ISIN}/../../portfolio"}, {})


def test_build_url_rejects_account_news_category():
    """The news capability cannot be built with category=holdings."""
    with pytest.raises(EgressDenied):
        build_url(policy.CAPABILITIES[C.NEWS_BY_INSTRUMENT], {}, {"category": "holdings"})


def _gate(handler, audit: AuditLog) -> EgressGate:
    """A gate whose wire is an in-memory mock transport (no real network)."""
    return EgressGate(AnalyticsToken("t" * 40), audit, transport=httpx.MockTransport(handler), sleep=lambda _: None)


def test_gate_sends_token_only_to_api_host():
    """The analytics token goes to api.upstox.com and never to the public assets host."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Record the request and return an empty success payload."""
        seen.append(request)
        return httpx.Response(200, content=b'{"status":"success","data":[]}')

    SEAL.open_data_network()
    audit = AuditLog()
    gate = _gate(handler, audit)
    gate.get(C.KEY_RATIOS, path_params={"isin": ISIN})
    gate.get(C.INSTRUMENT_MASTER)
    assert seen[0].headers["authorization"] == "Bearer " + "t" * 40
    assert "authorization" not in seen[1].headers
    assert [e["event"] for e in audit.entries("egress.")] == ["egress.allow", "egress.allow"]


def test_denied_request_never_reaches_the_wire():
    """A denied capability call raises, is audited, and the transport is never invoked."""
    calls = []
    SEAL.open_data_network()
    audit = AuditLog()
    gate = _gate(lambda r: calls.append(r) or httpx.Response(200), audit)
    with pytest.raises(EgressDenied):
        gate.get(C.NEWS_BY_INSTRUMENT, query_params={"category": "holdings"})
    assert calls == []
    assert audit.entries("egress.deny")


def test_redirects_are_not_followed():
    """A 302 is surfaced as an HTTP error rather than followed to another location."""
    SEAL.open_data_network()
    gate = _gate(lambda r: httpx.Response(302, headers={"location": "https://evil.example/"}), AuditLog())
    with pytest.raises(EgressHTTPError):
        gate.get(C.KEY_RATIOS, path_params={"isin": ISIN})


def test_gate_refuses_when_sealed():
    """After the seal drops, the gate cannot issue requests even with a valid token."""
    SEAL.seal()
    gate = _gate(lambda r: httpx.Response(200, content=b"{}"), AuditLog())
    with pytest.raises(SealViolation):
        gate.get(C.KEY_RATIOS, path_params={"isin": ISIN})


def test_response_size_cap(monkeypatch):
    """Bodies larger than the policy cap are refused mid-stream."""
    monkeypatch.setattr(policy, "MAX_RESPONSE_BYTES", 10)
    SEAL.open_data_network()
    gate = _gate(lambda r: httpx.Response(200, content=b"x" * 100), AuditLog())
    with pytest.raises(EgressDenied):
        gate.get(C.KEY_RATIOS, path_params={"isin": ISIN})


def test_close_drops_token():
    """Closing the gate (end of acquisition) makes the token unusable."""
    SEAL.open_data_network()
    token = AnalyticsToken("t" * 40)
    gate = EgressGate(token, AuditLog(), transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    gate.close()
    assert token.dropped
