"""Tests for the fixes prompted by the first live Nifty 500 acquisition (15 Sep 2026).

Covers: bank ratios and removal of quarterly year-on-year columns (derived-v2), bank-aware
screening and ordering (screen-v2), per-page news pagination audit, rate-limit seeding from
earlier runs, the session-close guard, and discarding checkpoints saved before the close.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import date, datetime, timedelta, timezone

import httpx

from conftest import AS_OF, PROJECT_ROOT
from sealed_window.snapshot import indicators as ind
from sealed_window.acquire.checkpoint import CheckpointedSource
from sealed_window.acquire.etl import session_is_final
from sealed_window.acquire.fixture_source import SyntheticMarket, synthetic_isin
from sealed_window.acquire.upstox_adapter import UpstoxAdapter
from sealed_window.governance.audit import AuditLog
from sealed_window.governance.credentials import AnalyticsToken
from sealed_window.governance.egress import EgressGate
from sealed_window.governance.ratelimit import RollingRateLimiter, recent_request_ages
from sealed_window.governance.seal import SEAL
from sealed_window.screen.config import ScreenConfig
from sealed_window.screen.screen import evaluate_row, return_on_capital
from sealed_window.snapshot.columns import COLUMNS

IST = timezone(timedelta(hours=5, minutes=30))

HDFCBANK_RATIOS = [  # as returned by Upstox on 14 Sep 2026
    {"company_value": "13.23", "name": "P/E", "sector_value": "3.64"},
    {"company_value": "1.81", "name": "P/B", "sector_value": "1.54"},
    {"company_value": "3.28%", "name": "NIM", "sector_value": "4.18%"},
    {"company_value": "1.94%", "name": "ROA", "sector_value": "0.92%"},
    {"company_value": "13.61%", "name": "ROE", "sector_value": "8.84%"},
    {"company_value": "0.38%", "name": "Net NPA", "sector_value": "0.79%"},
    {"company_value": "34.0", "name": "CASA", "sector_value": "30.96"},
]
FRESH = {"last_candle_age_days": 3.0, "fundamentals_age_days": 76.0}


def test_bank_ratios_are_parsed_and_quarterly_yoy_is_gone():
    """Bank ratios populate their columns and flag the bank; quarterly YoY columns no longer exist."""
    bank = ind.fundamental_row(HDFCBANK_RATIOS, None, None, None, date(2026, 9, 14))
    assert (bank["is_bank"], bank["nim_pct"], bank["net_npa_pct"], bank["casa_pct"]) == (1.0, 3.28, 0.38, 34.0)
    assert bank["roce_pct"] is None and bank["sector_net_npa_pct"] == 0.79
    corporate = ind.fundamental_row([{"name": "ROCE", "company_value": "10.39%", "sector_value": "16.9%"}],
                                    None, None, None, date(2026, 9, 14))
    assert corporate["is_bank"] == 0.0
    assert ind.fundamental_row(None, None, None, None, date(2026, 9, 14))["is_bank"] is None
    assert not {"revenue_growth_q_yoy_pct", "net_profit_growth_q_yoy_pct"} & set(COLUMNS)


def test_banks_are_screened_on_bank_metrics():
    """ROCE and leverage filters skip banks; the NPA ceiling applies only to banks; unknowns fail closed."""
    config = ScreenConfig(roce_min_pct=12, liabilities_to_equity_max=2, net_npa_max_pct=1.0)
    bank = {**FRESH, "is_bank": 1.0, "roce_pct": None, "liabilities_to_equity": 9.5, "net_npa_pct": 0.38}
    assert evaluate_row(bank, config) == []
    assert "missing net_npa_pct" in evaluate_row({**bank, "net_npa_pct": None}, config)
    assert any("net NPA above ceiling" in r for r in evaluate_row({**bank, "net_npa_pct": 2.5}, config))
    corporate = {**FRESH, "is_bank": 0.0, "roce_pct": 15.0, "liabilities_to_equity": 1.0}
    assert evaluate_row(corporate, config) == []
    unknown = {**FRESH, "roce_pct": None, "liabilities_to_equity": 1.0}
    assert "missing roce_pct" in evaluate_row(unknown, config)


def test_candidate_ordering_falls_back_to_roe_for_banks():
    """Banks are ordered by ROE rather than sorting last for lack of ROCE."""
    assert return_on_capital({"roce_pct": None, "roe_pct": 13.61}) == 13.61
    assert return_on_capital({"roce_pct": 20.0, "roe_pct": 13.61}) == 20.0


def test_screen_config_hash_changed_with_semantics():
    """screen-v2 hashes differ from screen-v1 for identical sliders (hash recorded on 15 Sep 2026)."""
    v1_default_hash = "112b348c7e2776b4b09b35d0ee69cf28b360869de4f3db77d7dc33e8a0cece41"
    config = ScreenConfig(roe_min_pct=12, roce_min_pct=12, revenue_growth_min_pct=0, pe_max=80,
                          rsi_min=30, rsi_max=80, atr_pct_max=5, max_candidates=20)
    assert config.config_hash() != v1_default_hash


def test_every_news_page_is_audited():
    """Each news page records article counts and the vendor's pagination metadata."""
    SEAL.open_data_network()
    audit = AuditLog()
    key = "NSE_EQ|INE002A01018"
    body = {"status": "success", "data": {key: [{"heading": "h", "published_time": 1}]},
            "metadata": {"page": {"page_number": 1, "total_pages": 4}}}
    gate = EgressGate(AnalyticsToken("t" * 40), audit, sleep=lambda _: None,
                      transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body)))
    assert len(UpstoxAdapter(gate, audit=audit).news([key])[key]) == 1
    page = audit.entries("news.page")[0]["detail"]
    assert page["articles"] == 1 and page["instruments_with_articles"] == 1 and "total_pages" in page["metadata"]


def test_rate_limiter_counts_requests_from_earlier_runs(tmp_path):
    """Requests recorded by earlier runs' audit logs fill the new run's rate windows."""
    earlier = AuditLog(tmp_path / "earlier.jsonl")
    for _ in range(3):
        earlier.record("egress.allow", {"capability": "key_ratios"})
    earlier.record("egress.deny", {"reason": "not a request"})
    assert len(recent_request_ages(tmp_path)) == 3
    assert recent_request_ages(tmp_path, now=datetime.now(timezone.utc) + timedelta(hours=1)) == []

    clock = {"now": 1000.0}
    limiter = RollingRateLimiter([(60.0, 3)], clock=lambda: clock["now"],
                                 sleep=lambda s: clock.__setitem__("now", clock["now"] + s))
    assert limiter.seed([10.0, 20.0, 5000.0]) == 2  # the 5,000-second-old request is outside every window
    assert limiter.acquire() == 0.0  # third slot of the minute
    assert limiter.acquire() == 40.0  # waits until the request made 20 s ago leaves the window


def test_session_close_guard():
    """A date's data is final only from 16:00 IST that day; future dates are never final."""
    assert not session_is_final(date(2026, 9, 15), datetime(2026, 9, 15, 14, 15, tzinfo=IST))
    assert session_is_final(date(2026, 9, 15), datetime(2026, 9, 15, 16, 1, tzinfo=IST))
    assert session_is_final(date(2026, 9, 14), datetime(2026, 9, 15, 9, 0, tzinfo=IST))
    assert not session_is_final(date(2026, 9, 16), datetime(2026, 9, 15, 17, 0, tzinfo=IST))


def test_checkpoints_saved_before_the_close_are_discarded(tmp_path):
    """A checkpoint created before not_before is wiped; a newer one is reused."""
    directory = tmp_path / "ckpt"
    CheckpointedSource(SyntheticMarket(AS_OF), directory).key_ratios(synthetic_isin(1))
    assert list(directory.glob("*.json"))
    kept = CheckpointedSource(SyntheticMarket(AS_OF), directory,
                              not_before=datetime.now(timezone.utc) - timedelta(days=1))
    assert not kept.discarded_stale and list(directory.glob("*.json"))
    wiped = CheckpointedSource(SyntheticMarket(AS_OF), directory,
                               not_before=datetime.now(timezone.utc) + timedelta(hours=1))
    assert wiped.discarded_stale and not list(directory.glob("*.json"))


def test_cli_refuses_an_unfinished_session(tmp_path):
    """The acquire command exits 2 for a date whose session is not final, before touching credentials."""
    result = subprocess.run(
        [sys.executable, "-m", "sealed_window", "--snapshots", str(tmp_path / "snapshots"),
         "acquire", "--source", "upstox", "--as-of", "2099-01-01"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 2 and "not final" in result.stderr
