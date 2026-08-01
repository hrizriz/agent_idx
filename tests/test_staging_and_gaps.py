from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from agent_idx import browser_stockbit, fund_farm, stockbit_reports
from agent_idx.ma_squeeze_study import _add_ma_features


def test_stream_scrape_is_staged_without_execution() -> None:
    key = 8123
    stockbit_reports.clear_pending_stream_scrape(key)

    result = stockbit_reports.stage_stream_scrape(
        key,
        {"source": "both", "days": 3},
    )

    assert result.startswith("STAGED")
    assert stockbit_reports.has_pending_stream_scrape(key)
    assert stockbit_reports.peek_pending_stream_scrape(key) == {
        "days": 3,
        "date_from": None,
        "date_to": None,
        "url": None,
        "source": "both",
    }
    assert stockbit_reports.clear_pending_stream_scrape(key)


def test_stream_scrape_rejects_unknown_source() -> None:
    key = 8124
    stockbit_reports.clear_pending_stream_scrape(key)

    result = stockbit_reports.stage_stream_scrape(key, {"source": "unknown"})

    assert result.startswith("ERROR:")
    assert not stockbit_reports.has_pending_stream_scrape(key)


def test_missing_foreign_values_remain_missing() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["BBCA"] * 2,
            "close": [100.0, 101.0],
            "volume": [10.0, 12.0],
            "foreign_buy": [None, 15.0],
            "foreign_sell": [None, 10.0],
        }
    )

    result = _add_ma_features(frame)

    assert pd.isna(result.loc[0, "nf"])
    assert pd.isna(result.loc[0, "nf_pos"])
    assert result.loc[1, "nf"] == 5.0
    assert bool(result.loc[1, "nf_pos"])


def test_closed_browser_is_marked_dead_without_cleanup(monkeypatch) -> None:
    class FakeBrowser:
        marked = False
        closed = False

        def _mark_dead(self) -> None:
            self.marked = True

        def close(self) -> None:
            self.closed = True

    fake = FakeBrowser()
    monkeypatch.setattr(browser_stockbit, "_BROWSER", fake)

    def closed_target() -> None:
        raise RuntimeError("Target page, context or browser has been closed")

    with pytest.raises(RuntimeError, match="BROWSER_CLOSED"):
        browser_stockbit.run_sync(closed_target)

    assert fake.marked
    assert not fake.closed


def test_fund_farm_skips_recent_complete_sidecar(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(fund_farm, "FUND_DIR", tmp_path)
    (tmp_path / "BBCA.json").write_text(
        json.dumps(
            {
                "symbol": "BBCA",
                "scraped_at": "2026-07-31T00:30:00+07:00",
                "sections": ["overview", "keystats", "financials", "profile"],
            }
        ),
        encoding="utf-8",
    )
    now = datetime(2026, 7, 31, 8, 0, tzinfo=ZoneInfo("Asia/Jakarta"))

    stale, fresh = fund_farm.filter_fresh_symbols(
        ["BBCA", "BMRI"],
        ["keystats", "financials", "profile"],
        fresh_days=7,
        now=now,
    )

    assert stale == ["BMRI"]
    assert fresh == ["BBCA"]


def test_fund_farm_does_not_skip_incomplete_sections(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(fund_farm, "FUND_DIR", tmp_path)
    (tmp_path / "BBCA.json").write_text(
        json.dumps(
            {
                "symbol": "BBCA",
                "scraped_at": "2026-07-31T00:30:00+07:00",
                "sections": ["keystats", "profile"],
            }
        ),
        encoding="utf-8",
    )
    now = datetime(2026, 7, 31, 8, 0, tzinfo=ZoneInfo("Asia/Jakarta"))

    stale, fresh = fund_farm.filter_fresh_symbols(
        ["BBCA"],
        ["keystats", "financials", "profile"],
        fresh_days=7,
        now=now,
    )

    assert stale == ["BBCA"]
    assert fresh == []


def test_fund_farm_rescrapes_at_seven_calendar_days(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(fund_farm, "FUND_DIR", tmp_path)
    (tmp_path / "BBCA.json").write_text(
        json.dumps(
            {
                "symbol": "BBCA",
                "scraped_at": "2026-07-24T09:00:00+07:00",
                "sections": ["keystats", "financials", "profile"],
            }
        ),
        encoding="utf-8",
    )
    now = datetime(2026, 7, 31, 7, 0, tzinfo=ZoneInfo("Asia/Jakarta"))

    stale, fresh = fund_farm.filter_fresh_symbols(
        ["BBCA"],
        ["keystats", "financials", "profile"],
        fresh_days=7,
        now=now,
    )

    assert stale == ["BBCA"]
    assert fresh == []
