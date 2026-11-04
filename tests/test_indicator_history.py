"""An indicator without enough history must read N/A, not a partial-window value.

stockstats rolls with ``min_periods=1``, so a "200-day" average over a recent
listing's 100 sessions was the mean of those 100 closes, and the agents cited it
as a long-term trend.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents.dataflows.vendors.yahoo import market, snapshot

AS_OF = "2026-11-03"


def _frame(rows: int) -> pd.DataFrame:
    dates = pd.bdate_range(end=AS_OF, periods=rows)
    close = 100 + np.sin(np.arange(rows) / 5) * 10 + np.arange(rows) * 0.1
    return pd.DataFrame({
        "Date": dates, "Open": close, "High": close + 1, "Low": close - 1,
        "Close": close, "Volume": 1_000_000.0,
    })


@pytest.fixture
def ohlcv(monkeypatch):
    def use(rows: int) -> pd.DataFrame:
        frame = _frame(rows)
        loader = lambda *args, **kwargs: frame.copy()  # noqa: E731
        monkeypatch.setattr(market, "load_ohlcv", loader)
        monkeypatch.setattr(snapshot, "load_ohlcv", loader)
        return frame
    return use


def _snapshot_value(text: str, indicator: str) -> str:
    return next(line for line in text.splitlines()
                if line.startswith(f"| {indicator} |")).split("|")[2].strip()


@pytest.mark.unit
def test_recent_listing_has_no_200_day_average(ohlcv):
    ohlcv(100)
    assert market.get_stockstats_indicator("SPCX", "close_200_sma", AS_OF) == "N/A"
    assert set(market._get_stock_stats_bulk("SPCX", "close_200_sma", AS_OF).values()) == {"N/A"}
    assert _snapshot_value(snapshot.build_verified_market_snapshot("SPCX", AS_OF),
                           "close_200_sma") == "N/A"


@pytest.mark.unit
def test_shorter_windows_still_compute_on_a_recent_listing(ohlcv):
    frame = ohlcv(100)
    value = float(market.get_stockstats_indicator("SPCX", "close_50_sma", AS_OF))
    assert value == pytest.approx(frame["Close"].tail(50).mean())


@pytest.mark.unit
def test_a_full_window_reports_the_true_average(ohlcv):
    frame = ohlcv(200)
    value = float(market.get_stockstats_indicator("SPCX", "close_200_sma", AS_OF))
    assert value == pytest.approx(frame["Close"].mean())


@pytest.mark.unit
def test_early_dates_read_na_until_the_window_fills(ohlcv):
    frame = ohlcv(60)
    values = market._get_stock_stats_bulk("SPCX", "close_50_sma", AS_OF)
    dates = frame["Date"].dt.strftime("%Y-%m-%d")
    assert values[dates.iloc[48]] == "N/A"
    assert float(values[dates.iloc[49]]) == pytest.approx(frame["Close"].head(50).mean())


@pytest.mark.unit
@pytest.mark.parametrize("indicator", [
    "close_10_ema", "rsi", "boll", "boll_ub", "boll_lb",
    "macd", "macds", "macdh", "atr", "vwma", "mfi",
])
def test_every_windowed_indicator_needs_its_history(ohlcv, indicator):
    ohlcv(5)
    assert market.get_stockstats_indicator("SPCX", indicator, AS_OF) == "N/A"


@pytest.mark.unit
def test_snapshot_says_how_much_history_it_had(ohlcv):
    ohlcv(100)
    text = snapshot.build_verified_market_snapshot("SPCX", AS_OF)
    assert "100 trading rows" in text
