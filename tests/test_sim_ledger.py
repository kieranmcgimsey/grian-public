"""Tests for grian.dispatch.ledger — trade logging and P&L computation."""

import pandas as pd
import pytest

from grian.dispatch.ledger import (
    append_record,
    cumulative_pnl,
    daily_pnl,
    empty_ledger,
    forecast_accuracy,
    peak_drawdown,
    sharpe_ratio,
    summarise,
    to_dataframe,
    total_revenue,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_ledger():
    """A ledger with 10 intervals of mixed trading activity."""
    ledger = empty_ledger()
    timestamps = pd.date_range("2023-07-01", periods=10, freq="5min")
    prices = [50, 60, 40, 80, 30, 70, 55, 45, 65, 50]
    forecasts = [48, 62, 42, 75, 35, 68, 52, 48, 60, 53]

    for i, ts in enumerate(timestamps):
        # Alternate between charging and discharging
        if prices[i] < 50:
            charge, discharge = 50.0, 0.0
        else:
            charge, discharge = 0.0, 50.0

        append_record(
            ledger,
            timestamp=ts,
            actual_price=prices[i],
            forecast_price=forecasts[i],
            charge_mw=charge,
            discharge_mw=discharge,
            soc_mwh=100.0,
        )
    return ledger


@pytest.fixture
def sample_ledger_df(sample_ledger):
    """The sample ledger as a DataFrame."""
    return to_dataframe(sample_ledger)


# ---------------------------------------------------------------------------
# Ledger creation and appending
# ---------------------------------------------------------------------------

class TestLedgerCreation:
    """Tests for creating and appending to ledgers."""

    def test_empty_ledger_is_list(self):
        """empty_ledger returns a plain list."""
        ledger = empty_ledger()
        assert isinstance(ledger, list)
        assert len(ledger) == 0

    def test_append_adds_record(self):
        """Appending a record increases length by 1."""
        ledger = empty_ledger()
        append_record(
            ledger,
            timestamp=pd.Timestamp("2023-07-01"),
            actual_price=50.0,
            forecast_price=48.0,
            charge_mw=0.0,
            discharge_mw=100.0,
            soc_mwh=50.0,
        )
        assert len(ledger) == 1

    def test_revenue_calculation(self):
        """Revenue = (discharge - charge) * price * dt."""
        ledger = empty_ledger()
        append_record(
            ledger,
            timestamp=pd.Timestamp("2023-07-01"),
            actual_price=100.0,
            forecast_price=95.0,
            charge_mw=0.0,
            discharge_mw=50.0,
            soc_mwh=80.0,
            interval_minutes=5.0,
        )
        # revenue = 100 * (50 - 0) * (5/60) = 100 * 50 * 0.08333 = 416.67
        expected = 100.0 * 50.0 * (5.0 / 60.0)
        assert abs(ledger[0]["revenue"] - expected) < 0.01

    def test_charge_produces_negative_revenue(self):
        """Charging costs money (negative revenue)."""
        ledger = empty_ledger()
        append_record(
            ledger,
            timestamp=pd.Timestamp("2023-07-01"),
            actual_price=50.0,
            forecast_price=50.0,
            charge_mw=100.0,
            discharge_mw=0.0,
            soc_mwh=100.0,
        )
        assert ledger[0]["revenue"] < 0

    def test_30min_interval_revenue(self):
        """Revenue scales correctly for 30-minute intervals."""
        ledger = empty_ledger()
        append_record(
            ledger,
            timestamp=pd.Timestamp("2023-07-01"),
            actual_price=100.0,
            forecast_price=100.0,
            charge_mw=0.0,
            discharge_mw=50.0,
            soc_mwh=50.0,
            interval_minutes=30.0,
        )
        expected = 100.0 * 50.0 * 0.5
        assert abs(ledger[0]["revenue"] - expected) < 0.01

    def test_record_has_all_fields(self):
        """Appended record contains all expected fields."""
        ledger = empty_ledger()
        append_record(
            ledger,
            timestamp=pd.Timestamp("2023-07-01"),
            actual_price=50.0,
            forecast_price=48.0,
            charge_mw=10.0,
            discharge_mw=0.0,
            soc_mwh=90.0,
        )
        expected_keys = {
            "timestamp", "actual_price", "forecast_price",
            "charge_mw", "discharge_mw", "net_mw", "soc_mwh",
            "revenue", "interval_minutes",
        }
        assert set(ledger[0].keys()) == expected_keys


# ---------------------------------------------------------------------------
# DataFrame conversion
# ---------------------------------------------------------------------------

class TestToDataframe:
    """Tests for converting ledger lists to DataFrames."""

    def test_empty_ledger_to_df(self):
        """Empty ledger produces empty DataFrame with correct columns."""
        df = to_dataframe(empty_ledger())
        assert len(df) == 0
        assert "revenue" in df.columns

    def test_index_is_timestamp(self, sample_ledger_df):
        """DataFrame is indexed by timestamp."""
        assert sample_ledger_df.index.name == "timestamp"

    def test_correct_row_count(self, sample_ledger, sample_ledger_df):
        """DataFrame has same number of rows as ledger."""
        assert len(sample_ledger_df) == len(sample_ledger)


# ---------------------------------------------------------------------------
# P&L functions
# ---------------------------------------------------------------------------

class TestPnL:
    """Tests for P&L computation functions."""

    def test_cumulative_pnl_monotonic_check(self, sample_ledger_df):
        """Cumulative P&L starts at first interval's revenue."""
        cum = cumulative_pnl(sample_ledger_df)
        assert len(cum) == len(sample_ledger_df)
        assert cum.iloc[0] == sample_ledger_df["revenue"].iloc[0]

    def test_cumulative_pnl_final_equals_total(self, sample_ledger_df):
        """Last cumulative value equals total revenue."""
        cum = cumulative_pnl(sample_ledger_df)
        assert abs(cum.iloc[-1] - total_revenue(sample_ledger_df)) < 1e-10

    def test_daily_pnl_sums_correctly(self):
        """Daily P&L sums intervals within each day."""
        ledger = empty_ledger()
        # Two intervals on day 1, one on day 2
        for ts, rev_price in [
            ("2023-07-01 00:00", 50.0),
            ("2023-07-01 00:05", 60.0),
            ("2023-07-02 00:00", 40.0),
        ]:
            append_record(
                ledger, timestamp=pd.Timestamp(ts),
                actual_price=rev_price, forecast_price=rev_price,
                charge_mw=0.0, discharge_mw=10.0, soc_mwh=100.0,
            )
        df = to_dataframe(ledger)
        daily = daily_pnl(df)
        assert len(daily) == 2

    def test_total_revenue_type(self, sample_ledger_df):
        """total_revenue returns a float."""
        rev = total_revenue(sample_ledger_df)
        assert isinstance(rev, float)

    def test_peak_drawdown_non_negative(self, sample_ledger_df):
        """Peak drawdown is always non-negative."""
        dd = peak_drawdown(sample_ledger_df)
        assert dd >= 0

    def test_peak_drawdown_on_always_profitable(self):
        """Zero drawdown when every interval is profitable."""
        ledger = empty_ledger()
        for i in range(5):
            append_record(
                ledger,
                timestamp=pd.Timestamp("2023-07-01") + pd.Timedelta(minutes=5 * i),
                actual_price=100.0, forecast_price=100.0,
                charge_mw=0.0, discharge_mw=50.0, soc_mwh=50.0,
            )
        df = to_dataframe(ledger)
        assert peak_drawdown(df) == 0.0

    def test_sharpe_ratio_type(self, sample_ledger_df):
        """sharpe_ratio returns a float."""
        sr = sharpe_ratio(sample_ledger_df)
        assert isinstance(sr, float)

    def test_sharpe_zero_with_no_variation(self):
        """Sharpe is 0 when all daily returns are identical."""
        ledger = empty_ledger()
        for i in range(10):
            append_record(
                ledger,
                timestamp=pd.Timestamp("2023-07-01") + pd.Timedelta(minutes=5 * i),
                actual_price=50.0, forecast_price=50.0,
                charge_mw=0.0, discharge_mw=50.0, soc_mwh=100.0,
            )
        df = to_dataframe(ledger)
        # All on same day, so only 1 daily return — std is 0
        assert sharpe_ratio(df) == 0.0


# ---------------------------------------------------------------------------
# Forecast accuracy
# ---------------------------------------------------------------------------

class TestForecastAccuracy:
    """Tests for forecast accuracy metrics."""

    def test_perfect_forecast(self):
        """Zero error when forecast matches actual."""
        ledger = empty_ledger()
        for i in range(5):
            append_record(
                ledger,
                timestamp=pd.Timestamp("2023-07-01") + pd.Timedelta(minutes=5 * i),
                actual_price=50.0, forecast_price=50.0,
                charge_mw=0.0, discharge_mw=0.0, soc_mwh=0.0,
            )
        df = to_dataframe(ledger)
        acc = forecast_accuracy(df)
        assert acc["mae"] == 0.0
        assert acc["rmse"] == 0.0

    def test_mae_positive(self, sample_ledger_df):
        """MAE is positive when forecast differs from actual."""
        acc = forecast_accuracy(sample_ledger_df)
        assert acc["mae"] > 0

    def test_rmse_ge_mae(self, sample_ledger_df):
        """RMSE >= MAE (always true by Jensen's inequality)."""
        acc = forecast_accuracy(sample_ledger_df)
        assert acc["rmse"] >= acc["mae"]


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestSummarise:
    """Tests for the full summary function."""

    def test_summary_has_all_keys(self, sample_ledger_df):
        """summarise returns all expected metric keys."""
        s = summarise(sample_ledger_df)
        expected_keys = {
            "total_revenue", "daily_mean_revenue", "daily_std_revenue",
            "sharpe_ratio", "peak_drawdown", "n_intervals", "n_days",
            "mae", "rmse", "mape",
        }
        assert expected_keys <= set(s.keys())

    def test_n_intervals_correct(self, sample_ledger_df):
        """n_intervals matches ledger length."""
        s = summarise(sample_ledger_df)
        assert s["n_intervals"] == len(sample_ledger_df)
