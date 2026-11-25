import logging

import pytest

from dead_reckoning import telemetry


def test_rolling_and_stage_rates_preserve_total_request_count(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(telemetry, "monotonic", lambda: now[0])
    stats = telemetry.IngestionTelemetry(target_rpm=180)
    stats.start_stage("daily_bars")
    for instant in (10, 30, 50):
        now[0] = instant
        stats.record_request()
        stats.record_rows(1000)
    now[0] = 75
    snapshot = stats.snapshot()
    assert snapshot["http_requests"] == 3
    assert snapshot["current_rpm"] == 2
    assert snapshot["average_rpm"] == pytest.approx(2.4)
    assert snapshot["rows_per_second"] == 40
    assert snapshot["elapsed_seconds"] == 75

    now[0] = 80
    stats.start_stage("minute_bars")
    now[0] = 85
    stats.record_request()
    stats.record_rows(600)
    now[0] = 90
    snapshot = stats.snapshot()
    assert snapshot["http_requests"] == 1
    assert snapshot["http_requests_total"] == 4
    assert snapshot["rows_received"] == 600
    assert snapshot["average_rpm"] == 6
    assert snapshot["rows_per_second"] == 60
    now[0] = 150
    assert stats.snapshot()["current_rpm"] == 0


def test_reports_are_throttled_and_short_runs_have_safe_rates(monkeypatch, caplog):
    now = [0.0]
    monkeypatch.setattr(telemetry, "monotonic", lambda: now[0])
    stats = telemetry.IngestionTelemetry(target_rpm=180)
    assert stats.snapshot()["average_rpm"] == 0
    assert stats.snapshot()["current_rpm"] == 0
    with caplog.at_level(logging.INFO, logger="dead_reckoning.telemetry"):
        stats.record_request()
        stats.log_progress()
        assert not caplog.records
        now[0] = 30
        stats.log_progress()
        assert len(caplog.records) == 1
        assert stats.snapshot()["current_rpm"] == 2
        now[0] = 31
        stats.log_progress()
        assert len(caplog.records) == 1
        stats.log_progress(force=True)
        assert len(caplog.records) == 2
        assert "target_rpm=180" in caplog.text
        assert "rows_received=0" in caplog.text
