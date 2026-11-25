"""Small in-process counters for synchronous ingestion; no external metrics service."""

import logging
from collections import deque
from datetime import timedelta
from time import monotonic

log = logging.getLogger(__name__)
REPORT_INTERVAL_SECONDS = 30
RPM_WINDOW_SECONDS = 60


class IngestionTelemetry:
    def __init__(self, target_rpm: int):
        self.target_rpm = target_rpm
        self.http_requests_total = 0
        self.rows_received_total = 0
        self._started = monotonic()
        self._recent_requests: deque[float] = deque()
        self.start_stage("http")

    def start_stage(self, stage: str) -> None:
        self.stage = stage
        self._stage_started = monotonic()
        self._requests_before = self.http_requests_total
        self._rows_before = self.rows_received_total
        self._last_report = self._stage_started

    def _prune_requests(self, now: float) -> None:
        while self._recent_requests and self._recent_requests[0] <= now - RPM_WINDOW_SECONDS:
            self._recent_requests.popleft()

    def record_request(self) -> None:
        now = monotonic()
        self.http_requests_total += 1
        self._recent_requests.append(now)
        self._prune_requests(now)

    def record_rows(self, rows: int) -> None:
        self.rows_received_total += rows

    def snapshot(self) -> dict[str, int | float]:
        now = monotonic()
        self._prune_requests(now)
        elapsed = now - self._stage_started
        window = min(RPM_WINDOW_SECONDS, now - self._started)
        requests = self.http_requests_total - self._requests_before
        rows = self.rows_received_total - self._rows_before
        return {
            "http_requests": requests,
            "http_requests_total": self.http_requests_total,
            "rows_received": rows,
            "elapsed_seconds": elapsed,
            "current_rpm": len(self._recent_requests) * 60 / window if window > 0 else 0.0,
            "average_rpm": requests * 60 / elapsed if elapsed > 0 else 0.0,
            "rows_per_second": rows / elapsed if elapsed > 0 else 0.0,
        }

    def log_progress(self, *, force: bool = False) -> None:
        now = monotonic()
        if not force and now - self._last_report < REPORT_INTERVAL_SECONDS:
            return
        self._last_report = now
        stats = self.snapshot()
        log.info(
            "%s throughput: elapsed=%s HTTP requests=%s (total=%s) "
            "rpm_current_60s=%.1f rpm_avg=%.1f target_rpm=%s "
            "rows_received=%s rows/s=%.1f",
            self.stage,
            timedelta(seconds=int(stats["elapsed_seconds"])),
            f"{stats['http_requests']:,}",
            f"{stats['http_requests_total']:,}",
            stats["current_rpm"],
            stats["average_rpm"],
            self.target_rpm,
            f"{stats['rows_received']:,}",
            stats["rows_per_second"],
        )
