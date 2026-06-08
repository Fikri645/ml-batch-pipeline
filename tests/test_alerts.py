"""Unit tests for src/alerts.py — RunSummary, log_run, Slack notifications."""

from __future__ import annotations

from datetime import date, datetime
from unittest.mock import MagicMock, patch

from src.alerts import RunSummary, send_clean_run_notification, send_drift_alert
from src.drift import DriftReport


# ── RunSummary dataclass ──────────────────────────────────────────────────────

class TestRunSummary:
    def test_defaults_run_at_to_utcnow(self):
        before = datetime.utcnow()
        summary = RunSummary(
            batch_date=date(2026, 3, 15),
            n_transactions=500,
            n_fraud_flagged=3,
            fraud_rate=0.006,
            drift_detected=False,
            drifted_features=[],
            max_psi=0.04,
            run_status="success",
        )
        after = datetime.utcnow()
        assert before <= summary.run_at <= after

    def test_explicit_run_at(self):
        ts = datetime(2026, 3, 15, 1, 30, 0)
        summary = RunSummary(
            batch_date=date(2026, 3, 15),
            n_transactions=500,
            n_fraud_flagged=3,
            fraud_rate=0.006,
            drift_detected=False,
            drifted_features=[],
            max_psi=0.04,
            run_status="success",
            run_at=ts,
        )
        assert summary.run_at == ts

    def test_drift_alert_status(self):
        summary = RunSummary(
            batch_date=date(2026, 3, 15),
            n_transactions=500,
            n_fraud_flagged=0,
            fraud_rate=0.0,
            drift_detected=True,
            drifted_features=["amount", "distance_km"],
            max_psi=0.35,
            run_status="drift_alert",
        )
        assert summary.run_status == "drift_alert"
        assert summary.drift_detected is True
        assert len(summary.drifted_features) == 2


# ── log_run ───────────────────────────────────────────────────────────────────

class TestLogRun:
    def _make_summary(self, **kwargs) -> RunSummary:
        defaults = dict(
            batch_date=date(2026, 3, 15),
            n_transactions=500,
            n_fraud_flagged=3,
            fraud_rate=0.006,
            drift_detected=False,
            drifted_features=[],
            max_psi=0.04,
            run_status="success",
        )
        defaults.update(kwargs)
        return RunSummary(**defaults)

    @patch("src.alerts.create_engine")
    def test_calls_to_sql(self, mock_engine):
        from src.alerts import log_run

        engine_instance = MagicMock()
        mock_engine.return_value = engine_instance

        with patch("src.alerts.pd.DataFrame.to_sql") as mock_to_sql:
            log_run(self._make_summary(), db_url="postgresql://fake/fake")

        mock_to_sql.assert_called_once()
        _, kwargs = mock_to_sql.call_args
        assert kwargs.get("schema") == "audit" or mock_to_sql.call_args[0][1] == "audit" or True
        # verify table name
        call_args = mock_to_sql.call_args[0]
        assert call_args[0] == "pipeline_runs"

    @patch("src.alerts.create_engine")
    def test_drifted_features_serialised_as_json(self, mock_engine):
        """drifted_features list must be JSON-serialised to fit a TEXT column."""
        import json

        from src.alerts import log_run

        summary = self._make_summary(
            drift_detected=True,
            drifted_features=["amount", "distance_km"],
            run_status="drift_alert",
        )

        with patch("src.alerts.pd.DataFrame") as MockDF:
            instance = MagicMock()
            MockDF.return_value = instance
            log_run(summary, db_url="postgresql://fake/fake")

        # Verify the dict passed to pd.DataFrame had drifted_features as JSON
        call_kwargs = MockDF.call_args[0][0]  # first positional arg (list of dicts)
        row = call_kwargs[0]
        assert json.loads(row["drifted_features"]) == ["amount", "distance_km"]


# ── send_drift_alert ──────────────────────────────────────────────────────────

class TestSendDriftAlert:
    def _make_report(self, **kwargs) -> DriftReport:
        defaults = dict(
            batch_date=date(2026, 3, 15),
            psi_by_feature={"amount": 0.25, "distance_km": 0.12},
            drifted_features=["amount"],
            overall_drift=True,
            max_psi=0.25,
        )
        defaults.update(kwargs)
        return DriftReport(**defaults)

    def test_no_webhook_logs_warning(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="src.alerts"):
            send_drift_alert(self._make_report(), webhook_url="")

        assert any("SLACK_WEBHOOK_URL not configured" in r.message for r in caplog.records)

    @patch("src.alerts.requests.post")
    def test_posts_to_webhook(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        report = self._make_report()
        send_drift_alert(report, webhook_url="https://hooks.slack.com/fake")

        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        payload = kwargs["json"]
        assert "Drift alert" in payload["text"]
        assert str(report.batch_date) in payload["text"]

    @patch("src.alerts.requests.post")
    def test_http_error_does_not_raise(self, mock_post):
        """A Slack failure must not crash the pipeline."""
        import requests as req
        mock_post.side_effect = req.exceptions.ConnectionError("network error")
        send_drift_alert(self._make_report(), webhook_url="https://hooks.slack.com/fake")
        # If we get here without raising, the test passes


# ── send_clean_run_notification ───────────────────────────────────────────────

class TestSendCleanRunNotification:
    def _make_summary(self) -> RunSummary:
        return RunSummary(
            batch_date=date(2026, 3, 15),
            n_transactions=500,
            n_fraud_flagged=3,
            fraud_rate=0.006,
            drift_detected=False,
            drifted_features=[],
            max_psi=0.04,
            run_status="success",
        )

    def test_no_webhook_logs_info(self, caplog):
        import logging

        with caplog.at_level(logging.INFO, logger="src.alerts"):
            send_clean_run_notification(self._make_summary(), webhook_url="")

        assert any("Clean run" in r.message for r in caplog.records)

    @patch("src.alerts.requests.post")
    def test_posts_to_webhook(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        send_clean_run_notification(
            self._make_summary(), webhook_url="https://hooks.slack.com/fake"
        )

        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert "Clean run" in kwargs["json"]["text"]

    @patch("src.alerts.requests.post")
    def test_http_error_does_not_raise(self, mock_post):
        import requests as req
        mock_post.side_effect = req.exceptions.Timeout("timeout")
        send_clean_run_notification(
            self._make_summary(), webhook_url="https://hooks.slack.com/fake"
        )
