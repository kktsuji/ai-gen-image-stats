"""Tests for MetricsWriter utility."""

import csv

import pytest

from src.utils.metrics_writer import MetricsWriter


@pytest.mark.unit
class TestMetricsWriterCSV:
    """Test CSV writing functionality."""

    def test_csv_init_on_first_write(self, tmp_path):
        """CSV file is created with header on first write."""
        metrics_file = tmp_path / "metrics.csv"
        writer = MetricsWriter(metrics_file=metrics_file)

        writer.write_metrics({"step": 1, "loss": 0.5})

        assert metrics_file.exists()
        with open(metrics_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert reader.fieldnames == ["step", "loss"]
            assert len(rows) == 1
            assert rows[0]["step"] == "1"
            assert rows[0]["loss"] == "0.5"

    def test_csv_append(self, tmp_path):
        """Subsequent writes append to CSV."""
        metrics_file = tmp_path / "metrics.csv"
        writer = MetricsWriter(metrics_file=metrics_file)

        writer.write_metrics({"step": 1, "loss": 0.5})
        writer.write_metrics({"step": 2, "loss": 0.3})

        with open(metrics_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 2

    def test_csv_field_expansion(self, tmp_path):
        """New fields are added dynamically to CSV."""
        metrics_file = tmp_path / "metrics.csv"
        writer = MetricsWriter(metrics_file=metrics_file)

        writer.write_metrics({"step": 1, "loss": 0.5})
        writer.write_metrics({"step": 2, "loss": 0.3, "accuracy": 0.9})

        with open(metrics_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert reader.fieldnames is not None
            assert "accuracy" in reader.fieldnames
            assert len(rows) == 2
            # First row should have empty accuracy
            assert rows[0]["accuracy"] == "" or rows[0]["accuracy"] is None

    def test_csv_load_existing(self, tmp_path):
        """Existing CSV file fieldnames are loaded on init."""
        metrics_file = tmp_path / "metrics.csv"

        # Write initial CSV
        writer1 = MetricsWriter(metrics_file=metrics_file)
        writer1.write_metrics({"step": 1, "loss": 0.5})

        # Create new writer that loads existing CSV
        writer2 = MetricsWriter(metrics_file=metrics_file)
        writer2.write_metrics({"step": 2, "loss": 0.3})

        with open(metrics_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 2


@pytest.mark.unit
class TestMetricsWriterTensorBoard:
    """Test TensorBoard integration."""

    def test_tb_disabled_by_default(self, tmp_path):
        """TensorBoard is disabled when no config is provided."""
        writer = MetricsWriter(
            metrics_file=tmp_path / "metrics.csv",
        )
        assert writer.tb_writer is None

    def test_tb_enabled_with_config(self, tmp_path):
        """TensorBoard writer is created when enabled in config."""
        writer = MetricsWriter(
            metrics_file=tmp_path / "metrics.csv",
            tensorboard_config={"enabled": True},
            tb_log_dir=tmp_path / "tb",
        )
        # May be None if tensorboard is not installed
        # Just verify no error
        writer.close()

    def test_log_scalars_no_crash_without_tb(self, tmp_path):
        """log_scalars does not crash when TB is disabled."""
        writer = MetricsWriter(metrics_file=tmp_path / "metrics.csv")
        writer.log_scalars({"loss": 0.5}, step=1)  # Should not raise


@pytest.mark.unit
class TestMetricsWriterClose:
    """Test cleanup."""

    def test_close(self, tmp_path):
        """close() sets tb_writer to None."""
        writer = MetricsWriter(metrics_file=tmp_path / "metrics.csv")
        writer.close()
        assert writer.tb_writer is None
