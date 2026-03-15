"""Metrics Writer Utility

Composable utility for writing metrics to CSV files and TensorBoard.
Extracted from shared logic between DiffusionLogger and ClassifierLogger.
"""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.utils.tensorboard import (
    close_tensorboard_writer,
    create_tensorboard_writer,
    safe_log_scalar,
)


class MetricsWriter:
    """Composable utility for CSV and TensorBoard metrics writing.

    Handles:
    - CSV file initialization, writing, and dynamic field expansion
    - TensorBoard scalar logging
    - Cleanup of TensorBoard writer

    Args:
        metrics_file: Path to the CSV metrics file
        tensorboard_config: Optional TensorBoard configuration dict
        tb_log_dir: Optional directory for TensorBoard event logs
    """

    def __init__(
        self,
        metrics_file: Path,
        tensorboard_config: Optional[Dict[str, Any]] = None,
        tb_log_dir: Optional[Union[str, Path]] = None,
    ):
        self.metrics_file = metrics_file
        self.csv_initialized = self.metrics_file.exists()
        self.csv_fieldnames: Optional[List[str]] = None

        # If CSV exists, load existing fieldnames
        if self.csv_initialized:
            with open(self.metrics_file, "r") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    self.csv_fieldnames = list(reader.fieldnames)
                else:
                    self.csv_initialized = False
                    self.csv_fieldnames = None

        # Initialize TensorBoard
        tensorboard_config = tensorboard_config or {}
        self.tb_enabled = tensorboard_config.get("enabled", False)
        flush_secs = tensorboard_config.get("flush_secs", 30)

        if tb_log_dir is not None:
            self.tb_writer = create_tensorboard_writer(
                log_dir=tb_log_dir,
                flush_secs=flush_secs,
                enabled=self.tb_enabled,
            )
        else:
            self.tb_writer = None

    def write_metrics(self, log_entry: Dict[str, Any]) -> None:
        """Write a single metrics entry to CSV file.

        Args:
            log_entry: Dictionary of metrics to write
        """
        if not self.csv_initialized:
            self.csv_fieldnames = list(log_entry.keys())
            with open(self.metrics_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
                writer.writeheader()
            self.csv_initialized = True
        else:
            assert self.csv_fieldnames is not None
            new_fields = set(log_entry.keys()) - set(self.csv_fieldnames)
            if new_fields:
                self.csv_fieldnames.extend(sorted(new_fields))
                self._rewrite_csv_with_new_fields()

        assert self.csv_fieldnames is not None
        with open(self.metrics_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
            row = {field: log_entry.get(field) for field in self.csv_fieldnames}
            writer.writerow(row)

    def log_scalars(self, metrics: Dict[str, Any], step: int) -> None:
        """Log scalar metrics to TensorBoard.

        Args:
            metrics: Dictionary of metric names to values
            step: Current training step
        """
        if self.tb_writer is not None:
            for key, value in metrics.items():
                safe_log_scalar(self.tb_writer, f"metrics/{key}", value, step)

    def _rewrite_csv_with_new_fields(self) -> None:
        """Re-write CSV file with updated field names."""
        assert self.csv_fieldnames is not None
        existing_data = []
        if self.metrics_file.exists():
            with open(self.metrics_file, "r") as f:
                reader = csv.DictReader(f)
                existing_data = list(reader)

        with open(self.metrics_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
            writer.writeheader()
            for row in existing_data:
                updated_row = {field: row.get(field) for field in self.csv_fieldnames}
                writer.writerow(updated_row)

    def close(self) -> None:
        """Close TensorBoard writer and cleanup."""
        close_tensorboard_writer(self.tb_writer)
        self.tb_writer = None
