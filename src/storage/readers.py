"""
Partitioned data readers for the volatility infrastructure platform.

Reads JSON-line partitioned data written by the PartitionedWriter.
Supports queries by date range, underlying, and layer.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Type
from dataclasses import fields as dc_fields


class PartitionedReader:
    """
    Reader for partitioned data stores.

    Scans partition directories and deserializes records
    from JSON-line files.
    """

    def __init__(self, base_path: str):
        """
        Args:
            base_path: Root data directory.
        """
        self.base_path = Path(base_path)

    def read_partition(
        self,
        layer: str,
        trade_date: str,
        underlying: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Read all records from a specific partition.

        Args:
            layer: Data layer (raw, snapshots, forwards, etc.).
            trade_date: Trade date (YYYY-MM-DD).
            underlying: Optional underlying filter.

        Returns:
            List of record dictionaries.
        """
        partition_dir = self.base_path / layer / f"dt={trade_date}"
        if underlying:
            partition_dir = partition_dir / f"underlying={underlying}"

        if not partition_dir.exists():
            return []

        records = []
        for jsonl_file in sorted(partition_dir.glob("*.jsonl")):
            if "quarantine" in jsonl_file.name:
                continue
            with open(jsonl_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        return records

    def read_date_range(
        self,
        layer: str,
        start_date: str,
        end_date: str,
        underlying: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Read records across a date range.

        Args:
            layer: Data layer.
            start_date: Start date inclusive (YYYY-MM-DD).
            end_date: End date inclusive (YYYY-MM-DD).
            underlying: Optional underlying filter.

        Returns:
            List of record dictionaries ordered by date.
        """
        layer_dir = self.base_path / layer
        if not layer_dir.exists():
            return []

        records = []
        for dt_dir in sorted(layer_dir.glob("dt=*")):
            date_str = dt_dir.name.replace("dt=", "")
            if start_date <= date_str <= end_date:
                partition_records = self.read_partition(
                    layer, date_str, underlying
                )
                records.extend(partition_records)
        return records

    def list_partitions(self, layer: str) -> List[str]:
        """List available trade dates for a layer."""
        layer_dir = self.base_path / layer
        if not layer_dir.exists():
            return []
        return sorted([
            d.name.replace("dt=", "")
            for d in layer_dir.iterdir()
            if d.is_dir() and d.name.startswith("dt=")
        ])

    def list_underlyings(self, layer: str, trade_date: str) -> List[str]:
        """List available underlyings for a layer and date."""
        partition_dir = self.base_path / layer / f"dt={trade_date}"
        if not partition_dir.exists():
            return []
        return sorted([
            d.name.replace("underlying=", "")
            for d in partition_dir.iterdir()
            if d.is_dir() and d.name.startswith("underlying=")
        ])

    def read_manifest(
        self,
        layer: str,
        trade_date: str,
    ) -> Optional[Dict[str, Any]]:
        """Read the manifest for a partition if it exists."""
        manifest_path = self.base_path / layer / f"dt={trade_date}" / "manifest.json"
        if not manifest_path.exists():
            return None
        with open(manifest_path) as f:
            return json.load(f)

    def partition_exists(self, layer: str, trade_date: str) -> bool:
        """Check if a partition exists."""
        return (self.base_path / layer / f"dt={trade_date}").exists()
