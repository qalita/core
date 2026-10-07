"""
# QALITA (c) COPYRIGHT 2025 - ALL RIGHTS RESERVED -
"""

import os
import re
import json
import base64
import logging
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
from qalita_core.data_source_opener import (
    get_data_source,
    cleanup_parquet_files,
)
from urllib.parse import urlsplit
import math
from decimal import Decimal
import datetime as _dt

# Polars is a hard requirement of qalita_core: it is how packs read data at all.
# It is deliberately imported unguarded so a broken install fails loudly here
# rather than silently degrading to a pandas path that cannot stream.
import polars as pl

POLARS_AVAILABLE = True

if TYPE_CHECKING:
    import polars as pl


# `<source>_<object>_part_<n>.parquet` — written by
# data_source_opener._build_parquet_path. The base name identifies the logical
# object the parts belong to.
_PART_SUFFIX = re.compile(r"_part_\d+\.parquet$", re.IGNORECASE)


def _object_key(path: str) -> str:
    """Logical object name a parquet part belongs to."""
    return _PART_SUFFIX.sub("", os.path.basename(path))


class Pack:
    """
    Represents a pack in the system, handling configurations and data loading.
    """

    # Default configuration paths
    default_configs = {
        "pack_conf": "pack_conf.json",
        "source_conf": "source_conf.json",
        "target_conf": "target_conf.json",
        "agent_file": "~/.qalita/.worker",
    }

    def __init__(self, configs=None):
        self.logger = logging.getLogger(self.__class__.__name__)

        if configs is None:
            configs = {}

        # Update default paths with any provided configurations
        self.config_paths = {**self.default_configs, **configs}
        self.pack_config = ConfigLoader.load_config(
            self.config_paths["pack_conf"]
        )
        self.source_config = ConfigLoader.load_config(
            self.config_paths["source_conf"]
        )
        self.target_config = ConfigLoader.load_config(
            self.config_paths["target_conf"]
        )
        self.agent_config = self.load_agent_config(
            self.config_paths["agent_file"]
        )
        self.metrics = PlatformAsset("metrics")
        self.recommendations = PlatformAsset("recommendations")
        self.schemas = PlatformAsset("schemas")

        # Import local : figures.py dérive de PlatformAsset, défini plus bas
        # dans ce même module — un import en tête créerait un cycle.
        from qalita_core.figures import FiguresAsset

        self.figures = FiguresAsset()

        # What the last load_data() call returned, per trigger. Packs read
        # these as "what I just loaded"; cleanup does NOT use them, see
        # _staged below.
        self.paths_source = None
        self.paths_target = None
        # Every parquet part load_data() wrote, across all its calls. Cleanup
        # deletes these and only these: paths_source keeps the last call
        # only (a pack loading table by table leaked every earlier table),
        # and holds the user's own file for a parquet source, which cleanup
        # used to delete.
        self._staged: Dict[str, List[str]] = {"source": [], "target": []}
        self.df_source = None
        self.df_target = None
        # Logical object name -> its parquet parts. Built by load_data(). This
        # mapping is what makes zip(table_names, parquet_paths) unnecessary: the
        # pairing is recorded at load time instead of guessed afterwards, which
        # is how chunks 2..N used to be silently dropped or relabelled.
        self.objects_source: Dict[str, List[str]] = {}
        self.objects_target: Dict[str, List[str]] = {}
        self.skipped_source_objects: List[Dict[str, str]] = []
        self.skipped_target_objects: List[Dict[str, str]] = []

        # Validate configurations
        if not self.source_config:
            self.logger.error("Source configuration is empty.")
        elif "type" not in self.source_config:
            self.logger.error(
                "Source configuration is missing the 'type' key."
            )

    def __enter__(self):
        """Context manager entry - returns self for use in 'with' statements."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup of temporary parquet files."""
        self.cleanup()
        # Don't suppress exceptions - return False or None
        return False

    def cleanup(self):
        """
        Remove temporary parquet files created during data loading.

        This method should be called when the analysis is complete (success or failure)
        to free up disk space. It is automatically called when using Pack as a context manager.

        Returns:
            int: Total number of files removed.
        """
        total_removed = 0
        for trigger, staged in self._staged.items():
            if not staged:
                continue
            removed = cleanup_parquet_files(staged, self.logger)
            self.logger.debug(
                f"Cleaned up {removed} {trigger} parquet file(s)"
            )
            total_removed += removed
            staged.clear()
        if total_removed > 0:
            self.logger.info(
                f"Cleaned up {total_removed} temporary parquet file(s)"
            )
        return total_removed

    def load_agent_config(self, agent_file_path):
        try:
            abs_agent_file_path = os.path.expanduser(
                agent_file_path
            )  # Resolve any user-relative paths
            with open(abs_agent_file_path, "r") as agent_file:
                encoded_content = agent_file.read()
                decoded_content = base64.b64decode(encoded_content).decode(
                    "utf-8"
                )
                data = json.loads(decoded_content)
                # Normalize local context URL to scheme://host without trailing API paths
                try:
                    local_ctx = data.get("context", {}).get("local", {})
                    original_url = local_ctx.get("url")
                    if isinstance(original_url, str) and original_url:
                        parts = urlsplit(original_url)
                        if parts.scheme and parts.netloc:
                            base_url = f"{parts.scheme}://{parts.netloc}"
                            local_ctx["url"] = base_url
                except Exception:
                    # Do not fail if normalization cannot be applied
                    pass
                return data
        except Exception as e:
            self.logger.error(f"Error loading agent configuration: {e}")
            return {}

    def load_data(self, trigger, table_or_query=None) -> List[str]:
        """
        Load data from source/target and return list of parquet file paths.

        Args:
            trigger: "source" or "target"
            table_or_query: Optional table name or SQL query

        Returns:
            List of parquet file paths containing the loaded data.
        """
        source_conf = (
            self.source_config if trigger == "source" else self.target_config
        )
        pack_conf = self.pack_config
        ds = get_data_source(source_conf)
        table_or_query = table_or_query or source_conf.get("config", {}).get(
            "table_or_query"
        )
        # Enrich pack_config with deterministic output directory and chunking hints
        job_cfg = (pack_conf or {}).get("job", {})
        trigger_cfg = (
            (job_cfg.get(trigger) or {}) if isinstance(job_cfg, dict) else {}
        )
        parquet_output_dir = (
            trigger_cfg.get("parquet_output_dir")
            or job_cfg.get("parquet_output_dir")
            or "./parquet"
        )
        chunk_rows = (
            trigger_cfg.get("chunk_rows")
            or job_cfg.get("chunk_rows")
            or 100000
        )
        effective_pack_conf = {
            **(pack_conf or {}),
            "parquet_output_dir": parquet_output_dir,
            "chunk_rows": chunk_rows,
            "_trigger": trigger,
        }
        paths = ds.get_data(table_or_query, pack_config=effective_pack_conf)
        self._track_staged(trigger, ds, paths)
        objects = self._group_by_object(ds, paths)
        skipped_objects = [
            dict(item) for item in getattr(ds, "skipped_objects", [])
        ]
        if trigger == "source":
            # Keep legacy attribute names for backward compatibility
            self.paths_source = paths
            self.df_source = paths
            self.objects_source = objects
            self.skipped_source_objects = skipped_objects
            return self.paths_source
        elif trigger == "target":
            self.paths_target = paths
            self.df_target = paths
            self.objects_target = objects
            self.skipped_target_objects = skipped_objects
            return self.paths_target

    def _track_staged(self, trigger: str, ds, paths) -> None:
        """Remember the parts this call wrote, never the source's own files."""
        staged = self._staged.setdefault(trigger, [])
        borrowed = {
            os.path.realpath(path)
            for path in (getattr(ds, "borrowed_paths", None) or ())
            if isinstance(path, str)
        }
        known = set(staged)
        for path in paths or []:
            if not isinstance(path, str) or path in known:
                continue
            if os.path.realpath(path) in borrowed:
                continue
            staged.append(path)
            known.add(path)

    @staticmethod
    def _group_by_object(ds, paths: List[str]) -> Dict[str, List[str]]:
        """Map each logical object to the parquet parts that hold it.

        A source that records the mapping while writing wins; otherwise it is
        recovered from the part file names, which encode the object they came
        from.
        """
        recorded = getattr(ds, "object_paths", None)
        if isinstance(recorded, dict) and recorded:
            return {name: list(parts) for name, parts in recorded.items()}

        grouped: Dict[str, List[str]] = {}
        for path in paths or []:
            if not isinstance(path, str):
                continue
            grouped.setdefault(_object_key(path), []).append(path)
        return grouped

    def _objects(self, trigger: str) -> Dict[str, List[str]]:
        if trigger not in ("source", "target"):
            raise ValueError(
                f"trigger must be 'source' or 'target', got {trigger!r}"
            )
        objects = (
            self.objects_source if trigger == "source" else self.objects_target
        )
        if not objects:
            raise RuntimeError(
                f"No data loaded for trigger '{trigger}'. "
                f"Call load_data('{trigger}') first."
            )
        return objects

    def tables(self, trigger: str) -> List[str]:
        """Logical objects available for a trigger, in load order.

        For a single-table source this is a one-element list. For a database
        scanned with ``*`` it is one entry per table.
        """
        return list(self._objects(trigger).keys())

    def scan(
        self, trigger: str, table: Optional[str] = None
    ) -> "pl.LazyFrame":
        """Lazily scan every parquet part of one logical object.

        This is the only door through which a pack should reach data. Nothing is
        read here — the returned LazyFrame is a query plan, and the parts of a
        chunked object are seen by Polars as one dataset, so the streaming
        engine owns the cross-chunk state instead of the pack.

        Args:
            trigger: "source" or "target".
            table: which object to scan. May be omitted only when the trigger
                holds exactly one object; with several, omitting it raises
                rather than silently picking the first one.

        Example:
            ```python
            with Pack() as pack:
                pack.load_data("source")
                stats = analytics.agg(pack.scan("source"), {
                    "rows": pl.len(),
                    "nulls": pl.col("email").null_count(),
                })
            ```
        """
        objects = self._objects(trigger)

        if table is None:
            if len(objects) > 1:
                raise ValueError(
                    f"'{trigger}' holds {len(objects)} objects "
                    f"({', '.join(sorted(objects))}). Pass table= to choose "
                    f"one, or iterate over pack.tables('{trigger}')."
                )
            table = next(iter(objects))

        parts = objects.get(table)
        if not parts:
            raise KeyError(
                f"unknown object {table!r} for trigger '{trigger}'. "
                f"Available: {', '.join(sorted(objects))}"
            )
        return pl.scan_parquet(parts)

    def scan_all(self, trigger: str) -> Dict[str, "pl.LazyFrame"]:
        """One LazyFrame per logical object, keyed by object name."""
        return {
            name: pl.scan_parquet(parts)
            for name, parts in self._objects(trigger).items()
        }

    def schema(
        self, trigger: str, table: Optional[str] = None
    ) -> Dict[str, Any]:
        """Column names and dtypes, read from the parquet footers.

        No data page is touched, so this is effectively free whatever the
        dataset size — which is the whole of what schema_scanner_pack needs.
        """
        return dict(self.scan(trigger, table).collect_schema())

    def scan_data(self, trigger: str) -> "pl.LazyFrame":
        """Lazily scan every parquet part of every object for a trigger.

        Kept for packs written against the previous API. Prefer
        :meth:`scan`, which is explicit about which object it reads and cannot
        silently concatenate unrelated tables.
        """
        paths = self.paths_source if trigger == "source" else self.paths_target
        if not paths:
            raise RuntimeError(
                f"No data loaded for trigger '{trigger}'. "
                f"Call load_data('{trigger}') first."
            )
        return pl.scan_parquet(paths)

    def get_row_count(self, trigger: str, table: Optional[str] = None) -> int:
        """Exact row count, answered from parquet footers."""
        objects = (
            self.objects_source if trigger == "source" else self.objects_target
        )
        if not objects:
            return 0
        if table is None and len(objects) > 1:
            return sum(
                int(
                    pl.scan_parquet(parts)
                    .select(pl.len())
                    .collect(engine="streaming")
                    .item()
                )
                for parts in objects.values()
            )
        return int(
            self.scan(trigger, table)
            .select(pl.len())
            .collect(engine="streaming")
            .item()
        )


class ConfigLoader:
    """Utility class for loading configuration files."""

    @staticmethod
    def load_config(file_name):
        # logger = logging.getLogger("ConfigLoader")
        try:
            with open(file_name, "r", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError as e:
            # logger.warning(f"Configuration file not found: {file_name}")
            return {}


def _sanitize_for_json(obj):
    """Recursively convert objects to JSON-serializable forms.

    Handles pandas NA/NaT, numpy scalars/arrays, datetimes, Decimals, sets, and
    non-string dict keys.
    """
    # Fast-path for basic JSON types
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj

    # Floats: normalize non-finite values
    if isinstance(obj, float):
        if math.isfinite(obj):
            return obj
        return None

    # pandas NA / NaT without importing pandas.
    # Must be checked BEFORE the datetime branch: pd.NaT is an instance of
    # datetime.datetime, so the datetime branch would otherwise serialize a
    # missing date as the string "NaT" instead of null.
    tname = type(obj).__name__
    if tname in ("NAType", "NaTType"):
        return None

    # Datetime-like objects
    if isinstance(obj, (_dt.datetime, _dt.date, _dt.time)):
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)

    # Decimal -> float (or str if not finite)
    if isinstance(obj, Decimal):
        try:
            return float(obj)
        except Exception:
            return str(obj)

    # numpy scalars and arrays without importing numpy
    # Many numpy scalars have .item(); arrays often have .tolist()
    try:
        if hasattr(obj, "tolist"):
            return _sanitize_for_json(obj.tolist())
    except Exception:
        pass
    try:
        if hasattr(obj, "item"):
            return _sanitize_for_json(obj.item())
    except Exception:
        pass

    # Dict: ensure keys are strings and values sanitized
    if isinstance(obj, dict):
        sanitized = {}
        for k, v in obj.items():
            if isinstance(k, str):
                key = k
            else:
                try:
                    key = str(k)
                except Exception:
                    key = repr(k)
            sanitized[key] = _sanitize_for_json(v)
        return sanitized

    # Iterables (list/tuple/set)
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_for_json(x) for x in obj]

    # Fallback: best-effort string conversion
    try:
        return str(obj)
    except Exception:
        return None


class PlatformAsset:
    """
    A platform asset is a json formated data that can be pushed to the platform
    """

    def __init__(self, type):
        self.type = type
        self.data = []

    def save(self):
        # Writing data to metrics.json
        with open(self.type + ".json", "w", encoding="utf-8") as file:
            json.dump(_sanitize_for_json(self.data), file, indent=4)
