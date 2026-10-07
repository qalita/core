"""
# QALITA (c) COPYRIGHT 2025 - ALL RIGHTS RESERVED -

Ingestion: turn a configured source into Parquet parts on local disk.

This layer is the first thing that breaks on a large source, and it used to
break before Python saw a row: every SQL class called
``pandas.read_sql(sql, engine, chunksize=N)``, whose ``chunksize`` chunks
*client-side only* — psycopg2, pymysql and pymssql buffer the entire result set
in the driver before the first chunk is yielded. Everything here now goes
through :mod:`qalita_core.polars_io`, which asks for a server-side cursor where
the driver has one and streams Arrow batches straight to Parquet.

Three properties the rest of the system depends on:

* ``get_data()`` returns ``List[str]`` — the parquet parts, in order.
* ``object_paths`` maps each logical object (table, collection, index, file) to
  its parts. ``Pack`` prefers it over parsing part-file names, which is what
  makes ``zip(table_names, parquet_paths)`` — the idiom that silently dropped
  chunks 2..N — unnecessary at the root.
* ``skipped_objects`` lists what a multi-object scan could not read and why. A
  scan that silently returns half a schema is worse than one that fails, so a
  partial result always comes with the objects that are missing from it.
"""

import glob
import hashlib
import json
import logging
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import DBAPIError

from qalita_core import polars_io as pio

# Re-exported so a caller can catch the pre-flight disk refusal without
# reaching into polars_io.
from qalita_core.polars_io import (  # noqa: F401
    DEFAULT_CHUNK_ROWS,
    InsufficientDiskSpaceError,
)
from qalita_core.utils import slugify

import polars as pl

logger = logging.getLogger(__name__)

# Kept for callers that still test for it. Polars is a hard requirement of
# qalita_core now: there is no pandas path left to fall back to.
POLARS_AVAILABLE = True

DEFAULT_PORTS = {
    "5432": "postgresql",
    "3306": "mysql",
    "1433": "mssql+pymssql",  # Also used by Azure Synapse
    "1521": "oracle",
    "27017": "mongodb",
    "6379": "redis",
    "50000": "db2",
    "1434": "sybase",
    "3307": "mariadb",
    "5433": "greenplum",
    "5000": "sqlite",
    # Data warehouse ports
    "443": "snowflake",  # Snowflake uses HTTPS
    "5439": "redshift",
    "8123": "clickhouse",  # ClickHouse HTTP
    "9000": "clickhouse_native",  # ClickHouse native
    "8080": "trino",
    # Additional database ports
    "1025": "teradata",
    "30015": "sap_hana",
    "9042": "cassandra",
    "9200": "elasticsearch",
    "50001": "ibm_db2",
    "444": "athena",  # Athena uses HTTPS
}

# Object stores Polars reads natively. Anything else has to be staged through
# fsspec to a local file first.
_NATIVE_SCHEMES = (
    "s3://",
    "s3a://",
    "gs://",
    "gcs://",
    "az://",
    "abfs://",
    "abfss://",
    "http://",
    "https://",
)

_SQL_STARTERS = ("select", "with", "show", "describe", "pragma", "explain")


class DataSource(ABC):
    @abstractmethod
    def get_data(self, table_or_query=None, pack_config=None):
        """Return a list of parquet file paths for the requested data."""
        pass

    @property
    def object_paths(self) -> Dict[str, List[str]]:
        """Logical object name -> the parquet parts holding it.

        Recorded while writing, so the pairing between an object and its chunks
        is never reconstructed by position afterwards.
        """
        existing = getattr(self, "_object_paths", None)
        if existing is None:
            existing = {}
            self._object_paths = existing
        return existing

    @property
    def borrowed_paths(self) -> Set[str]:
        """Paths handed back as-is instead of staged: the user's own files.

        A parquet file is already the staging format, so it is returned
        without a copy. Those paths belong to the source, not to the pack run:
        ``Pack.cleanup`` must never delete them.
        """
        existing = getattr(self, "_borrowed_paths", None)
        if existing is None:
            existing = set()
            self._borrowed_paths = existing
        return existing

    @property
    def skipped_objects(self) -> List[Dict[str, str]]:
        """Objects skipped by the current multi-object scan."""
        existing = getattr(self, "_skipped_objects", None)
        if existing is None:
            existing = []
            self._skipped_objects = existing
        return existing

    @property
    def _base_names(self) -> Dict[str, str]:
        """Base name already handed out -> the raw object it was built for."""
        existing = getattr(self, "_base_name_owners", None)
        if existing is None:
            existing = {}
            self._base_name_owners = existing
        return existing

    def _object_base_name(
        self, source_type: str, object_identifier: str
    ) -> str:
        """The part-file prefix *and* ``object_paths`` key of one object.

        ``slugify`` folds case, accents and every run of separators into ``_``,
        so ``logs-2024.01`` and ``logs_2024_01`` — or ``Orders`` and ``orders``,
        or ``données`` and ``donnees`` — produce the same base name. Both the
        on-disk prefix and the object key derive from it, so a clash makes the
        second object truncate the first one's ``_part_1`` and concatenates the
        two path lists under one key: one object vanishes and the other is
        counted twice, silently.

        Disambiguation has to happen here, before the first part is opened —
        doing it when the paths are recorded is too late, the files have already
        overwritten each other. Only an actual clash is renamed, so the object
        names every pack and every stored metric already uses are untouched.
        """
        base = _build_base_name(source_type, object_identifier)
        identifier = str(object_identifier)
        registry = self._base_names

        candidate = base
        owner = registry.get(candidate)
        if owner is not None and owner != identifier:
            # A digest keeps the suffix stable across runs and independent of
            # the order the objects happen to be listed in. blake2s rather than
            # sha1: nothing here is security-sensitive, but a weak-hash warning
            # on every scan trains people to ignore the scanner.
            digest = hashlib.blake2s(
                identifier.encode("utf-8", "replace"), digest_size=4
            ).hexdigest()
            candidate = f"{base}_{digest}"
            index = 1
            while registry.get(candidate, identifier) != identifier:
                index += 1
                candidate = f"{base}_{digest}_{index}"
            logger.warning(
                "%r and %r have the same normalized name %r; %r is exported "
                "as %r so the two objects do not overwrite each other.",
                owner,
                identifier,
                base,
                identifier,
                candidate,
            )

        registry[candidate] = identifier
        return candidate

    def _record_object(
        self, name: str, paths: List[str], staged: bool = True
    ) -> List[str]:
        """Record an object's parts; ``staged=False`` marks borrowed paths."""
        if paths:
            self.object_paths.setdefault(name, []).extend(paths)
            if not staged:
                self.borrowed_paths.update(paths)
        return paths


# -----------------------------
# Helper utilities for Parquet
# -----------------------------


def _ensure_output_dir(pack_config: Optional[dict]) -> str:
    base_dir = (pack_config or {}).get("parquet_output_dir") or "./parquet"
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    return str(base_dir)


def _build_base_name(source_type: str, object_identifier: str) -> str:
    normalized_source = slugify(source_type or "source")
    normalized_object = slugify(object_identifier or "data")
    return f"{normalized_source}_{normalized_object}"


def _build_parquet_path(
    output_dir: str, base_name: str, part_index: int
) -> str:
    return pio.part_path(output_dir, base_name, part_index)


def _chunk_rows(pack_config: Optional[dict]) -> int:
    return int((pack_config or {}).get("chunk_rows") or DEFAULT_CHUNK_ROWS)


def _fetch_rows(pack_config: Optional[dict]) -> Optional[int]:
    """Rows a driver is asked for at a time, independent of the part size."""
    value = (pack_config or {}).get("fetch_rows")
    return int(value) if value else None


def _min_free_bytes(pack_config: Optional[dict]) -> int:
    return int(
        (pack_config or {}).get("min_free_disk_bytes")
        or pio.DEFAULT_MIN_FREE_BYTES
    )


def cleanup_parquet_files(paths: List[str], logger=None) -> int:
    """
    Remove parquet files from the filesystem.

    Args:
        paths: List of parquet file paths to remove.
        logger: Optional logger for debug/error messages.

    Returns:
        Number of files successfully removed.
    """
    removed_count = 0
    for path in paths or []:
        try:
            if os.path.exists(path):
                os.remove(path)
                removed_count += 1
                if logger:
                    logger.debug(f"Removed temporary parquet file: {path}")
        except OSError as e:
            if logger:
                logger.warning(f"Failed to remove parquet file {path}: {e}")
    return removed_count


def _is_sql_query(candidate: str) -> bool:
    """Heuristic: SQL statement rather than a bare object name."""
    sql = candidate.strip().lower()
    if ";" in sql or "\n" in sql:
        return True
    return any(sql.startswith(token) for token in _SQL_STARTERS)


def _column_type_hints(
    engine, table: Optional[str], schema: Optional[str]
) -> Dict[str, Any]:
    """Arrow types for a table, from the catalog.

    Only used to type columns that are entirely NULL in the first batch. Reading
    them from the catalog is what keeps ``*_part_1`` and ``*_part_7`` of the same
    object identically typed when the nulls happen to be at the front.
    """
    if not table:
        return {}
    try:
        return pio.arrow_type_hints(
            inspect(engine).get_columns(table, schema=schema)
        )
    except Exception:  # noqa: BLE001 - catalogs are best effort
        return {}


def _sql_to_parquet(
    source: DataSource,
    engine,
    sql: str,
    output_dir: str,
    base_name: str,
    chunk_rows: int,
    *,
    table: Optional[str] = None,
    schema: Optional[str] = None,
    pack_config: Optional[dict] = None,
) -> List[str]:
    """Stream one SQL result set to parquet parts and record the object."""
    paths = pio.stream_sql_to_parquet(
        engine,
        sql,
        output_dir,
        base_name,
        chunk_rows=chunk_rows,
        fetch_rows=_fetch_rows(pack_config),
        type_hints=_column_type_hints(engine, table, schema),
        min_free_bytes=_min_free_bytes(pack_config),
    )
    return source._record_object(base_name, paths)


def _sink_to_parquet(
    source: DataSource,
    lf: "pl.LazyFrame",
    output_dir: str,
    base_name: str,
    chunk_rows: int,
    *,
    pack_config: Optional[dict] = None,
) -> List[str]:
    """Sink a LazyFrame to parquet parts and record the object."""
    paths = pio.sink_parts(
        lf,
        output_dir,
        base_name,
        chunk_rows=chunk_rows,
        min_free_bytes=_min_free_bytes(pack_config),
    )
    return source._record_object(base_name, paths)


def _is_skippable_object_error(exc: Exception, dialect: Optional[str]) -> bool:
    """Whether *exc* is a structured, object-level permission refusal.

    SQLAlchemy wraps driver failures in ``DBAPIError`` and preserves the
    driver's structured status on ``orig``. SQLSTATE 42501 is the standard
    insufficient-privilege signal used by PostgreSQL and other drivers. No
    exception message is parsed: unknown SQL states and non-database failures
    remain fatal.
    """
    if not isinstance(exc, DBAPIError) or exc.connection_invalidated:
        return False

    original = exc.orig
    sqlstate = getattr(original, "sqlstate", None) or getattr(
        original, "pgcode", None
    )

    # redshift-connector exposes PostgreSQL protocol fields as a mapping in
    # args[0], rather than as attributes on the exception.
    dialect_name = (dialect or "").lower().split("+", 1)[0]
    if sqlstate is None and dialect_name == "redshift" and original.args:
        fields = original.args[0]
        if isinstance(fields, dict):
            sqlstate = fields.get("C")

    return str(sqlstate).upper() == "42501"


class _SqlAlchemySource(DataSource):
    """Shared table/query dispatch for every SQLAlchemy-backed source.

    Each warehouse used to carry its own copy of this ~60-line block; they had
    already drifted (different quoting, different naming, one missing the query
    branch entirely) while sharing the same ``pd.read_sql(chunksize=...)`` bug.
    """

    dialect_name = "db"

    def _qualify(
        self, table_name: str, schema: Optional[str]
    ) -> tuple[str, str]:
        """Return ``(sql_identifier, display_name)`` for a table."""
        if schema:
            return f"{schema}.{table_name}", f"{schema}.{table_name}"
        return table_name, table_name

    def _list_tables(self, engine, schema: Optional[str]) -> List[str]:
        return list(inspect(engine).get_table_names(schema=schema) or [])

    def _is_sql_query(self, s: str) -> bool:
        return _is_sql_query(s)

    def _read_tables(
        self,
        engine,
        table_names,
        schema,
        output_dir,
        chunk_rows,
        dialect_name=None,
        pack_config=None,
    ) -> List[str]:
        """Read each table, skipping the ones the connection cannot read.

        A ``"*"`` scan covers every table *and view* of the schema, technical
        objects included, so it used to require ``SELECT`` on the whole schema:
        the first refusal aborted the job after it had already streamed every
        readable table. A refused table is a property of that table, not of the
        run, so it is skipped.

        Skipping silently would be worse than the abort it replaces — a scan
        missing half its tables would look like a clean one — hence the
        per-object warning, the end-of-scan recap, and the refusal to return an
        empty scan as a success.
        """
        self._skipped_objects = []
        table_names = list(table_names)
        if not table_names:
            raise ValueError("No objects were selected for this scan.")

        all_paths: List[str] = []
        read_count = 0
        skipped: List[Dict[str, str]] = []
        first_error: Optional[BaseException] = None

        for table_name in table_names:
            try:
                all_paths.extend(
                    self._read_table_to_parquet(
                        engine,
                        table_name,
                        schema,
                        output_dir,
                        chunk_rows,
                        dialect_name,
                        pack_config=pack_config,
                    )
                )
            except InsufficientDiskSpaceError:
                # Not a property of this table: every remaining one would fail
                # the same way, and burying the cause under N identical
                # warnings would turn a full disk into a partial scan.
                raise
            except Exception as exc:  # noqa: BLE001 - classify driver errors
                if not _is_skippable_object_error(exc, dialect_name):
                    raise
                # Driver messages carry the offending SQL on their own lines;
                # folded into one so a skip stays a single greppable log line.
                reason = " ".join(str(exc).split()) or exc.__class__.__name__
                skipped.append(
                    {
                        "object": str(table_name),
                        "error": exc.__class__.__name__,
                        "reason": reason,
                    }
                )
                if first_error is None:
                    first_error = exc
                logger.warning(
                    "skipping %s: it could not be read (%s: %s)",
                    table_name,
                    exc.__class__.__name__,
                    reason,
                )
            else:
                # Counted rather than derived from all_paths: an empty table
                # was read successfully and yields no part.
                read_count += 1

        if skipped:
            self.skipped_objects.extend(skipped)
            if not read_count:
                # Every object failing is not a permission story any more — a
                # dropped connection would otherwise surface as an empty scan.
                details = "; ".join(
                    f"{item['object']} ({item['error']}: {item['reason']})"
                    for item in skipped
                )
                raise RuntimeError(
                    f"None of the {len(skipped)} objects in the scan could be "
                    f"read: {details}"
                ) from first_error
            logger.warning(
                "%d of %d objects were skipped and are absent from this scan: %s",
                len(skipped),
                len(skipped) + read_count,
                ", ".join(item["object"] for item in skipped),
            )

        return all_paths

    def _load_data(
        self,
        engine,
        table_or_query,
        schema,
        output_dir,
        chunk_rows,
        dialect_name=None,
        pack_config=None,
    ) -> List[str]:
        dialect_name = dialect_name or self.dialect_name

        if table_or_query is None or (
            isinstance(table_or_query, str) and table_or_query.strip() == "*"
        ):
            tables = self._list_tables(engine, schema)
        elif isinstance(table_or_query, (list, tuple, set)):
            tables = list(table_or_query)
        elif isinstance(table_or_query, str):
            if self._is_sql_query(table_or_query):
                return _sql_to_parquet(
                    self,
                    engine,
                    table_or_query,
                    output_dir,
                    self._object_base_name(dialect_name, "query"),
                    chunk_rows,
                    pack_config=pack_config,
                )
            # One explicitly named table: there is nothing to fall back on, so
            # the driver's own error is more useful than a scan summary.
            return self._read_table_to_parquet(
                engine,
                table_or_query,
                schema,
                output_dir,
                chunk_rows,
                dialect_name,
                pack_config=pack_config,
            )
        else:
            raise TypeError(
                "table_or_query must be None, '*', a string, or a list of "
                "table names."
            )

        return self._read_tables(
            engine,
            tables,
            schema,
            output_dir,
            chunk_rows,
            dialect_name,
            pack_config=pack_config,
        )

    def _read_table_to_parquet(
        self,
        engine,
        table_name,
        schema,
        output_dir,
        chunk_rows,
        dialect_name=None,
        pack_config=None,
    ) -> List[str]:
        qualified, display = self._qualify(table_name, schema)
        return _sql_to_parquet(
            self,
            engine,
            f"SELECT * FROM {qualified}",
            output_dir,
            self._object_base_name(dialect_name or self.dialect_name, display),
            chunk_rows,
            table=table_name,
            schema=schema,
            pack_config=pack_config,
        )


# -----------------------------
# Remote object stores
# -----------------------------


def _stage_remote_file(path: str, storage_options: Optional[dict]) -> str:
    """Copy a remote file to local disk in bounded blocks.

    Used for the formats Polars cannot scan in place (Excel, whole-document
    JSON) and for schemes its object store does not speak (HDFS). The copy is
    streamed, so the file never exists in memory.
    """
    try:
        import fsspec
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise ImportError(
            f"reading {path} requires fsspec (and the filesystem package for "
            f"its scheme). Install with: pip install fsspec"
        ) from exc

    local_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "qalita-staging"
    local_dir.mkdir(parents=True, exist_ok=True)

    # The basename alone collides: s3://a/data.csv and s3://b/data.csv stage to
    # the same file, and the second source silently analyses the first one's
    # bytes. The digest of the full path disambiguates without making the name
    # unreadable. blake2s, not sha1: a weak-hash warning on every staged file
    # trains people to ignore the scanner.
    base = os.path.basename(path.rstrip("/")) or "object"
    digest = hashlib.blake2s(path.encode("utf-8"), digest_size=4).hexdigest()
    local_path = local_dir / f"{digest}-{base}"

    opened = fsspec.open(path, "rb", **(storage_options or {}))
    with opened as remote:
        # Ask the remote how big it is before writing a byte. Without this a
        # 100 GiB object fills the staging volume and the failure surfaces as
        # ENOSPC halfway through, leaving a truncated file behind that looks
        # like a complete one.
        size = None
        try:
            size = opened.fs.size(path)
        except Exception:  # noqa: BLE001 - not every filesystem reports size
            logger.debug("could not determine remote size of %s", path)
        pio.check_disk_space(str(local_dir), size)

        try:
            with open(local_path, "wb") as local:
                shutil.copyfileobj(remote, local, length=8 * 1024 * 1024)
        except BaseException:
            # A partial copy is worse than none: it is a valid-looking file the
            # next run would happily scan.
            local_path.unlink(missing_ok=True)
            raise
    return str(local_path)


def _is_native_remote(path: str) -> bool:
    return path.lower().startswith(_NATIVE_SCHEMES)


def _infer_format_from_path(
    path: str, explicit_format: Optional[str] = None
) -> str:
    if explicit_format:
        return explicit_format.lower()
    lower = path.lower()
    if lower.endswith(".csv"):
        return "csv"
    if lower.endswith(".json"):
        return "json"
    if lower.endswith(".parquet") or lower.endswith(".pq"):
        return "parquet"
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        return "excel"
    return "csv"


# -----------------------------
# Files
# -----------------------------


def _excel_to_parquet(
    file_path: str,
    output_dir: str,
    base_name: str,
    *,
    skip_rows: int,
    chunk_rows: int,
    fetch_rows: Optional[int] = None,
    min_free_bytes: int = pio.DEFAULT_MIN_FREE_BYTES,
) -> List[str]:
    """Stream an xlsx sheet to parquet through openpyxl's read-only iterator.

    Neither Polars nor pandas streams Excel — both build the entire sheet first
    — so this is the only Excel path. The schema is fixed from the first batch
    and every later batch is cast to it.
    """
    headers, rows = pio.iter_excel_rows(file_path, skip_rows=skip_rows)
    if not headers:
        return []
    return pio.write_row_batches(
        rows,
        headers,
        output_dir,
        base_name,
        chunk_rows=chunk_rows,
        fetch_rows=fetch_rows,
        min_free_bytes=min_free_bytes,
    )


def _json_to_parquet(
    source: DataSource,
    file_path: str,
    output_dir: str,
    base_name: str,
    *,
    chunk_rows: int,
    pack_config: Optional[dict],
) -> List[str]:
    """Load a .json file, whatever shape it is in.

    The format is sniffed rather than taken from a ``json_lines`` flag that
    defaults to False: that default sent every NDJSON file through
    ``read_json``, which parses the whole document.
    """
    explicit = (pack_config or {}).get("json_lines")
    if explicit is None:
        shape = pio.sniff_json_format(file_path)
    else:
        shape = "ndjson" if explicit else "array"

    if shape == "ndjson":
        return _sink_to_parquet(
            source,
            pio.scan_ndjson(file_path),
            output_dir,
            base_name,
            chunk_rows,
            pack_config=pack_config,
        )

    # A top-level array is streamed element by element; only the batch being
    # written is alive at any point.
    paths = pio.write_dict_rows(
        _as_documents(pio.iter_json_array(file_path)),
        output_dir,
        base_name,
        fetch_rows=_fetch_rows(pack_config),
        chunk_rows=chunk_rows,
        min_free_bytes=_min_free_bytes(pack_config),
    )
    return source._record_object(base_name, paths)


def _as_documents(values: Iterable[Any]) -> Iterator[Dict[str, Any]]:
    """Normalize JSON array elements to one record per row."""
    for value in values:
        if isinstance(value, dict):
            yield value
        else:
            yield {"value": value}


def _csv_read_options(config: Optional[dict]) -> Dict[str, Any]:
    """Translate a file source config into ``pio.scan_csv`` keyword options.

    The keys are the ones the platform stores on a CSV source and the CLI
    worker reads for its own preview (qalita/cli#143): the two readers have to
    agree, or the pack profiles a file the user never saw in the Studio.

    Every unusable value raises instead of falling back to the default. A
    silent fallback is what makes this class of bug expensive: reading a ``;``
    file as comma-separated does not fail, it yields one column and a green job
    whose metrics describe nothing.
    """
    config = config or {}
    options: Dict[str, Any] = {}

    delimiter = config.get("delimiter")
    if delimiter not in (None, ""):
        # The form field is free text, so a tabulation is typed as the two
        # characters ``\t`` far more often than as a real tabulation.
        separator = "\t" if delimiter == "\\t" else str(delimiter)
        if len(separator) != 1:
            raise ValueError(
                f"Invalid CSV delimiter {delimiter!r}: a separator must be "
                "exactly one character (write a tabulation as '\\t')."
            )
        options["separator"] = separator

    encoding = config.get("encoding")
    if encoding not in (None, ""):
        if str(encoding).lower().replace("_", "-") not in pio.UTF8_ENCODINGS:
            raise ValueError(
                f"Unsupported CSV encoding {encoding!r}: the loader only reads "
                "the utf-8 family. Convert the file to UTF-8 and declare "
                "'utf-8' on the source."
            )
        options["encoding"] = "utf8"

    has_header = config.get("has_header")
    if has_header is not None:
        options["has_header"] = _config_flag(has_header, default=True)

    decimal_separator = config.get("decimal_separator")
    if decimal_separator not in (None, ""):
        if decimal_separator not in (".", ","):
            raise ValueError(
                f"Invalid CSV decimal separator {decimal_separator!r}: "
                "expected '.' or ','."
            )
        options["decimal_comma"] = decimal_separator == ","
    elif config.get("decimal_comma") is not None:
        options["decimal_comma"] = _config_flag(
            config["decimal_comma"], default=False
        )

    return options


def _config_flag(value: Any, default: bool) -> bool:
    """Parse a platform flag using the CLI's legacy-compatible semantics."""
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in ("", "false", "0", "no")
    return bool(value)


class FileSource(DataSource):
    def __init__(self, file_path, config=None):
        self.file_path = file_path
        # Optional so that the dozens of ``FileSource(path)`` call sites keep
        # working; only the CSV reader looks at it.
        self.config = config or {}

    def get_data(self, table_or_query=None, pack_config=None):
        output_dir = _ensure_output_dir(pack_config)
        if os.path.isfile(self.file_path):
            return self._load_file(self.file_path, pack_config, output_dir)
        if os.path.isdir(self.file_path):
            data_files = glob.glob(
                os.path.join(self.file_path, "*.csv")
            ) + glob.glob(os.path.join(self.file_path, "*.xlsx"))
            if not data_files:
                raise FileNotFoundError(
                    "No CSV or XLSX files found in the provided path."
                )
            return self._load_file(data_files[0], pack_config, output_dir)
        raise FileNotFoundError(
            f"The path {self.file_path} is neither a file nor a directory, or it can't be reached."
        )

    def _load_file(
        self,
        file_path,
        pack_config,
        output_dir: str,
        identifier: Optional[str] = None,
    ) -> List[str]:
        """Stage one file; ``identifier`` names the object (default: stem)."""
        skiprows = 0
        chunk_rows = _chunk_rows(pack_config)
        if pack_config:
            skiprows = (
                pack_config.get("job", {}).get("source", {}).get("skiprows", 0)
            )

        base_name = self._object_base_name(
            "file",
            identifier or os.path.splitext(os.path.basename(file_path))[0],
        )
        lower = file_path.lower()

        # Parquet is already the staging format: no copy, no conversion.
        if lower.endswith((".parquet", ".pq")):
            return self._record_object(base_name, [file_path], staged=False)

        # Text formats stage to roughly a fifth to a half of their own size
        # once zstd is applied; 0.6 keeps the guard on the safe side of that
        # without refusing loads that would in fact have fit.
        try:
            estimate = int(os.path.getsize(file_path) * 0.6)
        except OSError:
            estimate = None
        pio.check_disk_space(
            output_dir,
            estimate,
            min_free_bytes=_min_free_bytes(pack_config),
        )

        if lower.endswith(".csv"):
            return _sink_to_parquet(
                self,
                pio.scan_csv(
                    file_path,
                    skip_rows=int(skiprows),
                    **_csv_read_options(self.config),
                ),
                output_dir,
                base_name,
                chunk_rows,
                pack_config=pack_config,
            )

        if lower.endswith((".xlsx", ".xlsm")):
            paths = _excel_to_parquet(
                file_path,
                output_dir,
                base_name,
                skip_rows=int(skiprows),
                chunk_rows=chunk_rows,
                fetch_rows=_fetch_rows(pack_config),
                min_free_bytes=_min_free_bytes(pack_config),
            )
            return self._record_object(base_name, paths)

        if lower.endswith((".ndjson", ".jsonl")):
            return _sink_to_parquet(
                self,
                pio.scan_ndjson(file_path),
                output_dir,
                base_name,
                chunk_rows,
                pack_config=pack_config,
            )

        if lower.endswith(".json"):
            return _json_to_parquet(
                self,
                file_path,
                output_dir,
                base_name,
                chunk_rows=chunk_rows,
                pack_config=pack_config,
            )

        raise ValueError(
            f"Unsupported file extension or missing 'skiprows' for file: {file_path}"
        )


class DatabaseSource(_SqlAlchemySource):
    def __init__(self, connection_string=None, config=None):
        # Keep the config available for schema preference and other options
        self.config = config or {}

        if connection_string:
            self.engine = create_engine(connection_string)
        elif config:
            db_type = config.get("type") or DEFAULT_PORTS.get(
                str(config.get("port")), "unknown"
            )
            if db_type == "unknown":
                raise ValueError(
                    f"Unsupported or unknown database port: {config.get('port')}"
                )
            elif db_type == "oracle":
                db_type = "oracle+oracledb"
                # Use URL.create to avoid password-in-URL pattern detection
                url = URL.create(
                    drivername=db_type,
                    username=config["username"],
                    password=config["password"],
                    host=config["host"],
                    port=config["port"],
                    query={"service_name": config["database"]},
                )
                self.engine = create_engine(url)
            elif db_type.startswith("sqlite"):
                database_path = config.get("database") or ":memory:"
                if database_path == ":memory:":
                    url = URL.create(drivername="sqlite", database=":memory:")
                else:
                    # Accept absolute or relative filesystem path
                    url = URL.create(
                        drivername="sqlite", database=database_path
                    )
                self.engine = create_engine(url)
            else:
                # Use URL.create to avoid password-in-URL pattern detection
                url = URL.create(
                    drivername=db_type,
                    username=config["username"],
                    password=config["password"],
                    host=config["host"],
                    port=config["port"],
                    database=config["database"],
                )
                self.engine = create_engine(url)
        else:
            raise ValueError(
                "DatabaseSource requires a connection_string or a config dict."
            )

    def get_data(self, table_or_query=None, pack_config=None):
        """
        Load data from the database and write Parquet chunks.

        - If table_or_query is a string table name: returns list of parquet paths for that table
        - If table_or_query is a SQL query string: returns list of parquet paths for the query result
        - If table_or_query is a list/tuple/set of table names: returns parquet paths for each table
        - If table_or_query is '*' or None: scan all tables and return parquet paths for each

        When several tables are read, one the connection cannot read is
        skipped and recorded in ``skipped_objects`` rather than aborting the
        scan; see :meth:`_SqlAlchemySource._read_tables`.
        """

        # Determine schema: prefer source config over pack config; both are optional
        schema = None
        cfg_schema = None
        try:
            cfg_schema = (self.config or {}).get("schema")
        except Exception:
            cfg_schema = None
        if cfg_schema:
            schema = cfg_schema
        elif pack_config:
            schema = pack_config.get("job", {}).get("source", {}).get("schema")

        output_dir = _ensure_output_dir(pack_config)
        chunk_rows = _chunk_rows(pack_config)
        try:
            dialect_name = self.engine.dialect.name
        except Exception:
            dialect_name = None

        # Default behavior: scan all tables
        if table_or_query is None or (
            isinstance(table_or_query, str) and table_or_query.strip() == "*"
        ):
            table_names = self._get_all_table_names(schema)
            if not table_names:
                raise ValueError(
                    "No tables found in the database for the given schema."
                )
            return self._read_tables(
                self.engine,
                table_names,
                schema,
                output_dir,
                chunk_rows,
                dialect_name,
                pack_config=pack_config,
            )

        # If a list/tuple/set of table names is provided
        if isinstance(table_or_query, (list, tuple, set)):
            return self._read_tables(
                self.engine,
                table_or_query,
                schema,
                output_dir,
                chunk_rows,
                dialect_name,
                pack_config=pack_config,
            )

        # If a single string is provided, determine if it's a table name or SQL query
        if isinstance(table_or_query, str):
            if self._is_sql_query(table_or_query):
                return _sql_to_parquet(
                    self,
                    self.engine,
                    table_or_query,
                    output_dir,
                    self._object_base_name(dialect_name or "db", "query"),
                    chunk_rows,
                    pack_config=pack_config,
                )
            return self._read_table_to_parquet(
                self.engine,
                table_or_query,
                schema,
                output_dir,
                chunk_rows,
                dialect_name,
                pack_config=pack_config,
            )

        raise TypeError(
            "table_or_query must be None, '*', a string (table name or SQL), or a list/tuple/set of table names."
        )

    def _qualify(self, table_name, schema):
        # A fully-qualified "SCHEMA.TABLE" is accepted when no schema was given.
        effective_schema = schema
        effective_table = table_name
        if not schema and "." in table_name:
            effective_schema, effective_table = table_name.split(".", 1)
        qualified = (
            f"{effective_schema}.{effective_table}"
            if effective_schema
            else effective_table
        )
        return qualified, qualified

    def _read_table_to_parquet(
        self,
        engine,
        table_name,
        schema,
        output_dir,
        chunk_rows,
        dialect_name=None,
        pack_config=None,
    ) -> List[str]:
        effective_schema = schema
        effective_table = table_name
        if not schema and "." in table_name:
            effective_schema, effective_table = table_name.split(".", 1)

        qualified, display = self._qualify(table_name, schema)
        return _sql_to_parquet(
            self,
            engine,
            f"SELECT * FROM {qualified}",
            output_dir,
            self._object_base_name(dialect_name or "db", display),
            chunk_rows,
            table=effective_table,
            schema=effective_schema,
            pack_config=pack_config,
        )

    def _get_all_table_names(self, schema: Optional[str] = None) -> List[str]:
        """Return all table names (and views) in the database for the given schema, sorted alphabetically.
        For Oracle and PostgreSQL, if no schema is provided and none are found in the default schema,
        iterate over accessible schemas and return fully-qualified names ("SCHEMA.TABLE").
        """
        inspector = inspect(self.engine)

        def _collect_for_schema(target_schema: Optional[str]) -> List[str]:
            try:
                tables = inspector.get_table_names(schema=target_schema)
            except NotImplementedError:
                tables = []
            try:
                views = inspector.get_view_names(schema=target_schema)
            except NotImplementedError:
                views = []
            return list(set((tables or []) + (views or [])))

        # First pass: use the provided schema (or None)
        initial = sorted(_collect_for_schema(schema))
        if initial:
            return initial

        # Special handling for Oracle when no schema specified and nothing found
        dialect_name = self.engine.dialect.name

        if dialect_name == "oracle" and schema is None:
            try:
                schemas = inspector.get_schema_names()
            except NotImplementedError:
                schemas = []

            system_schemas = {
                "SYS",
                "SYSTEM",
                "OUTLN",
                "XDB",
                "MDSYS",
                "CTXSYS",
                "ORDSYS",
                "ORDDATA",
                "DBSNMP",
                "APPQOSSYS",
                "WMSYS",
                "OLAPSYS",
                "LBACSYS",
                "GSMADMIN_INTERNAL",
                "OJVMSYS",
                "DVF",
                "DVSYS",
                "REMOTE_SCHEDULER_AGENT",
                "SYS$UMF",
                "GGSYS",
                "AUDSYS",
                "ANONYMOUS",
            }

            qualified: List[str] = []
            for sch in schemas or []:
                if not sch or sch.upper() in system_schemas:
                    continue
                names = _collect_for_schema(sch)
                for n in names:
                    qualified.append(f"{sch}.{n}")

            qualified = sorted(set(qualified))
            if qualified:
                return qualified

            # Final fallback: try CURRENT_SCHEMA if available
            try:
                with self.engine.connect() as conn:
                    result = conn.execute(
                        text(
                            "SELECT sys_context('USERENV','CURRENT_SCHEMA') FROM dual"
                        )
                    )
                    row = result.fetchone()
                    current_schema = row[0] if row and row[0] else None
                if current_schema:
                    names = _collect_for_schema(current_schema)
                    if names:
                        return sorted([f"{current_schema}.{n}" for n in names])
            except NotImplementedError:
                pass

        # Special handling for PostgreSQL when no schema specified and nothing found
        if dialect_name == "postgresql" and schema is None:
            try:
                schemas = inspector.get_schema_names()
            except NotImplementedError:
                schemas = []

            # Filter out system schemas
            system_prefixes = ("pg_temp_", "pg_toast_temp_")
            system_schemas = {
                "information_schema",
                "pg_catalog",
                "pg_toast",
            }

            qualified: List[str] = []
            for sch in schemas or []:
                if not sch:
                    continue
                if sch in system_schemas or any(
                    sch.startswith(pfx) for pfx in system_prefixes
                ):
                    continue
                names = _collect_for_schema(sch)
                for n in names:
                    qualified.append(f"{sch}.{n}")

            qualified = sorted(set(qualified))
            if qualified:
                return qualified

        # Return whatever was found initially (likely empty)
        return initial


def _materialize_remote_to_parquet(
    source: DataSource,
    path: str,
    fmt: str,
    storage_options: Optional[dict],
    pack_config: Optional[dict],
) -> List[str]:
    """Stage a remote object to local parquet parts.

    Remote Parquet is passed through untouched — the one zero-copy path in the
    layer — but only when no credentials are needed: ``get_data`` returns bare
    paths, so credentials handed to this function could not travel with them and
    the pack would scan the bucket anonymously. With credentials, the object is
    staged locally instead of silently failing later.
    """
    output_dir = _ensure_output_dir(pack_config)
    chunk_rows = _chunk_rows(pack_config)
    min_free = _min_free_bytes(pack_config)
    skiprows = 0
    if pack_config:
        skiprows = (
            pack_config.get("job", {}).get("source", {}).get("skiprows", 0)
        )

    base_name = source._object_base_name(
        "remote", os.path.splitext(os.path.basename(path))[0]
    )

    if fmt == "parquet" and not storage_options:
        return source._record_object(base_name, [path], staged=False)

    # Polars scans s3/gs/abfs/http in place; anything else (HDFS above all) has
    # to come through fsspec, which streams it to a local file first.
    native = _is_native_remote(path) or os.path.exists(path)

    if fmt == "parquet":
        return _sink_to_parquet(
            source,
            pio.scan_parquet(path, storage_options=storage_options),
            output_dir,
            base_name,
            chunk_rows,
            pack_config=pack_config,
        )

    if fmt == "csv":
        # Object stores hold the same European CSVs as local disks do, and the
        # source config carries the same keys, so the reader options apply here
        # too.
        csv_options = _csv_read_options(getattr(source, "config", None))
        if native:
            lf = pio.scan_csv(
                path,
                skip_rows=int(skiprows),
                storage_options=storage_options,
                **csv_options,
            )
        else:
            lf = pio.scan_csv(
                _stage_remote_file(path, storage_options),
                skip_rows=int(skiprows),
                **csv_options,
            )
        return _sink_to_parquet(
            source,
            lf,
            output_dir,
            base_name,
            chunk_rows,
            pack_config=pack_config,
        )

    if fmt == "json":
        # The shape can only be sniffed on a seekable file, and openpyxl-style
        # local access is needed for the array case anyway.
        local_path = (
            path
            if os.path.exists(path)
            else _stage_remote_file(path, storage_options)
        )
        return _json_to_parquet(
            source,
            local_path,
            output_dir,
            base_name,
            chunk_rows=chunk_rows,
            pack_config=pack_config,
        )

    if fmt == "excel":
        local_path = (
            path
            if os.path.exists(path)
            else _stage_remote_file(path, storage_options)
        )
        paths = _excel_to_parquet(
            local_path,
            output_dir,
            base_name,
            skip_rows=int(skiprows),
            chunk_rows=chunk_rows,
            fetch_rows=_fetch_rows(pack_config),
            min_free_bytes=min_free,
        )
        return source._record_object(base_name, paths)

    raise ValueError(f"Unsupported remote format {fmt!r} for {path}")


def _first_of(config: dict, *keys: str) -> Any:
    for key in keys:
        if config.get(key) not in (None, ""):
            return config[key]
    return None


class S3Source(DataSource):
    def __init__(self, config):
        self.config = config or {}

    def _object_key_is_the_path(self) -> bool:
        """Whether ``config['key']`` names the object rather than the account.

        ``key`` means two different things in the configs this class accepts:
        fsspec spells the access key ``key``, and :meth:`get_data` builds
        ``s3://{bucket}/{key}`` when no explicit ``path`` is given. Reading it
        as a credential in that second case sent the object's own name to S3 as
        ``aws_access_key_id`` — an authentication failure on every private
        bucket configured with bucket+key, reported as a credentials problem
        the user could not find because the credentials were correct.
        """
        return not self.config.get("path") and bool(self.config.get("bucket"))

    def _storage_options(self) -> Optional[dict]:
        """Credentials in the naming Polars' object store expects.

        The previous implementation built fsspec-style options and then dropped
        them entirely on the Parquet path, so private buckets failed on the one
        path that was supposed to be free.
        """
        config = self.config
        client_kwargs = config.get("client_kwargs") or {}
        access_key_aliases = ["access_key", "aws_access_key_id"]
        if not self._object_key_is_the_path():
            access_key_aliases.insert(0, "key")
        options = {
            "aws_access_key_id": _first_of(config, *access_key_aliases),
            "aws_secret_access_key": _first_of(
                config, "secret", "secret_key", "aws_secret_access_key"
            ),
            "aws_session_token": _first_of(
                config, "token", "aws_session_token"
            ),
            "aws_region": _first_of(config, "region", "region_name")
            or client_kwargs.get("region_name"),
            "aws_endpoint_url": config.get("endpoint_url")
            or client_kwargs.get("endpoint_url"),
        }
        options = {k: v for k, v in options.items() if v}
        return options or None

    def get_data(self, table_or_query=None, pack_config=None):
        # Expect either a full s3 path in config['path'] or bucket/key
        path = self.config.get("path")
        if not path:
            bucket = self.config.get("bucket")
            key = self.config.get("key")
            if bucket and key:
                path = f"s3://{bucket}/{key}"
        if not path:
            raise ValueError(
                "S3Source requires either 'path' or 'bucket'+'key' in config."
            )

        fmt = _infer_format_from_path(path, self.config.get("format"))
        return _materialize_remote_to_parquet(
            self, path, fmt, self._storage_options(), pack_config
        )


class GCSSource(DataSource):
    def __init__(self, config):
        self.config = config or {}

    def _storage_options(self) -> Optional[dict]:
        config = self.config
        options: Dict[str, Any] = {}
        token = config.get("token")
        if isinstance(token, dict):
            options["google_service_account_key"] = json.dumps(token)
        elif isinstance(token, str):
            options["google_service_account"] = token
        if config.get("service_account_path"):
            options["google_service_account"] = config["service_account_path"]
        return options or None

    def get_data(self, table_or_query=None, pack_config=None):
        # Expect gs:// style path or bucket/object
        path = self.config.get("path")
        if not path:
            bucket = self.config.get("bucket")
            blob = self.config.get("blob") or self.config.get("key")
            if bucket and blob:
                path = f"gs://{bucket}/{blob}"
        if not path:
            raise ValueError(
                "GCSSource requires either 'path' or 'bucket'+'blob' in config."
            )

        fmt = _infer_format_from_path(path, self.config.get("format"))
        return _materialize_remote_to_parquet(
            self, path, fmt, self._storage_options(), pack_config
        )


class AzureBlobSource(DataSource):
    def __init__(self, config):
        self.config = config or {}

    def _storage_options(self) -> Optional[dict]:
        config = self.config
        options = {
            "azure_storage_account_name": config.get("account_name"),
            "azure_storage_account_key": config.get("account_key"),
            "azure_storage_sas_key": config.get("sas_token"),
            "azure_tenant_id": config.get("tenant_id"),
            "azure_client_id": config.get("client_id"),
            "azure_client_secret": config.get("client_secret"),
        }
        options = {k: v for k, v in options.items() if v}
        return options or None

    def get_data(self, table_or_query=None, pack_config=None):
        # Accept full abfs(s):// path or account/container/blob components
        path = self.config.get("path")
        if not path:
            account_name = self.config.get("account_name")
            container = self.config.get("container")
            blob = self.config.get("blob") or self.config.get("key")
            if account_name and container and blob:
                path = f"abfs://{container}@{account_name}.dfs.core.windows.net/{blob}"
        if not path:
            raise ValueError(
                "AzureBlobSource requires either 'path' or 'account_name'+'container'+'blob'."
            )

        fmt = _infer_format_from_path(path, self.config.get("format"))
        return _materialize_remote_to_parquet(
            self, path, fmt, self._storage_options(), pack_config
        )


class HDFSSource(DataSource):
    def __init__(self, config):
        self.config = config or {}

    def _storage_options(self) -> Optional[dict]:
        # HDFS is not spoken by Polars' object store, so these stay in fsspec
        # naming: the file is staged locally through fsspec first.
        options = {
            key: self.config[key]
            for key in ("host", "port", "user", "kerb_kwargs")
            if key in self.config
        }
        return options or None

    def get_data(self, table_or_query=None, pack_config=None):
        # Expect hdfs://host:port/path
        path = self.config.get("path")
        if not path:
            host = self.config.get("host")
            port = self.config.get("port") or 8020
            hdfs_path = self.config.get("hdfs_path") or self.config.get("key")
            if host and hdfs_path:
                path = f"hdfs://{host}:{port}/{hdfs_path.lstrip('/')}"
        if not path:
            raise ValueError(
                "HDFSSource requires 'path' or 'host'+'hdfs_path' in config."
            )

        fmt = _infer_format_from_path(path, self.config.get("format"))
        if fmt == "parquet":
            # No pass-through: a pack scanning hdfs:// directly has no way to
            # authenticate, so the object is staged locally instead.
            local_path = _stage_remote_file(path, self._storage_options())
            return _materialize_remote_to_parquet(
                self, local_path, fmt, None, pack_config
            )
        return _materialize_remote_to_parquet(
            self, path, fmt, self._storage_options(), pack_config
        )


FOLDER_DATA_SUFFIXES = (
    ".csv",
    ".xlsx",
    ".xlsm",
    ".parquet",
    ".pq",
    ".json",
    ".ndjson",
    ".jsonl",
)


class FolderSource(FileSource):
    """A directory of data files: one logical object per file.

    The folder counterpart of a database scanned with ``*``. Every file is
    staged by :meth:`FileSource._load_file`, so it reads exactly the formats
    and CSV options a single-file source does, and a parquet file is read in
    place (borrowed, never deleted by ``Pack.cleanup``).

    Config: ``path`` (the directory), ``recursive`` (default ``False``; nested
    files are then named after their relative path, ``2024/visits`` ->
    ``file_2024_visits``), plus the CSV options of a file source. Hidden files
    and directories and Excel lock files (``~$...``) are ignored.

    ``table_or_query`` selects files: ``None`` or ``"*"`` for all of them, a
    name or a list of names otherwise. A name is the file's identifier with or
    without its extension, case-insensitively.

    In a scan of several files, a file that cannot be read is skipped and
    recorded in ``skipped_objects``, as a database scan skips a table it cannot
    read; a full disk still aborts, and so does a scan where nothing could be
    read. A file named explicitly raises its own error.
    """

    def __init__(self, config):
        config = config or {}
        super().__init__(config.get("path"), config)

    def _data_files(self) -> Dict[str, str]:
        """Identifier -> file path, sorted.

        The identifier is the relative path without its suffix, unless two
        files share it (``visits.csv`` and ``visits.parquet``): both then keep
        their suffix, rather than one silently replacing the other.
        """
        root = Path(self.file_path)
        recursive = _config_flag(self.config.get("recursive"), default=False)
        candidates = []
        for path in sorted(root.glob("**/*" if recursive else "*")):
            relative = path.relative_to(root)
            if (
                path.is_file()
                and path.suffix.lower() in FOLDER_DATA_SUFFIXES
                and not any(part.startswith(".") for part in relative.parts)
                and not path.name.startswith("~$")
            ):
                candidates.append(relative)
        stems = [
            relative.with_suffix("").as_posix() for relative in candidates
        ]
        return {
            (stem if stems.count(stem) == 1 else relative.as_posix()): str(
                root / relative
            )
            for stem, relative in zip(stems, candidates)
        }

    def _select(self, files: Dict[str, str], table_or_query) -> Dict[str, str]:
        root = Path(self.file_path)
        if table_or_query is None or (
            isinstance(table_or_query, str) and table_or_query.strip() == "*"
        ):
            return files
        if isinstance(table_or_query, str):
            names = [table_or_query]
        elif isinstance(table_or_query, (list, tuple, set)):
            names = list(table_or_query)
        else:
            raise TypeError(
                "table_or_query must be None, '*', a file name, or a list of "
                "file names."
            )

        # A file answers to its identifier and to its relative file name,
        # extension included, case-insensitively.
        lookup = {}
        for identifier, path in files.items():
            lookup.setdefault(identifier.lower(), identifier)
            lookup.setdefault(
                Path(path).relative_to(root).as_posix().lower(), identifier
            )
        selected = {}
        for name in names:
            identifier = lookup.get(str(name).strip().lower())
            if identifier is None:
                raise FileNotFoundError(
                    f"No data file named {name!r} in the folder; available: "
                    f"{', '.join(files)}"
                )
            selected[identifier] = files[identifier]
        return selected

    def get_data(self, table_or_query=None, pack_config=None):
        if not self.file_path or not os.path.isdir(self.file_path):
            raise FileNotFoundError(
                f"Folder source path {self.file_path!r} is not a directory, "
                "or it can't be reached."
            )
        files = self._data_files()
        if not files:
            raise FileNotFoundError(
                f"No data file ({', '.join(FOLDER_DATA_SUFFIXES)}) in "
                f"{self.file_path}."
            )
        selected = self._select(files, table_or_query)
        explicit = not (
            table_or_query is None
            or (
                isinstance(table_or_query, str)
                and table_or_query.strip() == "*"
            )
        )
        output_dir = _ensure_output_dir(pack_config)

        self._skipped_objects = []
        all_paths: List[str] = []
        read_count = 0
        skipped: List[Dict[str, str]] = []
        first_error: Optional[BaseException] = None
        for identifier, path in selected.items():
            try:
                if path.lower().endswith((".parquet", ".pq")):
                    # Read in place, so nothing else would open it before the
                    # pack does: check the footer now, while a bad file can
                    # still be skipped instead of failing the analysis.
                    pl.read_parquet_schema(path)
                all_paths.extend(
                    self._load_file(path, pack_config, output_dir, identifier)
                )
            except InsufficientDiskSpaceError:
                raise
            except Exception as exc:  # noqa: BLE001 - a bad file is skipped
                if explicit:
                    raise
                reason = " ".join(str(exc).split()) or exc.__class__.__name__
                skipped.append(
                    {
                        "object": os.path.relpath(path, self.file_path),
                        "error": exc.__class__.__name__,
                        "reason": reason,
                    }
                )
                if first_error is None:
                    first_error = exc
                logger.warning(
                    "skipping %s: it could not be read (%s: %s)",
                    path,
                    exc.__class__.__name__,
                    reason,
                )
            else:
                read_count += 1

        if skipped:
            self.skipped_objects.extend(skipped)
            if not read_count:
                details = "; ".join(
                    f"{item['object']} ({item['error']}: {item['reason']})"
                    for item in skipped
                )
                raise RuntimeError(
                    f"None of the {len(skipped)} files in the folder could be "
                    f"read: {details}"
                ) from first_error
            logger.warning(
                "%d of %d files were skipped and are absent from this scan: %s",
                len(skipped),
                len(skipped) + read_count,
                ", ".join(item["object"] for item in skipped),
            )
        return all_paths


class MongoDBSource(DataSource):
    """MongoDB data source using pymongo."""

    def __init__(self, config):
        self.config = config or {}

    def get_data(self, table_or_query=None, pack_config=None):
        """
        Load data from MongoDB and write Parquet chunks.

        - If table_or_query is a string collection name: returns parquet paths for that collection
        - If table_or_query is a list of collection names: returns parquet paths for each
        - If table_or_query is '*' or None: scan all collections
        """
        try:
            from pymongo import MongoClient
        except ImportError:
            raise ImportError(
                "pymongo is required for MongoDBSource. Install with: pip install pymongo"
            )

        output_dir = _ensure_output_dir(pack_config)
        chunk_rows = _chunk_rows(pack_config)

        # Build connection
        connection_string = self.config.get("connection_string")
        if connection_string:
            client = MongoClient(connection_string)
        else:
            host = self.config.get("host", "localhost")
            port = int(self.config.get("port", 27017))
            username = self.config.get("username")
            password = self.config.get("password")
            database = self.config.get("database")

            if username and password:
                client = MongoClient(
                    host=host,
                    port=port,
                    username=username,
                    password=password,
                    authSource=database or "admin",
                )
            else:
                client = MongoClient(host=host, port=port)

        database_name = self.config.get("database")
        if not database_name:
            raise ValueError("MongoDBSource requires 'database' in config.")

        db = client[database_name]

        # Determine collections to process
        if table_or_query is None or (
            isinstance(table_or_query, str) and table_or_query.strip() == "*"
        ):
            collections = db.list_collection_names()
        elif isinstance(table_or_query, (list, tuple, set)):
            collections = list(table_or_query)
        elif isinstance(table_or_query, str):
            collections = [table_or_query]
        else:
            raise TypeError(
                "table_or_query must be None, '*', a string, or a list of collection names."
            )

        all_paths: List[str] = []
        try:
            for collection_name in collections:
                collection = db[collection_name]
                base_name = self._object_base_name("mongodb", collection_name)
                cursor = collection.find(batch_size=chunk_rows)
                paths = pio.write_dict_rows(
                    _mongo_documents(cursor),
                    output_dir,
                    base_name,
                    fetch_rows=_fetch_rows(pack_config),
                    chunk_rows=chunk_rows,
                    min_free_bytes=_min_free_bytes(pack_config),
                )
                all_paths.extend(self._record_object(base_name, paths))
        finally:
            client.close()
        return all_paths


def _mongo_documents(cursor) -> Iterator[Dict[str, Any]]:
    """Yield documents with an ObjectId that Arrow can represent."""
    for document in cursor:
        if "_id" in document:
            document["_id"] = str(document["_id"])
        yield document


class SnowflakeSource(_SqlAlchemySource):
    """Snowflake data warehouse source using snowflake-sqlalchemy."""

    dialect_name = "snowflake"

    def __init__(self, config):
        self.config = config or {}

    def get_data(self, table_or_query=None, pack_config=None):
        """
        Load data from Snowflake and write Parquet chunks.
        """
        output_dir = _ensure_output_dir(pack_config)
        chunk_rows = _chunk_rows(pack_config)

        # Build connection string
        connection_string = self.config.get("connection_string")
        if not connection_string:
            account = self.config.get("account")
            user = self.config.get("user") or self.config.get("username")
            password = self.config.get("password")
            database = self.config.get("database")
            schema = self.config.get("schema", "PUBLIC")
            warehouse = self.config.get("warehouse")
            role = self.config.get("role")

            if not all([account, user, password]):
                raise ValueError(
                    "SnowflakeSource requires 'account', 'user', and 'password' in config."
                )

            # Use URL.create to avoid password-in-URL pattern detection
            query_params = {}
            if warehouse:
                query_params["warehouse"] = warehouse
            if role:
                query_params["role"] = role

            # Build database path for Snowflake (database/schema)
            db_path = database if database else None
            if database and schema:
                db_path = f"{database}/{schema}"

            url = URL.create(
                drivername="snowflake",
                username=user,
                password=password,
                host=account,
                database=db_path,
                query=query_params if query_params else None,
            )
            connection_string = url

        engine = create_engine(connection_string)
        schema = self.config.get("schema", "PUBLIC")

        return self._load_data(
            engine,
            table_or_query,
            schema,
            output_dir,
            chunk_rows,
            self.dialect_name,
            pack_config=pack_config,
        )


class BigQuerySource(_SqlAlchemySource):
    """Google BigQuery data source using sqlalchemy-bigquery."""

    dialect_name = "bigquery"

    def __init__(self, config):
        self.config = config or {}

    def _qualify(self, table_name, schema):
        qualified = f"{schema}.{table_name}" if schema else table_name
        return f"`{qualified}`", qualified

    def get_data(self, table_or_query=None, pack_config=None):
        """
        Load data from BigQuery and write Parquet chunks.
        """
        output_dir = _ensure_output_dir(pack_config)
        chunk_rows = _chunk_rows(pack_config)

        # Build connection string
        connection_string = self.config.get("connection_string")
        if not connection_string:
            project = self.config.get("project")
            dataset = self.config.get("dataset")
            credentials_path = self.config.get(
                "credentials_json"
            ) or self.config.get("credentials")

            if not project:
                raise ValueError(
                    "BigQuerySource requires 'project' in config."
                )

            # bigquery://project/dataset
            connection_string = f"bigquery://{project}"
            if dataset:
                connection_string += f"/{dataset}"

            if credentials_path:
                connection_string += f"?credentials_path={credentials_path}"

        engine = create_engine(connection_string)
        dataset = self.config.get("dataset")

        return self._load_data(
            engine,
            table_or_query,
            dataset,
            output_dir,
            chunk_rows,
            self.dialect_name,
            pack_config=pack_config,
        )


class DatabricksSource(DataSource):
    """Databricks data source using databricks-sql-connector."""

    def __init__(self, config):
        self.config = config or {}

    def get_data(self, table_or_query=None, pack_config=None):
        """
        Load data from Databricks and write Parquet chunks.
        """
        try:
            from databricks import sql as databricks_sql
        except ImportError:
            raise ImportError(
                "databricks-sql-connector is required for DatabricksSource. Install with: pip install databricks-sql-connector"
            )

        output_dir = _ensure_output_dir(pack_config)
        chunk_rows = _chunk_rows(pack_config)

        server_hostname = self.config.get(
            "server_hostname"
        ) or self.config.get("host")
        http_path = self.config.get("http_path")
        access_token = self.config.get("access_token") or self.config.get(
            "token"
        )
        catalog = self.config.get("catalog")
        schema = self.config.get("schema")

        if not all([server_hostname, http_path, access_token]):
            raise ValueError(
                "DatabricksSource requires 'server_hostname', 'http_path', and 'access_token' in config."
            )

        connection = databricks_sql.connect(
            server_hostname=server_hostname,
            http_path=http_path,
            access_token=access_token,
        )

        cursor = connection.cursor()
        all_paths: List[str] = []
        try:
            # Set catalog and schema if provided
            if catalog:
                cursor.execute(f"USE CATALOG {catalog}")
            if schema:
                cursor.execute(f"USE SCHEMA {schema}")

            # Determine tables to process
            if table_or_query is None or (
                isinstance(table_or_query, str)
                and table_or_query.strip() == "*"
            ):
                cursor.execute("SHOW TABLES")
                tables = [
                    row[1] for row in cursor.fetchall()
                ]  # table name is typically second column
            elif isinstance(table_or_query, (list, tuple, set)):
                tables = list(table_or_query)
            elif isinstance(table_or_query, str):
                if _is_sql_query(table_or_query):
                    base_name = self._object_base_name("databricks", "query")
                    cursor.execute(table_or_query)
                    return self._record_object(
                        base_name,
                        self._drain(
                            cursor,
                            output_dir,
                            base_name,
                            chunk_rows,
                            pack_config,
                        ),
                    )
                tables = [table_or_query]
            else:
                raise TypeError(
                    "table_or_query must be None, '*', a string, or a list of table names."
                )

            for table_name in tables:
                base_name = self._object_base_name("databricks", table_name)
                cursor.execute(f"SELECT * FROM {table_name}")
                all_paths.extend(
                    self._record_object(
                        base_name,
                        self._drain(
                            cursor,
                            output_dir,
                            base_name,
                            chunk_rows,
                            pack_config,
                        ),
                    )
                )
        finally:
            cursor.close()
            connection.close()
        return all_paths

    def _drain(
        self, cursor, output_dir, base_name, chunk_rows, pack_config
    ) -> List[str]:
        """Write a cursor's result set, using its Arrow API when it has one.

        databricks-sql-connector fetches Arrow off the wire; going through
        ``fetchmany_arrow`` skips the conversion to Python objects entirely.
        """
        min_free = _min_free_bytes(pack_config)
        if hasattr(cursor, "fetchmany_arrow"):
            return pio.write_arrow_batches(
                _iter_arrow(cursor, chunk_rows),
                output_dir,
                base_name,
                chunk_rows=chunk_rows,
                min_free_bytes=min_free,
            )
        columns = [description[0] for description in cursor.description]
        return pio.write_row_batches(
            cursor,
            columns,
            output_dir,
            base_name,
            fetch_rows=_fetch_rows(pack_config),
            chunk_rows=chunk_rows,
            min_free_bytes=min_free,
        )


def _iter_arrow(cursor, chunk_rows: int) -> Iterator[Any]:
    while True:
        table = cursor.fetchmany_arrow(chunk_rows)
        if table is None or table.num_rows == 0:
            return
        yield table


class RedshiftSource(DataSource):
    """Amazon Redshift data source using redshift-connector or psycopg2."""

    def __init__(self, config):
        self.config = config or {}

    def get_data(self, table_or_query=None, pack_config=None):
        """
        Load data from Redshift and write Parquet chunks.
        """
        # Build connection string - Redshift is PostgreSQL-compatible
        connection_string = self.config.get("connection_string")
        if not connection_string:
            host = self.config.get("host")
            port = self.config.get("port", 5439)
            user = self.config.get("user") or self.config.get("username")
            password = self.config.get("password")
            database = self.config.get("database")

            if not all([host, user, password, database]):
                raise ValueError(
                    "RedshiftSource requires 'host', 'user', 'password', and 'database' in config."
                )

            # Try redshift+redshift_connector first, fall back to postgresql
            # Use URL.create to avoid password-in-URL pattern detection
            try:
                import redshift_connector  # noqa: F401

                url = URL.create(
                    drivername="redshift+redshift_connector",
                    username=user,
                    password=password,
                    host=host,
                    port=port,
                    database=database,
                )
            except ImportError:
                # Fall back to PostgreSQL driver
                url = URL.create(
                    drivername="postgresql",
                    username=user,
                    password=password,
                    host=host,
                    port=port,
                    database=database,
                )
            connection_string = url

        # The URL object is passed as-is: str(URL) masks the password as
        # "user:***@host", so stringifying it here produced an engine that
        # could not authenticate.
        db_source = DatabaseSource(
            connection_string=connection_string, config=self.config
        )
        paths = db_source.get_data(
            table_or_query=table_or_query, pack_config=pack_config
        )
        self._object_paths = db_source.object_paths
        self._borrowed_paths = set(getattr(db_source, "borrowed_paths", ()))
        self._skipped_objects = [
            dict(item) for item in db_source.skipped_objects
        ]
        return paths


class ClickHouseSource(_SqlAlchemySource):
    """ClickHouse data source using clickhouse-sqlalchemy."""

    dialect_name = "clickhouse"

    def __init__(self, config):
        self.config = config or {}

    def _qualify(self, table_name, schema):
        # ClickHouse databases are selected by the connection, not qualified.
        return table_name, table_name

    def _list_tables(self, engine, schema):
        try:
            tables = super()._list_tables(engine, None)
        except NotImplementedError:
            tables = []
        if tables:
            return tables
        database = self.config.get("database", "default")
        with engine.connect() as conn:
            result = conn.execute(text(f"SHOW TABLES FROM {database}"))
            return [row[0] for row in result]

    def get_data(self, table_or_query=None, pack_config=None):
        """
        Load data from ClickHouse and write Parquet chunks.
        """
        output_dir = _ensure_output_dir(pack_config)
        chunk_rows = _chunk_rows(pack_config)

        # Build connection string
        connection_string = self.config.get("connection_string")
        if not connection_string:
            host = self.config.get("host", "localhost")
            port = self.config.get("port", 8123)
            user = self.config.get("user") or self.config.get(
                "username", "default"
            )
            password = self.config.get("password", "")
            database = self.config.get("database", "default")
            protocol = self.config.get("protocol", "http")  # http or native

            # Use URL.create to avoid password-in-URL pattern detection
            # clickhouse+http or clickhouse+native
            driver = f"clickhouse+{protocol}"
            url = URL.create(
                drivername=driver,
                username=user,
                password=password if password else None,
                host=host,
                port=port,
                database=database,
            )
            connection_string = url

        engine = create_engine(connection_string)

        return self._load_data(
            engine,
            table_or_query,
            None,
            output_dir,
            chunk_rows,
            self.dialect_name,
            pack_config=pack_config,
        )


class DuckDBSource(DataSource):
    """DuckDB data source (local files or MotherDuck cloud).

    Optimized for big data (100GB+) using DuckDB's native COPY TO PARQUET
    which streams data directly to parquet without loading into memory.
    """

    def __init__(self, config):
        self.config = config or {}

    def get_data(self, table_or_query=None, pack_config=None):
        """
        Load data from DuckDB and write Parquet files using streaming export.

        IMPORTANT: Uses DuckDB's native COPY TO for memory-efficient export.
        Does NOT load data into a DataFrame for large datasets.
        """
        try:
            import duckdb
        except ImportError:
            raise ImportError(
                "duckdb is required for DuckDBSource. Install with: pip install duckdb"
            )

        output_dir = _ensure_output_dir(pack_config)
        chunk_rows = _chunk_rows(pack_config)

        # Connect to DuckDB
        db_path = self.config.get("path") or self.config.get(
            "database", ":memory:"
        )
        motherduck_token = self.config.get(
            "motherduck_token"
        ) or self.config.get("token")

        if motherduck_token:
            # MotherDuck cloud connection
            connection_string = (
                f"md:{db_path}?motherduck_token={motherduck_token}"
            )
            conn = duckdb.connect(connection_string)
        else:
            conn = duckdb.connect(db_path)

        schema = self.config.get("schema", "main")

        try:
            # Get tables
            if table_or_query is None or (
                isinstance(table_or_query, str)
                and table_or_query.strip() == "*"
            ):
                result = conn.execute(
                    f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{schema}'"
                )
                table_names = [row[0] for row in result.fetchall()]
            elif isinstance(table_or_query, (list, tuple, set)):
                table_names = list(table_or_query)
            elif isinstance(table_or_query, str):
                if _is_sql_query(table_or_query):
                    base_name = self._object_base_name("duckdb", "query")
                    return self._record_object(
                        base_name,
                        self._export_query_to_parquet(
                            conn,
                            table_or_query,
                            output_dir,
                            base_name,
                            chunk_rows,
                            pack_config,
                        ),
                    )
                table_names = [table_or_query]
            else:
                raise TypeError(
                    "table_or_query must be None, '*', a string, or a list of table names."
                )

            all_paths: List[str] = []
            for table_name in table_names:
                base_name = self._object_base_name("duckdb", table_name)
                qualified = (
                    f"{schema}.{table_name}"
                    if schema != "main"
                    else table_name
                )
                all_paths.extend(
                    self._record_object(
                        base_name,
                        self._export_query_to_parquet(
                            conn,
                            f"SELECT * FROM {qualified}",
                            output_dir,
                            base_name,
                            chunk_rows,
                            pack_config,
                        ),
                    )
                )
        finally:
            conn.close()
        return all_paths

    def _export_query_to_parquet(
        self,
        conn,
        query: str,
        output_dir: str,
        base_name: str,
        chunk_rows: int,
        pack_config: Optional[dict] = None,
    ) -> List[str]:
        """
        Export query results directly to parquet using DuckDB's COPY TO.

        This is memory-efficient as DuckDB streams data directly to parquet
        without loading into memory first.
        """
        pio.check_disk_space(
            output_dir, min_free_bytes=_min_free_bytes(pack_config)
        )
        output_path = _build_parquet_path(output_dir, base_name, 1)

        try:
            # DuckDB streams its own result set to Parquet; nothing crosses into
            # Python at all on this path.
            copy_sql = f"""
                COPY ({query}) TO '{output_path}'
                (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {chunk_rows})
            """
            conn.execute(copy_sql)
            logger.info(f"DuckDB streaming export to: {output_path}")
            return [output_path]

        except Exception as e:
            logger.warning(
                f"DuckDB COPY TO failed ({e}), falling back to chunked export"
            )
            return self._export_query_chunked(
                conn, query, output_dir, base_name, chunk_rows, pack_config
            )

    def _export_query_chunked(
        self,
        conn,
        query: str,
        output_dir: str,
        base_name: str,
        chunk_rows: int,
        pack_config: Optional[dict] = None,
    ) -> List[str]:
        """Fallback: pull the result as an Arrow record batch stream.

        ``fetch_record_batch`` yields batches as DuckDB produces them;
        ``fetch_arrow_table`` (what this used to call) materializes the entire
        result first, which defeats the purpose of the fallback.
        """
        result = conn.execute(query)
        reader = result.fetch_record_batch(chunk_rows)
        return pio.write_arrow_batches(
            reader,
            output_dir,
            base_name,
            chunk_rows=chunk_rows,
            min_free_bytes=_min_free_bytes(pack_config),
        )

    def _is_sql_query(self, s: str) -> bool:
        return _is_sql_query(s)


class TrinoSource(DataSource):
    """Trino/Presto data source using trino library."""

    def __init__(self, config):
        self.config = config or {}

    def get_data(self, table_or_query=None, pack_config=None):
        """
        Load data from Trino and write Parquet chunks.
        """
        try:
            from trino.dbapi import connect as trino_connect
        except ImportError:
            raise ImportError(
                "trino is required for TrinoSource. Install with: pip install trino"
            )

        output_dir = _ensure_output_dir(pack_config)
        chunk_rows = _chunk_rows(pack_config)

        host = self.config.get("host", "localhost")
        port = int(self.config.get("port", 8080))
        user = self.config.get("user") or self.config.get("username", "trino")
        catalog = self.config.get("catalog")
        schema = self.config.get("schema")
        http_scheme = self.config.get("http_scheme", "https")

        conn = trino_connect(
            host=host,
            port=port,
            user=user,
            catalog=catalog,
            schema=schema,
            http_scheme=http_scheme,
        )
        cursor = conn.cursor()
        all_paths: List[str] = []
        try:
            # Get tables
            if table_or_query is None or (
                isinstance(table_or_query, str)
                and table_or_query.strip() == "*"
            ):
                cursor.execute("SHOW TABLES")
                table_names = [row[0] for row in cursor.fetchall()]
            elif isinstance(table_or_query, (list, tuple, set)):
                table_names = list(table_or_query)
            elif isinstance(table_or_query, str):
                if _is_sql_query(table_or_query):
                    base_name = self._object_base_name("trino", "query")
                    cursor.execute(table_or_query)
                    return self._record_object(
                        base_name,
                        _drain_cursor(
                            cursor,
                            output_dir,
                            base_name,
                            chunk_rows,
                            pack_config,
                        ),
                    )
                table_names = [table_or_query]
            else:
                raise TypeError(
                    "table_or_query must be None, '*', a string, or a list of table names."
                )

            for table_name in table_names:
                base_name = self._object_base_name("trino", table_name)
                cursor.execute(f"SELECT * FROM {table_name}")
                all_paths.extend(
                    self._record_object(
                        base_name,
                        _drain_cursor(
                            cursor,
                            output_dir,
                            base_name,
                            chunk_rows,
                            pack_config,
                        ),
                    )
                )
        finally:
            cursor.close()
            conn.close()
        return all_paths

    def _is_sql_query(self, s: str) -> bool:
        return _is_sql_query(s)


def _drain_cursor(
    cursor,
    output_dir: str,
    base_name: str,
    chunk_rows: int,
    pack_config: Optional[dict],
) -> List[str]:
    """Write an executed DB-API cursor to parquet parts, column-major."""
    columns = [description[0] for description in cursor.description]
    return pio.write_row_batches(
        cursor,
        columns,
        output_dir,
        base_name,
        chunk_rows=chunk_rows,
        fetch_rows=_fetch_rows(pack_config),
        min_free_bytes=_min_free_bytes(pack_config),
    )


class TeradataSource(_SqlAlchemySource):
    """Teradata data source using teradatasqlalchemy."""

    dialect_name = "teradata"

    def __init__(self, config):
        self.config = config or {}

    def get_data(self, table_or_query=None, pack_config=None):
        """Load data from Teradata and write Parquet chunks."""
        output_dir = _ensure_output_dir(pack_config)
        chunk_rows = _chunk_rows(pack_config)

        connection_string = self.config.get("connection_string")
        if not connection_string:
            host = self.config.get("host")
            user = self.config.get("user") or self.config.get("username")
            password = self.config.get("password")
            database = self.config.get("database")

            if not all([host, user, password]):
                raise ValueError(
                    "TeradataSource requires 'host', 'user', and 'password' in config."
                )

            # Use URL.create to avoid password-in-URL pattern detection
            url = URL.create(
                drivername="teradatasql",
                username=user,
                password=password,
                host=host,
                database=database if database else None,
            )
            connection_string = url

        engine = create_engine(connection_string)
        schema = self.config.get("schema")

        return self._load_data(
            engine,
            table_or_query,
            schema,
            output_dir,
            chunk_rows,
            self.dialect_name,
            pack_config=pack_config,
        )


class SapHanaSource(_SqlAlchemySource):
    """SAP HANA data source using hdbcli/sqlalchemy-hana."""

    dialect_name = "sap_hana"

    def __init__(self, config):
        self.config = config or {}

    def _qualify(self, table_name, schema):
        if schema:
            return f'"{schema}"."{table_name}"', f"{schema}.{table_name}"
        return f'"{table_name}"', table_name

    def get_data(self, table_or_query=None, pack_config=None):
        """Load data from SAP HANA and write Parquet chunks."""
        output_dir = _ensure_output_dir(pack_config)
        chunk_rows = _chunk_rows(pack_config)

        connection_string = self.config.get("connection_string")
        if not connection_string:
            host = self.config.get("host")
            port = self.config.get("port", 30015)
            user = self.config.get("user") or self.config.get("username")
            password = self.config.get("password")

            if not all([host, user, password]):
                raise ValueError(
                    "SapHanaSource requires 'host', 'user', and 'password' in config."
                )

            # Use URL.create to avoid password-in-URL pattern detection
            url = URL.create(
                drivername="hana",
                username=user,
                password=password,
                host=host,
                port=port,
            )
            connection_string = url

        engine = create_engine(connection_string)
        schema = self.config.get("schema")

        return self._load_data(
            engine,
            table_or_query,
            schema,
            output_dir,
            chunk_rows,
            self.dialect_name,
            pack_config=pack_config,
        )


class CassandraSource(DataSource):
    """Apache Cassandra data source using cassandra-driver."""

    def __init__(self, config):
        self.config = config or {}

    def get_data(self, table_or_query=None, pack_config=None):
        """Load data from Cassandra and write Parquet chunks."""
        try:
            from cassandra.cluster import Cluster
            from cassandra.auth import PlainTextAuthProvider
        except ImportError:
            raise ImportError(
                "cassandra-driver is required for CassandraSource. Install with: pip install cassandra-driver"
            )

        output_dir = _ensure_output_dir(pack_config)
        chunk_rows = _chunk_rows(pack_config)
        min_free = _min_free_bytes(pack_config)

        hosts = self.config.get("hosts") or [
            self.config.get("host", "localhost")
        ]
        port = int(self.config.get("port", 9042))
        username = self.config.get("username")
        password = self.config.get("password")
        keyspace = self.config.get("keyspace") or self.config.get("database")

        if username and password:
            auth_provider = PlainTextAuthProvider(
                username=username, password=password
            )
            cluster = Cluster(hosts, port=port, auth_provider=auth_provider)
        else:
            cluster = Cluster(hosts, port=port)

        session = cluster.connect(keyspace)
        all_paths: List[str] = []
        try:
            # Get tables
            if table_or_query is None or (
                isinstance(table_or_query, str)
                and table_or_query.strip() == "*"
            ):
                rows = session.execute(
                    f"SELECT table_name FROM system_schema.tables WHERE keyspace_name = '{keyspace}'"
                )
                table_names = [row.table_name for row in rows]
            elif isinstance(table_or_query, (list, tuple, set)):
                table_names = list(table_or_query)
            elif isinstance(table_or_query, str):
                if _is_sql_query(table_or_query):
                    base_name = self._object_base_name("cassandra", "query")
                    rows = session.execute(table_or_query)
                    return self._record_object(
                        base_name,
                        pio.write_row_batches(
                            rows,
                            rows.column_names,
                            output_dir,
                            base_name,
                            fetch_rows=_fetch_rows(pack_config),
                            chunk_rows=chunk_rows,
                            min_free_bytes=min_free,
                        ),
                    )
                table_names = [table_or_query]
            else:
                raise TypeError(
                    "table_or_query must be None, '*', a string, or a list of table names."
                )

            for table_name in table_names:
                base_name = self._object_base_name("cassandra", table_name)
                # The driver pages the result set, so iterating it never holds
                # more than one page.
                rows = session.execute(f"SELECT * FROM {table_name}")
                all_paths.extend(
                    self._record_object(
                        base_name,
                        pio.write_row_batches(
                            rows,
                            rows.column_names,
                            output_dir,
                            base_name,
                            fetch_rows=_fetch_rows(pack_config),
                            chunk_rows=chunk_rows,
                            min_free_bytes=min_free,
                        ),
                    )
                )
        finally:
            cluster.shutdown()
        return all_paths

    def _is_sql_query(self, s: str) -> bool:
        sql = s.strip().lower()
        if ";" in sql or "\n" in sql:
            return True
        return sql.startswith(("select", "with"))


class ElasticsearchSource(DataSource):
    """Elasticsearch data source."""

    def __init__(self, config):
        self.config = config or {}

    def get_data(self, table_or_query=None, pack_config=None):
        """Load data from Elasticsearch and write Parquet chunks."""
        try:
            from elasticsearch import Elasticsearch
        except ImportError:
            raise ImportError(
                "elasticsearch is required for ElasticsearchSource. Install with: pip install elasticsearch"
            )

        output_dir = _ensure_output_dir(pack_config)
        chunk_rows = _chunk_rows(pack_config)

        hosts = self.config.get("hosts") or [
            self.config.get("host", "localhost")
        ]
        port = int(self.config.get("port", 9200))
        username = self.config.get("username")
        password = self.config.get("password")
        use_ssl = self.config.get("use_ssl", False)
        verify_certs = self.config.get("verify_certs", True)
        api_key = self.config.get("api_key")
        cloud_id = self.config.get("cloud_id")

        # Build hosts with port
        if (
            isinstance(hosts, list)
            and hosts
            and not any(":" in str(h) for h in hosts)
        ):
            scheme = "https" if use_ssl else "http"
            hosts = [f"{scheme}://{h}:{port}" for h in hosts]

        if cloud_id:
            es = (
                Elasticsearch(cloud_id=cloud_id, api_key=api_key)
                if api_key
                else Elasticsearch(
                    cloud_id=cloud_id, basic_auth=(username, password)
                )
            )
        elif api_key:
            es = Elasticsearch(
                hosts=hosts, api_key=api_key, verify_certs=verify_certs
            )
        elif username and password:
            es = Elasticsearch(
                hosts=hosts,
                basic_auth=(username, password),
                verify_certs=verify_certs,
            )
        else:
            es = Elasticsearch(hosts=hosts, verify_certs=verify_certs)

        # Get indices (table_or_query is the index pattern)
        if table_or_query is None or (
            isinstance(table_or_query, str) and table_or_query.strip() == "*"
        ):
            indices = list(es.indices.get(index="*").keys())
            # Filter out system indices
            indices = [i for i in indices if not i.startswith(".")]
        elif isinstance(table_or_query, (list, tuple, set)):
            indices = list(table_or_query)
        elif isinstance(table_or_query, str):
            indices = [table_or_query]
        else:
            raise TypeError(
                "table_or_query must be None, '*', a string (index name), or a list of index names."
            )

        all_paths: List[str] = []
        try:
            for index in indices:
                base_name = self._object_base_name("elasticsearch", index)
                paths = pio.write_dict_rows(
                    _scroll_documents(es, index, chunk_rows),
                    output_dir,
                    base_name,
                    fetch_rows=_fetch_rows(pack_config),
                    chunk_rows=chunk_rows,
                    min_free_bytes=_min_free_bytes(pack_config),
                )
                all_paths.extend(self._record_object(base_name, paths))
        finally:
            es.close()
        return all_paths


def _scroll_documents(es, index: str, page_size: int) -> Iterator[dict]:
    """Yield an index's documents through the scroll API, one page at a time."""
    response = es.search(
        index=index,
        scroll="2m",
        size=min(page_size, 10000),
        body={"query": {"match_all": {}}},
    )
    scroll_id = response["_scroll_id"]
    try:
        hits = response["hits"]["hits"]
        while hits:
            for hit in hits:
                document = hit["_source"]
                document["_id"] = hit["_id"]
                yield document
            response = es.scroll(scroll_id=scroll_id, scroll="2m")
            scroll_id = response["_scroll_id"]
            hits = response["hits"]["hits"]
    finally:
        try:
            es.clear_scroll(scroll_id=scroll_id)
        except Exception:  # noqa: BLE001 - the scroll expires on its own
            logger.debug("failed to clear scroll %s", scroll_id)


class IbmDb2Source(_SqlAlchemySource):
    """IBM DB2 data source using ibm_db_sa."""

    dialect_name = "ibm_db2"

    def __init__(self, config):
        self.config = config or {}

    def get_data(self, table_or_query=None, pack_config=None):
        """Load data from IBM DB2 and write Parquet chunks."""
        output_dir = _ensure_output_dir(pack_config)
        chunk_rows = _chunk_rows(pack_config)

        connection_string = self.config.get("connection_string")
        if not connection_string:
            host = self.config.get("host")
            port = self.config.get("port", 50000)
            user = self.config.get("user") or self.config.get("username")
            password = self.config.get("password")
            database = self.config.get("database")

            if not all([host, user, password, database]):
                raise ValueError(
                    "IbmDb2Source requires 'host', 'user', 'password', and 'database' in config."
                )

            # Use URL.create to avoid password-in-URL pattern detection
            url = URL.create(
                drivername="ibm_db_sa",
                username=user,
                password=password,
                host=host,
                port=port,
                database=database,
            )
            connection_string = url

        engine = create_engine(connection_string)
        schema = self.config.get("schema")

        return self._load_data(
            engine,
            table_or_query,
            schema,
            output_dir,
            chunk_rows,
            self.dialect_name,
            pack_config=pack_config,
        )


class AthenaSource(_SqlAlchemySource):
    """Amazon Athena data source using PyAthena."""

    dialect_name = "athena"

    def __init__(self, config):
        self.config = config or {}

    def _qualify(self, table_name, schema):
        if schema:
            return f'"{schema}"."{table_name}"', f"{schema}.{table_name}"
        return f'"{table_name}"', table_name

    def get_data(self, table_or_query=None, pack_config=None):
        """Load data from Amazon Athena and write Parquet chunks."""
        output_dir = _ensure_output_dir(pack_config)
        chunk_rows = _chunk_rows(pack_config)

        connection_string = self.config.get("connection_string")
        if not connection_string:
            region = self.config.get("region", "us-east-1")
            s3_staging_dir = self.config.get("s3_staging_dir")
            database = self.config.get("database")
            access_key = self.config.get("access_key") or self.config.get(
                "aws_access_key_id"
            )
            secret_key = self.config.get("secret_key") or self.config.get(
                "aws_secret_access_key"
            )
            workgroup = self.config.get("workgroup")

            if not s3_staging_dir:
                raise ValueError(
                    "AthenaSource requires 's3_staging_dir' in config."
                )

            # awsathena+rest://access_key:secret_key@athena.region.amazonaws.com:443/database?s3_staging_dir=s3://...
            if access_key and secret_key:
                connection_string = f"awsathena+rest://{access_key}:{secret_key}@athena.{region}.amazonaws.com:443/"
            else:
                connection_string = (
                    f"awsathena+rest://:@athena.{region}.amazonaws.com:443/"
                )

            if database:
                connection_string += database

            connection_string += f"?s3_staging_dir={s3_staging_dir}"
            if workgroup:
                connection_string += f"&work_group={workgroup}"

        engine = create_engine(connection_string)
        schema = self.config.get("schema") or self.config.get("database")

        return self._load_data(
            engine,
            table_or_query,
            schema,
            output_dir,
            chunk_rows,
            self.dialect_name,
            pack_config=pack_config,
        )


class SynapseSource(_SqlAlchemySource):
    """Azure Synapse Analytics data source."""

    dialect_name = "synapse"

    def __init__(self, config):
        self.config = config or {}

    def _qualify(self, table_name, schema):
        if schema:
            return f"[{schema}].[{table_name}]", f"{schema}.{table_name}"
        return f"[{table_name}]", table_name

    def get_data(self, table_or_query=None, pack_config=None):
        """Load data from Azure Synapse and write Parquet chunks."""
        output_dir = _ensure_output_dir(pack_config)
        chunk_rows = _chunk_rows(pack_config)

        connection_string = self.config.get("connection_string")
        if not connection_string:
            host = self.config.get("host")  # xxx.sql.azuresynapse.net
            port = self.config.get("port", 1433)
            user = self.config.get("user") or self.config.get("username")
            password = self.config.get("password")
            database = self.config.get("database")

            if not all([host, user, password, database]):
                raise ValueError(
                    "SynapseSource requires 'host', 'user', 'password', and 'database' in config."
                )

            # Use URL.create to avoid password-in-URL pattern detection
            # Uses pymssql driver for Azure Synapse
            url = URL.create(
                drivername="mssql+pymssql",
                username=user,
                password=password,
                host=host,
                port=port,
                database=database,
            )
            connection_string = url

        engine = create_engine(connection_string)
        schema = self.config.get("schema", "dbo")

        return self._load_data(
            engine,
            table_or_query,
            schema,
            output_dir,
            chunk_rows,
            self.dialect_name,
            pack_config=pack_config,
        )


class SqliteSource(DataSource):
    def __init__(self, config):
        self.config = config

    def get_data(self, table_or_query=None, pack_config=None):
        raise NotImplementedError("SqliteSource.get_data Not yet Implemented.")


def get_data_source(source_config):
    type_ = source_config.get("type")
    config = source_config.get("config", {})

    # File sources
    if type_ in ("file", "csv", "excel", "json", "parquet"):
        return FileSource(config.get("path"), config=config)
    elif type_ == "folder":
        return FolderSource(config)

    # Traditional relational databases (via SQLAlchemy)
    elif type_ == "postgresql":
        return DatabaseSource(
            connection_string=config.get("connection_string"),
            config=config,
        )
    elif type_ == "mysql":
        return DatabaseSource(
            connection_string=config.get("connection_string"),
            config=config,
        )
    elif type_ == "oracle":
        return DatabaseSource(
            connection_string=config.get("connection_string"),
            config=config,
        )
    elif type_ == "mssql":
        return DatabaseSource(
            connection_string=config.get("connection_string"),
            config=config,
        )
    elif type_ == "sqlite":
        return DatabaseSource(
            connection_string=config.get("connection_string"),
            config=config,
        )

    # NoSQL databases
    elif type_ == "mongodb":
        return MongoDBSource(config)

    # Cloud object storage
    elif type_ == "s3":
        return S3Source(config)
    elif type_ == "gcs":
        return GCSSource(config)
    elif type_ == "azure_blob":
        return AzureBlobSource(config)
    elif type_ == "hdfs":
        return HDFSSource(config)

    # Data warehouses
    elif type_ == "snowflake":
        return SnowflakeSource(config)
    elif type_ == "bigquery":
        return BigQuerySource(config)
    elif type_ == "databricks":
        return DatabricksSource(config)
    elif type_ == "redshift":
        return RedshiftSource(config)
    elif type_ == "clickhouse":
        return ClickHouseSource(config)
    elif type_ == "duckdb":
        return DuckDBSource(config)
    elif type_ == "trino":
        return TrinoSource(config)

    # Enterprise databases
    elif type_ == "teradata":
        return TeradataSource(config)
    elif type_ == "sap_hana":
        return SapHanaSource(config)
    elif type_ == "cassandra":
        return CassandraSource(config)
    elif type_ == "elasticsearch":
        return ElasticsearchSource(config)
    elif type_ == "ibm_db2":
        return IbmDb2Source(config)
    elif type_ == "athena":
        return AthenaSource(config)
    elif type_ == "synapse":
        return SynapseSource(config)

    else:
        raise ValueError(f"Unsupported source type: {type_}")
