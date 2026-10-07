"""
# QALITA (c) COPYRIGHT 2025 - ALL RIGHTS RESERVED -
Tests for qalita_core.data_source_opener module
"""

import pytest
import logging
import os
import json
import sqlite3
import tempfile
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError, ProgrammingError

from qalita_core.data_source_opener import (
    FileSource,
    DatabaseSource,
    _SqlAlchemySource,
    S3Source,
    GCSSource,
    AzureBlobSource,
    HDFSSource,
    FolderSource,
    MongoDBSource,
    RedshiftSource,
    ClickHouseSource,
    SqliteSource,
    get_data_source,
    _ensure_output_dir,
    _chunk_rows,
    _build_base_name,
    _build_parquet_path,
    _infer_format_from_path,
    _materialize_remote_to_parquet,
    _csv_read_options,
    DEFAULT_PORTS,
)


class TestDefaultPorts:
    """Tests for DEFAULT_PORTS constant."""

    def test_postgresql_port(self):
        assert DEFAULT_PORTS["5432"] == "postgresql"

    def test_mysql_port(self):
        assert DEFAULT_PORTS["3306"] == "mysql"

    def test_mssql_port(self):
        assert DEFAULT_PORTS["1433"] == "mssql+pymssql"

    def test_oracle_port(self):
        assert DEFAULT_PORTS["1521"] == "oracle"

    def test_mongodb_port(self):
        assert DEFAULT_PORTS["27017"] == "mongodb"

    def test_sqlite_port(self):
        assert DEFAULT_PORTS["5000"] == "sqlite"


class TestHelperFunctions:
    """Tests for helper utility functions."""

    def test_ensure_output_dir_creates_directory(self, tmp_path):
        pack_config = {"parquet_output_dir": str(tmp_path / "new_dir")}
        result = _ensure_output_dir(pack_config)
        assert os.path.exists(result)
        assert result == str(tmp_path / "new_dir")

    def test_ensure_output_dir_default(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _ensure_output_dir(None)
        assert result == "./parquet"
        assert os.path.exists(result)

    def test_ensure_output_dir_empty_config(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _ensure_output_dir({})
        assert result == "./parquet"

    def test_build_base_name(self):
        result = _build_base_name("file", "testdata")
        assert result == "file_testdata"

    def test_build_base_name_with_special_chars(self):
        result = _build_base_name("file", "Test Data!")
        # Should be slugified
        assert "_" in result or result == "file_test_data_"

    def test_build_parquet_path(self, tmp_path):
        result = _build_parquet_path(str(tmp_path), "data", 1)
        assert result.endswith("data_part_1.parquet")
        assert str(tmp_path) in result

    def test_infer_format_csv(self):
        assert _infer_format_from_path("data.csv") == "csv"

    def test_infer_format_json(self):
        assert _infer_format_from_path("data.json") == "json"

    def test_infer_format_parquet(self):
        assert _infer_format_from_path("data.parquet") == "parquet"
        assert _infer_format_from_path("data.pq") == "parquet"

    def test_infer_format_excel(self):
        assert _infer_format_from_path("data.xlsx") == "excel"
        assert _infer_format_from_path("data.xls") == "excel"

    def test_infer_format_explicit_override(self):
        result = _infer_format_from_path("data.csv", explicit_format="json")
        assert result == "json"

    def test_infer_format_unknown_defaults_to_csv(self):
        assert _infer_format_from_path("data.unknown") == "csv"


class TestFileSource:
    """Tests for FileSource class."""

    def test_init(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("col1,col2\n1,2\n3,4\n")
        source = FileSource(str(csv_path))
        assert source.file_path == str(csv_path)

    def test_get_data_csv(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("col1,col2\n1,2\n3,4\n5,6\n")

        source = FileSource(str(csv_path))
        out_dir = tmp_path / "output"
        pack_config = {"parquet_output_dir": str(out_dir)}

        paths = source.get_data(pack_config=pack_config)
        assert isinstance(paths, list)
        assert len(paths) > 0
        assert all(p.endswith(".parquet") for p in paths)

    def test_get_data_csv_large_chunked(self, tmp_path):
        """Test CSV chunking with large file."""
        csv_path = tmp_path / "large.csv"
        with open(csv_path, "w") as f:
            f.write("id,value\n")
            for i in range(2500):
                f.write(f"{i},{i*2}\n")

        source = FileSource(str(csv_path))
        out_dir = tmp_path / "output"
        pack_config = {
            "parquet_output_dir": str(out_dir),
            "chunk_rows": 1000,
        }

        paths = source.get_data(pack_config=pack_config)
        assert len(paths) == 3  # 2500 rows / 1000 chunks = 3 files

    def test_get_data_directory(self, tmp_path):
        """Test loading from directory."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("col1,col2\n1,2\n3,4\n")

        source = FileSource(str(tmp_path))
        out_dir = tmp_path / "output"
        pack_config = {"parquet_output_dir": str(out_dir)}

        paths = source.get_data(pack_config=pack_config)
        assert len(paths) > 0

    def test_get_data_nonexistent_file(self, tmp_path):
        source = FileSource(str(tmp_path / "nonexistent.csv"))
        with pytest.raises(FileNotFoundError):
            source.get_data()

    def test_get_data_unsupported_extension(self, tmp_path):
        txt_path = tmp_path / "test.txt"
        txt_path.write_text("some text content")

        source = FileSource(str(txt_path))
        with pytest.raises(ValueError):
            source.get_data()


class TestBorrowedPaths:
    """Paths handed back unstaged are flagged so cleanup never deletes them."""

    def test_parquet_file_is_borrowed(self, tmp_path):
        parquet_path = tmp_path / "test.parquet"
        pl.DataFrame({"a": [1]}).write_parquet(parquet_path)
        source = FileSource(str(parquet_path))
        paths = source.get_data(
            pack_config={"parquet_output_dir": str(tmp_path / "out")}
        )
        assert paths == [str(parquet_path)]
        assert source.borrowed_paths == {str(parquet_path)}

    def test_staged_csv_parts_are_not_borrowed(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("a\n1\n")
        source = FileSource(str(csv_path))
        paths = source.get_data(
            pack_config={"parquet_output_dir": str(tmp_path / "out")}
        )
        assert paths and source.borrowed_paths == set()

    def test_remote_parquet_read_in_place_is_borrowed(self, tmp_path):
        parquet_path = tmp_path / "remote.parquet"
        pl.DataFrame({"a": [1]}).write_parquet(parquet_path)
        source = FileSource(str(parquet_path))
        paths = _materialize_remote_to_parquet(
            source,
            str(parquet_path),
            "parquet",
            None,
            {"parquet_output_dir": str(tmp_path / "out")},
        )
        assert paths == [str(parquet_path)]
        assert source.borrowed_paths == {str(parquet_path)}


class TestDatabaseSource:
    """Tests for DatabaseSource class."""

    def test_init_with_connection_string(self, tmp_path):
        db_path = tmp_path / "test.db"
        # Create an empty database
        conn = sqlite3.connect(db_path)
        conn.close()

        source = DatabaseSource(connection_string=f"sqlite:///{db_path}")
        assert source.engine is not None

    def test_init_with_config(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        conn.close()

        config = {
            "port": "5000",  # SQLite port in DEFAULT_PORTS
            "database": str(db_path),
        }
        source = DatabaseSource(config=config)
        assert source.engine is not None

    def test_init_without_config_raises(self):
        with pytest.raises(ValueError):
            DatabaseSource()

    def test_get_data_from_table(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("CREATE TABLE items(id INTEGER PRIMARY KEY, val TEXT)")
        cur.executemany(
            "INSERT INTO items(val) VALUES (?)", [("a",), ("b",), ("c",)]
        )
        conn.commit()
        conn.close()

        source = DatabaseSource(connection_string=f"sqlite:///{db_path}")
        out_dir = tmp_path / "output"
        pack_config = {"parquet_output_dir": str(out_dir)}

        paths = source.get_data("items", pack_config=pack_config)
        assert len(paths) > 0
        assert all(p.endswith(".parquet") for p in paths)

    def test_get_data_from_query(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("CREATE TABLE t(id INTEGER PRIMARY KEY, val INTEGER)")
        cur.executemany(
            "INSERT INTO t(val) VALUES (?)", [(i,) for i in range(100)]
        )
        conn.commit()
        conn.close()

        source = DatabaseSource(connection_string=f"sqlite:///{db_path}")
        out_dir = tmp_path / "output"
        pack_config = {"parquet_output_dir": str(out_dir)}

        paths = source.get_data(
            "SELECT * FROM t WHERE val > 50", pack_config=pack_config
        )
        assert len(paths) > 0

    def test_get_data_all_tables(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("CREATE TABLE table1(id INTEGER)")
        cur.execute("INSERT INTO table1 VALUES (1)")
        cur.execute("CREATE TABLE table2(id INTEGER)")
        cur.execute("INSERT INTO table2 VALUES (2)")
        conn.commit()
        conn.close()

        source = DatabaseSource(connection_string=f"sqlite:///{db_path}")
        out_dir = tmp_path / "output"
        pack_config = {"parquet_output_dir": str(out_dir)}

        paths = source.get_data("*", pack_config=pack_config)
        assert len(paths) >= 2

    def test_is_sql_query_detection(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        conn.close()

        source = DatabaseSource(connection_string=f"sqlite:///{db_path}")

        assert source._is_sql_query("SELECT * FROM table") is True
        assert (
            source._is_sql_query("WITH cte AS (SELECT 1) SELECT * FROM cte")
            is True
        )
        assert source._is_sql_query("my_table") is False
        assert source._is_sql_query("SELECT;multiple") is True


def _make_scan_db(path, readable=("alpha", "beta"), unreadable=("forbidden",)):
    """A database with real readable tables and catalog-visible broken views.

    Permission tests replace only the broken view's read with a structured
    SQLAlchemy driver error; successful objects still use the real SQL writer.
    """
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    for index, name in enumerate(readable):
        cur.execute(f"CREATE TABLE {name}(id INTEGER)")
        cur.execute(f"INSERT INTO {name} VALUES ({index})")
    for name in unreadable:
        cur.execute(f"CREATE VIEW {name} AS SELECT * FROM missing_{name}")
    conn.commit()
    conn.close()


class _PostgresDriverError(Exception):
    """Driver-shaped error carrying PostgreSQL's structured SQLSTATE."""

    def __init__(self, message, sqlstate):
        super().__init__(message)
        self.pgcode = sqlstate
        self.sqlstate = sqlstate


def _permission_error(table_name):
    return ProgrammingError(
        f"SELECT * FROM {table_name}",
        {},
        _PostgresDriverError("permission denied", "42501"),
    )


def _connection_drop(table_name):
    return OperationalError(
        f"SELECT * FROM {table_name}",
        {},
        _PostgresDriverError("connection dropped", "08006"),
        connection_invalidated=True,
    )


def _fail_tables(monkeypatch, source, failures):
    """Keep successful reads real and inject driver/writer boundary failures."""

    original = source._read_table_to_parquet

    def _read_table(
        engine,
        table_name,
        schema,
        output_dir,
        chunk_rows,
        dialect_name=None,
        pack_config=None,
    ):
        failure = failures.get(table_name)
        if failure is not None:
            raise failure
        return original(
            engine,
            table_name,
            schema,
            output_dir,
            chunk_rows,
            dialect_name,
            pack_config=pack_config,
        )

    monkeypatch.setattr(source, "_read_table_to_parquet", _read_table)


class _WarehouseStub(_SqlAlchemySource):
    """Stands in for the sources that share ``_SqlAlchemySource._load_data``.

    Snowflake, BigQuery, ClickHouse, Teradata, SAP HANA, DB2, Athena and
    Synapse all dispatch through it; a sqlite engine exercises that shared loop
    without any of their drivers.
    """

    dialect_name = "warehouse"

    def __init__(self, engine):
        self.engine = engine

    def get_data(self, table_or_query=None, pack_config=None):
        return self._load_data(
            self.engine,
            table_or_query,
            None,
            _ensure_output_dir(pack_config),
            _chunk_rows(pack_config),
            pack_config=pack_config,
        )


class TestScanSkipsUnreadableObjects:
    """``table_or_query`` defaults to ``"*"``, which covers the whole schema.

    Every table *and view* of it, technical objects included, so on a base with
    fine-grained grants the first refused ``SELECT`` used to kill the job —
    after it had already streamed every readable table.
    """

    def test_unreadable_object_is_skipped_and_recorded(
        self, tmp_path, monkeypatch
    ):
        db_path = tmp_path / "partial.db"
        _make_scan_db(db_path)

        source = DatabaseSource(connection_string=f"sqlite:///{db_path}")
        _fail_tables(
            monkeypatch,
            source,
            {"forbidden": _permission_error("forbidden")},
        )
        paths = source.get_data(
            "*", pack_config={"parquet_output_dir": str(tmp_path / "out")}
        )

        assert set(source.object_paths) == {"sqlite_alpha", "sqlite_beta"}
        assert sorted(paths) == sorted(
            p for parts in source.object_paths.values() for p in parts
        )

        # The tables that are missing from the scan are named, with the reason.
        assert [item["object"] for item in source.skipped_objects] == [
            "forbidden"
        ]
        skipped = source.skipped_objects[0]
        assert skipped["error"] == "ProgrammingError"
        assert "permission denied" in skipped["reason"]

    def test_the_scan_logs_a_recap_of_what_it_skipped(
        self, tmp_path, monkeypatch, caplog
    ):
        # A partial scan nobody is told about is worse than the abort it
        # replaces, so the recap is part of the contract.
        db_path = tmp_path / "recap.db"
        _make_scan_db(db_path)

        source = DatabaseSource(connection_string=f"sqlite:///{db_path}")
        _fail_tables(
            monkeypatch,
            source,
            {"forbidden": _permission_error("forbidden")},
        )
        with caplog.at_level(
            logging.WARNING, logger="qalita_core.data_source_opener"
        ):
            source.get_data(
                "*", pack_config={"parquet_output_dir": str(tmp_path / "out")}
            )

        messages = [record.getMessage() for record in caplog.records]
        assert any(
            "skipping forbidden" in message and "ProgrammingError" in message
            for message in messages
        )
        recap = [m for m in messages if "absent from this scan" in m]
        assert len(recap) == 1
        assert "1 of 3 objects" in recap[0] and "forbidden" in recap[0]

    def test_scan_fails_when_no_object_could_be_read(
        self, tmp_path, monkeypatch
    ):
        # Otherwise a dropped connection would return an empty scan as a
        # success, which is the failure mode the per-table skip introduces.
        db_path = tmp_path / "denied.db"
        _make_scan_db(db_path, readable=(), unreadable=("forbidden", "other"))

        source = DatabaseSource(connection_string=f"sqlite:///{db_path}")
        _fail_tables(
            monkeypatch,
            source,
            {
                "forbidden": _permission_error("forbidden"),
                "other": _permission_error("other"),
            },
        )
        with pytest.raises(RuntimeError) as excinfo:
            source.get_data(
                "*", pack_config={"parquet_output_dir": str(tmp_path / "out")}
            )

        message = str(excinfo.value)
        assert "forbidden" in message and "other" in message
        assert "ProgrammingError" in message
        assert source.skipped_objects != []

    def test_empty_schema_still_raises_value_error(self, tmp_path):
        db_path = tmp_path / "empty.db"
        sqlite3.connect(db_path).close()

        source = DatabaseSource(connection_string=f"sqlite:///{db_path}")
        with pytest.raises(ValueError, match="No tables found"):
            source.get_data(
                "*", pack_config={"parquet_output_dir": str(tmp_path / "out")}
            )

    def test_a_fully_readable_scan_is_unchanged(self, tmp_path):
        db_path = tmp_path / "readable.db"
        _make_scan_db(db_path, unreadable=())

        source = DatabaseSource(connection_string=f"sqlite:///{db_path}")
        paths = source.get_data(
            "*", pack_config={"parquet_output_dir": str(tmp_path / "out")}
        )

        assert set(source.object_paths) == {"sqlite_alpha", "sqlite_beta"}
        assert len(paths) == 2
        assert source.skipped_objects == []

    def test_explicit_table_list_skips_the_unreadable_one(
        self, tmp_path, monkeypatch
    ):
        db_path = tmp_path / "listed.db"
        _make_scan_db(db_path)

        source = DatabaseSource(connection_string=f"sqlite:///{db_path}")
        _fail_tables(
            monkeypatch,
            source,
            {"forbidden": _permission_error("forbidden")},
        )
        paths = source.get_data(
            ["alpha", "forbidden", "beta"],
            pack_config={"parquet_output_dir": str(tmp_path / "out")},
        )

        assert set(source.object_paths) == {"sqlite_alpha", "sqlite_beta"}
        assert len(paths) == 2
        assert [item["object"] for item in source.skipped_objects] == [
            "forbidden"
        ]

    def test_a_single_named_table_keeps_its_own_error(self, tmp_path):
        # Nothing to fall back on when one table was asked for by name: the
        # driver's error is more useful than a one-line scan summary.
        db_path = tmp_path / "named.db"
        _make_scan_db(db_path)

        source = DatabaseSource(connection_string=f"sqlite:///{db_path}")
        with pytest.raises(OperationalError):
            source.get_data(
                "forbidden",
                pack_config={"parquet_output_dir": str(tmp_path / "out")},
            )
        assert source.skipped_objects == []

    def test_shared_warehouse_dispatch_skips_the_unreadable_one(
        self, tmp_path, monkeypatch
    ):
        db_path = tmp_path / "warehouse.db"
        _make_scan_db(db_path, readable=("alpha",))

        source = _WarehouseStub(create_engine(f"sqlite:///{db_path}"))
        _fail_tables(
            monkeypatch,
            source,
            {"forbidden": _permission_error("forbidden")},
        )
        paths = source.get_data(
            ["alpha", "forbidden"],
            pack_config={"parquet_output_dir": str(tmp_path / "out")},
        )

        assert list(source.object_paths) == ["warehouse_alpha"]
        assert len(paths) == 1
        assert [item["object"] for item in source.skipped_objects] == [
            "forbidden"
        ]

    def test_connection_drop_after_success_is_reraised(
        self, tmp_path, monkeypatch
    ):
        db_path = tmp_path / "connection-drop.db"
        _make_scan_db(db_path, unreadable=())
        source = DatabaseSource(connection_string=f"sqlite:///{db_path}")
        failure = _connection_drop("beta")
        _fail_tables(monkeypatch, source, {"beta": failure})

        with pytest.raises(OperationalError) as excinfo:
            source.get_data(
                ["alpha", "beta"],
                pack_config={"parquet_output_dir": str(tmp_path / "out")},
            )

        assert excinfo.value is failure

    def test_writer_error_after_success_is_reraised(
        self, tmp_path, monkeypatch
    ):
        db_path = tmp_path / "writer-error.db"
        _make_scan_db(db_path, unreadable=())
        source = DatabaseSource(connection_string=f"sqlite:///{db_path}")
        failure = OSError("parquet writer failed")
        _fail_tables(monkeypatch, source, {"beta": failure})

        with pytest.raises(OSError) as excinfo:
            source.get_data(
                ["alpha", "beta"],
                pack_config={"parquet_output_dir": str(tmp_path / "out")},
            )

        assert excinfo.value is failure

    def test_catalog_introspection_failure_is_reraised(
        self, tmp_path, monkeypatch
    ):
        failure = OperationalError(
            "catalog query",
            {},
            _PostgresDriverError("catalog unavailable", "08006"),
            connection_invalidated=True,
        )

        class _FailingInspector:
            def get_table_names(self, schema=None):
                raise failure

        source = _WarehouseStub(object())
        monkeypatch.setattr(
            "qalita_core.data_source_opener.inspect",
            lambda engine: _FailingInspector(),
        )

        with pytest.raises(OperationalError) as excinfo:
            source.get_data(
                "*",
                pack_config={"parquet_output_dir": str(tmp_path / "out")},
            )

        assert excinfo.value is failure

    def test_empty_explicit_table_list_is_rejected(self, tmp_path):
        source = _WarehouseStub(object())

        with pytest.raises(ValueError, match="No objects"):
            source.get_data(
                [],
                pack_config={"parquet_output_dir": str(tmp_path / "out")},
            )

    def test_second_scan_resets_skipped_objects(self, tmp_path, monkeypatch):
        db_path = tmp_path / "reset-skips.db"
        _make_scan_db(db_path)
        source = DatabaseSource(connection_string=f"sqlite:///{db_path}")
        _fail_tables(
            monkeypatch,
            source,
            {"forbidden": _permission_error("forbidden")},
        )

        source.get_data(
            ["alpha", "forbidden"],
            pack_config={"parquet_output_dir": str(tmp_path / "first")},
        )
        assert [item["object"] for item in source.skipped_objects] == [
            "forbidden"
        ]

        source.get_data(
            ["beta"],
            pack_config={"parquet_output_dir": str(tmp_path / "second")},
        )

        assert source.skipped_objects == []

    def test_redshift_propagates_a_snapshot_of_internal_skips(
        self, monkeypatch
    ):
        skipped = [
            {
                "object": "forbidden",
                "error": "ProgrammingError",
                "reason": "permission denied",
            }
        ]

        class _InternalSource:
            object_paths = {"redshift_alpha": ["alpha.parquet"]}
            skipped_objects = skipped

            def __init__(self, connection_string=None, config=None):
                pass

            def get_data(self, table_or_query=None, pack_config=None):
                return ["alpha.parquet"]

        monkeypatch.setattr(
            "qalita_core.data_source_opener.DatabaseSource", _InternalSource
        )
        source = RedshiftSource(
            {"connection_string": "postgresql://example.invalid/db"}
        )

        assert source.get_data(["alpha", "forbidden"]) == ["alpha.parquet"]
        assert source.skipped_objects == skipped

        skipped[0]["reason"] = "mutated after load"
        skipped.append({"object": "later"})
        assert source.skipped_objects == [
            {
                "object": "forbidden",
                "error": "ProgrammingError",
                "reason": "permission denied",
            }
        ]


class TestCatalogIntrospectionFailures:
    def test_database_source_preserves_catalog_connection_error(
        self, tmp_path, monkeypatch
    ):
        failure = OperationalError(
            "catalog query",
            {},
            _PostgresDriverError("connection dropped", "08006"),
            connection_invalidated=True,
        )

        class _Inspector:
            def get_table_names(self, schema=None):
                raise failure

            def get_view_names(self, schema=None):
                return []

        source = DatabaseSource(connection_string="sqlite://")
        monkeypatch.setattr(
            "qalita_core.data_source_opener.inspect",
            lambda engine: _Inspector(),
        )

        with pytest.raises(OperationalError) as excinfo:
            source.get_data(
                "*",
                pack_config={"parquet_output_dir": str(tmp_path / "out")},
            )

        assert excinfo.value is failure

    def test_clickhouse_uses_show_tables_when_inspection_is_unsupported(
        self, monkeypatch
    ):
        class _Inspector:
            def get_table_names(self, schema=None):
                raise NotImplementedError("ClickHouse inspection unsupported")

        class _Connection:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def execute(self, statement):
                assert str(statement) == "SHOW TABLES FROM analytics"
                return [("alpha",), ("beta",)]

        class _Engine:
            def connect(self):
                return _Connection()

        monkeypatch.setattr(
            "qalita_core.data_source_opener.inspect",
            lambda engine: _Inspector(),
        )
        source = ClickHouseSource({"database": "analytics"})

        assert source._list_tables(_Engine(), None) == ["alpha", "beta"]

    def test_clickhouse_preserves_show_tables_connection_error(
        self, monkeypatch
    ):
        failure = OperationalError(
            "SHOW TABLES",
            {},
            _PostgresDriverError("connection dropped", "08006"),
            connection_invalidated=True,
        )

        class _Inspector:
            def get_table_names(self, schema=None):
                raise NotImplementedError("ClickHouse inspection unsupported")

        class _Engine:
            def connect(self):
                raise failure

        monkeypatch.setattr(
            "qalita_core.data_source_opener.inspect",
            lambda engine: _Inspector(),
        )
        source = ClickHouseSource({})

        with pytest.raises(OperationalError) as excinfo:
            source._list_tables(_Engine(), None)

        assert excinfo.value is failure


class TestGetDataSource:
    """Tests for get_data_source factory function."""

    def test_file_source(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("col1\n1\n")

        source_config = {"type": "file", "config": {"path": str(csv_path)}}
        source = get_data_source(source_config)
        assert isinstance(source, FileSource)

    def test_csv_source(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("col1\n1\n")

        source_config = {"type": "csv", "config": {"path": str(csv_path)}}
        source = get_data_source(source_config)
        assert isinstance(source, FileSource)

    def test_postgresql_source(self):
        source_config = {
            "type": "postgresql",
            # Using placeholder URL format - no actual credentials
            "config": {"connection_string": "postgresql://localhost/testdb"},
        }
        source = get_data_source(source_config)
        assert isinstance(source, DatabaseSource)

    def test_mysql_source(self):
        """Test MySQL source creation (skipped if MySQLdb not available)."""
        pytest.importorskip("MySQLdb", reason="MySQLdb not available")
        source_config = {
            "type": "mysql",
            # Using placeholder URL format - no actual credentials
            "config": {"connection_string": "mysql://localhost/testdb"},
        }
        source = get_data_source(source_config)
        assert isinstance(source, DatabaseSource)

    def test_sqlite_source(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        conn.close()

        source_config = {
            "type": "sqlite",
            "config": {"connection_string": f"sqlite:///{db_path}"},
        }
        source = get_data_source(source_config)
        assert isinstance(source, DatabaseSource)

    def test_s3_source(self):
        source_config = {
            "type": "s3",
            "config": {"path": "s3://bucket/key.csv"},
        }
        source = get_data_source(source_config)
        assert isinstance(source, S3Source)

    def test_gcs_source(self):
        source_config = {
            "type": "gcs",
            "config": {"path": "gs://bucket/key.csv"},
        }
        source = get_data_source(source_config)
        assert isinstance(source, GCSSource)

    def test_azure_blob_source(self):
        source_config = {
            "type": "azure_blob",
            "config": {
                "path": "abfs://container@account.dfs.core.windows.net/key.csv"
            },
        }
        source = get_data_source(source_config)
        assert isinstance(source, AzureBlobSource)

    def test_hdfs_source(self):
        source_config = {
            "type": "hdfs",
            "config": {"path": "hdfs://host:8020/path/file.csv"},
        }
        source = get_data_source(source_config)
        assert isinstance(source, HDFSSource)

    def test_folder_source(self):
        source_config = {"type": "folder", "config": {"path": "/some/path"}}
        source = get_data_source(source_config)
        assert isinstance(source, FolderSource)

    def test_unsupported_source_type(self):
        source_config = {"type": "unsupported", "config": {}}
        with pytest.raises(ValueError, match="Unsupported source type"):
            get_data_source(source_config)


class TestS3Source:
    """Tests for S3Source class."""

    def test_init(self):
        config = {"path": "s3://bucket/key.csv"}
        source = S3Source(config)
        assert source.config == config

    def test_get_data_missing_path_raises(self):
        source = S3Source({})
        with pytest.raises(ValueError):
            source.get_data()

    def test_get_data_constructs_path_from_bucket_key(self):
        source = S3Source({"bucket": "mybucket", "key": "mykey.parquet"})
        # This will fail at actual S3 access, but we can verify path construction
        # by checking the exception message or using mocks in integration tests

    def test_credentials_reach_the_scan(self):
        """The options used to be built and then dropped on the parquet path."""
        source = S3Source(
            {
                "path": "s3://bucket/key.parquet",
                "key": "AKIA",
                "secret": "shh",
                "token": "sess",
                "client_kwargs": {"region_name": "eu-west-3"},
            }
        )
        assert source._storage_options() == {
            "aws_access_key_id": "AKIA",
            "aws_secret_access_key": "shh",
            "aws_session_token": "sess",
            "aws_region": "eu-west-3",
        }

    def test_no_credentials_means_no_options(self):
        assert (
            S3Source({"path": "s3://bucket/key.csv"})._storage_options()
            is None
        )


class TestRemoteMaterialization:
    """Remote parquet: pass through when it can be read, stage when it cannot."""

    def _parquet(self, tmp_path):
        path = tmp_path / "remote.parquet"
        pl.DataFrame({"a": [1, 2, 3]}).write_parquet(path)
        return str(path)

    def test_public_parquet_is_passed_through(self, tmp_path):
        source = S3Source({"path": "s3://bucket/remote.parquet"})
        paths = _materialize_remote_to_parquet(
            source,
            self._parquet(tmp_path),
            "parquet",
            None,
            {"parquet_output_dir": str(tmp_path / "out")},
        )
        assert paths == [self._parquet(tmp_path)]
        assert list(source.object_paths) == ["remote_remote"]

    def test_private_parquet_is_staged_with_its_credentials(self, tmp_path):
        # get_data returns bare paths, so credentials cannot travel with them:
        # the object is staged rather than silently scanned anonymously later.
        source = S3Source({"path": "s3://bucket/remote.parquet", "key": "AK"})
        paths = _materialize_remote_to_parquet(
            source,
            self._parquet(tmp_path),
            "parquet",
            source._storage_options(),
            {"parquet_output_dir": str(tmp_path / "out")},
        )
        assert [os.path.basename(p) for p in paths] == [
            "remote_remote_part_1.parquet"
        ]
        assert pl.scan_parquet(paths).collect().height == 3


class TestGCSSource:
    """Tests for GCSSource class."""

    def test_init(self):
        config = {"path": "gs://bucket/key.csv"}
        source = GCSSource(config)
        assert source.config == config

    def test_get_data_missing_path_raises(self):
        source = GCSSource({})
        with pytest.raises(ValueError):
            source.get_data()


class TestAzureBlobSource:
    """Tests for AzureBlobSource class."""

    def test_init(self):
        config = {
            "path": "abfs://container@account.dfs.core.windows.net/key.csv"
        }
        source = AzureBlobSource(config)
        assert source.config == config

    def test_get_data_missing_path_raises(self):
        source = AzureBlobSource({})
        with pytest.raises(ValueError):
            source.get_data()


class TestHDFSSource:
    """Tests for HDFSSource class."""

    def test_init(self):
        config = {"path": "hdfs://host:8020/path/file.csv"}
        source = HDFSSource(config)
        assert source.config == config

    def test_get_data_missing_path_raises(self):
        source = HDFSSource({})
        with pytest.raises(ValueError):
            source.get_data()


class TestFolderSource:
    """Tests for FolderSource class."""

    def test_init(self):
        config = {"path": "/some/path"}
        source = FolderSource(config)
        assert source.config == config

    @staticmethod
    def _folder(tmp_path):
        folder = tmp_path / "warehouse"
        folder.mkdir()
        (folder / "patients.csv").write_text("id,sex\n1,F\n2,M\n")
        pl.DataFrame({"stay": [10, 11, 12]}).write_parquet(
            folder / "stays.parquet"
        )
        (folder / "README.md").write_text("not data")
        (folder / ".hidden.csv").write_text("a\n1\n")
        (folder / "~$patients.xlsx").write_text("excel lock file")
        return folder

    @staticmethod
    def _config(tmp_path):
        return {"parquet_output_dir": str(tmp_path / "out")}

    def test_every_data_file_is_one_object(self, tmp_path):
        folder = self._folder(tmp_path)
        source = FolderSource({"path": str(folder)})
        paths = source.get_data(pack_config=self._config(tmp_path))

        assert sorted(source.object_paths) == ["file_patients", "file_stays"]
        assert sorted(paths) == sorted(
            p for parts in source.object_paths.values() for p in parts
        )
        patients = pl.scan_parquet(source.object_paths["file_patients"])
        assert patients.collect()["sex"].to_list() == ["F", "M"]

    def test_parquet_files_are_read_in_place_and_borrowed(self, tmp_path):
        folder = self._folder(tmp_path)
        source = FolderSource({"path": str(folder)})
        source.get_data(pack_config=self._config(tmp_path))
        stays = str(folder / "stays.parquet")
        assert source.object_paths["file_stays"] == [stays]
        assert source.borrowed_paths == {stays}

    def test_star_selects_everything(self, tmp_path):
        folder = self._folder(tmp_path)
        source = FolderSource({"path": str(folder)})
        source.get_data("*", pack_config=self._config(tmp_path))
        assert len(source.object_paths) == 2

    def test_select_by_name_with_or_without_extension(self, tmp_path):
        folder = self._folder(tmp_path)
        for name in ("patients", "patients.csv", "PATIENTS"):
            source = FolderSource({"path": str(folder)})
            source.get_data(name, pack_config=self._config(tmp_path))
            assert list(source.object_paths) == ["file_patients"], name

    def test_select_a_list(self, tmp_path):
        folder = self._folder(tmp_path)
        source = FolderSource({"path": str(folder)})
        source.get_data(
            ["stays", "patients"], pack_config=self._config(tmp_path)
        )
        assert sorted(source.object_paths) == ["file_patients", "file_stays"]

    def test_unknown_name_lists_what_exists(self, tmp_path):
        folder = self._folder(tmp_path)
        source = FolderSource({"path": str(folder)})
        with pytest.raises(FileNotFoundError, match="patients, stays"):
            source.get_data("visits", pack_config=self._config(tmp_path))

    def test_missing_or_empty_folder(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            FolderSource({"path": str(tmp_path / "nope")}).get_data(
                pack_config=self._config(tmp_path)
            )
        with pytest.raises(FileNotFoundError):
            FolderSource({}).get_data(pack_config=self._config(tmp_path))
        empty = tmp_path / "empty"
        empty.mkdir()
        (empty / "notes.txt").write_text("x")
        with pytest.raises(FileNotFoundError, match="No data file"):
            FolderSource({"path": str(empty)}).get_data(
                pack_config=self._config(tmp_path)
            )

    def test_unreadable_file_is_skipped_in_a_full_scan(self, tmp_path):
        folder = self._folder(tmp_path)
        (folder / "broken.parquet").write_text("not parquet at all")
        (folder / "broken.json").write_text("{ not json")
        source = FolderSource({"path": str(folder)})
        source.get_data(pack_config=self._config(tmp_path))
        assert sorted(source.object_paths) == ["file_patients", "file_stays"]
        assert sorted(item["object"] for item in source.skipped_objects) == [
            "broken.json",
            "broken.parquet",
        ]

    def test_files_sharing_a_stem_keep_their_suffix(self, tmp_path):
        folder = tmp_path / "f"
        folder.mkdir()
        (folder / "visits.csv").write_text("v\n1\n")
        pl.DataFrame({"v": [2]}).write_parquet(folder / "visits.parquet")
        source = FolderSource({"path": str(folder)})
        source.get_data(pack_config=self._config(tmp_path))
        assert sorted(source.object_paths) == [
            "file_visits_csv",
            "file_visits_parquet",
        ]

    def test_unreadable_file_selected_by_name_raises(self, tmp_path):
        folder = self._folder(tmp_path)
        (folder / "broken.json").write_text("{ not json")
        source = FolderSource({"path": str(folder)})
        with pytest.raises(Exception):
            source.get_data("broken", pack_config=self._config(tmp_path))

    def test_nothing_readable_raises(self, tmp_path):
        folder = tmp_path / "bad"
        folder.mkdir()
        (folder / "a.json").write_text("{ nope")
        source = FolderSource({"path": str(folder)})
        with pytest.raises(RuntimeError, match="None of the 1 files"):
            source.get_data(pack_config=self._config(tmp_path))

    def test_recursive_is_opt_in(self, tmp_path):
        folder = self._folder(tmp_path)
        (folder / "2024").mkdir()
        (folder / "2024" / "visits.csv").write_text("v\n1\n")
        (folder / ".git").mkdir()
        (folder / ".git" / "x.csv").write_text("v\n1\n")

        flat = FolderSource({"path": str(folder)})
        flat.get_data(pack_config=self._config(tmp_path))
        assert "file_2024_visits" not in flat.object_paths

        deep = FolderSource({"path": str(folder), "recursive": True})
        deep.get_data(pack_config=self._config(tmp_path))
        assert sorted(deep.object_paths) == [
            "file_2024_visits",
            "file_patients",
            "file_stays",
        ]
        deep_again = FolderSource({"path": str(folder), "recursive": True})
        deep_again.get_data("2024/visits", pack_config=self._config(tmp_path))
        assert list(deep_again.object_paths) == ["file_2024_visits"]


class TestMongoDBSource:
    """Tests for MongoDBSource class."""

    def test_init(self):
        config = {"connection_string": "mongodb://localhost:27017"}
        source = MongoDBSource(config)
        assert source.config == config

    def test_get_data_requires_database(self):
        """Test that MongoDBSource requires database in config."""
        source = MongoDBSource(
            {"connection_string": "mongodb://localhost:27017"}
        )
        with pytest.raises((ImportError, ValueError)):
            # Will raise ImportError if pymongo not available, or ValueError if database missing
            source.get_data()


class TestSqliteSource:
    """Tests for SqliteSource class."""

    def test_init(self):
        config = {"database": "test.db"}
        source = SqliteSource(config)
        assert source.config == config

    def test_get_data_not_implemented(self):
        source = SqliteSource({})
        with pytest.raises(NotImplementedError):
            source.get_data()


class TestDatabaseSourceSchemaHandling:
    """Tests for DatabaseSource schema handling."""

    def test_schema_from_config(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("CREATE TABLE t(id INTEGER)")
        cur.execute("INSERT INTO t VALUES (1)")
        conn.commit()
        conn.close()

        config = {"schema": "main"}  # SQLite default schema
        source = DatabaseSource(
            connection_string=f"sqlite:///{db_path}",
            config=config,
        )
        assert source.config["schema"] == "main"

    def test_fully_qualified_table_name(self, tmp_path):
        """Test handling of SCHEMA.TABLE format."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("CREATE TABLE items(id INTEGER)")
        cur.execute("INSERT INTO items VALUES (1)")
        conn.commit()
        conn.close()

        source = DatabaseSource(connection_string=f"sqlite:///{db_path}")
        out_dir = tmp_path / "output"
        pack_config = {"parquet_output_dir": str(out_dir)}

        # SQLite doesn't really support schemas the same way, but we test parsing
        # This should split "main.items" into schema="main", table="items"
        paths = source.get_data("items", pack_config=pack_config)
        assert len(paths) > 0


def _make_db(path, tables):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    for name, rows in tables.items():
        cur.execute(f"CREATE TABLE {name}(id INTEGER, v TEXT)")
        cur.executemany(f"INSERT INTO {name} VALUES (?,?)", rows)
    conn.commit()
    conn.close()


class TestObjectPaths:
    """Every source records which parts belong to which logical object."""

    def test_database_records_one_entry_per_table(self, tmp_path):
        db_path = tmp_path / "objects.db"
        _make_db(
            db_path,
            {
                "alpha": [(i, f"a{i}") for i in range(2500)],
                "beta": [(i, f"b{i}") for i in range(10)],
            },
        )

        source = DatabaseSource(connection_string=f"sqlite:///{db_path}")
        pack_config = {
            "parquet_output_dir": str(tmp_path / "out"),
            "chunk_rows": 1000,
        }
        paths = source.get_data("*", pack_config=pack_config)

        # The pairing is recorded while writing, so the three parts of "alpha"
        # cannot be mistaken for three separate tables.
        assert set(source.object_paths) == {"sqlite_alpha", "sqlite_beta"}
        assert len(source.object_paths["sqlite_alpha"]) == 3
        assert len(source.object_paths["sqlite_beta"]) == 1
        assert sorted(paths) == sorted(
            p for parts in source.object_paths.values() for p in parts
        )

    def test_file_records_its_object(self, tmp_path):
        csv_path = tmp_path / "people.csv"
        csv_path.write_text("id,name\n1,a\n2,b\n")

        source = FileSource(str(csv_path))
        source.get_data(
            pack_config={"parquet_output_dir": str(tmp_path / "out")}
        )
        assert list(source.object_paths) == ["file_people"]

    def test_two_tables_with_the_same_slug_stay_two_objects(self, tmp_path):
        """``slugify`` folds the accent, so both tables shared one base name.

        The base name is both the ``object_paths`` key and the part-file
        prefix, so the second table used to truncate the first one's
        ``_part_1`` and the two path lists were merged: one table disappeared
        from ``tables()`` and the survivor's rows were counted twice. Mongo
        collections and ES indices hit this with ``-`` and ``.``; SQL hits it
        with accents and with case.
        """
        db_path = tmp_path / "collide.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        for name, count in (("données", 5), ("donnees", 2)):
            cursor.execute(f'CREATE TABLE "{name}"(id INTEGER, v TEXT)')
            cursor.executemany(
                f'INSERT INTO "{name}" VALUES (?,?)',
                [(i, f"{name}-{i}") for i in range(count)],
            )
        conn.commit()
        conn.close()

        source = DatabaseSource(connection_string=f"sqlite:///{db_path}")
        paths = source.get_data(
            "*",
            pack_config={
                "parquet_output_dir": str(tmp_path / "out"),
                # Forces several parts, so a collision overwrites part 1 while
                # part 2 survives — the shape that silently loses rows.
                "chunk_rows": 3,
            },
        )

        assert len(source.object_paths) == 2
        # No part file is claimed by two objects, hence none was overwritten.
        assert len(set(paths)) == len(paths)
        rows = sum(
            pl.scan_parquet(parts).select(pl.len()).collect().item()
            for parts in source.object_paths.values()
        )
        assert rows == 7


class TestStreamingFileFormats:
    """Files always go through the Polars streaming path now."""

    def test_small_csv_is_written_with_zstd(self, tmp_path):
        # The pandas path used to write these with snappy, so parts of one
        # object could differ in compression from one another.
        csv_path = tmp_path / "small.csv"
        csv_path.write_text("id,value\n1,10\n2,20\n")

        source = FileSource(str(csv_path))
        paths = source.get_data(
            pack_config={"parquet_output_dir": str(tmp_path / "out")}
        )
        metadata = pq.ParquetFile(paths[0]).metadata
        compression = metadata.row_group(0).column(0).compression
        assert compression.upper() == "ZSTD"

    def test_ndjson_document_is_detected_without_a_flag(self, tmp_path):
        # json_lines defaulted to False, so NDJSON was parsed as one document.
        json_path = tmp_path / "events.json"
        json_path.write_text(
            "\n".join(json.dumps({"id": i}) for i in range(2500))
        )

        source = FileSource(str(json_path))
        paths = source.get_data(
            pack_config={
                "parquet_output_dir": str(tmp_path / "out"),
                "chunk_rows": 1000,
            }
        )
        assert len(paths) == 3
        assert pl.scan_parquet(paths).select(pl.len()).collect().item() == 2500

    def test_json_array_document_is_streamed(self, tmp_path):
        json_path = tmp_path / "records.json"
        json_path.write_text(
            json.dumps([{"id": i, "name": f"n{i}"} for i in range(2500)])
        )

        source = FileSource(str(json_path))
        paths = source.get_data(
            pack_config={
                "parquet_output_dir": str(tmp_path / "out"),
                "chunk_rows": 1000,
            }
        )
        assert len(paths) == 3
        frame = pl.scan_parquet(paths).collect()
        assert frame.height == 2500
        assert frame["name"][0] == "n0"


class TestSqlStreaming:
    """SQL result sets reach Parquet as Arrow batches, one schema per object."""

    def test_parts_of_a_table_share_a_schema(self, tmp_path):
        db_path = tmp_path / "drift.db"
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("CREATE TABLE t(id INTEGER, v TEXT)")
        # The first chunk is entirely NULL on `v`: inferring per part types it
        # Null there and String in the next one, and scanning the parts
        # together then raises SchemaError.
        cur.executemany(
            "INSERT INTO t VALUES (?,?)",
            [(i, None) for i in range(1000)]
            + [(i, f"v{i}") for i in range(1000, 2000)],
        )
        conn.commit()
        conn.close()

        source = DatabaseSource(connection_string=f"sqlite:///{db_path}")
        paths = source.get_data(
            "t",
            pack_config={
                "parquet_output_dir": str(tmp_path / "out"),
                "chunk_rows": 500,
            },
        )
        assert len(paths) == 4
        frame = pl.scan_parquet(paths).collect()
        assert frame.height == 2000
        assert frame["v"].null_count() == 1000

    def test_empty_table_yields_a_scannable_object(self, tmp_path):
        db_path = tmp_path / "empty.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE blank(id INTEGER, v TEXT)")
        conn.commit()
        conn.close()

        source = DatabaseSource(connection_string=f"sqlite:///{db_path}")
        paths = source.get_data(
            "blank",
            pack_config={"parquet_output_dir": str(tmp_path / "out")},
        )
        assert len(paths) == 1
        assert pl.scan_parquet(paths).select(pl.len()).collect().item() == 0


class TestCsvSourceOptions:
    """CSV reader options declared on the source reach the reader.

    Regression for qalita/core#47: the factory kept only ``config['path']``, so
    a source declaring ``;`` was read as if it were comma-separated.
    """

    # The European shape: semicolon-separated, comma as the decimal mark. Read
    # with the defaults, the header is one field and every data row is three,
    # which is the ComputeError seen on the on-prem deployment.
    EUROPEAN = (
        "bilan_id;patient_id;date_bilan;hba1c_pct;glucose_aj_gl\n"
        "BIO0001_01;PAT0001;2023-01-07;8,4;3,33\n"
        "BIO0002_01;PAT0002;2023-02-11;6,1;1,05\n"
        "BIO0003_01;PAT0003;2023-03-02;7,25;2,4\n"
    )

    def _european(self, tmp_path):
        csv_path = tmp_path / "bilans.csv"
        csv_path.write_text(self.EUROPEAN, encoding="utf-8")
        return csv_path

    @staticmethod
    def _collect(source, tmp_path):
        paths = source.get_data(
            pack_config={"parquet_output_dir": str(tmp_path / "out")}
        )
        return pl.scan_parquet(paths).collect()

    def test_delimiter_and_decimal_separator_are_both_applied(self, tmp_path):
        csv_path = self._european(tmp_path)
        source = FileSource(
            str(csv_path),
            config={
                "path": str(csv_path),
                "delimiter": ";",
                "decimal_separator": ",",
            },
        )

        frame = self._collect(source, tmp_path)

        assert frame.columns == [
            "bilan_id",
            "patient_id",
            "date_bilan",
            "hba1c_pct",
            "glucose_aj_gl",
        ]
        assert frame.height == 3
        # The schema is the assertion that matters: the separator alone gives
        # five columns too, but leaves these two as String and every numeric
        # metric computed from them wrong.
        assert frame.schema["hba1c_pct"] == pl.Float64
        assert frame.schema["glucose_aj_gl"] == pl.Float64
        assert frame["hba1c_pct"].to_list() == [8.4, 6.1, 7.25]

    def test_delimiter_alone_leaves_the_numbers_as_strings(self, tmp_path):
        """Pins the half-fix that turns a crash into silent wrong metrics."""
        csv_path = self._european(tmp_path)
        source = FileSource(
            str(csv_path), config={"path": str(csv_path), "delimiter": ";"}
        )

        frame = self._collect(source, tmp_path)

        assert frame.width == 5
        assert frame.schema["hba1c_pct"] == pl.String

    def test_semicolon_file_without_options_still_fails(self, tmp_path):
        """No ``truncate_ragged_lines`` on this path, on purpose.

        Tolerating ragged lines would return the file as one wide string
        column instead of failing — a green job describing nothing.
        """
        csv_path = self._european(tmp_path)
        source = FileSource(str(csv_path))

        with pytest.raises(pl.exceptions.ComputeError):
            self._collect(source, tmp_path)

    def test_has_header_false_keeps_the_first_line_as_data(self, tmp_path):
        csv_path = tmp_path / "headless.csv"
        csv_path.write_text("1;alpha\n2;beta\n", encoding="utf-8")
        source = FileSource(
            str(csv_path),
            config={
                "path": str(csv_path),
                "delimiter": ";",
                "has_header": False,
            },
        )

        frame = self._collect(source, tmp_path)

        assert frame.height == 2
        assert frame.columns == ["column_1", "column_2"]
        assert frame["column_1"].to_list() == [1, 2]

    def test_has_header_string_false_keeps_the_first_line_as_data(
        self, tmp_path
    ):
        csv_path = tmp_path / "headless-string.csv"
        csv_path.write_text("1;alpha\n2;beta\n", encoding="utf-8")
        source = FileSource(
            str(csv_path),
            config={
                "path": str(csv_path),
                "delimiter": ";",
                "has_header": "false",
            },
        )

        frame = self._collect(source, tmp_path)

        assert frame.height == 2
        assert frame.columns == ["column_1", "column_2"]
        assert frame["column_1"].to_list() == [1, 2]

    def test_decimal_comma_string_false_keeps_decimal_values_as_strings(
        self, tmp_path
    ):
        csv_path = self._european(tmp_path)
        source = FileSource(
            str(csv_path),
            config={
                "path": str(csv_path),
                "delimiter": ";",
                "decimal_comma": "false",
            },
        )

        frame = self._collect(source, tmp_path)

        assert frame.schema["hba1c_pct"] == pl.String
        assert frame["hba1c_pct"].to_list() == ["8,4", "6,1", "7,25"]

    def test_decimal_comma_string_true_infers_decimal_values(self, tmp_path):
        csv_path = self._european(tmp_path)
        source = FileSource(
            str(csv_path),
            config={
                "path": str(csv_path),
                "delimiter": ";",
                "decimal_comma": "true",
            },
        )

        frame = self._collect(source, tmp_path)

        assert frame.schema["hba1c_pct"] == pl.Float64
        assert frame["hba1c_pct"].to_list() == [8.4, 6.1, 7.25]

    def test_utf8_sig_is_accepted_and_the_bom_is_dropped(self, tmp_path):
        """Checked against Polars rather than assumed: it strips the BOM."""
        csv_path = tmp_path / "bom.csv"
        csv_path.write_bytes("id;prix\n1;2,5\n".encode("utf-8-sig"))
        source = FileSource(
            str(csv_path),
            config={
                "path": str(csv_path),
                "delimiter": ";",
                "encoding": "utf-8-sig",
                "decimal_separator": ",",
            },
        )

        frame = self._collect(source, tmp_path)

        assert frame.columns == ["id", "prix"]
        assert frame["prix"].to_list() == [2.5]

    def test_defaults_are_unchanged_when_nothing_is_declared(self, tmp_path):
        csv_path = tmp_path / "plain.csv"
        csv_path.write_text("id,value\n1,10.5\n2,20.25\n", encoding="utf-8")

        bare = self._collect(FileSource(str(csv_path)), tmp_path / "bare")
        configured = self._collect(
            FileSource(str(csv_path), config={"path": str(csv_path)}),
            tmp_path / "configured",
        )

        assert bare.columns == ["id", "value"]
        assert bare.schema["value"] == pl.Float64
        assert configured.equals(bare)

    def test_factory_hands_the_config_to_the_file_source(self, tmp_path):
        csv_path = self._european(tmp_path)
        source = get_data_source(
            {
                "type": "csv",
                "config": {
                    "path": str(csv_path),
                    "delimiter": ";",
                    "decimal_separator": ",",
                },
            }
        )

        assert isinstance(source, FileSource)
        assert self._collect(source, tmp_path).schema["hba1c_pct"] == (
            pl.Float64
        )


class TestCsvReadOptions:
    """Translation of a source config into reader options."""

    def test_empty_config_asks_for_nothing(self):
        assert _csv_read_options(None) == {}
        assert _csv_read_options({}) == {}
        assert _csv_read_options({"path": "/tmp/x.csv"}) == {}

    def test_tabulation_is_accepted_as_the_two_character_escape(self):
        # The form field is free text, so this is how a tab is usually typed.
        assert _csv_read_options({"delimiter": "\\t"}) == {"separator": "\t"}
        assert _csv_read_options({"delimiter": "\t"}) == {"separator": "\t"}

    def test_multi_character_delimiter_names_the_value(self):
        with pytest.raises(ValueError, match=r"'\|\|'"):
            _csv_read_options({"delimiter": "||"})

    def test_empty_delimiter_falls_back_to_the_default(self):
        assert _csv_read_options({"delimiter": ""}) == {}

    def test_utf8_family_maps_to_the_single_polars_name(self):
        for name in ("utf-8", "UTF8", "utf_8", "utf-8-sig", "ascii"):
            assert _csv_read_options({"encoding": name}) == {
                "encoding": "utf8"
            }

    def test_unsupported_encoding_says_what_to_do(self):
        with pytest.raises(ValueError) as excinfo:
            _csv_read_options({"encoding": "latin-1"})
        message = str(excinfo.value)
        assert "latin-1" in message
        assert "UTF-8" in message

    def test_decimal_separator_maps_to_decimal_comma(self):
        assert _csv_read_options({"decimal_separator": ","}) == {
            "decimal_comma": True
        }
        assert _csv_read_options({"decimal_separator": "."}) == {
            "decimal_comma": False
        }

    def test_decimal_comma_boolean_is_accepted_too(self):
        assert _csv_read_options({"decimal_comma": True}) == {
            "decimal_comma": True
        }

    def test_unknown_decimal_separator_names_the_value(self):
        with pytest.raises(ValueError, match=r"';'"):
            _csv_read_options({"decimal_separator": ";"})

    def test_has_header_is_only_sent_when_declared(self):
        assert _csv_read_options({"has_header": False}) == {
            "has_header": False
        }
        assert _csv_read_options({"has_header": True}) == {"has_header": True}
