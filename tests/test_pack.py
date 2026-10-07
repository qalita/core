"""
# QALITA (c) COPYRIGHT 2025 - ALL RIGHTS RESERVED -
Tests for qalita_core.pack module
"""

from qalita_core.pack import (
    Pack,
    ConfigLoader,
    PlatformAsset,
    _sanitize_for_json,
    _object_key,
)
from qalita_core.data_source_opener import cleanup_parquet_files
import pytest
import os
import json
import math
import base64
import datetime as dt
from decimal import Decimal
import polars as pl


@pytest.fixture(scope="session")
def pack(config_paths):
    pack = Pack(configs=config_paths)
    return pack


class TestPackInstantiation:
    """Tests for Pack class instantiation."""

    def test_pack_instantiation(self, pack):
        assert isinstance(pack, Pack), "Failed to instantiate Pack"

    def test_pack_has_metrics(self, pack):
        assert hasattr(pack, "metrics")
        assert isinstance(pack.metrics, PlatformAsset)

    def test_pack_has_recommendations(self, pack):
        assert hasattr(pack, "recommendations")
        assert isinstance(pack.recommendations, PlatformAsset)

    def test_pack_has_schemas(self, pack):
        assert hasattr(pack, "schemas")
        assert isinstance(pack.schemas, PlatformAsset)


class TestPackConfigLoading:
    """Tests for Pack configuration loading."""

    def test_pack_load_pack_config(self, pack):
        pack_config = pack.pack_config
        assert isinstance(
            pack_config, dict
        ), "Failed to load pack configuration"

        expected_config = {
            "job": {"id_columns": [], "source": {"skiprows": 0}},
            "charts": {
                "overview": [
                    {
                        "metric_key": "score",
                        "chart_type": "text",
                        "display_title": True,
                        "justify": True,
                    }
                ],
                "scoped": [
                    {
                        "metric_key": "decimal_precision",
                        "chart_type": "text",
                        "display_title": True,
                        "justify": True,
                    },
                    {
                        "metric_key": "proportion_score",
                        "chart_type": "text",
                        "display_title": True,
                        "justify": True,
                    },
                    {
                        "metric_key": "proportion_score",
                        "chart_type": "spark_area_chart",
                        "display_title": False,
                        "justify": False,
                    },
                ],
            },
        }

        assert (
            pack_config == expected_config
        ), "pack_config does not match the expected configuration"

    def test_pack_load_source_config(self, pack):
        source_config = pack.source_config
        assert isinstance(
            source_config, dict
        ), "Failed to load source configuration"

        expected_config = {
            "config": {"path": "./tests/data/METABRIC_RNA_Mutation.xlsx"},
            "description": "Clinical attributes, m-RNA levels z-score, and genes mutations for 1904 patients",
            "id": 9,
            "name": "Breast Cancer Gene Expression Profiles (METABRIC)",
            "owner": "admin",
            "owner_id": 1,
            "reference": True,
            "sensitive": False,
            "type": "file",
            "validate": "valid",
            "visibility": "internal",
        }
        assert (
            source_config == expected_config
        ), "source_config does not match the expected configuration"

    def test_pack_load_target_config(self, pack):
        target_config = pack.target_config
        assert isinstance(
            target_config, dict
        ), "Failed to load target configuration"

        expected_config = {
            "config": {"path": "./tests/data/ref_bio_data.xlsx"},
            "description": "Ref data for clinical dataset",
            "id": 11,
            "name": "Bio referential data",
            "owner": "admin",
            "owner_id": 1,
            "reference": True,
            "sensitive": False,
            "type": "file",
            "validate": "valid",
            "visibility": "public",
        }

        assert (
            target_config == expected_config
        ), "target_config does not match the expected configuration"

    def test_pack_load_agent_config(self, pack):
        agent_config = pack.agent_config

        assert isinstance(
            agent_config, dict
        ), "Failed to load agent configuration"

        expected_config = {
            "user": {
                "id": 2,
                "email": "armand.leopold@qalita.io",
                "login": "armand.leopold",
                "name": "Armand LEOPOLD",
                "language": None,
                "avatar": None,
                "theme": "light",
                "home": "/home/de/sources",
                "role_id": 3,
                "role": "dataengineer",
                "role_override": True,
                "is_active": True,
                "created_at": "2024-02-18T20:07:36.620938",
                "last_activity": "2024-02-18T20:12:02.632269",
                "habilitations": [],
            },
            "context": {
                "local": {
                    "name": "armand.leopold",
                    "mode": "worker",
                    "token": "",
                    "url": "https://api.dev.platform.qalita.io",
                    "verbose": False,
                },
                "remote": {
                    "name": "armand.leopold",
                    "mode": "worker",
                    "status": "online",
                    "id": 3,
                    "is_active": True,
                    "registered_at": "2024-02-22T11:34:08.736840",
                    "last_status_check": "2024-02-22T11:34:08.736842",
                },
            },
            "registries": [
                {
                    "name": "local",
                    "id": 1,
                    "url": "https://2829b56e82804f0c8acaab6521f17694-platform-dev-qalita-bucket.s3.gra.io.cloud.ovh.net",
                }
            ],
        }

        assert (
            agent_config == expected_config
        ), "agent_config does not match the expected configuration"


class TestPackDataLoading:
    """Tests for Pack data loading."""

    def test_pack_load_data_target(self, pack, tmp_path):
        # Ensure output dir is set via env by overriding pack config at runtime
        pack.pack_config["job"]["parquet_output_dir"] = str(tmp_path)
        data = pack.load_data("target")
        assert isinstance(data, list) and all(isinstance(p, str) for p in data)
        assert all(p.endswith(".parquet") for p in data)
        # Files should exist
        for p in data:
            assert os.path.exists(p)

    def test_pack_load_data_source(self, pack, tmp_path):
        pack.pack_config["job"]["parquet_output_dir"] = str(tmp_path)
        data = pack.load_data("source")
        assert isinstance(data, list) and all(isinstance(p, str) for p in data)
        assert all(p.endswith(".parquet") for p in data)
        for p in data:
            assert os.path.exists(p)

    def test_pack_preserves_source_and_target_skip_snapshots(
        self, config_paths, monkeypatch
    ):
        pack = Pack(configs=config_paths)
        source_skips = [
            {
                "object": "source_denied",
                "error": "ProgrammingError",
                "reason": "source permission denied",
            }
        ]
        target_skips = [
            {
                "object": "target_denied",
                "error": "ProgrammingError",
                "reason": "target permission denied",
            }
        ]

        class _SourceWithSkips:
            def __init__(self, path, skipped):
                self.path = path
                self.skipped_objects = skipped
                self.object_paths = {path: [f"{path}.parquet"]}

            def get_data(self, table_or_query=None, pack_config=None):
                return [f"{self.path}.parquet"]

        data_sources = [
            _SourceWithSkips("source_readable", source_skips),
            _SourceWithSkips("target_readable", target_skips),
        ]
        monkeypatch.setattr(
            "qalita_core.pack.get_data_source",
            lambda config: data_sources.pop(0),
        )

        pack.load_data("source")
        pack.load_data("target")

        assert pack.skipped_source_objects == source_skips
        assert pack.skipped_target_objects == target_skips

        source_skips[0]["reason"] = "mutated after load"
        target_skips.append({"object": "later"})
        assert pack.skipped_source_objects == [
            {
                "object": "source_denied",
                "error": "ProgrammingError",
                "reason": "source permission denied",
            }
        ]
        assert pack.skipped_target_objects == [
            {
                "object": "target_denied",
                "error": "ProgrammingError",
                "reason": "target permission denied",
            }
        ]


class TestConfigLoader:
    """Tests for ConfigLoader class."""

    def test_load_valid_config(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_data = {"key": "value", "number": 42}
        config_file.write_text(json.dumps(config_data))

        result = ConfigLoader.load_config(str(config_file))
        assert result == config_data

    def test_load_missing_config(self):
        result = ConfigLoader.load_config("/nonexistent/path/config.json")
        assert result == {}

    def test_load_empty_config(self, tmp_path):
        config_file = tmp_path / "empty.json"
        config_file.write_text("{}")

        result = ConfigLoader.load_config(str(config_file))
        assert result == {}


class TestSanitizeForJson:
    """Tests for _sanitize_for_json function."""

    def test_none_value(self):
        assert _sanitize_for_json(None) is None

    def test_bool_value(self):
        assert _sanitize_for_json(True) is True
        assert _sanitize_for_json(False) is False

    def test_int_value(self):
        assert _sanitize_for_json(42) == 42
        assert _sanitize_for_json(-10) == -10

    def test_str_value(self):
        assert _sanitize_for_json("hello") == "hello"

    def test_finite_float(self):
        assert _sanitize_for_json(3.14) == 3.14

    def test_nan_float(self):
        result = _sanitize_for_json(float("nan"))
        assert result is None

    def test_inf_float(self):
        assert _sanitize_for_json(float("inf")) is None
        assert _sanitize_for_json(float("-inf")) is None

    def test_datetime(self):
        dt_val = dt.datetime(2024, 1, 15, 10, 30, 0)
        result = _sanitize_for_json(dt_val)
        assert result == "2024-01-15T10:30:00"

    def test_date(self):
        d_val = dt.date(2024, 1, 15)
        result = _sanitize_for_json(d_val)
        assert result == "2024-01-15"

    def test_time(self):
        t_val = dt.time(10, 30, 0)
        result = _sanitize_for_json(t_val)
        assert result == "10:30:00"

    def test_pandas_nat_becomes_null_not_the_string_nat(self):
        import pandas as pd

        # pd.NaT is an instance of datetime.datetime, so the datetime branch
        # must not win: a missing date is null, never the category "NaT".
        assert _sanitize_for_json(pd.NaT) is None

    def test_pandas_na_becomes_null(self):
        import pandas as pd

        assert _sanitize_for_json(pd.NA) is None

    def test_pandas_timestamp_is_isoformatted(self):
        import pandas as pd

        result = _sanitize_for_json(pd.Timestamp("2026-01-15T10:30:00"))
        assert result == "2026-01-15T10:30:00"

    def test_decimal(self):
        dec_val = Decimal("3.14159")
        result = _sanitize_for_json(dec_val)
        assert isinstance(result, float)
        assert abs(result - 3.14159) < 0.0001

    def test_dict_with_string_keys(self):
        data = {"key1": "value1", "key2": 42}
        result = _sanitize_for_json(data)
        assert result == data

    def test_dict_with_non_string_keys(self):
        data = {1: "value1", 2: "value2"}
        result = _sanitize_for_json(data)
        assert "1" in result
        assert "2" in result

    def test_nested_dict(self):
        data = {"outer": {"inner": 42}}
        result = _sanitize_for_json(data)
        assert result == data

    def test_list(self):
        data = [1, 2, 3]
        result = _sanitize_for_json(data)
        assert result == data

    def test_tuple(self):
        data = (1, 2, 3)
        result = _sanitize_for_json(data)
        assert result == [1, 2, 3]  # Converted to list

    def test_set(self):
        data = {1, 2, 3}
        result = _sanitize_for_json(data)
        assert sorted(result) == [1, 2, 3]  # Converted to list

    def test_nested_structures(self):
        data = {
            "list": [1, 2, {"nested": "value"}],
            "number": 42,
        }
        result = _sanitize_for_json(data)
        assert result["list"][2]["nested"] == "value"

    def test_complex_object_fallback(self):
        class CustomClass:
            def __str__(self):
                return "custom_object"

        result = _sanitize_for_json(CustomClass())
        assert result == "custom_object"


class TestPlatformAsset:
    """Tests for PlatformAsset class."""

    def test_init_metrics(self):
        asset = PlatformAsset("metrics")
        assert asset.type == "metrics"
        assert asset.data == []

    def test_init_recommendations(self):
        asset = PlatformAsset("recommendations")
        assert asset.type == "recommendations"
        assert asset.data == []

    def test_init_schemas(self):
        asset = PlatformAsset("schemas")
        assert asset.type == "schemas"
        assert asset.data == []

    def test_data_manipulation(self):
        asset = PlatformAsset("metrics")
        asset.data.append({"key": "value", "score": 0.95})
        assert len(asset.data) == 1
        assert asset.data[0]["key"] == "value"

    def test_save_creates_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        asset = PlatformAsset("metrics")
        asset.data = [{"key": "test", "value": 42}]
        asset.save()

        output_file = tmp_path / "metrics.json"
        assert output_file.exists()

        with open(output_file) as f:
            saved_data = json.load(f)
        assert saved_data == [{"key": "test", "value": 42}]

    def test_save_sanitizes_data(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        asset = PlatformAsset("metrics")
        # Add data with non-JSON-serializable values
        asset.data = [{"value": float("nan")}, {"date": dt.date(2024, 1, 15)}]
        asset.save()

        output_file = tmp_path / "metrics.json"
        assert output_file.exists()

        with open(output_file) as f:
            saved_data = json.load(f)
        # NaN should be sanitized to None
        assert saved_data[0]["value"] is None
        # Date should be converted to ISO string
        assert saved_data[1]["date"] == "2024-01-15"

    def test_save_empty_data(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        asset = PlatformAsset("schemas")
        asset.save()

        output_file = tmp_path / "schemas.json"
        assert output_file.exists()

        with open(output_file) as f:
            saved_data = json.load(f)
        assert saved_data == []


class TestPackWithMissingConfigs:
    """Tests for Pack behavior with missing configurations."""

    def test_pack_with_missing_source_type(self, tmp_path):
        # Create minimal configs without 'type' in source
        pack_conf = tmp_path / "pack_conf.json"
        source_conf = tmp_path / "source_conf.json"
        target_conf = tmp_path / "target_conf.json"
        agent_file = tmp_path / ".worker"

        pack_conf.write_text('{"job": {}}')
        source_conf.write_text('{"config": {}}')  # Missing 'type'
        target_conf.write_text(
            '{"type": "file", "config": {"path": "test.csv"}}'
        )

        # Create a valid agent file
        import base64

        agent_data = json.dumps(
            {
                "user": {},
                "context": {"local": {}, "remote": {}},
                "registries": [],
            }
        )
        agent_file.write_text(base64.b64encode(agent_data.encode()).decode())

        # Should not crash, just log error
        pack = Pack(
            configs={
                "pack_conf": str(pack_conf),
                "source_conf": str(source_conf),
                "target_conf": str(target_conf),
                "agent_file": str(agent_file),
            }
        )
        assert pack.source_config == {"config": {}}

    def test_pack_with_empty_source_config(self, tmp_path):
        pack_conf = tmp_path / "pack_conf.json"
        source_conf = tmp_path / "source_conf.json"
        target_conf = tmp_path / "target_conf.json"
        agent_file = tmp_path / ".worker"

        pack_conf.write_text('{"job": {}}')
        source_conf.write_text("{}")  # Empty
        target_conf.write_text(
            '{"type": "file", "config": {"path": "test.csv"}}'
        )

        import base64

        agent_data = json.dumps(
            {
                "user": {},
                "context": {"local": {}, "remote": {}},
                "registries": [],
            }
        )
        agent_file.write_text(base64.b64encode(agent_data.encode()).decode())

        pack = Pack(
            configs={
                "pack_conf": str(pack_conf),
                "source_conf": str(source_conf),
                "target_conf": str(target_conf),
                "agent_file": str(agent_file),
            }
        )
        assert pack.source_config == {}


class TestCleanupParquetFiles:
    """Tests for cleanup_parquet_files function."""

    def test_cleanup_removes_files(self, tmp_path):
        # Create some fake parquet files
        files = []
        for i in range(3):
            f = tmp_path / f"test_part_{i}.parquet"
            f.write_text("fake parquet content")
            files.append(str(f))

        # Verify files exist
        for f in files:
            assert os.path.exists(f)

        # Cleanup
        removed = cleanup_parquet_files(files)

        # Verify files are removed
        assert removed == 3
        for f in files:
            assert not os.path.exists(f)

    def test_cleanup_handles_nonexistent_files(self, tmp_path):
        files = [
            str(tmp_path / "nonexistent1.parquet"),
            str(tmp_path / "nonexistent2.parquet"),
        ]

        # Should not raise an error
        removed = cleanup_parquet_files(files)
        assert removed == 0

    def test_cleanup_handles_empty_list(self):
        removed = cleanup_parquet_files([])
        assert removed == 0

    def test_cleanup_handles_none(self):
        removed = cleanup_parquet_files(None)
        assert removed == 0

    def test_cleanup_partial_success(self, tmp_path):
        # Create one real file
        real_file = tmp_path / "real.parquet"
        real_file.write_text("content")

        files = [
            str(real_file),
            str(tmp_path / "nonexistent.parquet"),
        ]

        removed = cleanup_parquet_files(files)
        assert removed == 1
        assert not os.path.exists(real_file)


class TestPackCleanup:
    """Tests for Pack cleanup method and context manager."""

    def test_cleanup_method(self, config_paths, tmp_path):
        pack = Pack(configs=config_paths)
        pack.pack_config["job"]["parquet_output_dir"] = str(tmp_path)

        # Load data
        pack.load_data("source")

        # Verify parquet files exist
        assert pack.paths_source is not None
        for p in pack.paths_source:
            assert os.path.exists(p)

        # Cleanup
        removed = pack.cleanup()

        # Verify files are removed
        assert removed > 0
        for p in pack.paths_source:
            assert not os.path.exists(p)

    def test_cleanup_source_and_target(self, config_paths, tmp_path):
        pack = Pack(configs=config_paths)
        pack.pack_config["job"]["parquet_output_dir"] = str(tmp_path)

        # Load both source and target
        pack.load_data("source")
        pack.load_data("target")

        # Verify parquet files exist
        all_paths = pack.paths_source + pack.paths_target
        for p in all_paths:
            assert os.path.exists(p)

        # Cleanup
        removed = pack.cleanup()

        # Verify all files are removed
        assert removed == len(all_paths)
        for p in all_paths:
            assert not os.path.exists(p)

    def test_cleanup_without_loading_data(self, config_paths):
        pack = Pack(configs=config_paths)

        # Should not raise an error
        removed = pack.cleanup()
        assert removed == 0

    def test_context_manager_cleanup(self, config_paths, tmp_path):
        paths_to_check = None

        with Pack(configs=config_paths) as pack:
            pack.pack_config["job"]["parquet_output_dir"] = str(tmp_path)
            pack.load_data("source")
            paths_to_check = list(pack.paths_source)

            # Verify files exist inside the context
            for p in paths_to_check:
                assert os.path.exists(p)

        # After exiting context, files should be cleaned up
        for p in paths_to_check:
            assert not os.path.exists(p)

    def test_context_manager_cleanup_on_exception(
        self, config_paths, tmp_path
    ):
        paths_to_check = None

        try:
            with Pack(configs=config_paths) as pack:
                pack.pack_config["job"]["parquet_output_dir"] = str(tmp_path)
                pack.load_data("source")
                paths_to_check = list(pack.paths_source)

                # Verify files exist inside the context
                for p in paths_to_check:
                    assert os.path.exists(p)

                # Raise an exception
                raise ValueError("Test exception")
        except ValueError:
            pass

        # After exception, files should still be cleaned up
        for p in paths_to_check:
            assert not os.path.exists(p)

    def test_double_cleanup_safe(self, config_paths, tmp_path):
        pack = Pack(configs=config_paths)
        pack.pack_config["job"]["parquet_output_dir"] = str(tmp_path)
        pack.load_data("source")

        # First cleanup
        removed1 = pack.cleanup()
        assert removed1 > 0

        # Second cleanup should be safe (files already removed)
        removed2 = pack.cleanup()
        assert removed2 == 0


def _pack_for(tmp_path, source_config):
    """A Pack whose source is ``source_config``, staging under tmp_path."""
    pack_conf = tmp_path / "pack_conf.json"
    pack_conf.write_text(
        json.dumps({"job": {"parquet_output_dir": str(tmp_path / "staged")}})
    )
    source_conf = tmp_path / "source_conf.json"
    source_conf.write_text(json.dumps(source_config))
    return Pack(
        configs={
            "pack_conf": str(pack_conf),
            "source_conf": str(source_conf),
            "target_conf": str(tmp_path / "absent_target_conf.json"),
            "agent_file": str(tmp_path / "absent_agent"),
        }
    )


class TestCleanupNeverTouchesSourceFiles:
    """Cleanup removes what load_data staged -- and nothing else.

    A parquet file source is not staged: load_data hands back the user's own
    file. Cleanup used to delete everything in paths_source, so every pack run
    on a parquet file destroyed the source it had just analysed.
    """

    def test_parquet_file_source_survives_the_pack(self, tmp_path):
        source = tmp_path / "patients.parquet"
        pl.DataFrame({"id": [1, 2]}).write_parquet(source)

        with _pack_for(
            tmp_path, {"type": "file", "config": {"path": str(source)}}
        ) as pack:
            assert pack.load_data("source") == [str(source)]

        assert source.exists()
        assert pl.read_parquet(source)["id"].to_list() == [1, 2]

    def test_cleanup_reports_nothing_removed_for_a_parquet_source(
        self, tmp_path
    ):
        source = tmp_path / "patients.parquet"
        pl.DataFrame({"id": [1]}).write_parquet(source)
        pack = _pack_for(
            tmp_path, {"type": "file", "config": {"path": str(source)}}
        )
        pack.load_data("source")
        assert pack.cleanup() == 0
        assert source.exists()

    def test_csv_source_kept_and_its_staged_parts_removed(self, tmp_path):
        source = tmp_path / "patients.csv"
        source.write_text("id\n1\n2\n")

        with _pack_for(
            tmp_path, {"type": "file", "config": {"path": str(source)}}
        ) as pack:
            staged = pack.load_data("source")
            assert staged and all(os.path.exists(p) for p in staged)

        assert source.exists()
        assert not any(os.path.exists(p) for p in staged)


class TestCleanupCoversEveryLoad:
    """A pack loading table by table calls load_data once per table.

    paths_source only holds the last call's parts (packs read it as "what I
    just loaded"), so cleanup must not rely on it: every earlier table's
    parts used to be left on disk.
    """

    def test_parts_of_every_load_data_call_are_removed(self, tmp_path):
        import sqlite3

        database = tmp_path / "warehouse.db"
        with sqlite3.connect(database) as connection:
            connection.execute("create table a (x integer)")
            connection.execute("insert into a values (1)")
            connection.execute("create table b (y integer)")
            connection.execute("insert into b values (2)")

        with _pack_for(
            tmp_path,
            {
                "type": "sqlite",
                "config": {"connection_string": f"sqlite:///{database}"},
            },
        ) as pack:
            first = pack.load_data("source", table_or_query="a")
            second = pack.load_data("source", table_or_query="b")
            assert pack.paths_source == second  # last call, as before
            staged = first + second
            assert all(os.path.exists(p) for p in staged)

        assert not any(os.path.exists(p) for p in staged)
        assert database.exists()


class TestFolderSourceThroughPack:
    """A folder is a multi-object source, like a database scanned with *."""

    def test_tables_scan_and_cleanup(self, tmp_path):
        folder = tmp_path / "warehouse"
        folder.mkdir()
        (folder / "patients.csv").write_text("id\n1\n2\n")
        stays = folder / "stays.parquet"
        pl.DataFrame({"stay": [10]}).write_parquet(stays)

        with _pack_for(
            tmp_path, {"type": "folder", "config": {"path": str(folder)}}
        ) as pack:
            pack.load_data("source")
            assert sorted(pack.tables("source")) == [
                "file_patients",
                "file_stays",
            ]
            assert pack.get_row_count("source", "file_patients") == 2
            assert pack.schema("source", "file_stays") == {"stay": pl.Int64}
            staged = pack.objects_source["file_patients"]

        assert stays.exists() and (folder / "patients.csv").exists()
        assert not any(os.path.exists(p) for p in staged)


class TestObjectKey:
    """Tests for the _object_key helper."""

    def test_without_part_suffix(self):
        assert _object_key("foo.parquet") == "foo.parquet"

    def test_with_part_suffix(self):
        assert _object_key("tbl_part_3.parquet") == "tbl"

    def test_with_uppercase_part_suffix(self):
        assert _object_key("tbl_PART_3.PARQUET") == "tbl"


class TestPackConfigsNone:
    """Pack(configs=None) must fall back to the default config paths."""

    def test_configs_none_uses_defaults(self, monkeypatch):
        monkeypatch.chdir("/tmp")
        pack = Pack(configs=None)
        assert pack.config_paths == Pack.default_configs
        # The default pack/source/target files do not exist, so they load empty.
        assert pack.pack_config == {}
        assert pack.source_config == {}
        assert pack.target_config == {}
        # The default agent file may or may not exist on this machine; only the
        # fallback to the default path matters here.
        assert isinstance(pack.agent_config, dict)


class TestLoadAgentConfig:
    """Tests for Pack.load_agent_config edge cases."""

    def test_missing_file_returns_empty(self, config_paths, tmp_path):
        pack = Pack(configs=config_paths)
        assert pack.load_agent_config(str(tmp_path / "nonexistent")) == {}

    def test_invalid_base64_returns_empty(self, config_paths, tmp_path):
        f = tmp_path / "agent"
        f.write_text("this is not base64!!!")
        pack = Pack(configs=config_paths)
        assert pack.load_agent_config(str(f)) == {}

    def test_normalizes_local_context_url(self, config_paths, tmp_path):
        data = {
            "context": {
                "local": {"url": "https://api.dev.platform.qalita.io/api/v1"}
            }
        }
        f = tmp_path / "agent"
        f.write_text(base64.b64encode(json.dumps(data).encode()).decode())
        pack = Pack(configs=config_paths)
        result = pack.load_agent_config(str(f))
        assert (
            result["context"]["local"]["url"]
            == "https://api.dev.platform.qalita.io"
        )

    def test_keeps_url_without_path_unchanged(self, config_paths, tmp_path):
        data = {"context": {"local": {"url": "https://api.example.com"}}}
        f = tmp_path / "agent"
        f.write_text(base64.b64encode(json.dumps(data).encode()).decode())
        pack = Pack(configs=config_paths)
        result = pack.load_agent_config(str(f))
        assert result["context"]["local"]["url"] == "https://api.example.com"


class TestGroupByObject:
    """Tests for Pack._group_by_object."""

    def test_uses_recorded_object_paths_when_present(self):
        ds = type("DS", (), {"object_paths": {"x": ["x.parquet"]}})()
        assert Pack._group_by_object(ds, ["x.parquet"]) == {"x": ["x.parquet"]}

    def test_fallback_groups_by_part_suffix(self):
        ds = type("DS", (), {})()
        paths = [
            "a_part_1.parquet",
            "a_part_2.parquet",
            "b_part_1.parquet",
        ]
        assert Pack._group_by_object(ds, paths) == {
            "a": ["a_part_1.parquet", "a_part_2.parquet"],
            "b": ["b_part_1.parquet"],
        }

    def test_fallback_skips_non_string_paths(self):
        ds = type("DS", (), {})()
        assert Pack._group_by_object(ds, ["a_part_1.parquet", 123, None]) == {
            "a": ["a_part_1.parquet"]
        }


class TestObjectsAccessors:
    """Tests for _objects, tables, scan, scan_all, schema, scan_data."""

    def test_objects_invalid_trigger(self, config_paths):
        pack = Pack(configs=config_paths)
        with pytest.raises(ValueError):
            pack._objects("bogus")

    def test_objects_no_data(self, config_paths):
        pack = Pack(configs=config_paths)
        with pytest.raises(RuntimeError):
            pack._objects("source")

    def test_tables(self, config_paths):
        pack = Pack(configs=config_paths)
        pack.objects_source = {"x": ["x.parquet"], "y": ["y.parquet"]}
        assert set(pack.tables("source")) == {"x", "y"}

    def test_scan_multiple_objects_requires_table(self, config_paths):
        pack = Pack(configs=config_paths)
        pack.objects_source = {"x": ["x.parquet"], "y": ["y.parquet"]}
        with pytest.raises(ValueError):
            pack.scan("source")

    def test_scan_unknown_object_raises(self, config_paths):
        pack = Pack(configs=config_paths)
        pack.objects_source = {"x": ["x.parquet"]}
        with pytest.raises(KeyError):
            pack.scan("source", "nope")

    def test_scan_single_object_omits_table(self, config_paths, tmp_path):
        pack = Pack(configs=config_paths)
        p = tmp_path / "x.parquet"
        pl.DataFrame({"a": [1, 2]}).write_parquet(p)
        pack.objects_source = {"x": [str(p)]}
        assert pack.scan("source").collect().height == 2

    def test_scan_all(self, config_paths, tmp_path):
        pack = Pack(configs=config_paths)
        p1 = tmp_path / "x.parquet"
        p2 = tmp_path / "y.parquet"
        pl.DataFrame({"a": [1]}).write_parquet(p1)
        pl.DataFrame({"a": [2]}).write_parquet(p2)
        pack.objects_source = {"x": [str(p1)], "y": [str(p2)]}
        frames = pack.scan_all("source")
        assert set(frames) == {"x", "y"}

    def test_schema(self, config_paths, tmp_path):
        pack = Pack(configs=config_paths)
        p = tmp_path / "x.parquet"
        pl.DataFrame({"a": [1]}).write_parquet(p)
        pack.objects_source = {"x": [str(p)]}
        assert pack.schema("source") == {"a": pl.Int64}

    def test_scan_data_no_data(self, config_paths):
        pack = Pack(configs=config_paths)
        with pytest.raises(RuntimeError):
            pack.scan_data("source")

    def test_scan_data(self, config_paths, tmp_path):
        pack = Pack(configs=config_paths)
        p = tmp_path / "x.parquet"
        pl.DataFrame({"a": [1, 2]}).write_parquet(p)
        pack.paths_source = [str(p)]
        assert pack.scan_data("source").collect().height == 2


class TestGetRowCount:
    """Tests for Pack.get_row_count."""

    def test_no_objects_returns_zero(self, config_paths):
        pack = Pack(configs=config_paths)
        assert pack.get_row_count("source") == 0

    def test_single_object(self, config_paths, tmp_path):
        pack = Pack(configs=config_paths)
        p = tmp_path / "x.parquet"
        pl.DataFrame({"a": [1, 2, 3]}).write_parquet(p)
        pack.objects_source = {"x": [str(p)]}
        assert pack.get_row_count("source") == 3

    def test_multiple_objects_sums(self, config_paths, tmp_path):
        pack = Pack(configs=config_paths)
        p1 = tmp_path / "x.parquet"
        p2 = tmp_path / "y.parquet"
        pl.DataFrame({"a": [1, 2]}).write_parquet(p1)
        pl.DataFrame({"a": [3, 4, 5]}).write_parquet(p2)
        pack.objects_source = {"x": [str(p1)], "y": [str(p2)]}
        assert pack.get_row_count("source") == 5


class TestSanitizeForJsonExtraBranches:
    """Covers the remaining error branches of _sanitize_for_json."""

    def test_decimal_non_finite(self):
        assert _sanitize_for_json(Decimal("sNaN")) == "sNaN"

    def test_tolist_raises(self):
        class BadTolist:
            def tolist(self):
                raise ValueError("no")

        assert isinstance(_sanitize_for_json(BadTolist()), str)

    def test_item_raises(self):
        class BadItem:
            def item(self):
                raise ValueError("no")

        assert isinstance(_sanitize_for_json(BadItem()), str)

    def test_dict_key_str_raises_uses_repr(self):
        class BadKey:
            def __str__(self):
                raise ValueError("no")

        result = _sanitize_for_json({BadKey(): 1})
        assert list(result.values()) == [1]

    def test_fallback_str_raises_returns_none(self):
        class BadStr:
            def __str__(self):
                raise ValueError("no")

        assert _sanitize_for_json(BadStr()) is None
