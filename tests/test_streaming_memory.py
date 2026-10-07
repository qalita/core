"""
# QALITA (c) COPYRIGHT 2025 - ALL RIGHTS RESERVED -

Memory regression harness for the streaming primitives.

Correctness tests cannot catch the failure mode this project exists to prevent:
a helper that quietly materializes its input returns exactly the right answer
on a fixture and gets OOM-killed on a real source. So these tests assert a
resource, not a result — every primitive must describe a dataset whose eager
form is several times the ceiling, while staying under it.

How the measurement is made, and why:

- **Peak RSS**, a high-water mark, so it cannot be gamed by freeing before the
  assertion. Read from ``/proc/self/status`` ``VmHWM`` where available and from
  ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` otherwise — ``ru_maxrss`` alone
  is wrong here, see :func:`peak_kb` in the generated child.
- **In a fresh subprocess per operation.** A high-water mark never goes down,
  so two operations measured in one process would report the larger of the two
  forever. A subprocess also gives each operation a clean baseline.
- **With the Polars thread pool pinned to one thread.** The streaming engine
  keeps per-thread morsel buffers, so peak RSS otherwise scales with the core
  count of whatever machine runs CI and the ceiling would mean a different
  thing on every runner. Pinning removes the machine from the measurement; the
  property under test is that memory does not scale with *rows*.
- **On the delta above the post-import baseline**, so the ~40 MiB that
  importing polars/pandas costs is not charged to the operation.

``RLIMIT_AS`` (``ulimit -v``) was tried first and rejected: the Rust allocator
and OpenBLAS reserve address space far in excess of resident memory — with a
1.2 GiB ``RLIMIT_AS`` the child dies in ``import numpy`` on thread-stack
allocation, long before any data is touched. It would have measured address
space reservation, not memory use.

Sizing: the default dataset is 16M rows, ~1.1 GiB as an eager frame, generated
in half a second. Set ``QALITA_BIGDATA_ROWS`` for an opt-in big run; the
ceiling is a fixed number of MiB, so a bigger dataset makes these tests
strictly harder, which is the point.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.bigdata import (  # noqa: F401 - imported for pytest fixture lookup
    DEFAULT_PARTS,
    Dataset,
    bigdata,
    dataset_rows,
    generate,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# The ceiling every primitive must respect, in MiB of peak RSS above the
# interpreter baseline. Chosen so the default dataset's eager form is more than
# three times as large (asserted by `test_the_ceiling_is_a_real_constraint`)
# with real headroom over the worst primitive. Measured at 16M rows, one
# thread: row_count 14, value_counts 25, quantiles 40, failures 44, agg 55,
# profile 57, approx_n_unique 69, Pack.scan 69, sink 208 -- against an eager
# read at 1145.
CEILING_MB = 320

# `analytics.sample()` gets its own, higher ceiling: it builds a row index over
# the whole dataset before filtering, so its peak is markedly higher than the
# other primitives (263 MiB at 16M rows) and, unlike them, still tracks the row
# count. See `test_sample_memory_growth_is_recorded` for the measurement.
SAMPLE_CEILING_MB = 384

# Peak RSS must not follow the row count. Everything listed in GROWTH_OPERATIONS
# is measured at rows/4 and at rows. Measured over 4M -> 16M rows: failures
# x1.02, row_count x1.09, profile x1.16, agg x1.21, quantiles x1.23,
# approx_n_unique x1.77 -- against an eager read at x3.85. The limit sits
# between the two populations, nearer the streaming one.
GROWTH_LIMIT = 2.25

# What the control experiment must show for the harness to mean anything.
MIN_EAGER_GROWTH = 3.0

# `sample()` is the exception, at x2.84 over the same range: it is not flat,
# but it must not become as bad as an eager read either.
SAMPLE_GROWTH_LIMIT = 3.5


# Every body runs with `pl`, `analytics`, `profiling`, `FILES` and `scan()` in
# scope, and assigns something small to `result`.
OPERATIONS: dict[str, str] = {
    "row_count": "result = analytics.row_count(scan())",
    "agg_every_column": """
        columns = list(scan().collect_schema())
        exprs = {"__rows": pl.len()}
        for column in columns:
            exprs[f"nulls|{column}"] = pl.col(column).null_count()
            exprs[f"count|{column}"] = pl.col(column).count()
            exprs[f"distinct|{column}"] = pl.col(column).approx_n_unique()
        result = len(analytics.agg(scan(), exprs))
    """,
    "approx_n_unique_high_cardinality": """
        result = analytics.approx_n_unique(scan(), ["uid", "id", "key"])
    """,
    "quantiles_histogram": """
        result = analytics.quantiles(
            scan(), ["id", "value", "amount"], [0.05, 0.25, 0.5, 0.75, 0.95]
        )
    """,
    "failures_with_millions_failing": """
        count, rows = analytics.failures(scan(), pl.col("flag"), limit=10)
        result = (count, rows.height)
    """,
    "value_counts": """
        result = analytics.value_counts(scan(), "key", 20, other=True).height
    """,
    # The near-unique column, which is what the fixture was built for. Grouping
    # it exactly is what exhausts the machine, so the guard must refuse it —
    # and refusing must itself stay under the ceiling, not blow up on the way.
    "value_counts_high_cardinality": """
        try:
            analytics.value_counts(scan(), "uid", 20)
            result = "NOT REFUSED"
        except analytics.CardinalityTooHigh:
            result = "refused"
    """,
    "profile": "result = len(profiling.profile(scan(), top_k=0))",
    "sink_partitioned": """
        import tempfile
        target = tempfile.mkdtemp()
        # 250k rows per part is the order of magnitude ingestion actually uses
        # (`chunk_rows` defaults to 100k). The writer buffers one part, so this
        # bound is what the sink costs -- it must not scale with the source.
        analytics.sink(
            scan(), target + "/out", max_rows_per_file=250_000
        )
        result = len(list(pathlib.Path(target).rglob("*.parquet")))
    """,
    "pack_scan_path": """
        from qalita_core.pack import Pack
        pack = Pack(configs={
            "pack_conf": "/nonexistent/pack_conf.json",
            "source_conf": "/nonexistent/source_conf.json",
            "target_conf": "/nonexistent/target_conf.json",
            "agent_file": "/nonexistent/.worker",
        })
        pack.objects_source = {"bigdata": FILES}
        result = (
            pack.tables("source"),
            len(pack.schema("source")),
            pack.get_row_count("source"),
            analytics.agg(pack.scan("source"), {"n": pl.len()})["n"],
        )
    """,
    "sample_reservoir": """
        result = analytics.sample(scan(), 10_000).height
    """,
    # The control: this is the thing the ceiling exists to forbid.
    "eager_collect": """
        result = scan().collect(engine="streaming").height
    """,
}


_CHILD = '''\
"""Generated by tests/test_streaming_memory.py - measures one operation."""

import json
import pathlib
import resource
import sys

sys.path.insert(0, {root!r})

import polars as pl

from qalita_core import analytics, profiling

FILES = json.loads(pathlib.Path({manifest!r}).read_text())


def scan():
    return pl.scan_parquet(FILES)


def peak_kb():
    """Peak resident set of THIS process, in KiB.

    /proc is consulted before getrusage because `ru_maxrss` is INHERITED
    through posix_spawn: glibc implements it with CLONE_VM|CLONE_VFORK, so the
    child starts life carrying the parent's high-water mark and would be
    charged the whole pytest session's peak. VmHWM is read from the mm created
    by exec, so it is this process's own figure. getrusage stays as the
    fallback for platforms without /proc.
    """
    try:
        with open("/proc/self/status", "r", encoding="ascii") as status:
            for line in status:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


baseline_kb = peak_kb()
result = None

{body}

print(
    "QALITA-MEASUREMENT "
    + json.dumps(
        {{
            "operation": {operation!r},
            "baseline_kb": baseline_kb,
            "peak_kb": peak_kb(),
            "result": repr(result)[:200],
        }}
    )
)
'''


@dataclass(frozen=True)
class Measurement:
    operation: str
    rows: int
    peak_mb: float
    baseline_mb: float
    result: str

    def __str__(self) -> str:
        return (
            f"{self.operation} on {self.rows:,} rows: "
            f"{self.peak_mb:.1f} MiB above a {self.baseline_mb:.1f} MiB "
            f"baseline (result: {self.result})"
        )


def measure(operation: str, dataset: Dataset, tmp_path: Path) -> Measurement:
    """Peak RSS of one operation, in MiB above the interpreter baseline."""
    if operation not in OPERATIONS:
        raise KeyError(f"unknown operation {operation!r}")

    workdir = tmp_path / f"measure-{operation}"
    workdir.mkdir(parents=True, exist_ok=True)
    manifest = workdir / "files.json"
    manifest.write_text(json.dumps(dataset.files), encoding="utf-8")

    script = workdir / "child.py"
    script.write_text(
        _CHILD.format(
            root=str(REPO_ROOT),
            manifest=str(manifest),
            operation=operation,
            body=textwrap.dedent(OPERATIONS[operation]).strip(),
        ),
        encoding="utf-8",
    )

    env = dict(os.environ)
    # See the module docstring: the ceiling has to mean the same thing on a
    # 2-core runner and on a 24-core workstation.
    env["POLARS_MAX_THREADS"] = "1"
    env.setdefault("OPENBLAS_NUM_THREADS", "1")

    completed = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        env=env,
        timeout=1800,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"measuring {operation!r} failed with exit code "
            f"{completed.returncode}:\n{completed.stderr[-4000:]}"
        )

    payload = None
    for line in completed.stdout.splitlines():
        if line.startswith("QALITA-MEASUREMENT "):
            payload = json.loads(line[len("QALITA-MEASUREMENT ") :])
    if payload is None:
        raise AssertionError(
            f"{operation!r} produced no measurement:\n{completed.stdout[-4000:]}"
        )

    return Measurement(
        operation=operation,
        rows=dataset.rows,
        peak_mb=(payload["peak_kb"] - payload["baseline_kb"]) / 1024,
        baseline_mb=payload["baseline_kb"] / 1024,
        result=payload["result"],
    )


@pytest.fixture(scope="session")
def quarter_dataset(tmp_path_factory) -> Dataset:
    """A quarter of the main dataset, for the growth comparison.

    A quarter of the *parts*, each the same size as the main dataset's: the
    growth tests vary the row count and nothing else, the way staged data
    grows (more parts of ``chunk_rows``, not bigger ones). This used to be
    four parts of half the size, which also halved the rows per file. Peak RSS
    of a multi-file scan depends on that layout -- polars 1.44 prefetches
    across files, measured at 16M rows: 30 MiB in 64 files, 105 in 2, 128 in
    8 -- so the comparison charged the layout change to the row count, and
    `approx_n_unique` sat at x2.2-2.8 around GROWTH_LIMIT, failing at random
    in CI (pip installs the newest polars, the lock does not).
    """
    root = tmp_path_factory.mktemp("qalita-bigdata-quarter")
    return generate(
        root,
        rows=max(dataset_rows() // 4, 100_000),
        parts=max(DEFAULT_PARTS // 4, 1),
    )


# --------------------------------------------------------------------------
# the ceiling
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_the_ceiling_is_a_real_constraint(bigdata):
    """The dataset must dwarf the ceiling, or the ceiling proves nothing."""
    assert bigdata.in_memory_mb >= 3 * CEILING_MB, (
        f"{bigdata.rows:,} rows are only {bigdata.in_memory_mb:.0f} MiB in "
        f"memory; raise QALITA_BIGDATA_ROWS or lower CEILING_MB"
    )
    assert bigdata.parts > 1, "a single-part dataset hides chunk handling bugs"


@pytest.mark.slow
@pytest.mark.parametrize(
    "operation",
    [
        "row_count",
        "agg_every_column",
        "approx_n_unique_high_cardinality",
        "quantiles_histogram",
        "failures_with_millions_failing",
        "value_counts",
        "profile",
        "sink_partitioned",
        "pack_scan_path",
    ],
)
def test_primitive_stays_under_the_ceiling(operation, bigdata, tmp_path):
    measurement = measure(operation, bigdata, tmp_path)
    assert measurement.peak_mb < CEILING_MB, (
        f"{measurement} exceeded the {CEILING_MB} MiB ceiling while the "
        f"dataset is {bigdata.in_memory_mb:.0f} MiB in memory"
    )


@pytest.mark.slow
def test_sample_stays_under_its_ceiling(bigdata, tmp_path):
    measurement = measure("sample_reservoir", bigdata, tmp_path)
    assert (
        measurement.peak_mb < SAMPLE_CEILING_MB
    ), f"{measurement} exceeded the {SAMPLE_CEILING_MB} MiB sample ceiling"
    assert measurement.result == "10000", measurement


@pytest.mark.slow
def test_an_eager_read_blows_the_ceiling(bigdata, tmp_path):
    """The control experiment.

    If this ever passes, the harness has stopped measuring anything: either the
    dataset shrank below the ceiling or the measurement broke. A green suite
    with a broken control is worse than a red one.
    """
    measurement = measure("eager_collect", bigdata, tmp_path)
    assert measurement.peak_mb > CEILING_MB, (
        f"{measurement} did NOT exceed the {CEILING_MB} MiB ceiling. The "
        f"harness can no longer tell a streaming pass from an eager read."
    )


# --------------------------------------------------------------------------
# growth
# --------------------------------------------------------------------------


GROWTH_OPERATIONS = [
    "row_count",
    "agg_every_column",
    "approx_n_unique_high_cardinality",
    "quantiles_histogram",
    "failures_with_millions_failing",
    "profile",
]


@pytest.mark.slow
@pytest.mark.parametrize("operation", GROWTH_OPERATIONS)
def test_memory_does_not_follow_the_row_count(
    operation, bigdata, quarter_dataset, tmp_path
):
    """Four times the rows must not cost four times the memory.

    This is the assertion that survives a change of machine, of Polars version
    or of dataset size: an implementation that materializes scales with rows,
    a streaming one does not.
    """
    small = measure(operation, quarter_dataset, tmp_path / "small")
    large = measure(operation, bigdata, tmp_path / "large")

    ratio = large.peak_mb / max(small.peak_mb, 1.0)
    assert ratio < GROWTH_LIMIT, (
        f"{operation}: {small.peak_mb:.1f} MiB at {small.rows:,} rows -> "
        f"{large.peak_mb:.1f} MiB at {large.rows:,} rows (x{ratio:.2f}). "
        f"Memory is tracking the row count, which is what streaming is for."
    )


@pytest.mark.slow
def test_an_eager_read_does_follow_the_row_count(
    bigdata, quarter_dataset, tmp_path
):
    """Calibration for :data:`GROWTH_LIMIT`.

    Establishes what "memory follows the data" actually measures on this
    machine, so the limit above is not an arbitrary number.
    """
    small = measure("eager_collect", quarter_dataset, tmp_path / "small")
    large = measure("eager_collect", bigdata, tmp_path / "large")

    ratio = large.peak_mb / max(small.peak_mb, 1.0)
    assert ratio > MIN_EAGER_GROWTH, (
        f"eager collect only grew x{ratio:.2f} for 4x the rows "
        f"({small.peak_mb:.1f} -> {large.peak_mb:.1f} MiB); the growth test "
        f"cannot distinguish streaming from eager on this machine"
    )


@pytest.mark.slow
def test_sample_memory_growth_is_recorded(bigdata, quarter_dataset, tmp_path):
    """``analytics.sample()`` is the one primitive whose memory tracks rows.

    It builds a row index over the whole dataset before filtering, so the peak
    tracks the row count (measured with one thread: 61 MiB at 2M rows, 141 at
    8M, 263 at 16M, 495 at 32M). It is deliberately excluded from
    :func:`test_memory_does_not_follow_the_row_count` rather than silently
    covered by a loose limit, and pinned here so a fix shows up as a change
    instead of going unnoticed.
    """
    small = measure("sample_reservoir", quarter_dataset, tmp_path / "small")
    large = measure("sample_reservoir", bigdata, tmp_path / "large")

    ratio = large.peak_mb / max(small.peak_mb, 1.0)
    assert ratio < SAMPLE_GROWTH_LIMIT, (
        f"sample() grew x{ratio:.2f} for 4x the rows "
        f"({small.peak_mb:.1f} -> {large.peak_mb:.1f} MiB), which is eager "
        f"behaviour, not a bounded draw"
    )


# --------------------------------------------------------------------------
# the generator itself
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_the_generator_does_not_materialize_what_it_writes(tmp_path):
    """A harness whose fixture needs the data in RAM is not a harness.

    Generating four times the rows must not cost four times the memory either,
    or every ceiling asserted above is bounded by the fixture instead of by the
    code under test.
    """
    script = tmp_path / "generate.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import json, resource, sys
            sys.path.insert(0, {str(REPO_ROOT)!r})
            from tests.bigdata import generate

            baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            dataset = generate(sys.argv[1], rows=int(sys.argv[2]), parts=8)
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            print(json.dumps({{
                "rows": dataset.rows,
                "mb": (peak - baseline) / 1024,
            }}))
            """
        ),
        encoding="utf-8",
    )

    env = dict(os.environ)
    env["POLARS_MAX_THREADS"] = "1"

    def run(rows: int, name: str) -> float:
        completed = subprocess.run(
            [sys.executable, str(script), str(tmp_path / name), str(rows)],
            capture_output=True,
            text=True,
            env=env,
            timeout=1800,
        )
        assert completed.returncode == 0, completed.stderr[-4000:]
        return json.loads(completed.stdout.strip().splitlines()[-1])["mb"]

    rows = max(dataset_rows() // 4, 100_000)
    small = run(rows, "gen-small")
    large = run(rows * 4, "gen-large")

    ratio = large / max(small, 1.0)
    assert ratio < GROWTH_LIMIT, (
        f"the generator used {small:.1f} MiB for {rows:,} rows and "
        f"{large:.1f} MiB for {rows * 4:,} (x{ratio:.2f}): it is holding the "
        f"dataset it writes, so every ceiling measured against it is a lie"
    )
