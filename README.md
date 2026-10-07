# QALITA Core

<p align="center">
  <img width="250px" height="auto" src="https://app.platform.qalita.io/logo.svg" style="max-width:250px;"/>
</p>

QALITA Core is a lightweight helper library used by QALITA packs to load data from multiple sources, materialize them to Parquet in deterministic chunks, and share common utilities (sanitization and aggregation helpers).

## Key features

- Unified data access via a simple `DataSource` abstraction and factory
- File, database, and object storage loaders with streaming to Parquet
- Deterministic, size-bounded Parquet chunking with stable filenames
- `qalita_core.analytics` — streaming primitives (one pass, bounded results)
  that packs compute with instead of loading frames
- `qalita_core.profiling` — a whole-dataset profile in a handful of passes
- Shared aggregators for completeness, outliers, duplicates, and timeliness,
  all fed LazyFrames
- Minimal pack runtime with JSON config loading and simple asset persistence

pandas is **optional** (`pip install qalita_core[pandas]`): importing
`qalita_core` does not import it, and nothing on the data path uses it.

## Supported sources

- Files: CSV (`.csv`), Excel (`.xlsx`), JSON, Parquet (pass-through)
- Folders (`type: "folder"`): every data file of a directory, one logical
  object per file (`config.path`, optional `config.recursive`); select files
  with `table_or_query` (`"*"`, a name, or a list), unreadable files are
  skipped and listed in `skipped_objects`
- Databases: PostgreSQL, MySQL, Oracle, MS SQL Server, SQLite
- Object storage: Amazon S3, Google Cloud Storage, Azure Blob (via `abfs`), HDFS

Notes:
- A `file` source pointed at a directory still loads its first file only; use
  `type: "folder"` to load all of them.
- SQLite is supported through the generic `DatabaseSource` when selected via `type: "sqlite"`.

## Installation

Prerequisites: Python 3.10–3.12 and uv.

Install dependencies and set your environment:

```bash
pip install uv
uv sync
```

Open a uv shell when developing:

```bash
uv shell
```

## Quickstart

### Use within a Pack

`Pack` loads four JSON files by default (overridable) and provides `load_data()` for `source` or `target` triggers.

```python
from qalita_core.pack import Pack

pack = Pack(configs={
    "pack_conf": "./pack_conf.json",
    "source_conf": "./source_conf.json",
    "target_conf": "./target_conf.json",
    "agent_file": "~/.qalita/.worker",
})

# Ensure chunking/output are set (can be in pack_conf["job"] too)
pack.pack_config.setdefault("job", {})
pack.pack_config["job"]["parquet_output_dir"] = "./parquet"
pack.pack_config["job"]["chunk_rows"] = 100_000

# Load source
source_paths = pack.load_data("source")
# Load target (optional)
target_paths = pack.load_data("target")

# Persist custom metrics/recommendations/schemas to JSON files
pack.metrics.data.append({"key": "score", "value": "0.95", "scope": {"perimeter": "dataset", "value": "my_dataset"}})
pack.metrics.save()       # writes metrics.json
pack.recommendations.save()  # writes recommendations.json
pack.schemas.save()          # writes schemas.json
```

## Parquet chunking and filenames

- CSV/JSON/Excel are streamed with `chunksize` into multiple parquet files.
- Databases are read with chunked SQL via SQLAlchemy/`pandas.read_sql`.
- Filenames use a stable pattern: `<source>_<object>_part_<k>.parquet` where:
  - `<source>` is a slug of the source type (e.g. `file`, `sqlite`, `postgresql`).
  - `<object>` is a slug of the table name, query label, or file stem.
  - Example: `file_testdata_part_1.parquet`, `sqlite_items_part_3.parquet`, `sqlite_query_part_2.parquet`.

Configure output and size via `pack_config`:

- `parquet_output_dir` (default: `./parquet`)
- `chunk_rows` (default: `100000`)
- Optional `job.source.skiprows` applied to CSV/Excel

## Computing over data — `qalita_core.analytics`

A pack never reads a file and never holds a frame. It asks `Pack.scan()` for a
LazyFrame and computes with `analytics`, which is streaming-only: every collect
runs with `engine="streaming"` and **raises** (`StreamingCollectError`) instead
of falling back to the in-memory engine — on a large source that fallback is an
OOM kill dressed as resilience.

```python
import polars as pl
from qalita_core import analytics
from qalita_core.pack import Pack

with Pack() as pack:
    lf = pack.scan("source")            # LazyFrame; never a path, never a chunk
    schema = pack.schema("source")      # Parquet footers, no data read

    # ONE pass, every statistic at once. A `for column in columns` loop issuing
    # one query per column re-reads the whole source once per column.
    stats = analytics.agg(lf, {
        "rows": pl.len(),
        **{f"{c}__nulls": pl.col(c).null_count()
           for c in analytics.string_columns(schema)},
    })

    n_failed, examples = analytics.failures(   # count exact, rows bounded
        lf, pl.col("age") < 0, limit=10
    )
```

| Function | What it gives back |
|---|---|
| `row_count(lf)` | exact count; Parquet answers from the footers |
| `agg(lf, {name: expr})` | many aggregates in one streaming pass |
| `sample(lf, n, seed=…, method=…)` | at most `n` rows drawn from the WHOLE dataset |
| `approx_n_unique(lf, cols, exact=False)` | distinct counts (HyperLogLog by default) |
| `quantiles(lf, cols, qs, exact=False)` | quantiles (histogram by default, O(bins) memory) |
| `top_k(lf, by, k)` / `value_counts(lf, col, k)` | bounded ranked rows |
| `failures(lf, predicate, limit=…)` | exact failing count + bounded example rows |
| `sink(lf, path, max_rows_per_file=…)` | Parquet written without materializing |
| `numeric_columns` / `string_columns` / `temporal_columns` | column selection from a schema |

Approximate is the default for distinct counts and quantiles. Any metric derived
from an approximate statistic also emits a sibling metric `<key>_method` whose
value is `hyperloglog`, `histogram` or `exact`, so the UI can label the number.
`qalita_core.profiling.profile(lf, schema=…, exact=…)` applies the same contract
to a whole-dataset profile.

## Aggregation helpers (for packs)

Helpers centralize common result/metric aggregation logic. They take LazyFrames
and compute through `analytics`, so memory depends on the number of columns and
the requested top-K, never on the number of rows.

```python
from qalita_core import (
    detect_chunked_from_items,
    normalize_and_dedupe_recommendations,
    CompletenessAggregator,
    OutlierAggregator,
    DuplicateAggregator,
    TimelinessAggregator,
)
from qalita_core.aggregation import streaming_outliers
```

- `CompletenessAggregator.add_lf(lf)`: column/dataset completeness and schema
  extraction, one streaming pass per frame whatever its width.
- `DuplicateAggregator.add_lf(lf)`: registers a frame; the group-by runs once
  over every registered frame at `finalize_metrics()` / `duplicate_count()`
  time, so a key duplicated across two chunks is seen. `get_duplicate_keys(limit)`
  returns the **most duplicated** keys, capped (1000 by default).
- `TimelinessAggregator.add_lf(lf, date_columns)`: min/max of every date column
  in one batched pass. Temporal columns become date observations, numeric ones
  are read as years.
- `streaming_outliers(lf, columns, method="iqr"|"zscore", threshold=…, exact=…)`:
  two-pass global outlier detection — pass 1 computes the fences over the whole
  column, pass 2 counts the rows outside them in a single aggregation. Feed the
  result to `OutlierAggregator.add_streaming_outliers(results, rows=…)`.

```python
results = streaming_outliers(pack.scan("source"),
                             analytics.numeric_columns(schema))
agg = OutlierAggregator()
agg.add_streaming_outliers(results, rows=analytics.row_count(lf))
metrics, recommendations = agg.finalize_metrics_and_recommendations(name, 0.8)
```

`OutlierAggregator` no longer accepts per-chunk normality: a row-weighted mean of
per-chunk normality is not the normality of the dataset, and per-chunk fences
answer a different question than global ones. The legacy `add_column_stats()` /
`add_dataset_stats()` are gone; use `add_column_result()` /
`add_dataset_result()` / `add_streaming_outliers()`.

## pandas (optional extra)

pandas is not a dependency of `qalita_core`; install `qalita_core[pandas]` if a
pack needs it. The `add_df()` entry points of the aggregators still work on
frames that are already in memory.

Importing `qalita_core` no longer monkeypatches
`DataFrame.to_parquet`: the hook sanitizes into a **copy** of the frame, which
doubles peak RAM on every write. Ask for it explicitly, or call the sanitizer:

```python
from qalita_core import (
    sanitize_dataframe_for_parquet,       # resolved lazily, imports pandas
    install_pandas_parquet_sanitization,  # opt-in monkeypatch
)
clean_df = sanitize_dataframe_for_parquet(df)          # copy=False to avoid it
```

## Figures — `figures.json`

`FiguresAsset` (exposed as `pack.figures`) is the sibling of `pack.metrics`: metrics
carry the numbers, figures carry the aggregates that explain them. A pack declares
an **intention**, never a chart type — the platform picks the rendering.

```python
from qalita_core.pack import Pack

with Pack() as pack:
    pack.figures.declare_measure(
        "p_missing", unit="ratio", direction="lower_is_better", target=0.05
    )
    pack.figures.add(
        "missing_by_column",
        intent="breakdown",          # what contributes to the number
        of="p_cells_missing",        # the metric this figure explains
        frame=df,
        dims=["column"],
        measures=["p_missing"],
        scope={"perimeter": "dataset", "value": name},
    )
    pack.figures.save()
```

### API

```python
declare_measure(key, *, unit=None, direction=None, target=None,
                warn=None, label=None, description=None)
```
Declares the semantics of a measure — referenceable both by a figure's `measures`/`of`
and by a `metrics.json` key.

```python
add(key, *, intent, frame, dims, measures, scope, of=None, title=None,
    max_rows=5000)
```
Adds one figure. `intent` is one of `breakdown`, `composition`, `distribution`,
`trend`, `comparison`, `matrix`, `flow`. `frame` must be an **aggregate**: pandas
or polars `DataFrame`, or a list of dicts — at least one dimension, at least one
measure, exactly one row per dimension tuple. A polars `LazyFrame` (what
`pack.scan_data()` returns, the recommended path for 100 GB+ sources) is
rejected with a `TypeError` telling you to `.collect()` first — `figures.py`
never materializes your plan for you. Call `.collect()` (or
`.collect(engine="streaming")`) on your **aggregate**, not on the raw source,
before passing it; collecting inside `add()` would materialize the whole plan
in the worker and defeat the `max_rows` cap below, which exists to prevent
exactly that. `dims` items are a bare name
(`"column"`, nominal) or a `(name, type)` tuple (`type` one of `nominal`,
`ordinal`, `temporal`). Rows beyond `max_rows` are dropped and the figure is
flagged `truncated: true` rather than raising.

`add()` raises on an empty `dims`, an empty `measures`, or a duplicate dimension
tuple in `frame` (a strong signal the frame was never aggregated). It does
**not** catch every non-aggregate: a frame keyed by a unique identifier —
`dims=["patient_id"], measures=["age"]` — has no duplicate tuple by construction
and clears every guard, shipping one row per patient (up to `max_rows`, truncated
above that). Nothing but the caller enforces "aggregate only, never raw rows" in
that case — worth remembering on regulated data.

`save()` additionally verifies that every measure named by a figure's `measures`
or `of` was declared via `declare_measure`; if not, it raises rather than writing
a figure that can never be linked to its metric. This means an undeclared measure
fails the whole pack run, not just that figure.

```python
add_raw(key, *, option, scope, title=None)
```
Escape hatch: a raw ECharts `option` dict, bypassing intent/self-service
rendering and excluded from reporting/drill-down. Use only when no `intent`
fits.

```python
top_n(frame, by, n, other=False, label="Autres", dim=None)
```
Keeps the `n` largest rows by `by`, dropping the tail. On a polars `DataFrame`
the ranking runs inside the engine and only the `n` kept rows (plus the folded
one) cross into Python — `top_n` is usually fed a `group_by` result, i.e. the
high-cardinality case. `other=True` folds the
tail into one row labeled `label` instead — valid **only** for an additive
measure (a count): summing ratios into that row produces a wrong number, which
is why `other` defaults to `False`. `dim` names the column that receives
`label` on the folded row; it is inferred only when exactly one column other
than `by` exists, and `top_n` raises rather than guess when there is more than
one candidate (or none).

Note: folding an integer dimension column produces different element types
across backends — `["1", "2", "Autres"]` under polars vs. `[1, 2, "Autres"]`
under pandas, because the label forces a common supertype. Both are valid JSON;
this is intentional and covered by tests, not a bug to work around.

### Chunking and `figures.add`

Data above `chunk_rows` (default 100 000) is processed in chunks (see above). A
pack that calls `figures.add` once per chunk without aggregating across chunks
first will emit one row per rule *execution*, not per dimension value — producing
duplicate dimension tuples that `add()` rejects. Either accumulate into a dict
keyed by the dimension across the whole loop and call `add()` once at the end, or
concatenate chunks into a single frame before computing aggregates. Both are
legitimate; pick one per pack.

## Development

- Tests: `uv run pytest`
- Formatting: `uv run black .`
- Linting: `uv run flake8` and `uv run pylint <module>`
- Editable install while debugging:

```bash
uv sync
uv pip install -e .
```

## Documentation

Additional material can be found in the online documentation: `https://doc.qalita.io/`.
