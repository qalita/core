"""
# QALITA (c) COPYRIGHT 2025 - ALL RIGHTS RESERVED -

Unit tests for :mod:`qalita_core.profiling`.

The profiler replaced ydata-profiling, which could only describe what fitted in
RAM and in practice described ``head(500_000)``. So the assertions here are
about the two things that made the replacement necessary: the number of passes
is bounded and independent of the column count, and every statistic that is not
exact says so in ``methods`` rather than being presented as fact.
"""

from __future__ import annotations

import polars as pl
import pytest

from qalita_core import analytics, profiling

from tests.bigdata import (  # noqa: F401 - imported for pytest fixture lookup
    Dataset,
    generate,
    small_dataset,
)


@pytest.fixture(scope="module")
def profiled(small_dataset):
    """Default (approximate) profile of the shared dataset."""
    return profiling.profile(small_dataset.scan(), top_k=3)


# --------------------------------------------------------------------------
# shape and accounting
# --------------------------------------------------------------------------


def test_profile_describes_every_column(profiled, small_dataset):
    assert list(profiled) == list(small_dataset.columns)


def test_null_accounting_adds_up(profiled, small_dataset):
    for name, column in profiled.items():
        assert column["n"] == small_dataset.rows, name
        assert column["count"] + column["n_missing"] == column["n"], name
        assert column["p_missing"] == pytest.approx(
            column["n_missing"] / column["n"]
        ), name

    # The generator puts nulls in `value` and far fewer in `amount`; a
    # frame-wide null count instead of a per-column one would make these equal.
    assert profiled["id"]["n_missing"] == 0
    assert profiled["value"]["n_missing"] > profiled["amount"]["n_missing"] > 0


def test_profile_of_an_empty_schema_is_empty():
    assert profiling.profile(pl.LazyFrame({})) == {}


def _assert_same_profile(actual, expected, path="profile"):
    """Equal, floats within 1e-12: the streaming engine sums in parallel,
    so a mean or std can differ in the last bit from one run to the next."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert actual.keys() == expected.keys(), path
        for key, value in expected.items():
            _assert_same_profile(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, float) and not isinstance(expected, bool):
        assert actual == pytest.approx(expected, rel=1e-12, nan_ok=True), path
    else:
        assert actual == expected, path


def test_profile_accepts_a_precomputed_schema(small_dataset):
    schema = dict(small_dataset.scan().collect_schema())
    from_footer = profiling.profile(small_dataset.scan(), top_k=0)
    from_schema = profiling.profile(
        small_dataset.scan(), schema=schema, top_k=0
    )
    _assert_same_profile(from_schema, from_footer)


# --------------------------------------------------------------------------
# methods: the exact / approximate label
# --------------------------------------------------------------------------


def test_approximate_profile_labels_its_methods(profiled):
    for name, column in profiled.items():
        assert column["methods"]["n_distinct"] == "hyperloglog", name
    for name in ("id", "key", "value", "amount"):
        assert profiled[name]["methods"]["quantiles"] == "histogram", name


def test_exact_profile_labels_its_methods(small_dataset):
    exact = profiling.profile(small_dataset.scan(), exact=True, top_k=0)
    for name, column in exact.items():
        assert column["methods"]["n_distinct"] == "exact", name
    for name in ("id", "key", "value", "amount"):
        assert exact[name]["methods"]["quantiles"] == "exact", name


def test_non_numeric_columns_carry_no_quantile_method(profiled):
    for name in ("uid", "category", "flag", "ts"):
        assert "quantiles" not in profiled[name]["methods"], name
        assert "quantiles" not in profiled[name], name


def test_exact_distinct_counts_are_exact(small_dataset):
    exact = profiling.profile(small_dataset.scan(), exact=True, top_k=0)
    # One distinct value per row: this is the case an approximation gets wrong.
    assert exact["uid"]["n_distinct"] == small_dataset.rows
    assert exact["key"]["n_distinct"] == small_dataset.cardinality
    assert exact["flag"]["n_distinct"] == 2


def test_approximate_distinct_counts_stay_close(profiled, small_dataset):
    assert profiled["uid"]["n_distinct"] == pytest.approx(
        small_dataset.rows, rel=0.10
    )
    assert profiled["key"]["n_distinct"] == pytest.approx(
        small_dataset.cardinality, rel=0.10
    )


def test_p_distinct_never_exceeds_one(profiled):
    """HyperLogLog can overshoot; the ratio must still be reportable as a ratio."""
    for name, column in profiled.items():
        assert 0.0 <= column["p_distinct"] <= 1.0, name


# --------------------------------------------------------------------------
# per-dtype statistics
# --------------------------------------------------------------------------


def test_numeric_columns_get_numeric_statistics(profiled, small_dataset):
    ids = profiled["id"]
    assert ids["min"] == 0
    assert ids["max"] == small_dataset.rows - 1
    assert ids["mean"] == pytest.approx((small_dataset.rows - 1) / 2)
    assert ids["sum"] == small_dataset.rows * (small_dataset.rows - 1) // 2
    assert ids["n_zeros"] == 1
    assert ids["n_negative"] == 0
    assert ids["std"] > 0

    amounts = profiled["amount"]
    assert amounts["min"] < 0 < amounts["max"]
    assert amounts["n_negative"] > 0


def test_numeric_quantiles_are_ordered_and_yield_an_iqr(profiled):
    quantiles = profiled["id"]["quantiles"]
    values = [quantiles[str(q)] for q in profiling.DEFAULT_QUANTILES]
    assert values == sorted(values)
    assert profiled["id"]["iqr"] == pytest.approx(
        quantiles["0.75"] - quantiles["0.25"]
    )


def test_string_columns_get_length_statistics(profiled):
    category = profiled["category"]
    assert category["min_length"] == len("beta")
    assert category["max_length"] == len("epsilon")
    assert (
        category["min_length"]
        <= category["mean_length"]
        <= category["max_length"]
    )
    assert category["n_empty"] == 0
    assert "min" not in category  # numeric-only keys must not leak


def test_temporal_columns_get_bounds_only(profiled):
    ts = profiled["ts"]
    assert ts["min"] < ts["max"]
    assert "mean" not in ts
    assert "quantiles" not in ts


# --------------------------------------------------------------------------
# top-K
# --------------------------------------------------------------------------


def test_top_values_are_bounded_and_sorted(profiled):
    for name, column in profiled.items():
        top = column["top_values"]
        assert len(top) <= 3, name
        counts = [entry["count"] for entry in top]
        assert counts == sorted(counts, reverse=True), name


def test_top_values_report_the_real_frequencies(profiled, small_dataset):
    # `category` has exactly five values, evenly spread by construction.
    for entry in profiled["category"]["top_values"]:
        assert entry["count"] == pytest.approx(
            small_dataset.rows / 5, rel=0.01
        )


def test_profile_survives_a_column_named_count():
    """A dataset is allowed to have a column called "count".

    Nothing here is exotic — an inventory export with `sku,count` used to
    abort the whole profile instead of profiling the other columns.
    """
    lf = pl.LazyFrame({"sku": ["a", "b", "a"], "count": [3, 4, 5]})
    profiled = profiling.profile(lf, top_k=3)

    assert profiled["sku"]["top_values"][0] == {"value": "a", "count": 2}
    assert {entry["value"] for entry in profiled["count"]["top_values"]} == {
        3,
        4,
        5,
    }


def test_profile_survives_columns_named_after_internal_aliases():
    """A dataset is allowed to have a column called anything.

    Every internal alias — the row counter, the histogram bucket/size pair,
    the reserved frequency column — has to be immune to a user column of the
    same name, not merely unlikely to meet one.
    """
    lf = pl.LazyFrame(
        {
            "__rows": [1.0, 2.0, 3.0, 4.0],
            "bucket": [10.0, 20.0, 30.0, 40.0],
            "n": ["a", "b", "a", "c"],
            analytics.COUNT_COLUMN: [1, 1, 2, 2],
        }
    )
    profiled = profiling.profile(lf, top_k=2)

    assert set(profiled) == {"__rows", "bucket", "n", analytics.COUNT_COLUMN}
    # The row counter still describes the frame, not the column named "__rows".
    assert profiled["bucket"]["n"] == 4
    # The histogram ran on the column, not on an alias of the same name.
    assert 10.0 <= profiled["bucket"]["quantiles"]["0.5"] <= 40.0
    assert profiled["n"]["top_values"][0] == {"value": "a", "count": 2}


def test_profile_survives_nan_and_inf_in_a_numeric_column():
    """One NaN or Inf must degrade a statistic, not lose the analysis."""
    values = [float(i) for i in range(1, 1_001)]
    values += [float("nan"), float("inf")]
    lf = pl.LazyFrame({"price": values, "sku": ["a"] * len(values)})

    profiled = profiling.profile(lf, top_k=2)

    assert profiled["price"]["quantiles"]["0.5"] == pytest.approx(
        500.0, abs=2.0
    )
    # Non-finite values are excluded from the histogram, not from the row
    # accounting: they are values, not missing data.
    assert profiled["price"]["n"] == len(values)
    assert profiled["price"]["n_missing"] == 0


def test_top_k_zero_skips_the_per_column_pass(small_dataset):
    without = profiling.profile(small_dataset.scan(), top_k=0)
    assert all("top_values" not in column for column in without.values())


def test_wide_tables_skip_top_k(small_dataset):
    """The per-column pass is the only O(columns) cost; it must be capable of
    being turned off by width, not just by hand."""
    profiled = profiling.profile(
        small_dataset.scan(), top_k=5, max_topk_columns=3
    )
    assert profiled  # still profiled
    assert all("top_values" not in column for column in profiled.values())


# --------------------------------------------------------------------------
# passes
# --------------------------------------------------------------------------


def test_every_scalar_statistic_comes_from_a_single_pass(
    small_dataset, monkeypatch
):
    """One collect for all columns, whatever the column count.

    This is what separates the profiler from the ydata-profiling behaviour it
    replaced: the cost is one scan, not one scan per column.
    """
    collects: list[int] = []
    original = analytics._collect

    def counting(lf):
        collects.append(1)
        return original(lf)

    monkeypatch.setattr(analytics, "_collect", counting)

    profiling.profile(small_dataset.scan(), top_k=0, quantiles=())

    assert (
        sum(collects) == 1
    ), f"{len(small_dataset.columns)} columns cost {sum(collects)} scans"


def test_quantiles_cost_a_bounded_number_of_extra_passes(
    small_dataset, monkeypatch
):
    """Bounds pass plus one bucketing pass per numeric column, and no more."""
    collects: list[int] = []
    original = analytics._collect

    def counting(lf):
        collects.append(1)
        return original(lf)

    monkeypatch.setattr(analytics, "_collect", counting)

    profiling.profile(small_dataset.scan(), top_k=0)

    numeric = analytics.numeric_columns(
        dict(small_dataset.scan().collect_schema())
    )
    # 1 scalar pass + 1 bounds pass + 1 bucketing pass per numeric column.
    assert sum(collects) == 2 + len(numeric)
