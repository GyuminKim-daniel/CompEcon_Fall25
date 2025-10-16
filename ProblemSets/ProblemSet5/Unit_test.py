# test_analysis_pipeline.py
"""
Unit tests for analysis_pipeline.generate_stock_variables and concat_variants.

These tests verify the core data-transformation logic used in the ProblemSet5
pipeline:

- test_generate_stock_variables_basic_logic:
    Confirms that the function correctly flags event-day stockout patterns
    (s_sum/t_sum/c_sum), assigns OOS = 0/1 per the specification, and
    assigns treatment == 1 for treatment-size rows and treatment == 0 for
    control-size rows when treat_sum >= 1.

- test_generate_stock_variables_min_days_filter:
    Confirms the min_days filtering: when a group's in/out stock days are
    below min_days, OOS values are set to missing (NaN).

- test_concat_variants_removes_duplicates:
    Confirms that concat_variants concatenates variant DataFrames and removes
    exact duplicate rows.

Notes
-----
- Tests use a small synthetic DataFrame returned by _make_sample_df_basic().
- We pass min_days=0 for most checks so the min_days filter does not force NaNs
  in the tiny example; a separate test verifies the min_days behavior.
"""
import pandas as pd
import numpy as np

import pytest

# import target functions (adjust import if module name differs)
from analysis_pipeline import generate_stock_variables, concat_variants


def _make_sample_df_basic() -> pd.DataFrame:
    """
    Create a tiny dataset with three conceptual groups (A, B, C) used in tests.

    Group A: eventday=1, size 78 OUT (stock==0), sizes 80 & 82 IN -> OOS==1 for that eventday-group.
    Group B: eventday=1, all sizes IN -> OOS==0.
    Group C: has at least one size 78 OUT across eventdays -> treat_sum >= 1 for the (store,style,color) group.
    """
    rows = []
    # Group A (A, X, red) -- trigger out only (size 78 out)
    rows += [
        {"eventday": 1, "storecode": "A", "stylecode": "X", "color": "red",   "size": 78, "stock": 0},
        {"eventday": 1, "storecode": "A", "stylecode": "X", "color": "red",   "size": 80, "stock": 1},
        {"eventday": 1, "storecode": "A", "stylecode": "X", "color": "red",   "size": 82, "stock": 1},
    ]
    # Group B (B, Y, blue) -- all in-stock
    rows += [
        {"eventday": 1, "storecode": "B", "stylecode": "Y", "color": "blue",  "size": 78, "stock": 1},
        {"eventday": 1, "storecode": "B", "stylecode": "Y", "color": "blue",  "size": 80, "stock": 1},
        {"eventday": 1, "storecode": "B", "stylecode": "Y", "color": "blue",  "size": 82, "stock": 1},
    ]
    # Group C (C, Z, green) -- size78 out on eventday=1; add extra day rows to ensure treat_sum counts across days
    rows += [
        {"eventday": 1, "storecode": "C", "stylecode": "Z", "color": "green", "size": 78, "stock": 0},
        {"eventday": 1, "storecode": "C", "stylecode": "Z", "color": "green", "size": 80, "stock": 1},
        {"eventday": 1, "storecode": "C", "stylecode": "Z", "color": "green", "size": 82, "stock": 1},
        {"eventday": 2, "storecode": "C", "stylecode": "Z", "color": "green", "size": 80, "stock": 1},
        {"eventday": 2, "storecode": "C", "stylecode": "Z", "color": "green", "size": 82, "stock": 1},
    ]
    return pd.DataFrame(rows)


def test_generate_stock_variables_basic_logic():
    """
    Verify core logic:
      - event-day sums s_sum/t_sum/c_sum are correct,
      - OOS set to 1 for group A and 0 for group B,
      - treatment assignment for group C when treat_sum >= 1.
    Uses min_days=0 so OOS is not forced to NaN by the min_days rule.
    """
    df1 = _make_sample_df_basic()
    df2 = generate_stock_variables(df1, stockout_size=78, treat_size=80, control_size=82, min_days=0)

    a_mask = (df2["storecode"] == "A") & (df2["stylecode"] == "X") & (df2["color"] == "red")
    b_mask = (df2["storecode"] == "B") & (df2["stylecode"] == "Y") & (df2["color"] == "blue")
    c_mask = (df2["storecode"] == "C") & (df2["stylecode"] == "Z") & (df2["color"] == "green")

    # Group A: trigger out only -> s_sum==1, t_sum==0, c_sum==0, OOS==1
    assert (df2.loc[a_mask, "s_sum"] == 1).all()
    assert (df2.loc[a_mask, "t_sum"] == 0).all()
    assert (df2.loc[a_mask, "c_sum"] == 0).all()
    assert (df2.loc[a_mask, "OOS"] == 1).all()

    # Group B: all in-stock -> sums == 0 and OOS == 0
    assert (df2.loc[b_mask, ["s_sum", "t_sum", "c_sum"]].sum(axis=1) == 0).all()
    assert (df2.loc[b_mask, "OOS"] == 0).all()

    # Group C: treat_sum >= 1 -> treatment == 1 for treat_size rows and 0 for control_size rows
    cond_treat_rows = c_mask & (df2["treat_sum"] >= 1) & (df2["size"] == 80)
    cond_control_rows = c_mask & (df2["treat_sum"] >= 1) & (df2["size"] == 82)

    assert (df2.loc[cond_treat_rows, "treatment"] == 1).all()
    assert (df2.loc[cond_control_rows, "treatment"] == 0).all()


def test_generate_stock_variables_min_days_filter():
    """
    When min_days is large relative to the tiny sample, in_stock_day/out_stock_day counts
    are below min_days and OOS should be set to NaN everywhere.
    """
    df1 = _make_sample_df_basic()
    df2 = generate_stock_variables(df1, stockout_size=78, treat_size=80, control_size=82, min_days=10)

    assert df2["OOS"].isna().all()


def test_concat_variants_removes_duplicates():
    """
    concat_variants should return unique rows when concatenating identical variants.
    """
    df1 = _make_sample_df_basic()
    v1 = generate_stock_variables(df1, stockout_size=78, treat_size=80, control_size=82, min_days=0)
    v2 = generate_stock_variables(df1, stockout_size=78, treat_size=80, control_size=82, min_days=0)
    combined = concat_variants([v1, v2])

    # combined should not have more unique rows than a single variant
    assert combined.drop_duplicates().shape[0] == v1.drop_duplicates().shape[0]
