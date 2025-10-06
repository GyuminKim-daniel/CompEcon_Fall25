# test_analysis_pipeline.py
import numpy as np
import pandas as pd
import pytest

# adjust import if your functions live in another module:
from analysis_pipeline import generate_stock_variables, concat_variants


def _make_sample_df_basic():
    """
    Small sample covering three groups A/B/C used in tests.
    Group A: eventday=1, size78 out (stock==0), 80&82 in -> OOS==1 for eventday group A
    Group B: eventday=1, all in-stock -> OOS==0 for eventday group B
    Group C: eventday=1 has size78 out, so treat_sum >=1 for (C,Z,green) group
    """
    rows = []
    # Group A: size 78 out, 80 & 82 in (eventday=1)
    rows += [
        {"eventday": 1, "storecode": "A", "stylecode": "X", "color": "red",   "size": 78, "stock": 0},
        {"eventday": 1, "storecode": "A", "stylecode": "X", "color": "red",   "size": 80, "stock": 1},
        {"eventday": 1, "storecode": "A", "stylecode": "X", "color": "red",   "size": 82, "stock": 1},
    ]
    # Group B: all in-stock (eventday=1)
    rows += [
        {"eventday": 1, "storecode": "B", "stylecode": "Y", "color": "blue",  "size": 78, "stock": 1},
        {"eventday": 1, "storecode": "B", "stylecode": "Y", "color": "blue",  "size": 80, "stock": 1},
        {"eventday": 1, "storecode": "B", "stylecode": "Y", "color": "blue",  "size": 82, "stock": 1},
    ]
    # Group C: size78 out on eventday=1 (treat_sum >=1)
    rows += [
        {"eventday": 1, "storecode": "C", "stylecode": "Z", "color": "green", "size": 78, "stock": 0},
        {"eventday": 1, "storecode": "C", "stylecode": "Z", "color": "green", "size": 80, "stock": 1},
        {"eventday": 1, "storecode": "C", "stylecode": "Z", "color": "green", "size": 82, "stock": 1},
        # add a second day rows to show treat_sum computed over group (store,style,color)
        {"eventday": 2, "storecode": "C", "stylecode": "Z", "color": "green", "size": 80, "stock": 1},
        {"eventday": 2, "storecode": "C", "stylecode": "Z", "color": "green", "size": 82, "stock": 1},
    ]
    return pd.DataFrame(rows)


def test_generate_stock_variables_basic_logic():
    """
    Basic logic: check s_sum/t_sum/c_sum and OOS for small sample.
    Use min_days=0 so min_days filtering does not force NaN in this tiny example.
    """
    df1 = _make_sample_df_basic()
    df2 = generate_stock_variables(df1, stockout_size=78, treat_size=80, control_size=82, min_days=0)

    a_mask = (df2["storecode"] == "A") & (df2["stylecode"] == "X") & (df2["color"] == "red")
    b_mask = (df2["storecode"] == "B") & (df2["stylecode"] == "Y") & (df2["color"] == "blue")
    c_mask = (df2["storecode"] == "C") & (df2["stylecode"] == "Z") & (df2["color"] == "green")

    # Group A: expected s_sum==1, t_sum==0, c_sum==0, OOS==1
    assert (df2.loc[a_mask, "s_sum"] == 1).all()
    assert (df2.loc[a_mask, "t_sum"] == 0).all()
    assert (df2.loc[a_mask, "c_sum"] == 0).all()
    assert (df2.loc[a_mask, "OOS"] == 1).all()

    # Group B: all in-stock -> sums == 0 and OOS == 0
    assert (df2.loc[b_mask, ["s_sum", "t_sum", "c_sum"]].sum(axis=1) == 0).all()
    assert (df2.loc[b_mask, "OOS"] == 0).all()

    # Group C: treat_sum >=1 (because size78 was out at least once for that (store,style,color)),
    # so treatment==1 for size == treat_size rows and treatment==0 for control size rows.
    # use rows where treat_sum>=1 to avoid comparing rows where function leaves NaN
    cond_treat_rows = c_mask & (df2["treat_sum"] >= 1) & (df2["size"] == 80)
    cond_control_rows = c_mask & (df2["treat_sum"] >= 1) & (df2["size"] == 82)

    assert (df2.loc[cond_treat_rows, "treatment"] == 1).all()
    assert (df2.loc[cond_control_rows, "treatment"] == 0).all()


def test_generate_stock_variables_min_days_filter():
    """
    Ensure min_days filtering sets OOS to NaN when counts are below the threshold.
    We pick min_days=10 for the tiny example which should force NaN.
    """
    df1 = _make_sample_df_basic()
    df2 = generate_stock_variables(df1, stockout_size=78, treat_size=80, control_size=82, min_days=10)

    # OOS should be NaN everywhere because in_stock_day/out_stock_day are < 10
    assert df2["OOS"].isna().all()


def test_concat_variants_removes_duplicates():
    df1 = _make_sample_df_basic()
    v1 = generate_stock_variables(df1, stockout_size=78, treat_size=80, control_size=82, min_days=0)
    v2 = generate_stock_variables(df1, stockout_size=78, treat_size=80, control_size=82, min_days=0)
    combined = concat_variants([v1, v2])
    # concatenating identical variants yields the same unique rows as a single variant
    assert combined.drop_duplicates().shape[0] == v1.drop_duplicates().shape[0]
