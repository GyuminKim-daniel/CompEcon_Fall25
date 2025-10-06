"""Analysis pipeline: stock variable generation, plotting, estimation, and LaTeX export.

This module provides concise, documented functions to:
 - generate in-stock/out-of-stock and treatment indicators,
 - combine sub datasets,
 - save bar / trend plots,
 - fit OLS with cluster-robust SEs using a uniform interface,
 - extract selected coefficients (including interactions),
 - build and export a LaTeX table of selected coefficients.

Usage:
    Import functions or run the script directly:
        python analysis_pipeline.py
"""

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Tuple


# -------------------------
# Configuration (relative paths)
# -------------------------


# Get the path to the ProblemSet5 folder (this script's parent)
try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()

# Navigate two levels up, then into the shared Data folder
DATA = ROOT.parents[1] / "Data" / "Raw.dta"   # <- relative path to your Data folder
IMAGES = ROOT / "images"
IMAGES.mkdir(exist_ok=True)


def generate_stock_variables(
    df: pd.DataFrame,
    stockout_size: Any = 78,
    treat_size: Any = 80,
    control_size: Any = 82,
    min_days: int = 18,
) -> pd.DataFrame:
    """
    Produce stock / OOS / treatment indicators.

    Logic (matches your description):
      - `stockoutproduct` marks rows where size == stockout_size and stock == 0.
      - For each (eventday, storecode, stylecode, color) we compute:
          s_sum = sum(stockoutproduct)
          t_sum = sum(treatmentproduct)   # size == treat_size & stock==0 (used for OOS logic)
          c_sum = sum(controlproduct)     # size == control_size & stock==0 (used for OOS logic)
      - OOS for that eventday-group:
          OOS = 0  if s_sum + t_sum + c_sum == 0   (all in-stock)
          OOS = 1  if s_sum == 1 and t_sum == 0 and c_sum == 0
                   (i.e., size78 out while 80 & 82 in)
          Otherwise OOS = NaN (will be filtered by min_days below)
      - For treatment assignment across (storecode, stylecode, color):
          treat_sum = sum(stockoutproduct)  # presence of any size78 stockout in that (store,style,color)
          If treat_sum >= 1:
              rows with size == treat_size  => treatment = 1
              rows with size == control_size => treatment = 0
          else treatment = NaN
      - `in_stock_day` and `out_stock_day` count days with OOS==0 or OOS==1 per (store,style,size,color).
        If either count < min_days, set OOS = NaN for that (store,style,size,color).

    Parameters
    ----------
    df : pd.DataFrame
        Input frame; must contain 'stock','size','eventday','storecode','stylecode','color'.
    stockout_size, treat_size, control_size : scalar
        Size codes for the three roles (trigger, treatment, control).
    min_days : int
        Minimum number of in-stock or out-stock days required to keep OOS (default 18).

    Returns
    -------
    pd.DataFrame
        A copy of df with added columns:
          stockoutproduct, treatmentproduct, controlproduct,
          s_sum, t_sum, c_sum, OOS, in_stock_day, out_stock_day, treat_sum, treatment
    """
    required = {"stock", "size", "eventday", "storecode", "stylecode", "color"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    d = df.copy()

    # Indicator definitions
    d["stockoutproduct"]  = ((d["stock"] == 0) & (d["size"] == stockout_size)).astype(int)
    d["treatmentproduct"] = ((d["stock"] == 0) & (d["size"] == treat_size)).astype(int)
    d["controlproduct"]   = ((d["stock"] == 0) & (d["size"] == control_size)).astype(int)

    # Sums by event-day group (used to define OOS)
    g_day = ["eventday", "storecode", "stylecode", "color"]
    d["s_sum"] = d.groupby(g_day)["stockoutproduct"].transform("sum")
    d["t_sum"] = d.groupby(g_day)["treatmentproduct"].transform("sum")
    d["c_sum"] = d.groupby(g_day)["controlproduct"].transform("sum")

    # OOS logic: 0 if none out, 1 when size78 out and 80/82 not out
    d["OOS"] = np.nan
    d.loc[(d["s_sum"] + d["t_sum"] + d["c_sum"]) == 0, "OOS"] = 0
    d.loc[(d["s_sum"] == 1) & (d["t_sum"] == 0) & (d["c_sum"] == 0), "OOS"] = 1

    # Counts by (store,style,size,color) to apply min_days rule
    g_size = ["storecode", "stylecode", "size", "color"]
    d["in_stock_day"]  = d.groupby(g_size)["OOS"].transform(lambda s: s.eq(0).sum())
    d["out_stock_day"] = d.groupby(g_size)["OOS"].transform(lambda s: s.eq(1).sum())

    # If not enough days, mark OOS as missing (NaN)
    d.loc[d["in_stock_day"] < min_days, "OOS"] = np.nan
    d.loc[d["out_stock_day"] < min_days, "OOS"] = np.nan

    # Treatment assignment by (storecode, stylecode, color):
    # If any size78 stockout occurred in that (store,style,color), then
    # mark size==treat_size rows as treatment=1 and size==control_size rows as treatment=0.
    g_group = ["storecode", "stylecode", "color"]
    d["treat_sum"] = d.groupby(g_group)["stockoutproduct"].transform("sum")
    d["treatment"] = np.nan
    d.loc[(d["treat_sum"] >= 1) & (d["size"] == treat_size), "treatment"] = 1
    d.loc[(d["treat_sum"] >= 1) & (d["size"] == control_size), "treatment"] = 0

    return d


def concat_variants(dfs: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate dataframes, drop exact duplicates, reset index."""
    combined = pd.concat(list(dfs), ignore_index=True)
    return combined.drop_duplicates().reset_index(drop=True)


# -------------------------
# Plotting helpers
# -------------------------
def save_barplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str = None,
    outpath: Path = IMAGES / "plot.png",
    figsize: Tuple[int, int] = (8, 6),
    title: str = "",
):
    """Save a seaborn barplot to outpath (creates directory if needed)."""
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=figsize)
    sns.barplot(data=df, x=x, y=y, hue=hue)
    plt.title(title or f"{y} by {x}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


def save_line_trend_by_age(
    df: pd.DataFrame,
    age_col: str = "age",
    y_col: str = "ln_sales",
    group_col: str = "treatment",
    outpath: Path = IMAGES / "trend.png",
    descending: bool = True,
):
    """Save trend of mean(y_col) by age and group_col. Reverse x-axis if descending."""
    outpath.parent.mkdir(parents=True, exist_ok=True)
    df2 = df.dropna(subset=[age_col, y_col, group_col])
    trend = df2.groupby([age_col, group_col])[y_col].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=trend, x=age_col, y=y_col, hue=group_col, marker="o")
    if descending:
        plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


# -------------------------
# Modeling: uniform fit/extract/export interface
# -------------------------
def fit_model(df: pd.DataFrame, formula: str, cluster_col: str = "id_dum"):
    """
    Fit OLS with cluster-robust SEs.

    The function:
      - chooses required columns from a common set present in df,
      - drops rows with NA in those required columns,
      - converts common fixed-effect variables to category dtype on the modeling copy,
      - fits OLS and returns (result, df_model).

    Returns
    -------
    (result, df_model)
    """
    # common variables used in typical formulas
    common = [
        "ln_sales",
        "treatment",
        "OOS",
        "fit_in",
        "age",
        "price",
        "store_dum",
        "style_dum",
        "size",
        "year",
        "month",
        "date",
        "color_dum",
    ]
    required = [c for c in common if c in df.columns] + ([cluster_col] if cluster_col in df.columns else [])
    df_model = df.dropna(subset=required).copy()
    if df_model.empty:
        raise ValueError("No observations left after dropping missing values for required variables.")
    cats = ["treatment", "OOS", "store_dum", "style_dum", "size", "year", "month", "date", "color_dum"]
    for c in cats:
        if c in df_model.columns:
            df_model[c] = df_model[c].astype("category")
    model = smf.ols(formula=formula, data=df_model)
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df_model[cluster_col]})
    return result, df_model


def extract_params(res, substrings: List[str]) -> pd.DataFrame:
    """
    Return DataFrame of coef, se, t, p for params whose names contain any substring.
    Substring matching is case-insensitive.
    """
    names = [n for n in res.params.index if any(s.lower() in n.lower() for s in substrings)]
    df = pd.DataFrame(
        {
            "coef": res.params.loc[names],
            "se": res.bse.loc[names],
            "t": res.tvalues.loc[names],
            "p": res.pvalues.loc[names],
        }
    )
    return df


def _stars(p: float) -> str:
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.1:
        return "*"
    return ""


def build_coef_se_table(results: List, model_names: List[str], var_substrings: List[str]) -> pd.DataFrame:
    """
    Build DataFrame showing selected coefficients and standard errors (with stars).

    The table format: for each selected parameter we produce two rows:
      - coef (with significance stars)
      - (se)
    Additional rows: Observations and R-squared.
    """
    selected = []
    for res in results:
        for name in res.params.index:
            if any(s.lower() in name.lower() for s in var_substrings):
                if name not in selected:
                    selected.append(name)
    selected = sorted(selected)

    rows = []
    idx = []
    for pname in selected:
        coef_row = []
        se_row = []
        for res in results:
            if pname in res.params.index:
                pval = res.pvalues[pname]
                coef_row.append(f"{res.params[pname]:.3f}{_stars(pval)}")
                se_row.append(f"({res.bse[pname]:.3f})")
            else:
                coef_row.append("")
                se_row.append("")
        rows.append(coef_row)
        idx.append(pname)
        rows.append(se_row)
        idx.append(pname + " (se)")

    table = pd.DataFrame(rows, index=idx, columns=model_names)
    nobs_row = [int(getattr(r, "nobs", np.nan)) for r in results]
    r2_row = [f"{getattr(r, 'rsquared', np.nan):.3f}" for r in results]
    table.loc["Observations"] = nobs_row
    table.loc["R-squared"] = r2_row
    return table


def export_table_to_tex(table: pd.DataFrame, outpath: Path, caption: str = "Regression results", label: str = "tab:results"):
    """Save the DataFrame as a LaTeX table inside a table environment (escape underscores)."""
    outpath.parent.mkdir(parents=True, exist_ok=True)
    latex_tab = table.to_latex(escape=True, na_rep="", column_format="l" + "r" * (table.shape[1]))
    latex = f"\\begin{{table}}[htbp]\n\\centering\n{latex_tab}\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}\n"
    outpath.write_text(latex, encoding="utf-8")
    print(f"Saved LaTeX table to {outpath}")


# -------------------------
# Script flow when executed directly
# -------------------------
if __name__ == "__main__":
    # 1) load data
      from pathlib import Path
import sys

# Find the directory this script is in
try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()

# Move two folders up from ProblemSet5 to reach CompEcon_Fall25, then into Data/
DATA = ROOT.parents[1] / "Data" / "Raw.dta"
IMAGES = ROOT / "images"
IMAGES.mkdir(exist_ok=True)

# Verify path and load data
if not DATA.exists():
    print(f"\n❌ Data file not found at expected path:\n  {DATA}", file=sys.stderr)
    print("Please ensure Raw.dta is located in:\n  CompEcon_Fall25/Data/\n", file=sys.stderr)
    raise FileNotFoundError(f"Missing data file: {DATA}")
else:
    print(f"✅ Loading data from: {DATA}")
    df1 = pd.read_stata(DATA)


    # 2) generate variants and combine
    combos = [
        (78, 80, 82),
        (80, 82, 86),
        (74, 76, 78),
        (76, 78, 80),
        (82, 86, 90),
    ]
    dfs = [generate_stock_variables(df1, *c, min_days=18) for c in combos]
    df_combined = concat_variants(dfs)

    # 3) save plots
    save_barplot(df_combined, x="OOS", y="ln_sales", hue="treatment", outpath=IMAGES / "LogSales.png", title="Log Sales by OOS and Treatment")
    save_barplot(df_combined, x="OOS", y="fit_in", hue="treatment", outpath=IMAGES / "Fittingroomvisits.png", title="Fitting Room Visits by OOS and Treatment")
    save_line_trend_by_age(df_combined[df_combined["OOS"] == 0], age_col="age", y_col="ln_sales", group_col="treatment", outpath=IMAGES / "SalesTrend.png", descending=True)

    # 4) Fit three models (uniformly)
    did_formula = "ln_sales ~ C(treatment) * C(OOS) + fit_in + age + price + C(store_dum) + C(style_dum) + C(size) + C(year) + C(month) + C(date) + C(color_dum)"
    res_did, _ = fit_model(df_combined, did_formula, cluster_col="id_dum")

    three_way_formula = "ln_sales ~ C(treatment) * C(OOS) * fit_in + age + price + C(store_dum) + C(style_dum) + C(size) + C(year) + C(month) + C(date) + C(color_dum)"
    res_3way, _ = fit_model(df_combined, three_way_formula, cluster_col="id_dum")

    # restrict to OOS==0 and fit treatment×age model
    df_oos0 = df_combined[df_combined["OOS"] == 0].copy()
    res_ta, _ = fit_model(df_oos0, "ln_sales ~ C(treatment) * age + fit_in + price + C(store_dum) + C(style_dum) + C(size) + C(year) + C(month) + C(date) + C(color_dum)", cluster_col="id_dum")

    # 5) build and export a LaTeX table with selected variables and interactions
    models = [res_did, res_3way, res_ta]
    names = ["DID", "3-way", "Treat×Age"]
    vars_of_interest = ["treatment", "OOS", "fit_in", "price", "age", ":"]  # ':' helps catch interactions
    table = build_coef_se_table(models, names, vars_of_interest)
    export_table_to_tex(table, IMAGES / "selected_three_models.tex", caption="Selected coefficients", label="tab:selected_models")

    # print summaries (optional)
    print(res_did.summary())
    print(res_3way.summary())
    print(res_ta.summary())
