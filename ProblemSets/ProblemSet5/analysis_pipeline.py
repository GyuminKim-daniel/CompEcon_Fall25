# analysis_pipeline.py
"""
Analysis pipeline for ProblemSet5.

Provides:
 - generate_stock_variables(): create in-stock / out-of-stock / treatment flags
 - concat_variants(): combine multiple variants and deduplicate
 - save_barplot(), save_line_trend_by_age(): plotting helpers (saved to images/)
 - fit_model(): fit OLS with cluster-robust SEs (uniform interface)
 - extract_params(), build_coef_se_table(), export_table_to_tex(): build/export selected results

Usage:
    python analysis_pipeline.py
"""
from pathlib import Path
from typing import Any, Iterable, List, Tuple, Dict

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


# -------------------------
# Paths / configuration
# -------------------------
try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()

# Default data path: two levels up then Data/Raw.dta (adjust if your repo differs)
DATA = ROOT.parents[1] / "Data" / "Raw.dta"
IMAGES = ROOT / "images"
IMAGES.mkdir(exist_ok=True)


# -------------------------
# Data functions
# -------------------------
def generate_stock_variables(
    df: pd.DataFrame,
    stockout_size: Any = 78,
    treat_size: Any = 80,
    control_size: Any = 82,
    min_days: int = 18,
) -> pd.DataFrame:
    """
    Create stock/out-of-stock and treatment indicators.

    Parameters
    ----------
    df
        Input DataFrame. Required columns: 'stock','size','eventday','storecode','stylecode','color'.
    stockout_size, treat_size, control_size
        Size codes used to define trigger (stockout), treatment, and control products.
    min_days
        Minimum number of in-stock or out-stock days required to keep OOS (default 18).

    Returns
    -------
    pd.DataFrame
        Copy of df augmented with:
        stockoutproduct, treatmentproduct, controlproduct,
        s_sum, t_sum, c_sum, OOS, in_stock_day, out_stock_day, treat_sum, treatment.
    """
    required = {"stock", "size", "eventday", "storecode", "stylecode", "color"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    d = df.copy()

    # Basic indicators
    d["stockoutproduct"] = ((d["stock"] == 0) & (d["size"] == stockout_size)).astype(int)
    d["treatmentproduct"] = ((d["stock"] == 0) & (d["size"] == treat_size)).astype(int)
    d["controlproduct"] = ((d["stock"] == 0) & (d["size"] == control_size)).astype(int)

    # Sums by event-day group (eventday, storecode, stylecode, color)
    g_day = ["eventday", "storecode", "stylecode", "color"]
    d["s_sum"] = d.groupby(g_day)["stockoutproduct"].transform("sum")
    d["t_sum"] = d.groupby(g_day)["treatmentproduct"].transform("sum")
    d["c_sum"] = d.groupby(g_day)["controlproduct"].transform("sum")

    # OOS logic: 0 if all in-stock, 1 if trigger (stockout_size) out and others not out
    d["OOS"] = np.nan
    d.loc[(d["s_sum"] + d["t_sum"] + d["c_sum"]) == 0, "OOS"] = 0
    d.loc[(d["s_sum"] == 1) & (d["t_sum"] == 0) & (d["c_sum"] == 0), "OOS"] = 1

    # Counts by (storecode, stylecode, size, color)
    g_size = ["storecode", "stylecode", "size", "color"]
    d["in_stock_day"] = d.groupby(g_size)["OOS"].transform(lambda s: s.eq(0).sum())
    d["out_stock_day"] = d.groupby(g_size)["OOS"].transform(lambda s: s.eq(1).sum())

    # Apply min_days rule: not enough days -> mark OOS missing
    d.loc[d["in_stock_day"] < min_days, "OOS"] = np.nan
    d.loc[d["out_stock_day"] < min_days, "OOS"] = np.nan

    # Treatment assignment by (storecode, stylecode, color)
    g_group = ["storecode", "stylecode", "color"]
    d["treat_sum"] = d.groupby(g_group)["stockoutproduct"].transform("sum")
    d["treatment"] = np.nan
    d.loc[(d["treat_sum"] >= 1) & (d["size"] == treat_size), "treatment"] = 1
    d.loc[(d["treat_sum"] >= 1) & (d["size"] == control_size), "treatment"] = 0

    return d


def concat_variants(dfs: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """
    Vertically concatenate many DataFrame variants; drop exact duplicates.

    Returns
    -------
    pd.DataFrame
    """
    combined = pd.concat(list(dfs), ignore_index=True)
    return combined.drop_duplicates().reset_index(drop=True)


# -------------------------
# Plot helpers (save to images/)
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
    """
    Save a seaborn barplot to disk.
    """
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
    """
    Save a trend line of mean(y_col) by age and group_col to disk.
    """
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
# Modeling utilities
# -------------------------
def fit_model(df: pd.DataFrame, formula: str, cluster_col: str = "id_dum"):
    """
    Fit OLS (statsmodels) and return (result, df_model). Uses cluster-robust SEs.

    The function auto-detects a common set of columns and drops rows with NA in those columns.
    """
    # common variables often used in our formulas
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

    # convert categorical columns to category dtype on model copy
    cats = ["treatment", "OOS", "store_dum", "style_dum", "size", "year", "month", "date", "color_dum"]
    for c in cats:
        if c in df_model.columns:
            df_model[c] = df_model[c].astype("category")

    model = smf.ols(formula=formula, data=df_model)
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df_model[cluster_col]})
    return result, df_model


def extract_params(result, substrings: List[str]) -> pd.DataFrame:
    """
    Return DataFrame with coef, se, t, p for params whose names contain any substring (case-insensitive).
    """
    names = [n for n in result.params.index if any(s.lower() in n.lower() for s in substrings)]
    return pd.DataFrame(
        {
            "coef": result.params.loc[names],
            "se": result.bse.loc[names],
            "t": result.tvalues.loc[names],
            "p": result.pvalues.loc[names],
        }
    )


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
    Build a DataFrame table of selected coefficients (with significance stars) for multiple models.

    Rows: param name, param (se), repeated for each param, then Observations and R-squared.
    """
    selected = []
    for res in results:
        for name in res.params.index:
            if any(s.lower() in name.lower() for s in var_substrings) and name not in selected:
                selected.append(name)
    selected = sorted(selected)

    rows = []
    idx = []
    for pname in selected:
        coef_row, se_row = [], []
        for res in results:
            if pname in res.params.index:
                pval = res.pvalues[pname]
                coef_row.append(f"{res.params[pname]:.3f}{_stars(pval)}")
                se_row.append(f"({res.bse[pname]:.3f})")
            else:
                coef_row.append("")
                se_row.append("")
        rows.append(coef_row); idx.append(pname)
        rows.append(se_row); idx.append(pname + " (se)")

    table = pd.DataFrame(rows, index=idx, columns=model_names)
    nobs_row = [int(getattr(r, "nobs", np.nan)) for r in results]
    r2_row = [f"{getattr(r, 'rsquared', np.nan):.3f}" for r in results]
    table.loc["Observations"] = nobs_row
    table.loc["R-squared"] = r2_row
    return table


def export_table_to_tex(table: pd.DataFrame, outpath: Path, caption: str = "Regression results", label: str = "tab:results"):
    """
    Save a DataFrame as a LaTeX table inside a table environment at `outpath`.
    """
    outpath.parent.mkdir(parents=True, exist_ok=True)
    latex_tab = table.to_latex(escape=True, na_rep="", column_format="l" + "r" * (table.shape[1]))
    latex = f"\\begin{{table}}[htbp]\n\\centering\n{latex_tab}\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}\n"
    outpath.write_text(latex, encoding="utf-8")
    print(f"Saved LaTeX table to {outpath}")


# -------------------------
# Main pipeline
# -------------------------
def _run_pipeline():
    """Minimal runnable pipeline used when script is called directly."""
    if not DATA.exists():
        raise FileNotFoundError(f"Data file not found at expected path: {DATA}")

    df1 = pd.read_stata(DATA)

    combos = [
        (78, 80, 82),
        (80, 82, 86),
        (74, 76, 78),
        (76, 78, 80),
        (82, 86, 90),
    ]
    dfs = [generate_stock_variables(df1, *c, min_days=18) for c in combos]
    df_combined = concat_variants(dfs)

    # Plots
    save_barplot(df_combined, x="OOS", y="ln_sales", hue="treatment", outpath=IMAGES / "LogSales.png", title="Log Sales by OOS and Treatment")
    save_barplot(df_combined, x="OOS", y="fit_in", hue="treatment", outpath=IMAGES / "Fittingroomvisits.png", title="Fitting Room Visits by OOS and Treatment")
    save_line_trend_by_age(df_combined[df_combined["OOS"] == 0], age_col="age", y_col="ln_sales", group_col="treatment", outpath=IMAGES / "SalesTrend.png", descending=True)

    # Models
    did_formula = "ln_sales ~ C(treatment) * C(OOS) + fit_in + age + price + C(store_dum) + C(style_dum) + C(size) + C(year) + C(month) + C(date) + C(color_dum)"
    res_did, _ = fit_model(df_combined, did_formula, cluster_col="id_dum")

    three_way_formula = "ln_sales ~ C(treatment) * C(OOS) * fit_in + age + price + C(store_dum) + C(style_dum) + C(size) + C(year) + C(month) + C(date) + C(color_dum)"
    res_3way, _ = fit_model(df_combined, three_way_formula, cluster_col="id_dum")

    df_oos0 = df_combined[df_combined["OOS"] == 0].copy()
    res_ta, _ = fit_model(df_oos0, "ln_sales ~ C(treatment) * age + fit_in + price + C(store_dum) + C(style_dum) + C(size) + C(year) + C(month) + C(date) + C(color_dum)", cluster_col="id_dum")

    # Export selected coefficients to LaTeX
    models = [res_did, res_3way, res_ta]
    names = ["DID", "3-way", "Treat×Age"]
    vars_of_interest = ["treatment", "OOS", "fit_in", "price", "age", ":"]  # ':' catches interactions
    table = build_coef_se_table(models, names, vars_of_interest)
    export_table_to_tex(table, IMAGES / "selected_three_models.tex", caption="Selected coefficients", label="tab:selected_models")

    # Print brief summary
    print("Saved images and exported LaTeX table.")


if __name__ == "__main__":
    _run_pipeline()
