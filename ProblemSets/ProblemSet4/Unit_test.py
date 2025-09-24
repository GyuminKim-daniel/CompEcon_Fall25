import pandas as pd
import numpy as np
import statsmodels.api as sm
import pytest
from ProblemSet4_Kim import clean_and_prepare, mle_regression

# Load and clean dataset once for all tests
df = pd.read_stata("PSID_data.dta")
df_clean = clean_and_prepare(df)

def test_selection_criteria():
    """
    Test that all cleaned observations satisfy the selection restrictions.

    Checks
    ------
    - Only male heads of household (hsex == 1).
    - Ages between 25 and 60 inclusive.
    - Hourly wages strictly greater than $7.
    """
    assert (df_clean["hsex"] == 1).all()
    assert df_clean["age"].between(25, 60).all()
    assert (df_clean["wage"] > 7).all()

def test_indicator_variables():
    """
    Test that race dummies are coded correctly with White as the baseline.

    Checks
    ------
    - Each dummy (black, hispanic, otherrace) is binary (0 or 1).
    - No overlapping categories: row sums ∈ {0,1}.
      * Sum == 0 → White (baseline).
      * Sum == 1 → exactly one non-White category.
    """
    for col in ["black", "hispanic", "otherrace"]:
        assert set(df_clean[col].unique()).issubset({0, 1}), f"{col} not coded as 0/1"

    race_sum = df_clean[["black", "hispanic", "otherrace"]].sum(axis=1)
    assert set(race_sum.unique()).issubset({0, 1}), "Race indicators must sum to 0 or 1"

@pytest.mark.parametrize("year", [1971, 1980, 1990, 2000])
def test_mle_matches_ols(year):
    """
    Test that MLE and OLS produce (numerically) equivalent results.

    Steps
    -----
    1. Subset the cleaned data for the given year.
    2. Run OLS using statsmodels.
    3. Run MLE using custom mle_regression.
    4. Compare results:
       - Coefficients: MLE ≈ OLS (within 1e-4).
       - Variance: MLE σ² ≈ OLS SSR/n (within 1e-4).
    """
    df_year = df_clean[df_clean["year"] == year].dropna(
        subset=["ln_wage", "hyrsed", "age", "age2", "black", "hispanic", "otherrace"]
    )
    assert not df_year.empty, f"No data for year {year}"

    # OLS
    y = df_year["ln_wage"].to_numpy()
    X = sm.add_constant(df_year[["hyrsed","age","age2","black","hispanic","otherrace"]].to_numpy())
    ols_res = sm.OLS(y, X).fit()
    beta_ols = ols_res.params
    sigma2_ols_mle = np.sum((y - X @ beta_ols) ** 2) / len(y)  # SSR/n

    # MLE
    beta_mle, sigma_mle, res = mle_regression(df_clean, year, year_col="year", method="L-BFGS-B")
    assert res.success, f"MLE did not converge for year {year}"

    # Compare
    assert np.allclose(beta_mle, beta_ols, atol=1e-4), f"Beta mismatch in {year}"
    assert np.isclose(sigma_mle**2, sigma2_ols_mle, atol=1e-4), f"Variance mismatch in {year}"
