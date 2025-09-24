import pandas as pd
import numpy as np
from scipy.optimize import minimize

def clean_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataset and construct regression variables 
    for estimating the log-wage model.

    Steps
    -----
    1. Compute hourly wage:
       wage = hlabinc / hannhrs
       (drop observations with zero or missing hours).
    2. Apply sample restrictions:
       - male heads of household (hsex == 1),
       - ages 25–60 inclusive,
       - hourly wage > 7.
    3. Create regression variables:
       - ln_wage   : natural log of hourly wage
       - hyrsed    : years of education (already in df)
       - age2      : squared age
       - black     : 1 if hrace == 2, else 0
       - hispanic  : 1 if hrace == 5, else 0
       - otherrace : 1 if hrace in {3,4,6,7}, else 0
         (White, hrace == 1, is the omitted baseline)

    Parameters
    ----------
    df : pd.DataFrame
        Raw PSID dataset containing at least:
        hlabinc, hannhrs, hsex, age, hyrsed, hrace.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset restricted to the analysis sample, 
        with additional variables needed for estimation.
    """
    # Drop invalid or zero hours worked
    df = df[(df["hannhrs"] > 0)].copy()

    # Hourly wage and restrictions
    df["wage"] = df["hlabinc"] / df["hannhrs"]
    df = df[(df["hsex"] == 1) & df["age"].between(25, 60) & (df["wage"] > 7)]

    # Regression variables
    df["ln_wage"] = np.log(df["wage"])
    df["age2"] = df["age"] ** 2
    df["black"] = (df["hrace"] == 2).astype(int)
    df["hispanic"] = (df["hrace"] == 5).astype(int)
    df["otherrace"] = df["hrace"].isin([3, 4, 6, 7]).astype(int)

    return df


def mle_regression(
    df: pd.DataFrame, year: int, year_col: str = "year", method: str = "L-BFGS-B"
):
    """
    Estimate the log-wage regression via Maximum Likelihood Estimation (MLE)
    for a single survey year.

    Model
    -----
    ln(wage) = β0 + β1*hyrsed + β2*age + β3*age^2
               + β4*black + β5*hispanic + β6*otherrace + ε

    where ε ~ N(0, σ²). With normal errors, MLE is equivalent to OLS.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset (from clean_and_prepare).
    year : int
        Survey year to estimate the model for.
    year_col : str, optional
        Column identifying the survey year (default "year").
    method : str, optional
        Optimization method passed to scipy.optimize.minimize
        (default "L-BFGS-B").

    Returns
    -------
    beta_hat : np.ndarray
        Estimated coefficients in order:
        (const, hyrsed, age, age2, black, hispanic, otherrace).
    sigma_hat : float
        Estimated standard deviation of residuals.
    result : OptimizeResult
        Full optimization output from scipy.optimize.minimize.
    """
    # Subset for year and drop missing values
    df_year = df[df[year_col] == year].dropna(
        subset=["ln_wage","hyrsed","age","age2","black","hispanic","otherrace"]
    )
    if df_year.empty:
        raise ValueError(f"No data found for year {year}")

    y = df_year["ln_wage"].to_numpy()
    X = df_year[["hyrsed","age","age2","black","hispanic","otherrace"]].to_numpy()
    X = np.column_stack([np.ones(len(X)), X])  # add intercept
    n = len(y)

    # Log-likelihood (negative for minimization)
    def loglike(params):
        beta, sigma = params[:-1], params[-1]
        if sigma <= 0:
            return 1e10  # enforce positivity
        resid = y - X @ beta
        ll = -0.5 * n * np.log(2*np.pi*sigma**2) - (resid @ resid)/(2*sigma**2)
        return -ll

    # Initial guess from OLS
    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    sigma_ols = np.std(y - X @ beta_ols)
    init = np.append(beta_ols, sigma_ols)

    # Optimize
    bounds = [(None, None)] * X.shape[1] + [(1e-6, None)]
    result = minimize(loglike, init, method=method, bounds=bounds)

    return result.x[:-1], result.x[-1], result


if __name__ == "__main__":
    # Load raw data
    df = pd.read_stata("PSID_data.dta")
    df_clean = clean_and_prepare(df)

    # Estimate across survey years
    for yr in [1971, 1980, 1990, 2000]:
        betas, sigma, res = mle_regression(df_clean, yr)
        print(f"\n=== Results for {yr} ===")
        print("Betas (const, hyrsed, age, age2, black, hispanic, otherrace):")
        print(betas.round(4))
        print("Sigma:", round(sigma, 4))
        print("Converged:", res.success)
