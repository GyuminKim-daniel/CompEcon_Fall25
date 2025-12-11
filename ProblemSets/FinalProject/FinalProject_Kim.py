#!/usr/bin/env python3
"""
two_stage_mixedlogit_scaled_age.py

Final pipeline: aggregate -> clean -> estimate two-stage mixed-logit SML
(price scaled, age covariate).

CHANGES FROM PREVIOUS:
 - Do NOT write these files anymore:
     - cells_with_predictions_scaled_age_full.csv
     - two_stage_mixedlogit_results_scaled_age_full.csv
 - Keep all other outputs: results/ (table + LaTeX), images/ (plots),
   LaTeX figure snippets.
 - Added docstrings/comments for each main step.
 - No logic changed except removing the two specific CSV saves.

USAGE:
 - Edit DATA_PATH at the top if needed.
 - Run: python three_stage_mixedlogit_scaled_age.py
"""

import numpy as np
import pandas as pd
import time
import math
import warnings
import os
from scipy import stats, optimize
from scipy.special import logsumexp, expit
from scipy.stats import qmc
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# -----------------------------
# USER PARAMETERS (edit before running)
# -----------------------------
DATA_PATH = r"C:\Users\Gyumin Kim\Desktop\Coursework\25-2R\Computational Methods for Economists\CompEcon_Fall25\Data\Raw.dta"

# estimation settings
N_DRAWS = 500           # increase for final runs (500 -> 1000+)
HALTON_SCRAMBLE = True
SEED = 1

# sampling for debugging (keep False for full dataset)
DEBUG_SAMPLE = False
SAMPLE_N = 5000

# caching/rounding choices (controls # of unique keys)
ROUND_DECIMALS = 4      # rounding for scaled price cache keys
AGE_ROUND = 2           # rounding for age when caching

MAXITER = 200
HESSIAN_EPS = 1e-4

# price scaling: Won -> ten-thousands of Won
PRICE_SCALE = 10000.0

np.random.seed(SEED)

# -----------------------------
# STEP 1: Load raw data and basic checks
# -----------------------------
"""
Load the raw .dta data file into a pandas DataFrame and check that all
required columns exist in the data. This step does not mutate the data
beyond reading it.
"""
print("Loading data from", DATA_PATH)
df = pd.read_stata(DATA_PATH)
print("Raw rows:", df.shape[0])
print("Columns:", df.columns.tolist())

required = ['price','fit_in','sales_count','stock','store_dum','style_dum','eventday','age']
for c in required:
    if c not in df.columns:
        raise KeyError(f"Required column '{c}' not found in dataset")

# -----------------------------
# STEP 2: Basic cleaning: types & dates
# -----------------------------
"""
Convert numeric-like columns to numeric types and ensure eventday is a datetime.
Rows with invalid eventday are dropped because aggregation requires dates.
"""
for col in ['price','fit_in','sales_count','stock','age']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['eventday'] = pd.to_datetime(df['eventday'], errors='coerce')
df = df.dropna(subset=['eventday']).reset_index(drop=True)
print("After basic cleaning:", df.shape)

# -----------------------------
# STEP 3: Drop negative sales_count BEFORE aggregation
# -----------------------------
"""
Remove any rows where sales_count < 0 prior to aggregation. This follows
your instruction to filter invalid sales before aggregating to cells.
"""
neg_before = (df['sales_count'] < 0).sum()
print("Rows with sales_count < 0 (will be dropped):", neg_before)
df = df[df['sales_count'] >= 0].reset_index(drop=True)
print("After dropping negative sales_count:", df.shape)

# optional debug sampling (DO NOT use for final run)
if DEBUG_SAMPLE:
    df = df.sample(n=min(SAMPLE_N, len(df)), random_state=SEED).reset_index(drop=True)
    print("DEBUG SAMPLE rows:", df.shape)

# -----------------------------
# STEP 4: Aggregate FIRST: store_dum × style_dum × eventday
# -----------------------------
"""
Aggregate observations into 'cells' with keys (store_dum, style_dum, eventday).
We compute:
 - mean price
 - sum fit_in
 - sum sales_count -> saved as 'sales'
 - mean stock
 - mean age

Aggregation occurs BEFORE the post-aggregation filtering you requested.
"""
cells = df.groupby(['store_dum','style_dum','eventday']).agg({
    'price':'mean',
    'fit_in':'sum',
    'sales_count':'sum',
    'stock':'mean',
    'age':'mean'
}).reset_index().rename(columns={'sales_count':'sales'})

print("Cells BEFORE filtering:", cells.shape)

# -----------------------------
# STEP 5: Filter AFTER aggregation: drop invalid price or stock
# -----------------------------
"""
After aggregation, drop cells with missing or nonpositive price or stock.
Do NOT drop cells because of zero fit_in or zero sales — keep zeros.
Fill missing age by the cell mean.
"""
cells = cells.dropna(subset=['price','stock'])
cells = cells[(cells['price'] > 0) & (cells['stock'] > 0)].reset_index(drop=True)

# keep zeros for fit_in and sales
cells['fit_in'] = cells['fit_in'].clip(lower=0)
cells['sales'] = cells['sales'].clip(lower=0)

# Fill missing age with mean (do NOT drop cells because of missing age)
if cells['age'].isna().any():
    cells['age'] = cells['age'].fillna(cells['age'].mean())

print("Cells AFTER filtering:", cells.shape)
print("Share zero fit_in:", (cells['fit_in']==0).mean(), "Share zero sales:", (cells['sales']==0).mean())

# -----------------------------
# STEP 6: Price scaling and arrays for estimation
# -----------------------------
"""
Scale price by PRICE_SCALE (10,000) to improve numerical behavior.
Extract numpy arrays for the estimation routine.
"""
cells['price_scaled'] = cells['price'] / PRICE_SCALE
prices = cells['price_scaled'].astype(float).values
ages = cells['age'].astype(float).values
fits = cells['fit_in'].astype(float).values
sales = cells['sales'].astype(float).values

print("Sample scaled prices (first 8 unique):", np.unique(cells['price_scaled'].values)[:8])
print("Unique scaled prices (rounded):", np.unique(np.round(prices, ROUND_DECIMALS)).shape[0])

# -----------------------------
# STEP 7: Structural two-stage model (professor specification)
# -----------------------------
"""
Define the inclusive-value mappings and purchase probability functions
consistent with the professor feedback:
 - eps_j distributed Type I EV (implicit in the inclusive-value forms)
 - µ (mu_tilde) is the post-try precision parameter
"""
def iv_no_try(delta, mu=1.0):
    """Inclusive value for not trying."""
    return (1.0/mu) * np.log1p(np.exp(mu * delta))

def iv_try(delta, c, mu_tilde=2.0):
    """Inclusive value for trying (includes try cost c and precision mu_tilde)."""
    return -c + (1.0/mu_tilde) * np.log1p(np.exp(mu_tilde * delta))

def prob_buy_no_try(delta, mu=1.0):
    """Purchase probability if not tried (logistic on delta)."""
    return expit(mu * delta)

def prob_buy_try(delta, mu_tilde=2.0):
    """Purchase probability if tried (logistic on mu_tilde * delta)."""
    return expit(mu_tilde * delta)

# -----------------------------
# STEP 8: Mixed-logit π computation with age covariate
# -----------------------------
"""
compute_pi_price_age:
 - Inputs: parameter vector theta, scalar price_scaled, scalar age_val, array z_draws (R,)
 - Returns: π11, π10, π01, π00 averaged over draws
Random price coefficient is log-normal (negative sign ensures negative price effect).
"""
def compute_pi_price_age(theta, price_scaled, age_val, z_draws_local):
    v = theta[0]
    c = theta[1]
    mu_tilde = np.exp(theta[2])
    beta_mean = theta[3]
    beta_sd = np.exp(theta[4])
    gamma_age = theta[5]
    # draws for price coefficient (log-normal negative)
    beta_r = -np.exp(beta_mean + beta_sd * z_draws_local)    # shape (R,)
    delta_r = v + beta_r * price_scaled + gamma_age * age_val
    # inclusive values and probabilities per draw
    ivn_r = np.log1p(np.exp(delta_r))                        # IV_no
    ivt_r = -c + (1.0/mu_tilde) * np.log1p(np.exp(mu_tilde * delta_r))  # IV_try
    pr_try_r = np.exp(ivt_r - logsumexp(np.vstack([ivt_r, ivn_r]), axis=0))
    pr_buy_no_r = expit(delta_r)
    pr_buy_try_r = expit(mu_tilde * delta_r)
    pi11 = float(np.mean(pr_try_r * pr_buy_try_r))
    pi10 = float(np.mean(pr_try_r * (1.0 - pr_buy_try_r)))
    pi01 = float(np.mean((1.0 - pr_try_r) * pr_buy_no_r))
    pi00 = float(1.0 - (pi11 + pi10 + pi01))
    return pi11, pi10, pi01, pi00

# Negative log-likelihood using caching on rounded (price_scaled, age)
def negloglik_age(theta, prices_arr, ages_arr, fits_arr, sales_arr, z_draws_local):
    """
    Compute the negative log-likelihood.
    We cache π results for unique rounded (price, age) keys to avoid recomputation.
    The Poisson arrival structure implies:
      E[fit_in]  = Lambda * (pi11 + pi10)
      E[sales]   = Lambda * (pi11 + pi01)
    """
    keys = set((round(float(prices_arr[i]), ROUND_DECIMALS), round(float(ages_arr[i]), AGE_ROUND)) for i in range(len(prices_arr)))
    cache = {k: compute_pi_price_age(theta, k[0], k[1], z_draws_local) for k in keys}
    ll = 0.0
    Lambda = np.exp(theta[6])
    for i in range(len(prices_arr)):
        key = (round(float(prices_arr[i]), ROUND_DECIMALS), round(float(ages_arr[i]), AGE_ROUND))
        pi11, pi10, pi01, pi00 = cache[key]
        mean_fit = Lambda * (pi10 + pi11)
        mean_sales = Lambda * (pi01 + pi11)
        # avoid numerical zeros
        mean_fit = max(mean_fit, 1e-12)
        mean_sales = max(mean_sales, 1e-12)
        ll += fits_arr[i] * math.log(mean_fit) - mean_fit
        ll += sales_arr[i] * math.log(mean_sales) - mean_sales
    return -ll

# -----------------------------
# STEP 9: Halton draws setup and optimization
# -----------------------------
"""
Generate Halton draws for simulation and run L-BFGS-B to maximize the simulated
log-likelihood (minimize negative log-likelihood).
"""
R = N_DRAWS
sampler = qmc.Halton(d=1, scramble=HALTON_SCRAMBLE, seed=SEED)
draws_u = sampler.random(R)
z_draws = stats.norm.ppf(draws_u).reshape(R,)

# initial guess
theta0 = np.array([1.0, 0.5, np.log(2.0), np.log(0.1), np.log(0.5), 0.0, np.log(5.0)])
print("Initial negative log-likelihood (cached):", negloglik_age(theta0, prices, ages, fits, sales, z_draws))

t0 = time.time()
res = optimize.minimize(lambda th: negloglik_age(th, prices, ages, fits, sales, z_draws),
                        theta0, method='L-BFGS-B', options={'maxiter':MAXITER, 'disp':True})
t1 = time.time()
print("Optimization finished in {:.1f}s; success={} msg={}".format(t1-t0, res.success, res.message))
theta_hat = res.x
print("theta_hat:", theta_hat)

# -----------------------------
# STEP 10: Numerical Hessian (optional; may be singular)
# -----------------------------
"""
Attempt to compute numerical Hessian for standard errors. If Hessian is singular
or inversion fails, std errors are set to NaN.
"""
def numerical_hessian(fun, x0, eps=HESSIAN_EPS):
    x0 = np.asarray(x0, dtype=float)
    n = x0.size
    h = np.zeros((n,n), dtype=float)
    f0 = fun(x0)
    for i in range(n):
        xip = x0.copy(); xip[i] += eps; fip = fun(xip)
        xim = x0.copy(); xim[i] -= eps; fim = fun(xim)
        h[i,i] = (fip - 2.0*f0 + fim) / (eps*eps)
        for j in range(i+1, n):
            xijp = x0.copy(); xijp[i] += eps; xijp[j] += eps; fijp = fun(xijp)
            xijm = x0.copy(); xijm[i] += eps; xijm[j] -= eps; fijm = fun(xijm)
            xjim = x0.copy(); xjim[i] -= eps; xjim[j] += eps; fjim = fun(xjim)
            xjmm = x0.copy(); xjmm[i] -= eps; xjmm[j] -= eps; fjmm = fun(xjmm)
            h[i,j] = (fijp - fijm - fjim + fjmm) / (4.0*eps*eps)
            h[j,i] = h[i,j]
    return h

try:
    print("Computing numerical Hessian (may be slow)...")
    fun = lambda th: negloglik_age(th, prices, ages, fits, sales, z_draws)
    hess = numerical_hessian(fun, theta_hat, eps=HESSIAN_EPS)
    cov = np.linalg.inv(hess)
    se = np.sqrt(np.abs(np.diag(cov)))
except Exception as e:
    print("Hessian failed or singular:", e)
    se = np.full_like(theta_hat, np.nan)

# -----------------------------
# STEP 11: Report (in-memory) & do NOT save the two banned files
# -----------------------------
"""
Create res_table (DataFrame) with parameter estimates and standard errors.
We DO NOT write the previous two specific files:
 - cells_with_predictions_scaled_age_full.csv (removed)
 - two_stage_mixedlogit_results_scaled_age_full.csv (removed)

We will still write a results table under results/two_stage_results_table.csv and
other LaTeX/plots as requested.
"""
param_names = ['v_intercept','c_trycost','log_mu_tilde','beta_price_mean','log_beta_price_sd','gamma_age','log_Lambda']

res_table = pd.DataFrame({
    'param': param_names,
    'estimate': theta_hat,
    'std_err': se
})
# handle NaNs in se before computing z/p-values
def safe_div(a, b):
    try:
        if np.isnan(b) or b == 0:
            return np.nan
        return a / b
    except:
        return np.nan

res_table['z'] = [ safe_div(r['estimate'], r['std_err']) for _, r in res_table.iterrows() ]
res_table['pval'] = res_table['z'].apply(lambda z: 2 * (1 - stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan)

print("\n=== Estimation results ===")
print(res_table.to_string(index=False, float_format='%.6f'))

v_hat = theta_hat[0]; c_hat = theta_hat[1]; mu_tilde_hat = np.exp(theta_hat[2])
beta_mean_hat = theta_hat[3]; beta_sd_hat = np.exp(theta_hat[4]); gamma_age_hat = theta_hat[5]
Lambda_hat = np.exp(theta_hat[6])

print("\nInterpretable:")
print(f"v = {v_hat:.4f}")
print(f"c = {c_hat:.4f}")
print(f"mu_tilde = {mu_tilde_hat:.4f}")
print(f"price mean coef (log-scale) = {beta_mean_hat:.4f}")
print(f"price sd (exp) = {beta_sd_hat:.4f}")
print(f"gamma_age = {gamma_age_hat:.4f}")
print(f"Lambda = {Lambda_hat:.4f}")

# -----------------------------
# STEP 12: Predictions & diagnostics (in-memory)
# -----------------------------
"""
Compute predicted probabilities for each cell using the estimated parameters.
We DO compute predictions and use them for diagnostics, but we DO NOT save the
full 'cells_with_predictions_scaled_age_full.csv' file as per your instruction.
"""
keys = set((round(float(prices[i]), ROUND_DECIMALS), round(float(ages[i]), AGE_ROUND)) for i in range(len(prices)))
cache_hat = {k: compute_pi_price_age(theta_hat, k[0], k[1], z_draws) for k in keys}

def predict_cell_row(price_scaled, age_val):
    key = (round(float(price_scaled), ROUND_DECIMALS), round(float(age_val), AGE_ROUND))
    pi11, pi10, pi01, pi00 = cache_hat[key]
    pr_try = pi10 + pi11
    pr_buy = pi01 + pi11
    pr_buy_given_try = pi11 / (pi10 + pi11) if (pi10 + pi11) > 1e-12 else 0.0
    return pr_try, pr_buy, pr_buy_given_try

cells['pred_pr_try'], cells['pred_pr_buy'], cells['pred_buy_given_try'] = zip(*cells.apply(lambda r: predict_cell_row(r['price_scaled'], r['age']), axis=1))
# Note: we DO NOT save cells.to_csv(...) here (removed by request).

# Diagnostics by price bins
unique_prices_sorted = np.sort(cells['price_scaled'].unique())
if len(unique_prices_sorted) > 1:
    midpoints = (unique_prices_sorted[:-1] + unique_prices_sorted[1:]) / 2
    bins = np.concatenate(([-np.inf], midpoints, [np.inf]))
else:
    bins = [-np.inf, np.inf]

cells['price_bin'] = pd.cut(cells['price_scaled'], bins=bins)
agg = cells.groupby('price_bin').agg({'fit_in':'mean','sales':'mean','pred_pr_try':'mean','pred_pr_buy':'mean'}).reset_index()

print("\nBinned diagnostics (by scaled price):")
print(agg)

plt.figure(figsize=(8,5))
plt.plot(agg['pred_pr_try'], marker='o', label='pred_pr_try')
plt.plot(agg['fit_in'], marker='x', label='avg fit_in (mean)')
plt.legend(); plt.title('Binned by scaled price: predicted try vs observed mean fit_in')
plt.show()

plt.figure(figsize=(8,5))
plt.plot(agg['pred_pr_buy'], marker='o', label='pred_pr_buy')
plt.plot(agg['sales'], marker='x', label='avg sales (mean)')
plt.legend(); plt.title('Binned by scaled price: predicted buy vs observed mean sales')
plt.show()

print("Predictions computed (not saved to the two banned CSV files).")

# ============================================================
# STEP 13: CREATE FOLDERS IF NOT EXIST (results/ and images/)
# ============================================================
"""
Create the results/ and images/ directories when they are missing.
We will save the results table to results/ and plots to images/.
"""
RESULT_DIR = "results"
IMAGE_DIR  = "images"

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

print(f"Ensured folders: '{RESULT_DIR}/' and '{IMAGE_DIR}/'.")

# ============================================================
# STEP 14: SAVE RESULTS TABLES (kept)
# ============================================================
"""
Save a single, consolidated results table in results/.
This is NOT the banned two_stage_mixedlogit_results_scaled_age_full.csv file;
the file is named two_stage_results_table.csv per your earlier request.
"""
results_csv_path = os.path.join(RESULT_DIR, "two_stage_results_table.csv")
res_table.to_csv(results_csv_path, index=False)
print("Saved results CSV:", results_csv_path)

# Save results table as TXT (human-readable)
results_txt_path = os.path.join(RESULT_DIR, "two_stage_results_table.txt")
with open(results_txt_path, "w", encoding="utf-8") as f:
    f.write(res_table.to_string(index=False))
print("Saved results TXT:", results_txt_path)

# ============================================================
# STEP 15: SAVE PLOTS TO IMAGE FOLDER
# ============================================================
"""
Create and save two diagnostic plots into images/:
 - try_probability_plot.png
 - buy_probability_plot.png
These images are LaTeX-ready and used by your earlier LaTeX templates.
"""
# --- Try Probability Plot ---
plt.figure(figsize=(8,5))
plt.plot(agg['pred_pr_try'], marker='o', label='Predicted P(try)')
plt.plot(agg['fit_in'], marker='x', label='Observed mean fit_in')
plt.title("Try Probability vs Observed fit_in")
plt.xlabel("Price Bins")
plt.ylabel("Probability / Mean fit_in")
plt.legend()
plot_try_path = os.path.join(IMAGE_DIR, "try_probability_plot.png")
plt.savefig(plot_try_path, dpi=300, bbox_inches='tight')
print("Saved try plot:", plot_try_path)
plt.close()

# --- Buy Probability Plot ---
plt.figure(figsize=(8,5))
plt.plot(agg['pred_pr_buy'], marker='o', label='Predicted P(buy)')
plt.plot(agg['sales'], marker='x', label='Observed mean sales')
plt.title("Buy Probability vs Observed Sales")
plt.xlabel("Price Bins")
plt.ylabel("Probability / Mean Sales")
plt.legend()
plot_buy_path = os.path.join(IMAGE_DIR, "buy_probability_plot.png")
plt.savefig(plot_buy_path, dpi=300, bbox_inches='tight')
print("Saved buy plot:", plot_buy_path)
plt.close()

print("\nAll outputs saved successfully (except the two files you asked to omit).")

# ============================================================
# STEP 16: SAVE LATEX TABLE (MANUAL WRITER WITH ESCAPED UNDERSCORES)
# ============================================================

latex_path = os.path.join(RESULT_DIR, "two_stage_results_table.tex")
os.makedirs(RESULT_DIR, exist_ok=True)

with open(latex_path, "w", encoding="utf-8") as f:
    f.write("\\begin{table}[ht!]\n")
    f.write("\\centering\n")
    f.write("\\begin{tabular}{lcccc}\n")
    f.write("\\hline\n")
    f.write("param & estimate & std\\_err & z & pval \\\\\n")
    f.write("\\hline\n")

    for _, row in res_table.iterrows():
        param_name = row['param'].replace('_', '\\_')  # <--- FIX
        est  = f"{row['estimate']:.6f}"
        se   = f"{row['std_err']:.6f}" if not pd.isna(row['std_err']) else "nan"
        zval = f"{row['z']:.6f}"        if not pd.isna(row['z'])        else "nan"
        pval = f"{row['pval']:.6f}"     if not pd.isna(row['pval'])     else "nan"

        f.write(f"{param_name} & {est} & {se} & {zval} & {pval} \\\\\n")

    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.write("\\caption{Parameter Estimates from Two-Stage Mixed Logit Model}\n")
    f.write("\\label{tab:two_stage_results}\n")
    f.write("\\end{table}\n")

print("Saved LaTeX table:", latex_path)

# ============================================================
# STEP 17: SAVE LATEX FIGURE SNIPPETS
# ============================================================
"""
Save LaTeX-ready figure environments that include the saved PNGs.
These snippets are stored in results/ and can be pasted into your main .tex file.
"""
latex_try_fig = r"""
\begin{figure}[ht!]
    \centering
    \includegraphics[width=0.85\textwidth]{images/try_probability_plot.png}
    \caption{Predicted vs Observed Try Probability by Price Bin}
    \label{fig:try_probability}
\end{figure}
"""

latex_buy_fig = r"""
\begin{figure}[ht!]
    \centering
    \includegraphics[width=0.85\textwidth]{images/buy_probability_plot.png}
    \caption{Predicted vs Observed Purchase Probability by Price Bin}
    \label{fig:buy_probability}
\end{figure}
"""

try_fig_path = os.path.join(RESULT_DIR, "try_probability_plot.tex")
buy_fig_path = os.path.join(RESULT_DIR, "buy_probability_plot.tex")

with open(try_fig_path, "w", encoding="utf-8") as f:
    f.write(latex_try_fig)
with open(buy_fig_path, "w", encoding="utf-8") as f:
    f.write(latex_buy_fig)

print("Saved LaTeX figure snippet:", try_fig_path)
print("Saved LaTeX figure snippet:", buy_fig_path)

# ================= END OF SCRIPT ================
