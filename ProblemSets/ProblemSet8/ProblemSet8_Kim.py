"""
ProblemSet8_Kim.py

Modular implementation template for Problem Set 8 (Cooper & Ejarque replication).
Follows the SMM algorithm described in the problem set (solve DP -> simulate -> compute moments
-> minimize distance between simulated and target moments -> compute efficient weighting and SEs).

Dependencies
------------
numpy, scipy, numba (optional), pandas, tqdm (optional)

Usage
-----
Fill in 'TARGET_MOMENTS' with the Table 3 moments from the paper and calibrate constants.
Then run: estimate_smm(...)
"""

# --- Imports ---
import time
from typing import Dict, Any

import numpy as np
from scipy import optimize
from tqdm import trange

# FIX 1: missing pandas import
import pandas as pd

try:
    from quantecon import tauchen as qe_tauchen
except Exception:
    qe_tauchen = None

from pathlib import Path

# -------------------------
# Relative Paths / Folders
# -------------------------
try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)



# ============================================================
# Tauchen Wrapper
# ============================================================
def tauchen_QE(n: int, mu: float, rho: float, sigma: float, m: float = 3.0, return_logs: bool = False):
    """
    Discretizes the AR(1) productivity process using QuantEcon's Tauchen method.

    This implements:
        log(A_{t+1}) = mu + rho * log(A_t) + epsilon_t,
    where epsilon_t ~ N(0, sigma^2).

    Parameters
    ----------
    n : int
        Number of grid points.
    mu : float
        Mean of log productivity.
    rho : float
        AR(1) persistence parameter.
    sigma : float
        Shock standard deviation.
    m : float
        Tauchen grid width parameter.
    return_logs : bool
        If True, returns log-productivity grid; otherwise returns exp(logA).

    Notes
    -----
    - Keeps your original code logic (including rhoW typo).
    - Also normalizes transition matrix to avoid numerical negativity.
    """
    if qe_tauchen is None:
        raise ImportError("quantecon.tauchen is not available.")

    try:
        qe_res = qe_tauchen(n=n, mu=mu, rho=rho, sigma=sigma)
    except TypeError:
        # intentionally keeping your rhoW typo as requested
        qe_res = qe_tauchen(rho=rhoW, sigma=sigma, n=n, m=m)

    z_log = np.asarray(qe_res.state_values)
    P = np.asarray(qe_res.P)

    # ensure nonnegative
    P = np.where(P < 0, 0.0, P)
    P = P / P.sum(axis=1, keepdims=True)

    if return_logs:
        return z_log, P
    return np.exp(z_log), P



# ============================================================
# Dynamic Programming Solver
# ============================================================
def solve_dp(params: Dict[str, float],
             grid_k: np.ndarray,
             grid_i: np.ndarray,
             z_grid: np.ndarray,
             P_z: np.ndarray,
             tol: float = 1e-6,
             maxiter: int = 2000) -> Dict[str, Any]:
    """
    Solves the firm's dynamic programming problem with costly external finance.

    The Bellman equation is:
        V(k,z) = max_i { π(k,z,i) + β E[V(k',z')] }

    Profit function:
        π = Ak^α - i - φ₀ b - ½φ₁ b² - ½γ (i/k - δ)^2 k

    where:
        b = max(0, i - Ak^α)

    Parameters
    ----------
    params : dict
        Economic parameters.
    grid_k : ndarray
        Capital grid.
    grid_i : ndarray
        Investment grid.
    z_grid : ndarray
        Productivity grid (levels).
    P_z : ndarray
        Productivity transition matrix.
    tol : float
        VFI convergence tolerance.
    maxiter : int
        VFI iteration limit.

    Returns
    -------
    dict with:
        V : value function
        i_policy : optimal investment
        i_policy_idx : policy indices
    """

    grid_k = np.asarray(grid_k)
    grid_i = np.asarray(grid_i)
    z_grid = np.asarray(z_grid)
    P_z = np.asarray(P_z)

    beta = params.get('beta', 0.96)
    delta = params.get('delta', 0.10)
    alpha = params.get('alpha', 0.6956)
    gamma = params.get('gamma', 0.1331)
    phi0 = params.get('phi0', 0.0)
    phi1 = params.get('phi1', 0.0)

    nk = len(grid_k)
    nz = len(z_grid)
    ni = len(grid_i)

    V = np.zeros((nk, nz))
    i_policy_idx = np.zeros((nk, nz), dtype=int)

    KK = grid_k[:, None, None]
    ZZ = z_grid[None, :, None]
    II = grid_i[None, None, :]

    CF = ZZ * (KK ** alpha)
    external_funds = np.maximum(0.0, II - CF)
    finance_cost = phi0 * external_funds + 0.5 * phi1 * (external_funds ** 2)
    adj_cost = 0.5 * gamma * ((II / (KK + 1e-12)) - delta)**2 * KK
    profit = CF - II - finance_cost - adj_cost

    kprime = (1 - delta) * KK + II
    kp_idx = np.clip(np.searchsorted(grid_k, kprime.ravel()), 0, nk - 1)
    kp_idx = kp_idx.reshape(kprime.shape)

    for _ in range(maxiter):
        EV = V @ P_z.T
        EV_expanded = np.repeat(EV[:, :, None], ni, axis=2)
        EV_kprime = np.take_along_axis(EV_expanded, kp_idx, axis=0)
        RHS = profit + beta * EV_kprime

        V_new = np.max(RHS, axis=2)
        i_policy_idx = np.argmax(RHS, axis=2)

        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break

        V = V_new

    i_policy = grid_i[i_policy_idx]
    return {"V": V, "i_policy": i_policy, "i_policy_idx": i_policy_idx}


# === Simulation ===
def simulate_panel(policy, params, grid_k, z_grid, P_z, S=500, T=100, seed=12345):
   
    """
   Simulates a panel of firms (S x T) using the optimal policy from solve_dp().
   This produces artificial data needed for SMM:
        • Investment I
        • Tobin's Q
        • Cash flow CF
        • Capital stock paths K
        • Productivity indices z_idx

    Evolution equations:
        k_{t+1} = (1 - δ) k_t + i_t
        CF_t = A_t k_t^α - i_t - financing_cost - adjustment_cost

    Tobin's Q is approximated numerically from the value function:
        Q ≈ V(k+Δ) - V(k)

    All logic preserved exactly as in your submitted code.

    """


    rng = np.random.default_rng(seed)

    nk = len(grid_k)
    nz = len(z_grid)

    I = np.zeros((S, T))
    Q = np.zeros((S, T))
    CF = np.zeros((S, T))
    K = np.zeros((S, T+1))
    z_idx = np.zeros((S, T), int)

    K[:, 0] = np.random.choice(grid_k, S)
    z0 = rng.integers(0, nz, S)
    z_idx[:, 0] = z0

    alpha = params.get('alpha', 0.6956)
    delta = params.get('delta', 0.10)
    phi0  = params.get('phi0', 0.0)
    phi1  = params.get('phi1', 0.0)
    gamma = params.get('gamma', 0.1331)

    V = policy["V"]

    for s in range(S):
        z_curr = int(z0[s])
        for t in range(T):

            k_val = K[s, t]
            k_idx = min(np.searchsorted(grid_k, k_val), nk-1)

            i_choice = policy["i_policy"][k_idx, z_curr]
            I[s, t] = i_choice

            output = z_grid[z_curr] * (k_val ** alpha)
            CF_int = output

            ext_funds = max(0, i_choice - CF_int)
            finance_cost = phi0*ext_funds + 0.5*phi1*(ext_funds**2)
            adj_cost = 0.5 * gamma * ((i_choice/(k_val+1e-12)) - delta)**2 * k_val

            CF[s, t] = output - i_choice - finance_cost - adj_cost

            if k_idx == 0:
                Q[s, t] = V[1, z_curr] - V[0, z_curr]
            elif k_idx == nk - 1:
                Q[s, t] = V[k_idx, z_curr] - V[k_idx - 1, z_curr]
            else:
                Q[s, t] = 0.5*(V[k_idx+1, z_curr] - V[k_idx-1, z_curr])

            K[s, t+1] = np.clip((1-delta)*k_val + i_choice, grid_k[0], grid_k[-1])

            z_curr = int(rng.choice(np.arange(nz), p=P_z[z_curr]))
            if t < T-1:
                z_idx[s, t+1] = z_curr

    return {"I": I, "Q": Q, "CF": CF, "K": K, "z_idx": z_idx}

# === Compute Moments ===
def compute_moments(sim):
    """
    Computes the five simulated moments required for SMM:

        1. mean(I/K)
        2. std(I/K)
        3. corr(Q, I/K)
        4. corr(CF, I/K)
        5. autocorr(I/K)

    These match the structure of Table 3(b) moments in Cooper & Ejarque (2003).

    Notes
    -----
    • Uses safe correlation function to avoid NaN cases.
    • Follows your original computation exactly.
    """

    I, Q, CF, K = sim["I"], sim["Q"], sim["CF"], sim["K"][:, :-1]
    invest = I / (K + 1e-8)

    def safe_corr(a, b):
        if np.std(a)==0 or np.std(b)==0:
            return 0
        return np.corrcoef(a.ravel(), b.ravel())[0, 1]

    m1 = np.mean(invest)
    m2 = np.std(invest)
    m3 = safe_corr(Q, invest)
    m4 = safe_corr(CF, invest)
    m5 = safe_corr(invest[:, :-1], invest[:, 1:])

    return np.array([m1, m2, m3, m4, m5])

# === SMM objective ===
def smm_objective(theta, target, W, param_template,
                  grid_k, grid_i, nz, mu_log, S, T, seed):
    """
    Computes SMM distance:

        J(θ) = (m_data - m_sim(θ))' W (m_data - m_sim(θ))

    Steps:
        1. Fill parameter dict with θ
        2. Tauchen discretization of productivity
        3. Solve DP
        4. Simulate panel
        5. Compute simulated moments
        6. Return weighted squared distance

    No logic modified — only docstring added.
    """
  
    params = param_template.copy()
    est_names = [k for k,v in param_template.items() if v is None]
    for i, nm in enumerate(est_names):
        params[nm] = float(theta[i])

    rho = params["rho"]
    sigma = params["sigma"]

    z_grid, P_z = tauchen_QE(nz, mu_log, rho, sigma)

    policy = solve_dp(params, grid_k, grid_i, z_grid, P_z)
    sim = simulate_panel(policy, params, grid_k, z_grid, P_z, S=S, T=T, seed=seed)
    sim_moms = compute_moments(sim)

    diff = target - sim_moms
    return float(diff.T @ W @ diff)

# === Estimation ===
def estimate_smm(initial_theta, target, param_template,
                 grid_k, grid_i, z_grid, P_z, S, T,
                 W=None, seed=123, nz=None, mu_log=None):
    """
    Runs SMM estimation using Nelder–Mead.

    Parameters
    ----------
    initial_theta : np.ndarray
        Initial guess for the estimated parameters.
    target : np.ndarray
        Empirical moments being matched.
    param_template : dict
        Parameter structure showing which are estimated (None).
    W : ndarray
        Weighting matrix (identity for first stage).

    Returns
    -------
    dict with:
        theta_hat : np.ndarray
        obj : float objective minimum
        res : scipy OptimizeResult
    """
 
    if W is None:
        W = np.eye(len(target))

    obj = lambda th: smm_objective(th, target, W, param_template,
                                   grid_k, grid_i, nz, mu_log, S, T, seed)

    res = optimize.minimize(obj, initial_theta, method="Nelder-Mead",
                            options={"maxiter":2000,"disp":True})

    return {"theta_hat":res.x, "obj":res.fun, "res":res}

# === Weighting matrix ===
def compute_weighting_matrix(theta_hat, param_template, grid_k, grid_i,
                             nz, mu_log, target, S_sim=200, T=50, n_reps=200, seed=2025):
    """
    Computes efficient weighting matrix W = Cov(moments)^(-1)

    Procedure:
        1. Fix parameters at first-stage θ̂
        2. Run many simulations (n_reps)
        3. Compute covariance of simulated moments
        4. Invert covariance (or pseudoinverse)

    Exactly matches your implementation.
    """
  
    q = len(target)
    mat = np.zeros((n_reps, q))

    params = param_template.copy()
    names = [k for k,v in param_template.items() if v is None]
    for i, nm in enumerate(names):
        params[nm] = float(theta_hat[i])

    z_grid, P_z = tauchen_QE(nz, mu_log, params["rho"], params["sigma"])
    rng = np.random.default_rng(seed)

    for r in range(n_reps):
        policy = solve_dp(params, grid_k, grid_i, z_grid, P_z)
        sim = simulate_panel(policy, params, grid_k, z_grid, P_z,
                             S=S_sim, T=T, seed=int(rng.integers(1e6)))
        mat[r, :] = compute_moments(sim)

    cov = np.cov(mat, rowvar=False)
    cov += 1e-8 * np.eye(q)

    try:
        return np.linalg.inv(cov)
    except:
        return np.linalg.pinv(cov)

# === Standard errors ===
def compute_standard_errors(theta, param_template, grid_k, grid_i,
                             nz, mu_log, W, target, S_used=300, eps=1e-5):
    """
    Computes numerical Jacobian-based SMM standard errors.

    Steps:
        1. Perturb each parameter ± eps
        2. Recompute simulated moments
        3. Approximate derivative (central difference)
        4. Form covariance matrix: (D' W D)^(-1)
        5. Apply finite sample correction (1 + 1/S_used)

    No changes to your computational logic.
    """

    names = [k for k,v in param_template.items() if v is None]
    p = len(names); q = len(target)

    D = np.zeros((q, p))
    base = param_template.copy()

    for i, nm in enumerate(names):
        th_p = theta.copy()
        th_m = theta.copy()
        th_p[i] += eps
        th_m[i] -= eps

        p_params = base.copy()
        m_params = base.copy()
        for j, nm2 in enumerate(names):
            p_params[nm2] = float(th_p[j])
            m_params[nm2] = float(th_m[j])

        z_p, P_p = tauchen_QE(nz, mu_log, p_params["rho"], p_params["sigma"])
        z_m, P_m = tauchen_QE(nz, mu_log, m_params["rho"], m_params["sigma"])

        pol_p = solve_dp(p_params, grid_k, grid_i, z_p, P_p)
        pol_m = solve_dp(m_params, grid_k, grid_i, z_m, P_m)

        sim_p = simulate_panel(pol_p, p_params, grid_k, z_p, P_p, S=200, T=50)
        sim_m = simulate_panel(pol_m, m_params, grid_k, z_m, P_m, S=200, T=50)

        D[:, i] = (compute_moments(sim_p) - compute_moments(sim_m)) / (2 * eps)

    VC = np.linalg.pinv(D.T @ W @ D)
    VC *= (1 + 1/S_used)

    return np.sqrt(np.diag(VC))

# === Table Output ===
def make_results_table(theta_id, theta_eff, se, param_template, save=True):
    """
    Produces LaTeX table in results/estimates_table.tex.

    The table includes:
        • Identity-weight SMM estimates
        • Efficient-weight SMM estimates
        • Standard errors
        • t-statistics

    Table is pure tabular (no outer environments) so it can be included with \input{}.
    """


    est_names = [k for k, v in param_template.items() if v is None]

    df = pd.DataFrame({
        "Identity_W Estimate": theta_id,
        "Efficient_W Estimate": theta_eff,
        "Std.Error": se,
        "t-stat": theta_eff / (se + 1e-12)
    }, index=est_names)

    if save:
        tex_path = RESULTS / "estimates_table.tex"

        with open(tex_path, "w") as f:
            f.write("\\begin{tabular}{lrrrr}\n")
            f.write("\\toprule\n")
            f.write("Parameter & Identity\\_W Estimate & Efficient\\_W Estimate & Std.Error & t-stat \\\\\n")
            f.write("\\midrule\n")

            for param in df.index:
                row = df.loc[param]
                f.write(f"{param} & "
                        f"{row['Identity_W Estimate']:.6f} & "
                        f"{row['Efficient_W Estimate']:.6f} & "
                        f"{row['Std.Error']:.6f} & "
                        f"{row['t-stat']:.6f} \\\\\n")

            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")

        print(f"Saved LaTeX table to: {tex_path}")

    return df

# === Workflow ===
def run_smm_workflow():
    """
    Executes complete 2-step SMM estimation:

        Step 1: Identity-weight SMM
        Step 2: Compute efficient weighting matrix
        Step 3: Efficient SMM re-estimation
        Step 4: Compute standard errors
        Step 5: Output results table

    Targets moments from Cooper–Ejarque Table 3.
    All logic preserved exactly.
    """

    TARGET = np.array([0.6956, 0.1331, 0.0976, 0.8932, 0.0])

    param_template = {
        'alpha':None,'gamma':None,'rho':None,'sigma':None,'phi0':None,
        'delta':0.15,'beta':0.95,'phi1':0.0
    }

    initial_theta = np.array([0.6956,0.1331,0.0976,0.8932,0.01])

    grid_k = np.linspace(0.1,10,50)
    grid_i = np.linspace(0,2,25)
    nz = 5; mu_log = 0.0

    z0, P0 = tauchen_QE(nz, mu_log, rho=0.9, sigma=0.2)

    S, T = 50, 20
    W_I = np.eye(len(TARGET))

    print("=== Step 1 ===")
    res1 = estimate_smm(initial_theta, TARGET, param_template, grid_k, grid_i,
                        z0, P0, S, T, W=W_I, seed=123, nz=nz, mu_log=mu_log)
    theta1 = res1["theta_hat"]

    print("=== Compute W_eff ===")
    W_eff = compute_weighting_matrix(theta1, param_template, grid_k, grid_i,
                                     nz, mu_log, TARGET)

    print("=== Step 2 ===")
    res2 = estimate_smm(theta1, TARGET, param_template, grid_k, grid_i,
                        z0, P0, S, T, W=W_eff, seed=456, nz=nz, mu_log=mu_log)
    theta2 = res2["theta_hat"]

    print("=== SEs ===")
    se = compute_standard_errors(theta2, param_template, grid_k, grid_i,
                                 nz, mu_log, W_eff, TARGET)

    make_results_table(theta1, theta2, se, param_template, save=True)

    return {
        "theta_identity": theta1,
        "theta_efficient": theta2,
        "se": se,
        "res1": res1,
        "res2": res2
    }

if __name__ == "__main__":
    run_smm_workflow()
