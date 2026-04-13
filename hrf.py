from scipy.optimize import minimize
import numpy as np
from nitools import spm
import itertools
import pandas as pd

def calc_resms(SPM, y_scl, P, TR, T=16):
    """Return voxelwise ResMS for a given HRF parameter vector P."""
    bf, _ = spm.spm_hrf(TR, P, T)          # HRF for current P
    SPM.convolve_glm(bf)                      # rebuild design with HRF
    *_, residuals = SPM.rerun_glm(y_scl)      # refit GLM
    dof = residuals.shape[0] - np.linalg.matrix_rank(SPM.X)
    rss = np.nansum(residuals * residuals, axis=0)  # voxelwise RSS
    return rss / dof

def calc_R2(Y, Yhat):
    ss_res = np.nansum((Y - Yhat) ** 2)
    ss_tot = np.nansum((Y - np.nanmean(Y,axis=0,keepdims=True)) ** 2)
    return 1 - (ss_res / ss_tot)


def objective(P, SPM, y_scl, TR, T=16):
    """Objective function: mean ResMS for given P."""
    resms = calc_resms(SPM, y_scl, P, TR, T=T)

    return np.nanmean(resms)

def optimise_hrf(SPM, y_scl, P0=None, TR=1, T=16):
    """
    Minimize mean ResMS by adjusting HRF parameters P.
    Returns best_P, best_val, opt_result.
    """
    if P0 is None:
        P0 = np.array([6., 16., 1., 1., 6., 0., 32.], dtype=float)

    bounds = [
        (1.0, 20.0),   # delay (response)
        (5.0, 40.0),   # delay (undershoot)
        (0.5, 3.0),    # dispersion (response)
        (0.5, 3.0),    # dispersion (undershoot)
        (1.0, 12.0),   # ratio
        (-2.0, 4.0),   # onset
        (16.0, 64.0),  # length
    ]

    res = minimize(objective, P0, args=(SPM, y_scl, TR, T), bounds=bounds, options=dict(maxiter=150))

    return res.x, res.fun, res


def grid_search_hrf(SPM, y_raw, TR=1.0, T=16, P0=None, grid=None, agg="mean", verbose=True):
    """
    Brute-force grid search for HRF params P that minimize mean/median ResMS.
    Returns best_P, best_value, all_results (list of (P, value)).
    """
    if P0 is None:
        # SPM canonical defaults
        P0 = np.array([6., 16., 1., 1., 6., 0., 32.], dtype=float)

    # Default compact grid: vary peaks & ratio a bit; keep others fixed.
    # You can widen/narrow as needed.
    if grid is None:
        grid = {
            0: np.array([4., 6., 8.]),     # delay response
            1: np.array([14., 16., 18.]),  # delay undershoot
            2: np.array([1.0]),            # dispersion response
            3: np.array([1.0]),            # dispersion undershoot
            4: np.array([4., 6., 8.]),     # ratio
            5: np.array([0.0]),            # onset
            6: np.array([32.0])            # length
        }

    # Build candidate lists for each P index; use P0 if not specified
    candidates = [np.asarray(grid.get(i, [P0[i]]), dtype=float) for i in range(7)]

    best_P = P0.copy()
    best_val = -np.inf
    params = []

    # Evaluate every combination
    for combo in itertools.product(*candidates):
        P = np.array(combo, dtype=float)
        bf, _ = spm.spm_hrf(TR, P, T)
        SPM.convolve_glm(bf)
        _, info, y_filt, y_hat, y_adj, _ = SPM.rerun_glm(y_raw)
        R2 = calc_R2(y_adj, y_hat)
        params.append(np.r_[P, R2]) # dont update in a loop
        if R2 > best_val: # val < best_val:
            best_val = R2
            best_P = P
        print(f'tried P={P}, R2={R2}, best P={best_P}, best R2={best_val}') if verbose else None

    params_df = pd.DataFrame(np.array(params), 
        columns = ['delay_response', 'delay_undershoot', 'dispersion_response', 
        'dispersion_undershoot', 'ratio', 'onset', 'length', 'R_squared'])

    return best_P, best_val, params_df