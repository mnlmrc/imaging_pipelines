import numpy as np
import PcmPy as pcm

def bootstrap_summary(r_bootstrap, alpha=0.025):
    """
    Given the retained bootstrap correlations, return:
      - central (1-2*alpha) CI (so for alpha=.05 -> 90% CI)
      - functions for one-sided tests: r < x and r > x
    """
    r_bootstrap = np.asarray(r_bootstrap)
    if r_bootstrap.size == 0:
        raise ValueError("No valid bootstrap replicates retained.")

    lo = np.quantile(r_bootstrap, alpha)        # lower bound of central CI
    hi = np.quantile(r_bootstrap, 1 - alpha)    # upper bound of central CI

    def pval_r_less_than(x):
        # p ≈ proportion of bootstrap >= x  (upper tail)
        return float(np.mean(r_bootstrap >= x))

    def pval_r_greater_than(x):
        # p ≈ proportion of bootstrap <= x  (lower tail)
        # NOTE: tends to be liberal in the paper (lower bound too high)
        return float(np.mean(r_bootstrap <= x))

    return (lo, hi), pval_r_less_than, pval_r_greater_than


def bootstrap_correlation(idx, Y, Mflex, sigma_floor=1e-4):
    """

    Args:
        Y:
        Mflex:
        sigma_floor:

    Returns:

    """

    S = len(Y)
    y = [Y[i] for i in idx]

    _, theta_gr = pcm.fit_model_group(y, Mflex, fixed_effect=None, fit_scale=True, verbose=False)

    theta_gr, _ = pcm.group_to_individ_param(theta_gr[0], Mflex, S)

    sigma2_1 = np.exp(theta_gr[0, 0])
    sigma2_2 = np.exp(theta_gr[1, 0])
    sigma2_e = np.exp(theta_gr[-1])
    r = Mflex.get_correlation(theta_gr)

    sd = np.sqrt(sigma2_1 * sigma2_2)
    if sd < sigma_floor * np.sqrt(sigma2_e).max():
        print(f'No reliable signal, discarding bootstrap resample')
        return None
    else:
        return r[0]