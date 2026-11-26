"""
Modules related to pure numerical functions of two variables.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve


FITTING_TAGS = {
    'f': 0,
    'xdfdx': 1,
    'ydfdy': 2,
    'y2ddfdyy': 3,
    'xdfdx-xyddfdxy': 4,
}
"""Integer tags indicating which data type to fit to."""


def solve_regularized_lstsq(A_mat, b_vec, Q_mat=None, target=1.0):
    """Solves the linear least-squares problem `Ax = b`, with a possible
    regularization via a penalty matrix `Q`. The solution is therefore the
    vector `x` which minimizes the loss function `|Ax - b|² + λ²xᵀQx`, where
    the regularization parameter `λ` is automatically selected to keep the
    fit's reduced χ² below the target value.
    
    Parameters
    ----------
    A_mat : 2D array of floats of shape (N_rows, N_cols)
        Matrix `A` specifying the linear least-squares problem.
    b_vec : 1D array of floats of shape (N_rows,)
        Vector `b` specifying the linear least-squares problem. It is assumed
        that all entries have equal variance 1, and are uncorrelated; if this
        is not the case, both `A_mat` and `b_vec` should be normalized against
        the data covariance, so that the correct inverse-variance weighting is
        applied, and the correct value of χ² is calculated.
    Q_mat : 2D array of floats of shape (N_cols, N_cols), optional
        Penalty matrix `Q` for regularization; setting this to None (which is
        the default behaviour) disables regularization.
    target : float, optional
        Target reduced χ² value for automatic selection of the regularization
        parameter.

    Returns
    -------
    best_fit : 1D array of floats of shape (N_cols,)
        Best-fit value for the vector `x` which minimizes the regularized loss
        function.
    regularizer : float
        Value of the regularization parameter `λ` used in the loss function.
        This is chosen as the largest value, out of a preset list of values
        spanning 10⁻⁹ to 10³, which results in a fit reduced χ² lower than the
        target value; or, if no such value of `λ` is able to achieve this,
        then `λ` is chosen as the value which minimizes the generalized
        cross-validation. Defaults to 0 if regularization is disabled.
    chi_sq : float
        Fit reduced χ² corresponding to the best-fit solution.
    """

    N_rows, N_cols = A_mat.shape

    if Q_mat is None:
        best_fit = np.linalg.lstsq(A_mat, b_vec)[0]
        rss = np.sum((b_vec - (A_mat @ best_fit))**2)
        denom = max(N_rows - np.linalg.matrix_rank(A_mat), 1)
        return best_fit, 0.0, rss / denom
    
    G_mat = A_mat.T @ A_mat
    rhs = np.concat([G_mat, (A_mat.T @ b_vec)[:,None]], axis=1)
    jitter = (1e-12 * np.trace(G_mat) / N_cols) * np.eye(N_cols)

    results = []
    for log_regularizer in np.linspace(-9.0, 3.0, 49):
        regularizer = 10.0**log_regularizer
        M_mat = G_mat + ((regularizer**2) * Q_mat) + jitter
        try:
            sol = cho_solve(cho_factor(M_mat), rhs)
        except np.linalg.LinAlgError:
            continue
        best_fit = sol[:,-1]
        rss = np.sum((b_vec - (A_mat @ best_fit))**2)
        denom = N_rows - np.trace(sol[:,:-1])
        chi_sq = (rss / denom) if denom > 1.0 else np.inf
        gcv = (N_rows * rss / (denom**2)) if denom > 1.0 else np.inf
        results.append((regularizer, best_fit, chi_sq, gcv))

    good = [r for r in results if r[2] <= target]
    if len(good) > 0:
        regularizer, best_fit, chi_sq, _ = max(good, key=lambda r: r[0])
    else:
        regularizer, best_fit, chi_sq, _ = min(results, key=lambda r: r[3])
    return best_fit, regularizer, chi_sq