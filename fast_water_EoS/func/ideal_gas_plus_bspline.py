"""
A 2D B-spline of arbitrary order.
"""

import numpy as np
from . import FITTING_TAGS, solve_regularized_lstsq
from ..misc import castable, pos_divide, broadcast_and_flatten


def _generate_B(x, knots, p):
    """Generates the table of order-`p` basis functions `Bᵢ` of the given knot
    sequence, evaluated at positions `x`."""
    B = ((x >= knots[:-1,None]) & (x < knots[1:,None])).astype(float)
    B[:,x < knots[0]] = np.nan
    B[:,x > knots[-1]] = np.nan
    for k in range(1, p + 1):
        n = len(knots) - k - 1
        l_denom = knots[k:(k+n)] - knots[:n]
        l_numer = x - knots[:n,None]
        l_factor = l_numer * pos_divide(1.0, l_denom)[:,None]
        r_denom = knots[(k+1):(k+n+1)] - knots[1:(n+1)]
        r_numer = knots[(k+1):(k+n+1),None] - x
        r_factor = r_numer * pos_divide(1.0, r_denom)[:,None]
        B = (l_factor * B[:-1]) + (r_factor * B[1:])
    return B


def _calc_deriv_integral(knots, p, k):
    """Calculates the matrix `Gᵢⱼ = ∫ ∂ₖBᵢ(x) ∂ₖBⱼ(x) dx` for the order-`p`
    basis functions of the given knot sequence."""
    n = len(knots) - p - 1
    span = knots[-1] - knots[0]
    x = (knots - knots[0]) / (span if span > 0.0 else 1.0)

    coeffs = np.eye(n)
    for j in range(k):
        deg = p - j
        denom = x[deg:] - x[:-deg]
        padded = np.pad(coeffs, ((1, 1), (0, 0)))
        coeffs = (padded[1:] - padded[:-1]) * pos_divide(deg, denom)[:,None]

    nodes, weights = np.polynomial.legendre.leggauss(p - k + 1)
    a, b = x[:-1], x[1:]
    keep = b > a
    a, b = a[keep], b[keep]
    pts = a[:,None] + (0.5 * (b - a)[:,None] * (nodes[None,:] + 1.0))
    pts = pts.ravel()
    wts = (0.5 * (b - a)[:,None] * weights[None,:]).ravel()
    B = _generate_B(pts, x, p - k)

    return coeffs.T @ (B * wts) @ B.T @ coeffs


class IdealGasPlusBSpline:
    """Hybrid of `IdealGas` and `BSpline` classes. See documentations
    separately."""


    def __init__(
            self,
            dof: int | float = 6,
            order: int | tuple[int, int] = 4,
            reg_orders: list[tuple[int, int]] = None,
            knots: tuple[np.ndarray, np.ndarray] = None,
            coeffs: np.ndarray = None,
            coeffs_cov: np.ndarray = None,
        ):

        self._dof = float(dof)

        if castable(order, int):
            self._px = int(order)
            self._py = int(order)
        elif castable(order, tuple):
            self._px = int(order[0])
            self._py = int(order[1])
        else:
            raise RuntimeError('Unknown input type for order')
        
        if reg_orders is None:
            self._reg_orders = [(self._px, 0), (0, self._py)]
        elif castable(reg_orders, int):
            k = int(reg_orders)
            if k == -1:
                self._reg_orders = list()
            else:
                if k < 0:
                    raise RuntimeError('reg_orders must be nonnegative')
                if k > max(self._px, self._py):
                    raise RuntimeError('reg_orders must not exceed order')
                self._reg_orders = [(k, 0), (0, k)]
        else:
            self._reg_orders = list()
            for pair in reg_orders:
                k, l = int(pair[0]), int(pair[1])
                if (k < 0) or (l < 0):
                    raise RuntimeError('reg_orders must be nonnegative')
                if (k > self._px) or (l > self._py):
                    raise RuntimeError('reg_orders must not exceed order')
                self._reg_orders.append((k, l))
        
        if knots is None:
            self._xknots = None
            self._yknots = None
            self._coeffs = None
            self._coeffs_cov = None
        else:
            self._xknots = np.asarray(knots[0], dtype=float).ravel()
            self._yknots = np.asarray(knots[1], dtype=float).ravel()
            if ((not np.all(np.isfinite(self._xknots)))
                or (not np.all(np.isfinite(self._yknots)))):
                raise RuntimeError('Invalid value in knots')
            if (np.any(np.diff(self._xknots, 1) < 0.0)
                or np.any(np.diff(self._yknots, 1) < 0.0)):
                raise RuntimeError('knots must be strictly nondecreasing')
            nx = self._xknots.shape[0] - self._px - 1
            ny = self._yknots.shape[0] - self._py - 1
            if coeffs is None:
                self._coeffs = np.zeros((nx, ny), dtype=float)
                self._coeffs_cov = np.zeros((nx, ny, nx, ny), dtype=float)
            else:
                self._coeffs = np.asarray(coeffs, dtype=float)
                self._coeffs_cov = np.asarray(coeffs_cov, dtype=float)
                assert self._coeffs.shape == (nx, ny)
                assert self._coeffs_cov.shape == (nx, ny, nx, ny)


    def _diff_x(self, coeffs, p):
        """Internal helper to generate coefficients after differentiating in
        the x-axis."""
        padded = np.pad(coeffs, ((1, 1), (0, 0)))
        den = self._xknots[p:] - self._xknots[:-p]
        return (padded[1:,:] - padded[:-1,:]) * pos_divide(p, den)[:,None]


    def _diff_y(self, coeffs, p):
        """Internal helper to generate coefficients after differentiating in
        the y-axis."""
        padded = np.pad(coeffs, ((0, 0), (1, 1)))
        den = self._yknots[p:] - self._yknots[:-p]
        return (padded[:,1:] - padded[:,:-1]) * pos_divide(p, den)[None,:]


    def f(self, x, y):
        """Returns `f` at the given `(x, y)` points."""
        x, y = broadcast_and_flatten(x, y)
        x_basis = _generate_B(x, self._xknots, self._px)
        y_basis = _generate_B(y, self._yknots, self._py)
        return (np.log(x) + (self._dof * np.log(y) / 2.0)
            + np.einsum('ij,i...,j...->...', self._coeffs, x_basis, y_basis))


    def dfdx(self, x, y):
        """Returns `df/dx` at the given `(x, y)` points."""
        x, y = broadcast_and_flatten(x, y)
        if self._px < 1:
            return np.zeros_like(x)
        coeffs_dx = self._diff_x(self._coeffs, self._px)
        x_basis = _generate_B(x, self._xknots, self._px - 1)
        y_basis = _generate_B(y, self._yknots, self._py)
        return ((1.0 / x)
            + np.einsum('ij,i...,j...->...', coeffs_dx, x_basis, y_basis))


    def dfdy(self, x, y):
        """Returns `df/dy` at the given `(x, y)` points."""
        x, y = broadcast_and_flatten(x, y)
        if self._py < 1:
            return np.zeros_like(x)
        coeffs_dy = self._diff_y(self._coeffs, self._py)
        x_basis = _generate_B(x, self._xknots, self._px)
        y_basis = _generate_B(y, self._yknots, self._py - 1)
        return ((self._dof / (2.0 * y))
            + np.einsum('ij,i...,j...->...', coeffs_dy, x_basis, y_basis))


    def ddfdxx(self, x, y):
        """Returns `d²f/dx²` at the given `(x, y)` points."""
        x, y = broadcast_and_flatten(x, y)
        if self._px < 2:
            return np.zeros_like(x)
        coeffs_dx = self._diff_x(self._coeffs, self._px)
        coeffs_dxx = self._diff_x(coeffs_dx, self._px - 1)
        x_basis = _generate_B(x, self._xknots, self._px - 2)
        y_basis = _generate_B(y, self._yknots, self._py)
        return ((-1.0 / (x**2))
            + np.einsum('ij,i...,j...->...', coeffs_dxx, x_basis, y_basis))
    

    def ddfdxy(self, x, y):
        """Returns `d²f/dxdy` at the given `(x, y)` points."""
        x, y = broadcast_and_flatten(x, y)
        if (self._px < 1) or (self._py < 1):
            return np.zeros_like(x)
        coeffs_dx = self._diff_x(self._coeffs, self._px)
        coeffs_dxy = self._diff_y(coeffs_dx, self._py)
        x_basis = _generate_B(x, self._xknots, self._px - 1)
        y_basis = _generate_B(y, self._yknots, self._py - 1)
        return np.einsum('ij,i...,j...->...', coeffs_dxy, x_basis, y_basis)
    

    def ddfdyy(self, x, y):
        """Returns `d²f/dy²` at the given `(x, y)` points."""
        x, y = broadcast_and_flatten(x, y)
        if self._py < 2:
            return np.zeros_like(x)
        coeffs_dy = self._diff_y(self._coeffs, self._py)
        coeffs_dyy = self._diff_y(coeffs_dy, self._py - 1)
        x_basis = _generate_B(x, self._xknots, self._px)
        y_basis = _generate_B(y, self._yknots, self._py - 2)
        return ((-self._dof / (2.0 * (y**2)))
            + np.einsum('ij,i...,j...->...', coeffs_dyy, x_basis, y_basis))


    def get_params(self, prefix=''):
        """Returns the internal representation parameters as a dict, such that
        the full state of the function can be restored using the
        `load_params(...)` method."""
        params = {
            prefix + 'dof': float(self._dof),
            prefix + 'px': int(self._px),
            prefix + 'py': int(self._py),
            prefix + 'reg_orders': np.asarray(self._reg_orders),
        }
        if self._coeffs is not None:
            params[prefix + 'xknots'] = self._xknots
            params[prefix + 'yknots'] = self._yknots
            params[prefix + 'coeffs'] = self._coeffs
            params[prefix + 'coeffs_cov'] = self._coeffs_cov
        return params
    

    def load_params(self, params, prefix=''):
        """Loads the internal representation parameters from a dict, such that
        the full state of the function can be restored from the dict returned
        by the `get_params()` method."""
        self._dof = float(params[prefix + 'dof'])
        self._px = int(params[prefix + 'px'])
        self._py = int(params[prefix + 'py'])
        self._reg_orders = [(int(c[0]), int(c[1]))
                            for c in params[prefix + 'reg_orders']]
        if (prefix + 'coeffs') in params:
            self._xknots = np.asarray(params[prefix + 'xknots'], dtype=float)
            self._yknots = np.asarray(params[prefix + 'yknots'], dtype=float)
            self._coeffs = np.asarray(params[prefix + 'coeffs'], dtype=float)
            self._coeffs_cov = np.asarray(params[prefix + 'coeffs_cov'],
                                          dtype=float)
        else:
            self._xknots = None
            self._yknots = None
            self._coeffs = None
            self._coeffs_cov = None


    def _suggest_naive_knots(self, x, p):
        """Suggest naive knots for the order-`p` spline along the specified
        direction, provided data points `x`."""
        target_N_intervals = int(np.clip(
            np.ceil(0.5 * np.sqrt(x.shape[0])),
            p + 1,
            max(np.unique(x).shape[0], p + 1),
        ))
        x_min, x_max = np.min(x), np.max(x)
        l_end_knots = np.full(p + 1, x_min - (1e-9 * (x_max - x_min)))
        r_end_knots = np.full(p + 1, x_max + (1e-9 * (x_max - x_min)))
        N_interior_knots = target_N_intervals - p - 1
        if N_interior_knots <= 0:
            return np.concat([l_end_knots, r_end_knots])
        q = np.arange(1, N_interior_knots + 1) / (N_interior_knots + 1)
        interior = np.unique(np.quantile(x, q))
        interior = interior[(interior > x_min) & (interior < x_max)]
        return np.concat([l_end_knots, interior, r_end_knots])


    def _get_fitting_matrix(self, x, y, data_tags):
        """Get the fitting matrix `A` which enters into the regularized
        linear least-squares problem."""

        N_data = x.shape[0]
        nx = self._xknots.shape[0] - self._px - 1
        ny = self._yknots.shape[0] - self._py - 1

        A_mat = np.zeros((N_data, nx, ny), dtype=float)
        for tag in FITTING_TAGS.values():
            idx = (data_tags == tag)
            if np.count_nonzero(idx) < 1:
                continue
            fit_x, fit_y = x[idx], y[idx]
            if tag == FITTING_TAGS['f']:
                x_basis = _generate_B(fit_x, self._xknots, self._px)
                y_basis = _generate_B(fit_y, self._yknots, self._py)
                A_mat[idx,:,:] = x_basis.T[:,:,None] * y_basis.T[:,None,:]
            elif tag == FITTING_TAGS['xdfdx']:
                Dx = self._diff_x(np.eye(nx), self._px).T
                x_basis_dx = _generate_B(fit_x, self._xknots, self._px - 1)
                x_basis_dx = Dx @ x_basis_dx
                y_basis = _generate_B(fit_y, self._yknots, self._py)
                A_mat[idx,:,:] = (fit_x[:,None,None] * x_basis_dx.T[:,:,None]
                                  * y_basis.T[:,None,:])
            elif tag == FITTING_TAGS['ydfdy']:
                Dy = self._diff_y(np.eye(ny), self._py)
                x_basis = _generate_B(fit_x, self._xknots, self._px)
                y_basis_dy = _generate_B(fit_y, self._yknots, self._py - 1)
                y_basis_dy = Dy @ y_basis_dy
                A_mat[idx,:,:] = (fit_y[:,None,None] * x_basis.T[:,:,None]
                                  * y_basis_dy.T[:,None,:])
            elif tag == FITTING_TAGS['y2ddfdyy']:
                Dy = self._diff_y(np.eye(ny), self._py)
                Dyy = self._diff_y(Dy, self._py - 1)
                x_basis = _generate_B(fit_x, self._xknots, self._px)
                y_basis_dyy = _generate_B(fit_y, self._yknots, self._py - 2)
                y_basis_dyy = Dyy @ y_basis_dyy
                A_mat[idx,:,:] = ((fit_y**2)[:,None,None]
                                  * x_basis.T[:,:,None]
                                  * y_basis_dyy.T[:,None,:])
            elif tag == FITTING_TAGS['xdfdx-xyddfdxy']:
                Dx = self._diff_x(np.eye(nx), self._px).T
                Dy = self._diff_y(np.eye(ny), self._py)
                x_basis_dx = _generate_B(fit_x, self._xknots, self._px - 1)
                x_basis_dx = Dx @ x_basis_dx
                y_basis = _generate_B(fit_y, self._yknots, self._py)
                y_basis_dy = _generate_B(fit_y, self._yknots, self._py - 1)
                y_basis_dy = Dy @ y_basis_dy
                A_mat[idx,:,:] = ((fit_x[:,None,None] * x_basis_dx.T[:,:,None]
                                   * y_basis.T[:,None,:])
                                  - (fit_x[:,None,None] * fit_y[:,None,None]
                                     * x_basis_dx.T[:,:,None]
                                     * y_basis_dy.T[:,None,:]))
        
        return A_mat.reshape((N_data, nx*ny))
    

    def _get_zero_offset(self, x, y, data_tags):
        """Return offset for the vector `b` when all fitted parameters are
        zero."""
        results = np.zeros_like(x)
        for tag in FITTING_TAGS.values():
            idx = (data_tags == tag)
            if np.count_nonzero(idx) < 1:
                continue
            if tag == FITTING_TAGS['f']:
                results[idx] = (np.log(x[idx])
                                + (self._dof * np.log(y[idx]) / 2.0))
            elif tag == FITTING_TAGS['xdfdx']:
                results[idx] = 1.0
            elif tag == FITTING_TAGS['ydfdy']:
                results[idx] = self._dof / 2.0
            elif tag == FITTING_TAGS['y2ddfdyy']:
                results[idx] = -self._dof / 2.0
            elif tag == FITTING_TAGS['xdfdx-xyddfdxy']:
                results[idx] = 1.0
        return results
    

    def _get_penalty_matrix(self):
        """Get the penalty matrix `Q` which enters into the regularized
        linear least-squares problem."""
        if len(self._reg_orders) == 0:
            return None
        Q = 0.0
        for k, l in self._reg_orders:
            Gx = _calc_deriv_integral(self._xknots, self._px, k)
            Gy = _calc_deriv_integral(self._yknots, self._py, l)
            Q = Q + np.kron(Gx, Gy)
        return Q


    def _refine_knots(
            self, x, y, data, data_err, data_tags,
            target: float = 1.0,
            max_coeff_frac: float = 0.5,
            holdout: float = 0.2,
            frac_per_pass: float = 0.1,
            min_pts: int = 3,
            patience: int = 2,
            max_passes: int = 100,
        ):
        """Iteratively refine the knot sequences in the x and y directions,
        by greedily inserting interior knots where the fit is worst. This is
        achieved by splitting the data into a training set and validation set,
        and evaluating the quality-of-fit on the validation set when fitted
        only to the training set. Each distinct knot span containing at least
        `2 * min_pts` training points is then ranked based on this validation
        error, and the worst knot spans will have new interior knots added at
        the median of the contained points.

        Parameters
        ----------
        x, y, data, data_err, data_tags :
            As in the `fit(...)` method.
        target : float, optional
            Target value of the normalized χ² per point.
        max_coeff_frac : float, optional
            Upper limit on the total number of coefficients `nx * ny`, as a
            fraction of the number of training points; prevents the model
            from becoming underdetermined.
        holdout : float, optional
            Fraction of the data reserved for validation. If zero, no data is
            held out and the training error is used for validation (which
            gives no protection against overfitting).
        frac_per_pass : float, optional
            Fraction of the candidate knot insertions to carry out in each
            pass; at least one insertion is always attempted.
        min_pts : int, optional
            Minimum number of training points which each of the two halves of
            a split knot span must be able to contain; spans with fewer than
            `2 * min_pts` points are never subdivided.
        patience : int, optional
            Number of consecutive passes without significant improvement of
            the validation error, after which refinement is stopped.
        max_passes : int, optional
            Maximum number of refinement passes.
        """

        N_data = x.shape[0]
        rng = np.random.default_rng()
        perm = rng.permutation(N_data)
        N_valid = int(round(holdout * N_data)) if holdout else 0
        valid, train = perm[:N_valid], perm[N_valid:]
        max_coeffs = max_coeff_frac * train.shape[0]

        best_knots = (self._xknots.copy(), self._yknots.copy())
        best_val = np.inf
        stale = 0

        for _ in range(max_passes):

            nx = self._xknots.shape[0] - self._px - 1
            ny = self._yknots.shape[0] - self._py - 1

            A_train = (self._get_fitting_matrix(
                x[train],
                y[train],
                data_tags[train],
            ) / data_err[train, None])
            b_train = data[train] / data_err[train]
            best_fit, _, chi_sq_train = solve_regularized_lstsq(
                A_train, b_train, self._get_penalty_matrix(), target,
            )
            rss_train = (b_train - (A_train @ best_fit))**2

            if N_valid > 0:
                A_valid = (self._get_fitting_matrix(
                    x[valid],
                    y[valid],
                    data_tags[valid],
                ) / data_err[valid, None])
                b_valid = data[valid] / data_err[valid]
                chi_sq_valid = np.mean((b_valid - (A_valid @ best_fit))**2)
            else:
                chi_sq_valid = chi_sq_train

            if chi_sq_valid < best_val:
                if chi_sq_valid < (0.98 * best_val):
                    stale = 0
                else:
                    stale += 1
                best_val = chi_sq_valid
                best_knots = (self._xknots.copy(), self._yknots.copy())
            else:
                stale += 1
            if chi_sq_valid <= target or stale >= patience:
                break

            candidates = []
            for axis, u_data, knots in ((0, x[train], self._xknots),
                                        (1, y[train], self._yknots)):
                uniq = np.unique(knots)
                idx = np.clip(np.searchsorted(uniq, u_data, side='right') - 1,
                              0, len(uniq) - 2)
                sums = np.bincount(idx, weights=rss_train,
                                   minlength=(len(uniq) - 1))
                counts = np.bincount(idx, minlength=(len(uniq) - 1))
                for s in np.nonzero(counts >= (2 * min_pts))[0]:
                    new = np.median(u_data[idx == s])
                    if uniq[s] < new < uniq[s+1]:
                        candidates.append((sums[s], axis, new))
            if len(candidates) == 0:
                break
            candidates.sort(key=lambda c: -c[0])
            n_insert = max(1, int(frac_per_pass * len(candidates)))

            added = 0
            for _, axis, new in candidates[:n_insert]:
                dnx, dny = (1, 0) if axis == 0 else (0, 1)
                if (nx + dnx) * (ny + dny) > max_coeffs:
                    continue
                if axis == 0:
                    self._xknots = np.sort(np.append(self._xknots, new))
                    nx += 1
                else:
                    self._yknots = np.sort(np.append(self._yknots, new))
                    ny += 1
                added += 1
            if added == 0:
                break

        self._xknots, self._yknots = best_knots


    def fit(
            self,
            x: np.ndarray[tuple[int], np.dtype[float]],
            y: np.ndarray[tuple[int], np.dtype[float]],
            data: np.ndarray[tuple[int], np.dtype[float]],
            data_err: np.ndarray[tuple[int], np.dtype[float]],
            data_tags: np.ndarray[tuple[int], np.dtype[int]] = 0,
            refine: bool = True,
        ) -> None:
        """Fits this function to a set of known values.

        Parameters
        ----------
        x : array_like of floats of shape (N_data,)
            x-coordinates of the input data.
        y : array_like of floats of shape (N_data,)
            y-coordinates of the input data.
        data : array_like of floats of shape (N_data,)
            Known values at the given data points to be fitted to.
        data_err : array_like of floats of shape (N_data,)
            Uncertainties of the given data points; assumed to be independent,
            and numerically equal to the standard deviation. The least-squares
            fitting problem is normalized using inverse variance weightage, so
            it is important that `data_err` should contain realistic positive
            values.
        data_tags : array_like of int of shape (N_data,), optional
            String tags denoting which type of data was provided in `data`.
            Each tag must be one of the values in `_FITTING_TAGS`.
        refine : bool, optional
            Whether or not to perform iterative knot refinement.
        """
        
        inputs = broadcast_and_flatten(x, y, data, data_err, data_tags)
        x, y, data, data_err, data_tags = inputs
        data_tags = data_tags.astype(int)
        unique_tags = np.unique(data_tags)
        valid_tags = list(FITTING_TAGS.values())
        bad_tags = np.logical_not(np.isin(unique_tags, valid_tags))
        if np.any(data_err <= 0.0):
            raise ValueError('data_err must be strictly positive!')
        if np.any(bad_tags):
            raise ValueError(f'Unrecognized tags: {unique_tags[bad_tags]}')
        if np.isin(FITTING_TAGS['xdfdx'], unique_tags) and (self._px < 1):
            raise RuntimeError('Unable to fit to gradient data')
        if np.isin(FITTING_TAGS['ydfdy'], unique_tags) and (self._py < 1):
            raise RuntimeError('Unable to fit to gradient data')
        if np.isin(FITTING_TAGS['y2ddfdyy'], unique_tags) and (self._py < 2):
            raise RuntimeError('Unable to fit to curvature data')
        if (np.isin(FITTING_TAGS['xdfdx-xyddfdxy'], unique_tags)
            and (self._px < 1) and (self._py < 1)):
            raise RuntimeError('Unable to fit to curvature data')
        
        data -= self._get_zero_offset(x, y, data_tags)

        if self._xknots is None:
            self._xknots = self._suggest_naive_knots(x, self._px)
            self._yknots = self._suggest_naive_knots(y, self._py)
            if refine:
                self._refine_knots(x, y, data, data_err, data_tags)

        A_mat = self._get_fitting_matrix(x, y, data_tags)
        A_norm = A_mat / data_err[:,None]
        b_norm = data / data_err
        Q_norm = self._get_penalty_matrix()

        best_fit, regularizer, chi_sq = solve_regularized_lstsq(
            A_norm, b_norm, Q_norm,
        )

        M_mat = A_norm.T @ A_norm
        if Q_norm is not None:
            M_mat += (regularizer**2) * Q_norm

        # Frequentist uncertainty
        # M_inv = np.linalg.pinv(M_mat)
        # fit_cov = chi_sq * (M_inv @ (A_norm.T @ A_norm) @ M_inv)

        # Bayesian uncertainty
        fit_cov = np.linalg.pinv(M_mat)

        nx = self._xknots.shape[0] - self._px - 1
        ny = self._yknots.shape[0] - self._py - 1
        self._coeffs = best_fit.reshape((nx, ny))
        self._coeffs_cov = fit_cov.reshape((nx, ny, nx, ny))

    
    def get_bootstrap_model(self, rng: np.random.Generator):
        nx = self._xknots.shape[0] - self._px - 1
        ny = self._yknots.shape[0] - self._py - 1
        new_coeffs = rng.multivariate_normal(
            mean=self._coeffs.reshape(nx*ny),
            cov=self._coeffs_cov.reshape((nx*ny, nx*ny)),
        )
        return IdealGasPlusBSpline(
            dof=self._dof,
            order=(self._px, self._py),
            reg_orders=self._reg_orders,
            knots=(self._xknots, self._yknots),
            coeffs=new_coeffs.reshape((nx, ny)),
            coeffs_cov=self._coeffs_cov,
        )

        