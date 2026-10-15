"""
A 2D B-spline of arbitrary order.
"""

import warnings
import numpy as np
from . import ParameterizableFunc, FITTING_TAGS


def _generate_B(x, knots, delta, p):
    x = np.asarray(x, dtype=float)
    N_pieces = len(knots) - p - 1
    results = np.empty((N_pieces,) + x.shape)
    if p == 0:
        for i in range(N_pieces):
            results[i] = np.where(
                np.logical_and(x >= knots[i], x < knots[i+1]), 1.0, 0.0
            )
        return results
    prev_B = _generate_B(x, knots, delta, p - 1)
    denom = p * delta
    for i in range(N_pieces):
        results[i] = (((x - knots[i]) * prev_B[i])
                      + ((knots[i+p+1] - x) * prev_B[i+1])) / denom
    return results


class BSpline(ParameterizableFunc):
    """Represents a piecewise polynomial spline, spanning a rectangular
    domain, using a cardinal B-spline (i.e. B-spline with uniform knot spacing
    in both dimensions). Extrapolation beyond the endpoints results in smooth
    decay to zero. This subclass is a "flexible" function.

    Parameters
    ----------
    order : int, optional
        Polynomial degree for interpolation; defaults to 4.
    reg_order : int, optional
        Order at which to regularize derivatives, e.g. setting this to 4
        results in minimization of 4th-order derivatives during the fitting
        process. Defaults to the same value as the interpolation polynomial
        degree.
    extent : tuple[float, float, float, float], optional
        Rectangular domain for the BSpline, expressed as `(xa, xb, ya, yb)`
        for bounds `xa <= x <= xb` and `ya <= y <= yb`. If not specified, the
        bounds will be set to the data extremes during fitting.
    """

    def __init__(
            self,
            order: int = 4,
            reg_order: int = None,
            extent: tuple[float, float, float, float] = None,
        ):
        if order < 0:
            raise RuntimeError('order must be nonnegative')
        if (reg_order is not None) and (reg_order < 0):
            raise RuntimeError('reg_order must be nonnegative')
        self._n = 1
        if extent is None:
            self._xa, self._xb = 0.0, 1.0
            self._ya, self._yb = 0.0, 1.0
            self._fixed_bounds = False
        else:
            self._xa, self._xb = float(extent[0]), float(extent[1])
            self._ya, self._yb = float(extent[2]), float(extent[3])
            if self._xa >= self._xb:
                raise RuntimeError('xa must be less than xb')
            if self._ya >= self._yb:
                raise RuntimeError('ya must be less than yb')
            self._fixed_bounds = True
        self._p = int(order)
        self._make_knots()
        self._coeffs = np.zeros((self._n + self._p, self._n + self._p))
        self._reg_order = self._p if reg_order is None else int(reg_order)
        self._fitted = False


    def _make_knots(self):
        self._xdelta = (self._xb - self._xa) / self._n
        self._ydelta = (self._yb - self._ya) / self._n
        self._xknots = np.linspace(
            self._xa - (self._p * self._xdelta),
            self._xb + (self._p * self._xdelta),
            self._n + 1 + (2 * self._p),
        )
        self._yknots = np.linspace(
            self._ya - (self._p * self._ydelta),
            self._yb + (self._p * self._ydelta),
            self._n + 1 + (2 * self._p),
        )


    def f(self, x, y):
        if not self._fitted:
            warnings.warn('Function not yet fitted!', UserWarning)
        new_x, new_y = np.broadcast_arrays(x, y)
        x_basis = _generate_B(new_x, self._xknots, self._xdelta, self._p)
        y_basis = _generate_B(new_y, self._yknots, self._ydelta, self._p)
        return np.einsum('ij,i...,j...->...', self._coeffs, x_basis, y_basis)


    def dfdx(self, x, y):
        if not self._fitted:
            warnings.warn('Function not yet fitted!', UserWarning)
        new_x, new_y = np.broadcast_arrays(x, y)
        if self._p < 1:
            return np.zeros_like(new_x)
        padded = np.pad(self._coeffs, ((1, 1), (0, 0)))
        coeffs_p = (padded[1:,:] - padded[:-1,:]) / self._xdelta
        x_basis = _generate_B(new_x, self._xknots, self._xdelta, self._p - 1)
        y_basis = _generate_B(new_y, self._yknots, self._ydelta, self._p)
        return np.einsum('ij,i...,j...->...', coeffs_p, x_basis, y_basis)


    def dfdy(self, x, y):
        if not self._fitted:
            warnings.warn('Function not yet fitted!', UserWarning)
        new_x, new_y = np.broadcast_arrays(x, y)
        if self._p < 1:
            return np.zeros_like(new_x)
        padded = np.pad(self._coeffs, ((0, 0), (1, 1)))
        coeffs_p = (padded[:,1:] - padded[:,:-1]) / self._ydelta
        x_basis = _generate_B(new_x, self._xknots, self._xdelta, self._p)
        y_basis = _generate_B(new_y, self._yknots, self._ydelta, self._p - 1)
        return np.einsum('ij,i...,j...->...', coeffs_p, x_basis, y_basis)


    def ddfdxx(self, x, y):
        if not self._fitted:
            warnings.warn('Function not yet fitted!', UserWarning)
        new_x, new_y = np.broadcast_arrays(x, y)
        if self._p < 2:
            return np.zeros_like(new_x)
        padded = np.pad(self._coeffs, ((2, 2), (0, 0)))
        coeffs_p = (padded[1:,:] - padded[:-1,:]) / self._xdelta
        coeffs_pp = (coeffs_p[1:,:] - coeffs_p[:-1,:]) / self._xdelta
        x_basis = _generate_B(new_x, self._xknots, self._xdelta, self._p - 2)
        y_basis = _generate_B(new_y, self._yknots, self._ydelta, self._p)
        return np.einsum('ij,i...,j...->...', coeffs_pp, x_basis, y_basis)
    

    def ddfdxy(self, x, y):
        if not self._fitted:
            warnings.warn('Function not yet fitted!', UserWarning)
        new_x, new_y = np.broadcast_arrays(x, y)
        if self._p < 1:
            return np.zeros_like(new_x)
        padded = np.pad(self._coeffs, ((1, 1), (1, 1)))
        coeffs_p = (padded[1:,:] - padded[:-1,:]) / self._xdelta
        coeffs_pp = (coeffs_p[:,1:] - coeffs_p[:,:-1]) / self._ydelta
        x_basis = _generate_B(new_x, self._xknots, self._xdelta, self._p - 1)
        y_basis = _generate_B(new_y, self._yknots, self._ydelta, self._p - 1)
        return np.einsum('ij,i...,j...->...', coeffs_pp, x_basis, y_basis)
    

    def ddfdyy(self, x, y):
        if not self._fitted:
            warnings.warn('Function not yet fitted!', UserWarning)
        new_x, new_y = np.broadcast_arrays(x, y)
        if self._p < 2:
            return np.zeros_like(new_x)
        padded = np.pad(self._coeffs, ((0, 0), (2, 2)))
        coeffs_p = (padded[:,1:] - padded[:,:-1]) / self._ydelta
        coeffs_pp = (coeffs_p[:,1:] - coeffs_p[:,:-1]) / self._ydelta
        x_basis = _generate_B(new_x, self._xknots, self._xdelta, self._p)
        y_basis = _generate_B(new_y, self._yknots, self._ydelta, self._p - 2)
        return np.einsum('ij,i...,j...->...', coeffs_pp, x_basis, y_basis)


    def get_params(self):
        return {
            'n': int(self._n),
            'xa': float(self._xa),
            'xb': float(self._xb),
            'ya': float(self._ya),
            'yb': float(self._yb),
            'fixed_bounds': bool(self._fixed_bounds),
            'p': int(self._p),
            'coeffs': [[float(c) for c in r] for r in self._coeffs],
            'reg_order': int(self._reg_order),
            'fitted': bool(self._fitted),
        }
    

    def load_params(self, p):
        self._n = int(p['n'])
        self._xa, self._xb = float(p['xa']), float(p['xb'])
        self._ya, self._yb = float(p['ya']), float(p['yb'])
        self._fixed_bounds = bool(p['fixed_bounds'])
        self._p = int(p['p'])
        self._make_knots()
        self._coeffs = np.asarray(p['coeffs'], dtype=float)
        self._reg_order = int(p['reg_order'])
        self._fitted = bool(p['fitted'])


    def _get_fitting_matrix(self, x, y, data_tags):

        N_data = x.shape[0]
        self._n = int(round(np.sqrt(N_data)))
        if not self._fixed_bounds:
            self._xa, self._xb = np.min(x), np.max(x)
            self._ya, self._yb = np.min(y), np.max(y)
        self._make_knots()
        N_params = self._n + self._p

        A_mat = np.zeros((N_data, N_params, N_params), dtype=float)
        for tag in FITTING_TAGS:
            idx = (data_tags == tag)
            if tag == 'f':
                x_basis = _generate_B(x[idx], self._xknots, self._xdelta,
                                      self._p)
                y_basis = _generate_B(y[idx], self._yknots, self._ydelta,
                                      self._p)
                A_mat[idx,:] = x_basis.T[:,:,None] * y_basis.T[:,None,:]
            elif tag == 'dfdx':
                if self._p < 1:
                    raise RuntimeError('Unable to fit to gradient data, try '
                                       'increasing order')
                x_basis = _generate_B(x[idx], self._xknots, self._xdelta,
                                      self._p - 1)
                x_basis_p = (x_basis[:-1] - x_basis[1:]) / self._xdelta
                y_basis = _generate_B(y[idx], self._yknots, self._ydelta,
                                      self._p)
                A_mat[idx,:] = x_basis_p.T[:,:,None] * y_basis.T[:,None,:]
            elif tag == 'dfdy':
                if self._p < 1:
                    raise RuntimeError('Unable to fit to gradient data, try '
                                       'increasing order')
                x_basis = _generate_B(x[idx], self._xknots, self._xdelta,
                                      self._p)
                y_basis = _generate_B(y[idx], self._yknots, self._ydelta,
                                      self._p - 1)
                y_basis_p = (y_basis[:-1] - y_basis[1:]) / self._ydelta
                A_mat[idx,:] = x_basis.T[:,:,None] * y_basis_p.T[:,None,:]
            elif tag == 'ddfdxx':
                if self._p < 2:
                    raise RuntimeError('Unable to fit to gradient data, try '
                                       'increasing order')
                x_basis = _generate_B(x[idx], self._xknots, self._xdelta,
                                      self._p - 2)
                x_basis_p = (x_basis[:-1] - x_basis[1:]) / self._xdelta
                x_basis_pp = (x_basis_p[:-1] - x_basis_p[1:]) / self._xdelta
                y_basis = _generate_B(y[idx], self._yknots, self._ydelta,
                                      self._p)
                A_mat[idx,:] = x_basis_pp.T[:,:,None] * y_basis.T[:,None,:]
            elif tag == 'ddfdxy':
                if self._p < 1:
                    raise RuntimeError('Unable to fit to gradient data, try '
                                       'increasing order')
                x_basis = _generate_B(x[idx], self._xknots, self._xdelta,
                                      self._p - 1)
                x_basis_p = (x_basis[:-1] - x_basis[1:]) / self._xdelta
                y_basis = _generate_B(y[idx], self._yknots, self._ydelta,
                                      self._p - 1)
                y_basis_p = (y_basis[:-1] - y_basis[1:]) / self._ydelta
                A_mat[idx,:] = x_basis_p.T[:,:,None] * y_basis_p.T[:,None,:]
            elif tag == 'ddfdyy':
                if self._p < 2:
                    raise RuntimeError('Unable to fit to gradient data, try '
                                       'increasing order')
                x_basis = _generate_B(x[idx], self._xknots, self._xdelta,
                                      self._p)
                y_basis = _generate_B(y[idx], self._yknots, self._ydelta,
                                      self._p - 2)
                y_basis_p = (y_basis[:-1] - y_basis[1:]) / self._ydelta
                y_basis_pp = (y_basis_p[:-1] - y_basis_p[1:]) / self._ydelta
                A_mat[idx,:] = x_basis.T[:,:,None] * y_basis_pp.T[:,None,:]
        return A_mat.reshape((N_data, N_params**2))
    

    def _get_regularization_matrix(self):
        N_params = self._n + self._p
        if self._reg_order == 0:
            return np.eye(N_params**2, dtype=float)
        else:
            mat_x = np.eye(N_params**2, dtype=float).reshape((N_params,) * 4)
            mat_y = np.eye(N_params**2, dtype=float).reshape((N_params,) * 4)
            for _ in range(self._reg_order):
                mat_x -= np.roll(mat_x, shift=1, axis=2)
                mat_y -= np.roll(mat_y, shift=1, axis=3)
            mat_x = mat_x.reshape((N_params**2, N_params**2))
            mat_y = mat_y.reshape((N_params**2, N_params**2))
            return np.concat((mat_x, mat_y), axis=0)
    

    def _set_fitted_params(self, params):
        N_params = self._n + self._p
        self._coeffs = params.reshape((N_params, N_params))
        self._fitted = True

        