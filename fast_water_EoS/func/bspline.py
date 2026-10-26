"""
A 2D B-spline of arbitrary order.
"""

import warnings
import numpy as np
from . import ParameterizableFunc, FITTING_TAGS
from ..misc import castable


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
    reg_orders : list[tuple[int, int]], optional
        List of orders at which to regularize derivatives, where each entry
        `(k, l)` results in minimization of the mixed `(k,l)`ᵗʰ partial
        derivative of the B-spline during the fitting process. Providing a
        single nonnegative integer `k` is equivalent to `[(k, 0), (0, k)]`,
        i.e. minimizing the `k`ᵗʰ partial derivatives in both the x and y
        directions. Passing `k = -1` as a special integer disables all
        regularization. Defaults to `[(p, 0), (0, p)]`, where `p` is the
        interpolation polynomial degree.
    extent : tuple[float, float, float, float], optional
        Rectangular domain for the BSpline, expressed as `(xa, xb, ya, yb)`
        for bounds `xa <= x <= xb` and `ya <= y <= yb`. If not specified, the
        bounds will be set to the data extremes during fitting.
    n : int or tuple[int, int], optional
        Number of knots to divide the domain up into along each axis. Will be
        interpreted as `(nx, ny)` if provided as a tuple, or `nx = ny = n` if
        provided as a single integer. If not specified, will be automatically
        set during fitting, such that `nx * ny` is greater than or equal to
        the number of data points, and `ny / nx` is approximately close to the
        aspect ratio `(yb - ya) / (xb - xa)`.
    """

    def __init__(
            self,
            order: int = 4,
            reg_orders: list[tuple[int, int]] = None,
            extent: tuple[float, float, float, float] = None,
            n: int | tuple[int, int] = None,
        ):
        if order < 0:
            raise RuntimeError('order must be nonnegative')
        if n is None:
            self._nx = 1
            self._ny = 1
            self._fixed_n = False
        else:
            if castable(n, int):
                self._nx = int(n)
                self._ny = int(n)
            else:
                self._nx = int(n[0])
                self._ny = int(n[1])
            if (self._nx < 1) or (self._ny < 1):
                raise RuntimeError('n must be positive')
            self._fixed_n = True
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
        self._coeffs = np.zeros((self._nx + self._p, self._ny + self._p))
        if reg_orders is None:
            self._reg_orders = [(self._p, 0), (0, self._p)]
        elif castable(reg_orders, int):
            k = int(reg_orders)
            if k == -1:
                self._reg_orders = list()
            else:
                if k < 0:
                    raise RuntimeError('reg_orders must be nonnegative')
                self._reg_orders = [(k, 0), (0, k)]
        else:
            self._reg_orders = list()
            for pair in reg_orders:
                k, l = int(pair[0]), int(pair[1])
                if (k < 0) or (l < 0):
                    raise RuntimeError('reg_orders must be nonnegative')
                self._reg_orders.append((int(pair[0]), int(pair[1])))
        self._fitted = False


    def _make_knots(self):
        self._xdelta = (self._xb - self._xa) / self._nx
        self._ydelta = (self._yb - self._ya) / self._ny
        self._xknots = np.linspace(
            self._xa - (self._p * self._xdelta),
            self._xb + (self._p * self._xdelta),
            self._nx + 1 + (2 * self._p),
        )
        self._yknots = np.linspace(
            self._ya - (self._p * self._ydelta),
            self._yb + (self._p * self._ydelta),
            self._ny + 1 + (2 * self._p),
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
            'nx': int(self._nx),
            'ny': int(self._ny),
            'fixed_n': bool(self._fixed_n),
            'xa': float(self._xa),
            'xb': float(self._xb),
            'ya': float(self._ya),
            'yb': float(self._yb),
            'fixed_bounds': bool(self._fixed_bounds),
            'p': int(self._p),
            'coeffs': [[float(c) for c in r] for r in self._coeffs],
            'reg_orders': list(self._reg_orders),
            'fitted': bool(self._fitted),
        }
    

    def load_params(self, p):
        self._nx = int(p['nx'])
        self._ny = int(p['ny'])
        self._fixed_n = bool(p['fixed_n'])
        self._xa, self._xb = float(p['xa']), float(p['xb'])
        self._ya, self._yb = float(p['ya']), float(p['yb'])
        self._fixed_bounds = bool(p['fixed_bounds'])
        self._p = int(p['p'])
        self._make_knots()
        self._coeffs = np.asarray(p['coeffs'], dtype=float)
        self._reg_orders = list(p['reg_orders'])
        self._fitted = bool(p['fitted'])


    def _get_fitting_matrix(self, x, y, data_tags):

        N_data = x.shape[0]
        if not self._fixed_bounds:
            self._xa, self._xb = np.min(x), np.max(x)
            self._ya, self._yb = np.min(y), np.max(y)
        if not self._fixed_n:
            aspect_ratio = (self._yb - self._ya) / (self._xb - self._xa)
            self._nx = int(np.ceil(np.sqrt(N_data / aspect_ratio)))
            self._ny = int(np.ceil(aspect_ratio * self._nx))
        self._make_knots()
        N_params_x = self._nx + self._p
        N_params_y = self._ny + self._p

        A_mat = np.zeros((N_data, N_params_x, N_params_y), dtype=float)
        for tag in FITTING_TAGS:
            idx = (data_tags == tag)
            if tag == 'f':
                x_basis = _generate_B(x[idx], self._xknots, self._xdelta,
                                      self._p)
                y_basis = _generate_B(y[idx], self._yknots, self._ydelta,
                                      self._p)
                A_mat[idx,:,:] = x_basis.T[:,:,None] * y_basis.T[:,None,:]
            elif tag == 'dfdx':
                if self._p < 1:
                    raise RuntimeError('Unable to fit to gradient data, try '
                                       'increasing order')
                x_basis = _generate_B(x[idx], self._xknots, self._xdelta,
                                      self._p - 1)
                x_basis_p = (x_basis[:-1] - x_basis[1:]) / self._xdelta
                y_basis = _generate_B(y[idx], self._yknots, self._ydelta,
                                      self._p)
                A_mat[idx,:,:] = x_basis_p.T[:,:,None] * y_basis.T[:,None,:]
            elif tag == 'dfdy':
                if self._p < 1:
                    raise RuntimeError('Unable to fit to gradient data, try '
                                       'increasing order')
                x_basis = _generate_B(x[idx], self._xknots, self._xdelta,
                                      self._p)
                y_basis = _generate_B(y[idx], self._yknots, self._ydelta,
                                      self._p - 1)
                y_basis_p = (y_basis[:-1] - y_basis[1:]) / self._ydelta
                A_mat[idx,:,:] = x_basis.T[:,:,None] * y_basis_p.T[:,None,:]
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
                A_mat[idx,:,:] = x_basis_pp.T[:,:,None] * y_basis.T[:,None,:]
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
                A_mat[idx,:,:] = x_basis_p.T[:,:,None] * y_basis_p.T[:,None,:]
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
                A_mat[idx,:,:] = x_basis.T[:,:,None] * y_basis_pp.T[:,None,:]
        return A_mat.reshape((N_data, N_params_x * N_params_y))
    

    def _get_regularization_matrix(self):
        if len(self._reg_orders) == 0:
            return None
        N_params_x = self._nx + self._p
        N_params_y = self._ny + self._p
        matrices = list()
        for k, l in self._reg_orders:
            if (k + l) == 0:
                matrices.append(np.eye(N_params_x * N_params_y, dtype=float))
            else:
                mat = np.eye(N_params_x * N_params_y, dtype=float).reshape(
                    (N_params_x, N_params_y, N_params_x, N_params_y)
                )
                for _ in range(k):
                    mat -= np.roll(mat, shift=1, axis=2)
                for _ in range(l):
                    mat -= np.roll(mat, shift=1, axis=3)
                matrices.append(mat.reshape((N_params_x * N_params_y,
                                             N_params_x * N_params_y)))
        return np.concat(matrices, axis=0)
    

    def _set_fitted_params(self, params):
        N_params_x = self._nx + self._p
        N_params_y = self._ny + self._p
        self._coeffs = params.reshape((N_params_x, N_params_y))
        self._fitted = True

        