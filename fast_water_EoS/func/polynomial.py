"""
A 2D polynomial.
"""

import warnings
import numpy as np
from . import ParameterizableFunc, FITTING_TAGS


class Polynomial(ParameterizableFunc):
    """Represents a polynomial, centered around some (x, y) = (x₀, y₀), up to
    some order. This subclass is a "flexible" function.

    Parameters
    ----------
    order : int
        Maximum polynomial degree.
    origin : tuple[float, float], optional
        Polynomial center (x0, y0). If left blank, will be determined
        automatically during fitting.
    """

    def __init__(
            self,
            order: int,
            origin: tuple[float, float] = None,
        ):
        if order < 1:
            raise RuntimeError('order must be positive')
        self._p = int(order)
        self._coeffs = np.zeros((self._p + 1, self._p + 1), dtype=float)
        if origin is None:
            self._x0, self._y0 = 0.0, 0.0
            self._fixed_origin = False
        else:
            self._x0 = float(origin[0])
            self._y0 = float(origin[1])
            self._fixed_origin = True
        self._fitted = False


    def f(self, x, y):
        if not self._fitted:
            warnings.warn('Function not yet fitted!', UserWarning)
        new_x, new_y = np.broadcast_arrays(x, y)
        dx, dy = new_x - self._x0, new_y - self._y0
        result = np.zeros_like(new_x)
        for i in range(self._p + 1):
            for j in range(self._p + 1 - i):
                result += self._coeffs[i,j] * (dx**i) * (dy**j)
        return result


    def dfdx(self, x, y):
        if not self._fitted:
            warnings.warn('Function not yet fitted!', UserWarning)
        new_x, new_y = np.broadcast_arrays(x, y)
        dx, dy = new_x - self._x0, new_y - self._y0
        result = np.zeros_like(new_x)
        for i in range(1, self._p + 1):
            for j in range(self._p + 1 - i):
                result += self._coeffs[i,j] * i * (dx**(i - 1)) * (dy**j)
        return result


    def dfdy(self, x, y):
        if not self._fitted:
            warnings.warn('Function not yet fitted!', UserWarning)
        new_x, new_y = np.broadcast_arrays(x, y)
        dx, dy = new_x - self._x0, new_y - self._y0
        result = np.zeros_like(new_x)
        for i in range(self._p + 1):
            for j in range(1, self._p + 1 - i):
                result += self._coeffs[i,j] * j * (dx**i) * (dy**(j - 1))
        return result


    def ddfdxx(self, x, y):
        if not self._fitted:
            warnings.warn('Function not yet fitted!', UserWarning)
        new_x, new_y = np.broadcast_arrays(x, y)
        dx, dy = new_x - self._x0, new_y - self._y0
        result = np.zeros_like(new_x)
        for i in range(2, self._p + 1):
            for j in range(self._p + 1 - i):
                result += (self._coeffs[i,j] * i * (i - 1) * (dx**(i - 2))
                           * (dy**j))
        return result
    

    def ddfdxy(self, x, y):
        if not self._fitted:
            warnings.warn('Function not yet fitted!', UserWarning)
        new_x, new_y = np.broadcast_arrays(x, y)
        dx, dy = new_x - self._x0, new_y - self._y0
        result = np.zeros_like(new_x)
        for i in range(1, self._p + 1):
            for j in range(1, self._p + 1 - i):
                result += (self._coeffs[i,j] * i * j * (dx**(i - 1))
                           * (dy**(j - 1)))
        return result
    

    def ddfdyy(self, x, y):
        if not self._fitted:
            warnings.warn('Function not yet fitted!', UserWarning)
        new_x, new_y = np.broadcast_arrays(x, y)
        dx, dy = new_x - self._x0, new_y - self._y0
        result = np.zeros_like(new_x)
        for i in range(self._p + 1):
            for j in range(2, self._p + 1 - i):
                result += (self._coeffs[i,j] * j * (j - 1) * (dx**i)
                           * (dy**(j - 2)))
        return result
    

    def get_params(self):
        clean_coeffs = list()
        for i in range(self._p + 1):
            clean_coeffs.append([float(self._coeffs[i,j])
                                 for j in range(self._p + 1 - i)])
        return {
            'order': int(self._p),
            'coeffs': clean_coeffs,
            'x0': float(self._x0),
            'y0': float(self._y0),
            'fixed_origin': bool(self._fixed_origin),
            'fitted': bool(self._fitted),
        }
    

    def load_params(self, p):
        self._p = int(p['order'])
        self._coeffs = np.zeros((self._p + 1, self._p + 1), dtype=float)
        for i in range(self._p + 1):
            for j, c in enumerate(p['coeffs'][i]):
                self._coeffs[i,j] = c
        self._x0 = float(p['x0'])
        self._y0 = float(p['y0'])
        self._fixed_origin = bool(p['fixed_origin'])
        self._fitted = bool(p['fitted'])


    def _get_fitting_matrix(self, x, y, data_tags):

        N_data = x.shape[0]
        N_params = (self._p + 1) * (self._p + 2) // 2
        if N_data < N_params:
            raise RuntimeError('Insufficient data to fit parameters, try '
                               'reducing order')
        
        if not self._fixed_origin:
            self._x0, self._y0 = np.mean(x), np.mean(y)
        dx, dy = x - self._x0, y - self._y0
        A_mat_trans = list()
        for i in range(self._p + 1):
            for j in range(self._p + 1 - i):
                column = np.zeros(N_data, dtype=float)
                for tag in FITTING_TAGS:
                    idx = (data_tags == tag)
                    if tag == 'f':
                        column[idx] = (dx[idx]**i) * (dy[idx]**j)
                    elif tag == 'dfdx' and i > 0:
                        column[idx] = i * (dx[idx]**(i - 1)) * (dy[idx]**j)
                    elif tag == 'dfdy' and j > 0:
                        column[idx] = j * (dx[idx]**i) * (dy[idx]**(j - 1))
                    elif tag == 'ddfdxx' and i > 1:
                        column[idx] = (i * (i - 1) * (dx[idx]**(i - 2))
                                       * (dy[idx]**j))
                    elif tag == 'ddfdxy' and i > 0 and j > 0:
                        column[idx] = (i * j * (dx[idx]**(i - 1))
                                       * (dy[idx]**(j - 1)))
                    elif tag == 'ddfdyy' and j > 1:
                        column[idx] = (j * (j - 1) * (dx[idx]**i)
                                       * (dy[idx]**(j - 2)))
                A_mat_trans.append(column)
        return np.asarray(A_mat_trans, dtype=float).T
        

    def _set_fitted_params(self, params):
        self._coeffs = np.zeros((self._p + 1, self._p + 1), dtype=float)
        k = 0
        for i in range(self._p + 1):
            for j in range(self._p + 1 - i):
                self._coeffs[i,j] = params[k]
                k += 1
        self._fitted = True

        

