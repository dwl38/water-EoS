"""
The ideal gas dimensionless free energy as a function of reduced density and
inverse reduced temperature.
"""

import numpy as np
from . import ParameterizableFunc, FITTING_TAGS


class IdealGas(ParameterizableFunc):
    """Represents the dimensionless free energy function `f(x, y)` for an
    ideal gas.
    
    Parameters
    ----------
    dof : int or float, optional
        Number of degrees of freedom per particle; defaults to 3, which is the
        monatomic case.
    """

    def __init__(self, dof: int | float = 3):
        self._dof = float(dof)


    def f(self, x, y):
        new_x, new_y = np.broadcast_arrays(x, y)
        return np.log(new_x) + (self._dof * np.log(new_y) / 2.0)


    def dfdx(self, x, y):
        new_x, _ = np.broadcast_arrays(x, y)
        return 1.0 / new_x


    def dfdy(self, x, y):
        _, new_y = np.broadcast_arrays(x, y)
        return self._dof / (2.0 * new_y)


    def ddfdxx(self, x, y):
        new_x, _ = np.broadcast_arrays(x, y)
        return -1.0 / (new_x**2)
    

    def ddfdxy(self, x, y):
        _, _, zeros = np.broadcast_arrays(x, y, 0.0)
        return zeros
    

    def ddfdyy(self, x, y):
        _, new_y = np.broadcast_arrays(x, y)
        return -self._dof / (2.0 * (new_y**2))
    

    def get_params(self):
        return {'dof': float(self._dof)}
    

    def load_params(self, p):
        self._dof = float(p['dof'])


    def _get_fitting_matrix(self, x, y, data_tags):
        return None


    def _get_zero_offset(self, x, y, data_tags):
        results = np.zeros_like(x)
        for tag in FITTING_TAGS:
            idx = (data_tags == tag)
            if tag == 'f':
                results[idx] += self.f(x[idx], y[idx])
            elif tag == 'dfdx':
                results[idx] += self.dfdx(x[idx], y[idx])
            elif tag == 'dfdy':
                results[idx] += self.dfdy(x[idx], y[idx])
            elif tag == 'ddfdxx':
                results[idx] += self.ddfdxx(x[idx], y[idx])
            elif tag == 'ddfdxy':
                results[idx] += self.ddfdxy(x[idx], y[idx])
            elif tag == 'ddfdyy':
                results[idx] += self.ddfdyy(x[idx], y[idx])
        return results


    def _set_fitted_params(self, params):
        pass
        

