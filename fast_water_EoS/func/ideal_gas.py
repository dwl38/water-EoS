"""
The ideal gas dimensionless free energy as a function of reduced density and
inverse reduced temperature.
"""

import numpy as np
from ..misc import broadcast_and_flatten


class IdealGas:
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
        x, y = broadcast_and_flatten(x, y)
        return np.log(x) + (self._dof * np.log(y) / 2.0)


    def dfdx(self, x, y):
        x, _ = broadcast_and_flatten(x, y)
        return 1.0 / x


    def dfdy(self, x, y):
        _, y = broadcast_and_flatten(x, y)
        return self._dof / (2.0 * y)


    def ddfdxx(self, x, y):
        x, _ = broadcast_and_flatten(x, y)
        return -1.0 / (x**2)
    

    def ddfdxy(self, x, y):
        _, _, zeros = broadcast_and_flatten(x, y, 0.0)
        return zeros
    

    def ddfdyy(self, x, y):
        _, y = broadcast_and_flatten(x, y)
        return -self._dof / (2.0 * (y**2))
    

    def get_params(self, prefix=''):
        return {prefix + 'dof': float(self._dof)}
    

    def load_params(self, params, prefix=''):
        self._dof = float(params[prefix + 'dof'])

