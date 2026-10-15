"""
The GERG-2008 dimensionless free energy as a function of reduced density and
inverse reduced temperature.
"""

import numpy as np
from . import ParameterizableFunc, FITTING_TAGS


class GERG2008(ParameterizableFunc):
    """Represents the dimensionless free energy function `f(x, y)` from the
    GERG-2008 equation of state for liquid water, in terms of the reduced
    density `x` and inverse reduced temperature `y`. The critical point is at
    `x = y = 1` by construction."""

    def __init__(self):
        self._params = {
            'm_1':   8.20352069,
            'm_2': -11.996306443,        'm_3':   3.00392,
            'm_4':   0.01059,            't_4':   0.415386589,
            'm_5':   0.98763,            't_5':   1.763895929,
            'm_6':   3.06904,            't_6':   3.874803739,
            'n_1':   0.82728408749586,   'n_2':  -1.8602220416584,
            'n_3':  -1.1199009613744,    'n_4':   0.15635753976056,
            'n_5':   0.87375844859025,   'n_6':  -0.36674403715731,
            'n_7':   0.053987893432436,  'n_8':   1.0957690214499,
            'n_9':   0.053213037828563,  'n_10':  0.013050533930825,
            'n_11': -0.41079520434476,   'n_12':  0.1463744334412,
            'n_13': -0.055726838623719,  'n_14': -0.0112017741438,
            'n_15': -0.0066062758068099, 'n_16':  0.0046918522004538,
        }


    def f(self, x, y):
        x2 = x**2
        x3 = x**3
        x4 = x**4
        x5 = x**5
        return (np.log(x)
                + self._params['m_1']
                + (self._params['m_2'] * y)
                + (self._params['m_3'] * np.log(y))
                + (self._params['m_4']
                   * np.log(np.abs(np.sinh(self._params['t_4'] * y))))
                - (self._params['m_5']
                   * np.log(np.cosh(self._params['t_5'] * y)))
                + (self._params['m_6']
                   * np.log(np.abs(np.sinh(self._params['t_6'] * y))))
                + (self._params['n_1'] * (y**0.5) * x)
                + (self._params['n_2'] * (y**1.25) * x)
                + (self._params['n_3'] * (y**1.875) * x)
                + (self._params['n_4'] * (y**0.125) * x2)
                + (self._params['n_5'] * (y**1.5) * x2)
                + (self._params['n_6'] * y * x3)
                + (self._params['n_7'] * (y**0.75) * x4)
                + (self._params['n_8'] * (y**1.5) * x * np.exp(-x))
                + (self._params['n_9'] * (y**0.625) * x5 * np.exp(-x))
                + (self._params['n_10'] * (y**2.625) * x5 * np.exp(-x))
                + (self._params['n_11'] * (y**5) * x * np.exp(-x2))
                + (self._params['n_12'] * (y**4) * x2 * np.exp(-x2))
                + (self._params['n_13'] * (y**4.5) * x4 * np.exp(-x2))
                + (self._params['n_14'] * (y**3) * x4 * np.exp(-x3))
                + (self._params['n_15'] * (y**4) * x * np.exp(-x5))
                + (self._params['n_16'] * (y**6) * x * np.exp(-x5)))


    def dfdx(self, x, y):
        x2 = x**2
        x3 = x**3
        x4 = x**4
        x5 = x**5
        return ((1.0 / x)
                + (self._params['n_1'] * (y**0.5))
                + (self._params['n_2'] * (y**1.25))
                + (self._params['n_3'] * (y**1.875))
                + (2. * self._params['n_4'] * (y**0.125) * x)
                + (2. * self._params['n_5'] * (y**1.5) * x)
                + (3. * self._params['n_6'] * y * x2)
                + (4. * self._params['n_7'] * (y**0.75) * x3)
                + (self._params['n_8'] * (y**1.5) * (1.0 - x) * np.exp(-x))
                + (self._params['n_9'] * (y**0.625) * x4 * (5.0 - x)
                   * np.exp(-x))
                + (self._params['n_10'] * (y**2.625) * x4 * (5.0 - x)
                   * np.exp(-x))
                + (self._params['n_11'] * (y**5) * (1.0 - (2.0 * x2))
                   * np.exp(-x2))
                + (2. * self._params['n_12'] * (y**4) * x * (1.0 - x2)
                   * np.exp(-x2))
                + (2. * self._params['n_13'] * (y**4.5) * x3 * (2.0 - x2)
                   * np.exp(-x2))
                + (self._params['n_14'] * (y**3) * x3 * (4.0 - (3.0 * x3))
                   * np.exp(-x3))
                + (self._params['n_15'] * (y**4) * (1.0 - (5.0 * x5))
                   * np.exp(-x5))
                + (self._params['n_16'] * (y**6) * (1.0 - (5.0 * x5))
                   * np.exp(-x5)))


    def dfdy(self, x, y):
        x2 = x**2
        x3 = x**3
        x4 = x**4
        x5 = x**5
        return (self._params['m_2']
                + (self._params['m_3'] / y)
                + (self._params['m_4'] * self._params['t_4']
                   / np.tanh(self._params['t_4'] * y))
                - (self._params['m_5'] * self._params['t_5']
                   * np.tanh(self._params['t_5'] * y))
                + (self._params['m_6'] * self._params['t_6']
                   / np.tanh(self._params['t_6'] * y))
                + (0.5 * self._params['n_1'] * (y**(-0.5)) * x)
                + (1.25 * self._params['n_2'] * (y**0.25) * x)
                + (1.875 * self._params['n_3'] * (y**0.875) * x)
                + (0.125 * self._params['n_4'] * (y**(-0.875)) * x2)
                + (1.5 * self._params['n_5'] * (y**0.5) * x2)
                + (self._params['n_6'] * x3)
                + (0.75 * self._params['n_7'] * (y**(-0.25)) * x4)
                + (1.5 * self._params['n_8'] * (y**0.5) * x * np.exp(-x))
                + (0.625 * self._params['n_9'] * (y**(-0.375)) * x5
                   * np.exp(-x))
                + (2.625 * self._params['n_10'] * (y**1.625) * x5
                   * np.exp(-x))
                + (5.0 * self._params['n_11'] * (y**4) * x * np.exp(-x2))
                + (4.0 * self._params['n_12'] * (y**3) * x2 * np.exp(-x2))
                + (4.5 * self._params['n_13'] * (y**3.5) * x4 * np.exp(-x2))
                + (3.0 * self._params['n_14'] * (y**2) * x4 * np.exp(-x3))
                + (4.0 * self._params['n_15'] * (y**3) * x * np.exp(-x5))
                + (6.0 * self._params['n_16'] * (y**5) * x * np.exp(-x5)))


    def ddfdxx(self, x, y):
        x2 = x**2
        x3 = x**3
        x4 = x**4
        x5 = x**5
        x6 = x**6
        return ((-1.0 / x2)
                + (2.0 * self._params['n_4'] * (y**0.125))
                + (2.0 * self._params['n_5'] * (y**1.5))
                + (6.0 * self._params['n_6'] * y * x)
                + (12.0 * self._params['n_7'] * (y**0.75) * x2)
                + (self._params['n_8'] * (y**1.5) * (x - 2.0) * np.exp(-x))
                + (self._params['n_9'] * (y**0.625) * x3
                   * (x2 - (10.0 * x) + 20.0) * np.exp(-x))
                + (self._params['n_10'] * (y**2.625) * x3
                   * (x2 - (10.0 * x) + 20.0) * np.exp(-x))
                + (2.0 * self._params['n_11'] * (y**5) * x
                   * ((2.0 * x2) - 3.0) * np.exp(-x2))
                + (2.0 * self._params['n_12'] * (y**4)
                   * ((2.0 * x4) - (5.0 * x2) + 1.0) * np.exp(-x2))
                + (2.0 * self._params['n_13'] * (y**4.5) * x2
                   * ((2.0 * x4) - (9.0 * x2) + 6.0) * np.exp(-x2))
                + (3.0 * self._params['n_14'] * (y**3) * x2
                   * ((3.0 * x6) - (10.0 * x3) + 4.0) * np.exp(-x3))
                + (5.0 * self._params['n_15'] * (y**4) * x4
                   * ((5.0 * x5) - 6.0) * np.exp(-x5))
                + (5.0 * self._params['n_16'] * (y**6) * x4
                   * ((5.0 * x5) - 6.0) * np.exp(-x5)))
    

    def ddfdxy(self, x, y):
        x2 = x**2
        x3 = x**3
        x4 = x**4
        x5 = x**5
        return ((0.5 * self._params['n_1'] * (y**(-0.5)))
                + (1.25 * self._params['n_2'] * (y**0.25))
                + (1.875 * self._params['n_3'] * (y**0.875))
                + (0.25 * self._params['n_4'] * (y**(-0.875)) * x)
                + (3.0 * self._params['n_5'] * (y**0.5) * x)
                + (3.0 * self._params['n_6'] * x2)
                + (3.0 * self._params['n_7'] * (y**(-0.25)) * x3)
                + (1.5 * self._params['n_8'] * (y**0.5) * (1.0 - x)
                   * np.exp(-x))
                + (0.625 * self._params['n_9'] * (y**(-0.375)) * x4
                   * (5.0 - x) * np.exp(-x))
                + (2.625 * self._params['n_10'] * (y**1.625) * x4 * (5.0 - x)
                   * np.exp(-x))
                + (5.0 * self._params['n_11'] * (y**4) * (1.0 - (2.0 * x2))
                   * np.exp(-x2))
                + (8.0 * self._params['n_12'] * (y**3) * x * (1.0 - x2)
                   * np.exp(-x2))
                + (9.0 * self._params['n_13'] * (y**3.5) * x3 * (2.0 - x2)
                   * np.exp(-x2))
                + (3.0 * self._params['n_14'] * (y**2) * x3
                   * (4.0 - (3.0 * x3)) * np.exp(-x3))
                + (4.0 * self._params['n_15'] * (y**3) * (1.0 - (5.0 * x5))
                   * np.exp(-x5))
                + (6.0 * self._params['n_16'] * (y**5) * (1.0 - (5.0 * x5))
                   * np.exp(-x5)))
    

    def ddfdyy(self, x, y):
        x2 = x**2
        x3 = x**3
        x4 = x**4
        x5 = x**5
        return ((-self._params['m_3'] / (y**2))
                - (self._params['m_4'] * (self._params['t_4']**2)
                   / (np.sinh(self._params['t_4'] * y)**2))
                - (self._params['m_5'] * (self._params['t_5']**2)
                   / (np.cosh(self._params['t_5'] * y)**2))
                - (self._params['m_6'] * (self._params['t_6']**2)
                   / (np.sinh(self._params['t_6'] * y)**2))
                - (0.25 * self._params['n_1'] * (y**(-1.5)) * x)
                + (0.3125 * self._params['n_2'] * (y**(-0.75)) * x)
                + (1.640625 * self._params['n_3'] * (y**(-0.125)) * x)
                - (0.109375 * self._params['n_4'] * (y**(-1.875)) * x2)
                + (0.75 * self._params['n_5'] * (y**(-0.5)) * x2)
                - (0.1875 * self._params['n_7'] * (y**(-1.25)) * x4)
                + (0.75 * self._params['n_8'] * (y**(-0.5)) * x * np.exp(-x))
                - (0.234375 * self._params['n_9'] * (y**(-1.375)) * x5
                   * np.exp(-x))
                + (4.265625 * self._params['n_10'] * (y**0.625) * x5
                   * np.exp(-x))
                + (20.0 * self._params['n_11'] * (y**3) * x * np.exp(-x2))
                + (12.0 * self._params['n_12'] * (y**2) * x2 * np.exp(-x2))
                + (15.75 * self._params['n_13'] * (y**2.5) * x4 * np.exp(-x2))
                + (6.0 * self._params['n_14'] * y * x4 * np.exp(-x3))
                + (12.0 * self._params['n_15'] * (y**2) * x * np.exp(-x5))
                + (30.0 * self._params['n_16'] * (y**4) * x * np.exp(-x5)))
    

    def get_params(self):
        return dict()
    

    def load_params(self, p):
        pass


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

