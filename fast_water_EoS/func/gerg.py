"""
The GERG-2008 dimensionless free energy as a function of reduced density and
inverse reduced temperature.
"""

import numpy as np


class GERG2008:
    """Represents the dimensionless free energy function `f(x, y)` from the
    GERG-2008 equation of state for liquid water, in terms of the reduced
    density `x` and inverse reduced temperature `y`. The critical point is at
    `x = y = 1` by construction."""

    def __init__(self):
        
        self._m1 =   8.20352069
        self._m2 = -11.996306443
        self._m3 =   3.00392

        self._m4, self._t4 = 0.01059, 0.415386589
        self._m5, self._t5 = 0.98763, 1.763895929
        self._m6, self._t6 = 3.06904, 3.874803739

        self._n1 =   0.82728408749586
        self._n2 =  -1.8602220416584
        self._n3 =  -1.1199009613744
        self._n4 =   0.15635753976056
        self._n5 =   0.87375844859025
        self._n6 =  -0.36674403715731
        self._n7 =   0.053987893432436
        self._n8 =   1.0957690214499
        self._n9 =   0.053213037828563
        self._n10 =  0.013050533930825
        self._n11 = -0.41079520434476
        self._n12 =  0.1463744334412
        self._n13 = -0.055726838623719
        self._n14 = -0.0112017741438
        self._n15 = -0.0066062758068099
        self._n16 =  0.0046918522004538


    def f(self, x, y):
        x2 = x**2
        x3 = x**3
        x4 = x**4
        x5 = x**5
        return (np.log(x)
                + self._m1
                + (self._m2 * y)
                + (self._m3 * np.log(y))
                + (self._m4 * np.log(np.abs(np.sinh(self._t4 * y))))
                - (self._m5 * np.log(np.cosh(self._t5 * y)))
                + (self._m6 * np.log(np.abs(np.sinh(self._t6 * y))))
                + (self._n1 * (y**0.5) * x)
                + (self._n2 * (y**1.25) * x)
                + (self._n3 * (y**1.875) * x)
                + (self._n4 * (y**0.125) * x2)
                + (self._n5 * (y**1.5) * x2)
                + (self._n6 * y * x3)
                + (self._n7 * (y**0.75) * x4)
                + (self._n8 * (y**1.5) * x * np.exp(-x))
                + (self._n9 * (y**0.625) * x5 * np.exp(-x))
                + (self._n10 * (y**2.625) * x5 * np.exp(-x))
                + (self._n11 * (y**5) * x * np.exp(-x2))
                + (self._n12 * (y**4) * x2 * np.exp(-x2))
                + (self._n13 * (y**4.5) * x4 * np.exp(-x2))
                + (self._n14 * (y**3) * x4 * np.exp(-x3))
                + (self._n15 * (y**4) * x * np.exp(-x5))
                + (self._n16 * (y**6) * x * np.exp(-x5)))


    def dfdx(self, x, y):
        x2 = x**2
        x3 = x**3
        x4 = x**4
        x5 = x**5
        return ((1.0 / x)
                + (self._n1 * (y**0.5))
                + (self._n2 * (y**1.25))
                + (self._n3 * (y**1.875))
                + (2 * self._n4 * (y**0.125) * x)
                + (2 * self._n5 * (y**1.5) * x)
                + (3 * self._n6 * y * x2)
                + (4 * self._n7 * (y**0.75) * x3)
                + (self._n8 * (y**1.5) * (1 - x) * np.exp(-x))
                + (self._n9 * (y**0.625) * x4 * (5 - x) * np.exp(-x))
                + (self._n10 * (y**2.625) * x4 * (5 - x) * np.exp(-x))
                + (self._n11 * (y**5) * (1 - (2 * x2)) * np.exp(-x2))
                + (2 * self._n12 * (y**4) * x * (1 - x2) * np.exp(-x2))
                + (2 * self._n13 * (y**4.5) * x3 * (2 - x2) * np.exp(-x2))
                + (self._n14 * (y**3) * x3 * (4 - (3 * x3)) * np.exp(-x3))
                + (self._n15 * (y**4) * (1 - (5 * x5)) * np.exp(-x5))
                + (self._n16 * (y**6) * (1 - (5 * x5)) * np.exp(-x5)))


    def dfdy(self, x, y):
        x2 = x**2
        x3 = x**3
        x4 = x**4
        x5 = x**5
        return (self._m2
                + (self._m3 / y)
                + (self._m4 * self._t4 / np.tanh(self._t4 * y))
                - (self._m5 * self._t5 * np.tanh(self._t5 * y))
                + (self._m6 * self._t6 / np.tanh(self._t6 * y))
                + (0.5 * self._n1 * (y**(-0.5)) * x)
                + (1.25 * self._n2 * (y**0.25) * x)
                + (1.875 * self._n3 * (y**0.875) * x)
                + (0.125 * self._n4 * (y**(-0.875)) * x2)
                + (1.5 * self._n5 * (y**0.5) * x2)
                + (self._n6 * x3)
                + (0.75 * self._n7 * (y**(-0.25)) * x4)
                + (1.5 * self._n8 * (y**0.5) * x * np.exp(-x))
                + (0.625 * self._n9 * (y**(-0.375)) * x5 * np.exp(-x))
                + (2.625 * self._n10 * (y**1.625) * x5 * np.exp(-x))
                + (5 * self._n11 * (y**4) * x * np.exp(-x2))
                + (4 * self._n12 * (y**3) * x2 * np.exp(-x2))
                + (4 * self._n13 * (y**3.5) * x4 * np.exp(-x2))
                + (3 * self._n14 * (y**2) * x4 * np.exp(-x3))
                + (4 * self._n15 * (y**3) * x * np.exp(-x5))
                + (6 * self._n16 * (y**5) * x * np.exp(-x5)))


    def ddfdxx(self, x, y):
        x2 = x**2
        x3 = x**3
        x4 = x**4
        x5 = x**5
        x6 = x**6
        poly1 = x5 - (10 * x4) + (20 * x3)
        poly2 = (2 * x4) - (5 * x2) + 1
        poly3 = x2 * ((2 * x4) - (9 * x2) + 6)
        poly4 = x2 * ((3 * x6) - (10 * x3) + 4)
        poly5 = x4 * ((5 * x5) - 6)
        return ((-1.0 / x2)
                + (2 * self._n4 * (y**0.125))
                + (2 * self._n5 * (y**1.5))
                + (6 * self._n6 * y * x)
                + (12 * self._n7 * (y**0.75) * x2)
                + (self._n8 * (y**1.5) * (x - 2) * np.exp(-x))
                + (self._n9 * (y**0.625) * poly1 * np.exp(-x))
                + (self._n10 * (y**2.625) * poly1 * np.exp(-x))
                + (2 * self._n11 * (y**5) * x * ((2 * x2) - 3) * np.exp(-x2))
                + (2 * self._n12 * (y**4) * poly2 * np.exp(-x2))
                + (2 * self._n13 * (y**4.5) * poly3 * np.exp(-x2))
                + (3 * self._n14 * (y**3) * poly4 * np.exp(-x3))
                + (5 * self._n15 * (y**4) * poly5 * np.exp(-x5))
                + (5 * self._n16 * (y**6) * poly5 * np.exp(-x5)))
    

    def ddfdxy(self, x, y):
        x2 = x**2
        x3 = x**3
        x4 = x**4
        x5 = x**5
        poly1 = (5 * x4) - x5
        return ((0.5 * self._n1 * (y**(-0.5)))
                + (1.25 * self._n2 * (y**0.25))
                + (1.875 * self._n3 * (y**0.875))
                + (0.25 * self._n4 * (y**(-0.875)) * x)
                + (3 * self._n5 * (y**0.5) * x)
                + (3 * self._n6 * x2)
                + (3 * self._n7 * (y**(-0.25)) * x3)
                + (1.5 * self._n8 * (y**0.5) * (1 - x) * np.exp(-x))
                + (0.625 * self._n9 * (y**(-0.375)) * poly1 * np.exp(-x))
                + (2.625 * self._n10 * (y**1.625) * poly1 * np.exp(-x))
                + (5 * self._n11 * (y**4) * (1 - (2 * x2)) * np.exp(-x2))
                + (8 * self._n12 * (y**3) * x * (1 - x2) * np.exp(-x2))
                + (9 * self._n13 * (y**3.5) * x3 * (2 - x2) * np.exp(-x2))
                + (3 * self._n14 * (y**2) * x3 * (4 - (3 * x3)) * np.exp(-x3))
                + (4 * self._n15 * (y**3) * (1 - (5 * x5)) * np.exp(-x5))
                + (6 * self._n16 * (y**5) * (1 - (5 * x5)) * np.exp(-x5)))
    

    def ddfdyy(self, x, y):
        x2 = x**2
        x3 = x**3
        x4 = x**4
        x5 = x**5
        return ((-self._m3 / (y**2))
                - (self._m4 * (self._t4**2) / (np.sinh(self._t4 * y)**2))
                - (self._m5 * (self._t5**2) / (np.cosh(self._t5 * y)**2))
                - (self._m6 * (self._t6**2) / (np.sinh(self._t6 * y)**2))
                - (0.25 * self._n1 * (y**(-1.5)) * x)
                + (0.3125 * self._n2 * (y**(-0.75)) * x)
                + (1.640625 * self._n3 * (y**(-0.125)) * x)
                - (0.109375 * self._n4 * (y**(-1.875)) * x2)
                + (0.75 * self._n5 * (y**(-0.5)) * x2)
                - (0.1875 * self._n7 * (y**(-1.25)) * x4)
                + (0.75 * self._n8 * (y**(-0.5)) * x * np.exp(-x))
                - (0.234375 * self._n9 * (y**(-1.375)) * x5 * np.exp(-x))
                + (4.265625 * self._n10 * (y**0.625) * x5 * np.exp(-x))
                + (20 * self._n11 * (y**3) * x * np.exp(-x2))
                + (12 * self._n12 * (y**2) * x2 * np.exp(-x2))
                + (15.75 * self._n13 * (y**2.5) * x4 * np.exp(-x2))
                + (6 * self._n14 * y * x4 * np.exp(-x3))
                + (12 * self._n15 * (y**2) * x * np.exp(-x5))
                + (30 * self._n16 * (y**4) * x * np.exp(-x5)))

