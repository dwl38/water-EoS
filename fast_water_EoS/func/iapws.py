"""
The IAPWS-95 dimensionless free energy as a function of reduced density and
inverse reduced temperature.
"""

import numpy as np
from . import ParameterizableFunc


class IAPWS95(ParameterizableFunc):
    """Represents the dimensionless free energy function `f(x, y)` from the
    IAPWS-95 equation of state for liquid water, in terms of the reduced
    density `x` and inverse reduced temperature `y`. The critical point is at
    `x = y = 1` by construction."""

    def __init__(self):
        self._n01 = -8.3204464837497
        self._n02 =  6.6832105275932
        self._n03 =  3.00632
        self._ideal = [
        #   [   nᵒᵢ  ,     γᵒᵢ    ]
            [0.012436,  1.28728967],
            [0.97315 ,  3.53734222],
            [1.27950 ,  7.74073708],
            [0.96956 ,  9.24437796],
            [0.24873 , 27.5075105 ],
        ]
        self._poly = [
        #   [         nᵢ        , dᵢ,   tᵢ  ]
            [ 0.012533547935523 ,  1, -0.500],
            [ 7.8957634722828   ,  1,  0.875],
            [-8.7803203303561   ,  1,  1    ],
            [ 0.31802509345418  ,  2,  0.5  ],
            [-0.26145533859358  ,  2,  0.75 ],
            [-0.0078199751687981,  3,  0.375],
            [ 0.0088089493102134,  4,  1    ],
        ]
        self._poly_exp = [
        #   [         nᵢ         , dᵢ, tᵢ, cᵢ]
            [-0.66856572307965   ,  1,  4,  1],
            [ 0.20433810950965   ,  1,  6,  1],
            [-6.6212605039687e-5 ,  1, 12,  1],
            [-0.19232721156002   ,  2,  1,  1],
            [-0.25709043003438   ,  2,  5,  1],
            [ 0.16074868486251   ,  3,  4,  1],
            [-0.040092828925807  ,  4,  2,  1],
            [ 3.9343422603254e-7 ,  4, 13,  1],
            [-7.5941377088144e-6 ,  5,  9,  1],
            [ 5.6250979351888e-4 ,  7,  3,  1],
            [-1.5608652257135e-5 ,  9,  4,  1],
            [ 1.1537996422951e-9 , 10, 11,  1],
            [ 3.6582165144204e-7 , 11,  4,  1],
            [-1.3251180074668e-12, 13, 13,  1],
            [-6.2639586912454e-10, 15,  1,  1],
            [-0.10793600908932   ,  1,  7,  2],
            [ 0.017611491008752  ,  2,  1,  2],
            [ 0.22132295167546   ,  2,  9,  2],
            [-0.40247669763528   ,  2, 10,  2],
            [ 0.58083399985759   ,  3, 10,  2],
            [ 0.0049969146990806 ,  4,  3,  2],
            [-0.031358700712549  ,  4,  7,  2],
            [-0.74315929710341   ,  4, 10,  2],
            [ 0.47807329915480   ,  5, 10,  2],
            [ 0.020527940895948  ,  6,  6,  2],
            [-0.13636435110343   ,  6, 10,  2],
            [ 0.014180634400617  ,  7, 10,  2],
            [ 0.0083326504880713 ,  9,  1,  2],
            [-0.029052336009585  ,  9,  2,  2],
            [ 0.038615085574206  ,  9,  3,  2],
            [-0.020393486513704  ,  9,  4,  2],
            [-0.0016554050063734 ,  9,  8,  2],
            [ 0.0019955571979541 , 10,  6,  2],
            [ 1.5870308324157e-4 , 10,  9,  2],
            [-1.6388568342530e-5 , 12,  8,  2],
            [ 0.043613615723811  ,  3, 16,  3],
            [ 0.034994005463765  ,  4, 22,  3],
            [-0.076788197844621  ,  4, 23,  3],
            [ 0.022446277332006  ,  5, 23,  3],
            [-6.2689710414685e-5 , 14, 10,  4],
            [-5.5711118565645e-10,  3, 50,  6],
            [-0.19905718354408   ,  6, 44,  6],
            [ 0.31777497330738   ,  6, 46,  6],
            [-0.11841182425981   ,  6, 50,  6],
        ]
        self._gaussian = [
        #   [        nᵢ       , dᵢ, tᵢ, αᵢ,  βᵢ,  γᵢ , εᵢ]
            [ -31.306260323435,  3,  0, 20, 150, 1.21,  1],
            [  31.546140237781,  3,  1, 20, 150, 1.21,  1],
            [2521.3154341695  ,  3,  4, 20, 250, 1.25,  1],
        ]
        self._nonanalytic = [
        #   [        nᵢ       ,  aᵢ,  bᵢ ,  Bᵢ, Cᵢ,  Dᵢ,  Aᵢ ,  1/2βᵢ ]
            [-0.14874640856724, 3.5, 0.85, 0.2, 28, 700, 0.32, 1.0/0.6],
            [ 0.31806110878444, 3.5, 0.95, 0.2, 32, 800, 0.32, 1.0/0.6],
        ]
        

    def f(self, x, y):
        result = (np.log(x) + self._n01 + (self._n02 * y)
                  + (self._n03 * np.log(y)))
        for term in self._ideal:
            n, g = term
            result += n * np.log(1.0 - np.exp(-g * y))
        for term in self._poly:
            n, d, t = term
            result += n * (x**d) * (y**t)
        for term in self._poly_exp:
            n, d, t, c = term
            result += n * (x**d) * (y**t) * np.exp(-(x**c))
        for term in self._gaussian:
            n, d, t, kx, ky, y0, x0 = term
            tmp = (kx * ((x - x0)**2)) + (ky * ((y - y0)**2))
            result += n * (x**d) * (y**t) * np.exp(-tmp)
        for term in self._nonanalytic:
            n, a, b, Bfactor, kx, ky, Afactor, Aexp = term
            Dx2 = (x - 1.0)**2
            Dy = y - 1.0
            psi = np.exp(-(kx * Dx2) - (ky * (Dy**2)))
            theta = (Afactor * (Dx2**Aexp)) - Dy
            delta = (theta**2) + (Bfactor * (Dx2**a))
            result += n * (delta**b) * x * psi
        return result


    def dfdx(self, x, y):
        result = 1.0 / x
        for term in self._poly:
            n, d, t = term
            result += n * d * (x**(d - 1)) * (y**t)
        for term in self._poly_exp:
            n, d, t, c = term
            result += (n * (x**(d - 1)) * (y**t) * (d - (c * (x**c)))
                       * np.exp(-(x**c)))
        for term in self._gaussian:
            n, d, t, kx, ky, y0, x0 = term
            tmp = (kx * ((x - x0)**2)) + (ky * ((y - y0)**2))
            prefactor = (d / x) - (2.0 * kx * (x - x0))
            result += n * prefactor * (x**d) * (y**t) * np.exp(-tmp)
        for term in self._nonanalytic:
            n, a, b, Bfactor, kx, ky, Afactor, Aexp = term
            Dx2 = (x - 1.0)**2
            Dy = y - 1.0
            psi = np.exp(-(kx * Dx2) - (ky * (Dy**2)))
            theta = (Afactor * (Dx2**Aexp)) - Dy
            delta = (theta**2) + (Bfactor * (Dx2**a))
            dpsidx = -2.0 * kx * (x - 1.0) * psi
            ddeltadx = ((4.0 * Afactor * theta * Aexp * (Dx2**(Aexp - 1.0)))
                        + (2.0 * Bfactor * a * (Dx2**(a - 1.0)))) * (x - 1.0)
            ddeltabdx = b * (delta**(b - 1.0)) * ddeltadx
            result += n * (((delta**b) * (psi + (x * dpsidx)))
                           + (ddeltabdx * x * psi))
        return result


    def dfdy(self, x, y):
        result = self._n02 + (self._n03 / y)
        for term in self._ideal:
            n, g = term
            result += n * g / (np.exp(g * y) - 1.0)
        for term in self._poly:
            n, d, t = term
            result += n * t * (x**d) * (y**(t - 1))
        for term in self._poly_exp:
            n, d, t, c = term
            result += n * t * (x**d) * (y**(t - 1)) * np.exp(-(x**c))
        for term in self._gaussian:
            n, d, t, kx, ky, y0, x0 = term
            tmp = (kx * ((x - x0)**2)) + (ky * ((y - y0)**2))
            prefactor = (t / y) - (2.0 * ky * (y - y0))
            result += n * prefactor * (x**d) * (y**t) * np.exp(-tmp)
        for term in self._nonanalytic:
            n, a, b, Bfactor, kx, ky, Afactor, Aexp = term
            Dx2 = (x - 1.0)**2
            Dy = y - 1.0
            psi = np.exp(-(kx * Dx2) - (ky * (Dy**2)))
            theta = (Afactor * (Dx2**Aexp)) - Dy
            delta = (theta**2) + (Bfactor * (Dx2**a))
            dpsidy = -2.0 * ky * Dy * psi
            ddeltabdy = -2.0 * theta * b * (delta**(b - 1.0))
            result += n * x * ((ddeltabdy * psi) + ((delta**b) * dpsidy))
        return result


    def ddfdxx(self, x, y):
        result = -1.0 / (x**2)
        for term in self._poly:
            n, d, t = term
            result += n * d * (d - 1) * (x**(d - 2)) * (y**t)
        for term in self._poly_exp:
            n, d, t, c = term
            tmp = c * (x**c)
            result += (n * (x**(d - 2)) * (y**t) * np.exp(-(x**c))
                       * (((d - tmp) * (d - 1 - tmp)) - (c * tmp)))
        for term in self._gaussian:
            n, d, t, kx, ky, y0, x0 = term
            tmp = (kx * ((x - x0)**2)) + (ky * ((y - y0)**2))
            part1 = (d / x) - (2.0 * kx * (x - x0))
            prefactor = (part1**2) - (d / (x**2)) - (2.0 * kx)
            result += n * prefactor * (x**d) * (y**t) * np.exp(-tmp)
        for term in self._nonanalytic:
            n, a, b, Bfactor, kx, ky, Afactor, Aexp = term
            Dx2 = (x - 1.0)**2
            Dy = y - 1.0
            psi = np.exp(-(kx * Dx2) - (ky * (Dy**2)))
            theta = (Afactor * (Dx2**Aexp)) - Dy
            delta = (theta**2) + (Bfactor * (Dx2**a))
            dpsidx = -2.0 * kx * (x - 1.0) * psi
            ddpsidxx = ((2.0 * kx * Dx2) - 1.0) * 2.0 * kx * psi
            ddeltadx = ((4.0 * Afactor * theta * Aexp * (Dx2**(Aexp - 1.0)))
                        + (2.0 * Bfactor * a * (Dx2**(a - 1.0)))) * (x - 1.0)
            dddeltadxx = ((4.0 * Bfactor * a * (a - 1.0) * (Dx2**(a - 1.0)))
                          + (8.0 * Dx2 * ((Afactor * Aexp
                                           * (Dx2**(Aexp - 1.0)))**2))
                          + (8.0 * Afactor * theta * Aexp * (Aexp - 1.0)
                             * (Dx2**(Aexp - 1.0)))
                          + (ddeltadx / (x - 1.0)))
            ddeltabdx = b * (delta**(b - 1.0)) * ddeltadx
            dddeltabdxx = (b * (delta**(b - 2.0)) * ((delta * dddeltadxx)
                                                     + ((b - 1.0)
                                                        * (ddeltadx**2))))
            result += n * (((delta**b) * ((2.0 * dpsidx) + (x * ddpsidxx)))
                           + (2.0 * ddeltabdx * (psi + (x * dpsidx)))
                           + (dddeltabdxx * x * psi))
        return result
    

    def ddfdxy(self, x, y):
        result = 0.0
        for term in self._poly:
            n, d, t = term
            result += n * d * t * (x**(d - 1)) * (y**(t - 1))
        for term in self._poly_exp:
            n, d, t, c = term
            result += (n * t * (x**(d - 1)) * (y**(t - 1)) * np.exp(-(x**c))
                       * (d - (c * (x**c))))
        for term in self._gaussian:
            n, d, t, kx, ky, y0, x0 = term
            tmp = (kx * ((x - x0)**2)) + (ky * ((y - y0)**2))
            prefactorx = (d / x) - (2.0 * kx * (x - x0))
            prefactory = (t / x) - (2.0 * ky * (y - y0))
            result += (n * prefactorx * prefactory * (x**d) * (y**t)
                       * np.exp(-tmp))
        for term in self._nonanalytic:
            n, a, b, Bfactor, kx, ky, Afactor, Aexp = term
            Dx2 = (x - 1.0)**2
            Dy = y - 1.0
            psi = np.exp(-(kx * Dx2) - (ky * (Dy**2)))
            theta = (Afactor * (Dx2**Aexp)) - Dy
            delta = (theta**2) + (Bfactor * (Dx2**a))
            dpsidx = -2.0 * kx * (x - 1.0) * psi
            dpsidy = -2.0 * ky * Dy * psi
            ddpsidxy = 4.0 * kx * ky * (x - 1.0) * Dy * psi
            ddeltadx = ((4.0 * Afactor * theta * Aexp * (Dx2**(Aexp - 1.0)))
                        + (2.0 * Bfactor * a * (Dx2**(a - 1.0)))) * (x - 1.0)
            ddeltabdx = b * (delta**(b - 1.0)) * ddeltadx
            ddeltabdy = -2.0 * theta * b * (delta**(b - 1.0))
            dddeltabdxy = -(delta**(b - 1.0)) * ((4.0 * Afactor * b * Aexp
                                                  * (x - 1.0)
                                                  * (Dx2**(Aexp - 1.0)))
                                                 + (2.0 * theta * b
                                                    * (b - 1.0) * ddeltadx
                                                    / delta))
            result += n * (((delta**b) * (dpsidy + (x * ddpsidxy)))
                           + (x * ddeltabdx * dpsidy)
                           + (ddeltabdy * (psi + (x * dpsidx)))
                           + (dddeltabdxy * x * psi))
        return result
    

    def ddfdyy(self, x, y):
        result = -self._n03 / (y**2)
        for term in self._ideal:
            n, g = term
            result -= 0.5 * n * (g**2) / (np.cosh(g * y) - 1.0)
        for term in self._poly:
            n, d, t = term
            result += n * t * (t - 1) * (x**d) * (y**(t - 2))
        for term in self._poly_exp:
            n, d, t, c = term
            result += (n * t * (t - 1) * (x**d) * (y**(t - 2))
                       * np.exp(-(x**c)))
        for term in self._gaussian:
            n, d, t, kx, ky, y0, x0 = term
            tmp = (kx * ((x - x0)**2)) + (ky * ((y - y0)**2))
            part1 = (t / y) - (2.0 * ky * (y - y0))
            prefactor = (part1**2) - (t / (y**2)) - (2.0 * ky)
            result += n * prefactor * (x**d) * (y**t) * np.exp(-tmp)
        for term in self._nonanalytic:
            n, a, b, Bfactor, kx, ky, Afactor, Aexp = term
            Dx2 = (x - 1.0)**2
            Dy = y - 1.0
            psi = np.exp(-(kx * Dx2) - (ky * (Dy**2)))
            theta = (Afactor * (Dx2**Aexp)) - Dy
            delta = (theta**2) + (Bfactor * (Dx2**a))
            dpsidy = -2.0 * ky * Dy * psi
            ddpsidyy = ((2.0 * ky * (Dy**2)) - 1.0) * 2.0 * ky * psi
            ddeltabdy = -2.0 * theta * b * (delta**(b - 1.0))
            dddeltabdyy = ((2.0 * b * (delta**(b - 1.0)))
                           + (4.0 * (theta**2) * b * (b - 1.0)
                              * (delta*(b - 2.0))))
            result += n * x * ((dddeltabdyy * psi)
                               + (2.0 * ddeltabdy * dpsidy)
                               + ((delta**b) * ddpsidyy))
        return result
    

    def get_params(self):
        return dict()
    

    def load_params(self, p):
        pass


    def _get_fitting_matrix(self, x, y, data_tags):
        return None


    def _set_fitted_params(self, params):
        pass
        

