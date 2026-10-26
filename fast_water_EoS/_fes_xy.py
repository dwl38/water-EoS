"""
Constructs the free energy surface `F(V, T)` in terms of volume `V` and
temperature `T` from a series of unbiased NVT simulations at varying
thermodynamic state points.
"""

import numpy as np
from typing import Literal
from numpy._typing import _AnyShape
from .func.additive import Additive
from .func.polynomial import Polynomial
from .func.bspline import BSpline
from .func.ideal_gas_xy import IdealGasXY
from .constants import (K_B, PRESSURE_CONVERSION, AMBIENT_VOLUME,
                        AMBIENT_TEMPERATURE)


class _FES_XY:
    """Represents a free energy surface `F(V, T)`. The fit is performed in
    terms of the dimensionless free energy as a function of logarithmic
    density `x` and logarithmic inverse temperature `y`:
    ```
    x  =  ln(V₀ / V) ,
    y  =  ln(T₀ / T) ,
    F(V, T)  =  kT f(x, y) ,
    ```

    where `V₀`  and `T₀` are arbitrary reference points.

    Parameters
    ----------
    model : {'bspline', 'poly'}, optional
        Model for the equation of state `f(x, y)`. These choices all
        correspond to flexible models, which are designed to be fitted onto
        user-supplied thermodynamic data. Possible options are:

        - 'bspline': A flexible model based on B-splines, which are local
          basis functions guaranteeing well-regularized interpolation.
          Specifically, the free energy function `f(x, y)` is expressed as an
          ideal gas part plus a residual part, where the residual part is
          fitted using 4th-order B-splines; the resulting model is purely
          interpolatory, and reverts back to the ideal gas when far outside
          the rectangular domain bounded by the provided data.
        - 'poly': A flexible model based on a polynomial series. Specifically,
          the free energy function `f(x, y)` is expressed as the highest-order
          polynomial possible. Although the resulting model nominally has some
          extrapolatory power, in practice the Runge phenomenon results in
          rapid divergences especially when far away from the provided data's
          center-of-mass.

        The default option is `bspline`.
    """

    def __init__(
            self,
            method: Literal['bspline', 'poly'] = None,
        ):
        self.f = None
        self.V_ref = AMBIENT_VOLUME
        self.T_ref = AMBIENT_TEMPERATURE
        if method is None or method == 'bspline':
            self._method = 'bspline'
        elif method == 'poly':
            self._method = 'poly'
        else:
            raise RuntimeError(f'method "{method}" not recognized!')
    

    def fit_to_data(
            self,
            V: np.ndarray[tuple[int], np.dtype[float]],
            T: np.ndarray[tuple[int], np.dtype[float]],
            p: np.ndarray[tuple[int], np.dtype[float]],
            p_err: np.ndarray[tuple[int], np.dtype[float]],
            U: np.ndarray[tuple[int], np.dtype[float]],
            U_err: np.ndarray[tuple[int], np.dtype[float]],
        ) -> None:
        """Fits this free energy surface to a set of known values.

        Parameters
        ----------
        V : array_like of shape (N_data,)
            Volumes, in Å³.
        T : array_like of shape (N_data,)
            Temperatures, in K.
        p : array_like of shape (N_data,)
            Pressures corresponding to the provided volumes and temperatures,
            in bar.
        p_err : array_like of shape (N_data,)
            Uncertainties of pressures corresponding to the provided volumes
            and temperatures, in bar.
        U : array_like of shape (N_data,)
            Internal (total) energies corresponding to the provided volumes
            and temperatures, in eV.
        U_err : array_like of shape (N_data,)
            Uncertainties of internal (total) energies corresponding to the
            provided temperatures and volumes, in eV.
        """
        inputs = np.broadcast_arrays(V, T, p, p_err, U, U_err)
        fit_V = inputs[0].astype(float).ravel()
        fit_T = inputs[1].astype(float).ravel()
        fit_p = inputs[2].astype(float).ravel() / PRESSURE_CONVERSION
        sigma_p = inputs[3].astype(float).ravel() / PRESSURE_CONVERSION
        fit_U = inputs[4].astype(float).ravel()
        sigma_U = inputs[5].astype(float).ravel()

        self.V_ref = np.exp(np.mean(np.log(fit_V)))
        self.T_ref = np.exp(np.mean(np.log(fit_T)))

        x, y = np.log(self.V_ref / fit_V), np.log(self.T_ref / fit_T)
        dfdx = fit_p * fit_V / (K_B * fit_T)
        dfdx_err = sigma_p * fit_V / (K_B * fit_T)
        dfdy = fit_U / (K_B * fit_T)
        dfdy_err = sigma_U / (K_B * fit_T)
        
        xx = np.concat((x, x), axis=0)
        yy = np.concat((y, y), axis=0)
        data = np.concat((dfdx, dfdy), axis=0)
        data_err = np.concat((dfdx_err, dfdy_err), axis=0)
        data_tags = np.asarray(
            (['dfdx',] * dfdx.shape[0]) + (['dfdy',] * dfdy.shape[0]),
            dtype='<U6',
        )

        if self._method == 'bspline':
            xmin, xmax = float(np.min(x)), float(np.max(x))
            ymin, ymax = float(np.min(y)), float(np.max(y))
            nx = int(round((xmax - xmin) / 0.08))
            ny = int(round((ymax - ymin) / 0.08))
            self.f = Additive(
                [IdealGasXY, BSpline],
                [
                    {'dof': 6.0},
                    {'extent': (xmin, xmax, ymin, ymax), 'n': (nx, ny)},
                ],
            )
        elif self._method == 'poly':
            max_order = int(np.sqrt(2.0 * data.shape[0]) - 1.5)
            self.f = Polynomial(max_order)
        self.f.fit(xx, yy, data, data_err, data_tags)
        return None


    def load_from_dict(self, params):
        """Loads this free energy surface from a dict.

        Parameters
        ----------
        params : dict
            The dict to load from.
        """
        if 'method' not in params:
            raise IOError('"method" field missing from params!')
        if params['method'] == 'bspline':
            if 'f' in params:
                self.f = Additive([IdealGasXY, BSpline],
                                  [dict(), dict()])
                self.f.load_params(params['f'])
            else:
                self.f = None
        elif params['method'] == 'poly':
            if 'f' in params:
                self.f = Polynomial(2)
                self.f.load_params(params['f'])
            else:
                self.f = None
        else:
            raise RuntimeError(f'method "{params['method']}" not recognized!')
        self.V_ref = float(params['V_ref'])
        self.T_ref = float(params['T_ref'])
        self._method = str(params['method'])


    def save_to_dict(self):
        """Saves this free energy surface to a dict."""
        params = {
            'method': str(self._method),
            'V_ref': float(self.V_ref),
            'T_ref': float(self.T_ref),
        }
        if self.f is not None:
            params['f'] = self.f.get_params()
        return params


    def _check_valid_and_convert_to_xy(self, V, T):
        if self.f is None:
            raise RuntimeError('method must be specified at construction!')
        return np.log(self.V_ref / V), np.log(self.T_ref / T)


    def free_energy(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]] | float:
        """Returns the free energy `F(T, V)`, in units of eV, at the specified
        volume and temperature.
        
        Note that the absolute value of free energy is meaningless, as the
        underlying partition function is arbitrary up to a multiplicative
        constant; only differences in free energies are meaningful."""
        x, y = self._check_valid_and_convert_to_xy(V, T)
        return K_B * T * self.f.f(x, y)
        

    def internal_energy(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]] | float:
        """Returns the internal energy `U = F + TS`, in units of eV, at the
        specified volume and temperature."""
        x, y = self._check_valid_and_convert_to_xy(V, T)
        return K_B * T * self.f.dfdy(x, y)


    def entropy(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]] | float:
        """Returns the entropy `S = -∂F/∂T`, in units of eV/K, at the
        specified volume and temperature.
        
        Note that the absolute value of entropy is meaningless, as the
        underlying partition function is arbitrary up to a multiplicative
        constant; only differences in entropies are meaningful."""
        x, y = self._check_valid_and_convert_to_xy(V, T)
        return K_B * (self.f.dfdy(x, y) - self.f.f(x, y))
    

    def pressure(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]] | float:
        """Returns the pressure `p = -∂F/∂V`, in units of bar, at the
        specified volume and temperature."""
        x, y = self._check_valid_and_convert_to_xy(V, T)
        return PRESSURE_CONVERSION * K_B * T * self.f.dfdx(x, y) / V


    def heat_capacity(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]] | float:
        """Returns the isochoric heat capacity `C = ∂U/∂T`, in units of eV/K,
        at the specified volume and temperature."""
        x, y = self._check_valid_and_convert_to_xy(V, T)
        return K_B * (self.f.dfdy(x, y) - self.f.ddfdyy(x, y))
    

    def compressibility(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]] | float:
        """Returns the isothermal compressibility `β = -(∂V/∂p) / V`, in units
        of bar⁻¹, at the specified volume and temperature."""
        x, y = self._check_valid_and_convert_to_xy(V, T)
        denom = self.f.dfdx(x, y) + self.f.ddfdxx(x, y)
        return V / (PRESSURE_CONVERSION * K_B * T * denom)


    def thermal_pressure_coefficient(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]] | float:
        """Returns the isochoric thermal pressure coefficient `γ = ∂p/∂T`, in
        units of bar/K, at the specified volume and temperature."""
        x, y = self._check_valid_and_convert_to_xy(V, T)
        numer = self.f.dfdx(x, y) - self.f.ddfdxy(x, y)
        return PRESSURE_CONVERSION * K_B * numer / V


    def thermal_expansion_coefficient(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]] | float:
        """Returns the isobaric thermal expansion coefficient `α = (∂V/∂T)
        / V`, in units of K⁻¹, at the specified volume and temperature."""
        x, y = self._check_valid_and_convert_to_xy(V, T)
        numer = self.f.dfdx(x, y) - self.f.ddfdxy(x, y)
        denom = self.f.dfdx(x, y) + self.f.ddfdxx(x, y)
        return numer / (T * denom)


    def isobaric_heat_capacity(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]] | float:
        """Returns the isobaric heat capacity, in units of eV/K, at the
        specified volume and temperature."""
        x, y = self._check_valid_and_convert_to_xy(V, T)
        part1 = self.f.dfdy(x, y) - self.f.ddfdyy(x, y)
        numer = self.f.dfdx(x, y) - self.f.ddfdxy(x, y)
        denom = self.f.dfdx(x, y) + self.f.ddfdxx(x, y)
        return K_B * (part1 + ((numer**2) / denom))
    

    def isentropic_bulk_modulus(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]] | float:
        """Returns the isentropic bulk modulus `K = -V(∂p/∂V)`, in units of
        bar, at the specified volume and temperature."""
        x, y = self._check_valid_and_convert_to_xy(V, T)
        part1 = self.f.dfdx(x, y) + self.f.ddfdxx(x, y)
        numer = self.f.dfdx(x, y) - self.f.ddfdxy(x, y)
        denom = self.f.dfdy(x, y) - self.f.ddfdyy(x, y)
        result = part1 + ((numer**2) / denom)
        return PRESSURE_CONVERSION * K_B * T * result / V
    

    def volume_at_p(
        self,
        p: float,
        T: float,
        *,
        init_guess : float = None,
        tol : float = 1.e-12,
        max_iter : int = 10000,
    ) -> float:
        """Returns the volume V at which the pressure `p(V, T)` is equal to
        the specified value.

        Parameters
        ----------
        p : float
            Pressure, in bar.
        T : float
            Temperature, in K.
        init_guess : float, optional
            Initial guess for the volume, in Å³.
        tol : float, optional
            Tolerance for the root-finding algorithm.
        max_iter : int, optional
            Maximum number of iterations for the root-finding algorithm.
        
        Returns
        -------
        V : float
            Volume, in Å³. Will default to NaN if no root is found.
        """
        target = p * self.V_ref / (PRESSURE_CONVERSION * K_B * T)
        x = 0.0 if init_guess is None else np.log(self.V_ref / init_guess)
        y = np.log(self.T_ref / init_guess)
        max_value = 1.0 / abs(tol)
        for _ in range(max_iter):
            dfdx = self.f.dfdx(x, y)
            ddfdxx = self.f.ddfdxx(x, y)
            delta = (dfdx - (target * np.exp(-x))) / (dfdx + ddfdxx)
            x -= delta
            if x > max_value:
                return 0.0
            if x < -max_value:
                return np.inf
            if abs(delta) < tol:
                return self.V_ref * np.exp(-x)
        return np.nan
