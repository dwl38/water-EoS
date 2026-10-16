"""
Constructs the free energy surface `F(V, T)` in terms of volume `V` and
temperature `T` from a series of unbiased NVT simulations at varying
thermodynamic state points.
"""

import warnings
import numpy as np
from typing import Literal
from numpy._typing import _AnyShape
from .func.additive import Additive
from .func.polynomial import Polynomial
from .func.bspline import BSpline
from .func.ideal_gas import IdealGas
from .func.iapws import IAPWS95
from .func.gerg import GERG2008
from .constants import (K_B, PRESSURE_CONVERSION, CRITICAL_VOLUME,
                        CRITICAL_TEMPERATURE)


class _FES:
    """Represents a free energy surface `F(V, T)`. The fit is performed in
    terms of the dimensionless free energy as a function of reduced density
    and reduced inverse temperature:
    ```
    x  =  V𞁞 / V ,
    y  =  T𞁞 / T ,
    F(V, T)  =  kT f(x, y) ,
    ```

    where `V𞁞 = 92.9039635 Å³`  and `T𞁞 = 647.096 K` represent the volume and
    temperature respectively of water's liquid-vapour critical point. This
    choice of nondimensionalizing `V` and `T` relative to the critical point
    is meaningful for the preset free energy surfaces (e.g. IAWPS-95 and
    GERG-2008) as they fix divergences at `x = y = 1`, but completely
    arbitrary for fitted free energy surfaces as it simply represents a
    rescaling of the variables.

    Parameters
    ----------
    model : {'bspline', 'poly', 'iapws', 'gerg'}, optional
        Model for the equation of state `f(x, y)`. Some of these choices
        correspond to flexible models, which are designed to be fitted onto
        user-supplied thermodynamic data, whereas some correspond to fixed
        presets from literature (typically based on experimental data).
        Possible options are:

        - 'bspline': A flexible model based on B-splines, which are local
          basis functions guaranteeing well-regularized interpolation.
          Specifically, the free energy function `f(x, y)` is expressed as an
          ideal gas part plus a residual part, where the residual part is
          fitted using 4th-order B-splines; the resulting model is purely
          interpolatory, and reverts back to the ideal gas when far outside
          the rectangular domain bounded by the provided data.
        - 'poly': A flexible model based on a polynomial series. Specifically,
          the free energy function `f(x, y)` is expressed as an ideal gas part
          plus a residual part, where the residual part is fitted using the
          highest-order polynomial possible. Although the resulting model
          nominally has some extrapolatory power, in practice the Runge
          phenomenon results in rapid divergences especially when far away
          from the provided data's center-of-mass.
        - 'iapws': The IAPWS-95 equation of state, a fixed preset.
        - 'gerg': The GERG-2008 equation of state, a fixed preset.

        The default option is `bspline`.
    dof : int or float, optional
        Number of degrees of freedom per particle, for the ideal gas baseline
        used in the flexible models. This parameter is ignored for fixed
        presets. Defaults to 6, which is the rigid triatomic case.
    """

    def __init__(
            self,
            method: Literal['bspline', 'poly', 'iapws', 'gerg'] = None,
            dof: int | float = 6,
        ):
        if method is None or method == 'bspline':
            self.f = None
            self._dof = float(dof)
            self._method = 'bspline'
            self._fixed_preset = False
        elif method == 'poly':
            self.f = None
            self._dof = float(dof)
            self._method = 'poly'
            self._fixed_preset = False
        elif method == 'iapws':
            self.f = IAPWS95()
            self._method = 'iapws'
            self._fixed_preset = True
        elif method == 'gerg':
            self.f = GERG2008()
            self._method = 'gerg'
            self._fixed_preset = True
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
        if self._fixed_preset:
            warnings.warn('Cannot fit onto fixed functional form!',
                          UserWarning)
            return None
        
        inputs = np.broadcast_arrays(V, T, p, p_err, U, U_err)
        fit_V = inputs[0].astype(float).ravel()
        fit_T = inputs[1].astype(float).ravel()
        fit_p = inputs[2].astype(float).ravel() / PRESSURE_CONVERSION
        sigma_p = inputs[3].astype(float).ravel() / PRESSURE_CONVERSION
        fit_U = inputs[4].astype(float).ravel()
        sigma_U = inputs[5].astype(float).ravel()

        x = CRITICAL_VOLUME / fit_V
        y = CRITICAL_TEMPERATURE / fit_T
        dfdx = fit_p * fit_V / (K_B * fit_T * x)
        dfdx_err = sigma_p * fit_V / (K_B * fit_T * x)
        dfdy = fit_U / (K_B * CRITICAL_TEMPERATURE)
        dfdy_err = sigma_U / (K_B * CRITICAL_TEMPERATURE)
        
        xx = np.concat((x, x), axis=0)
        yy = np.concat((y, y), axis=0)
        data = np.concat((dfdx, dfdy), axis=0)
        data_err = np.concat((dfdx_err, dfdy_err), axis=0)
        data_tags = np.asarray(
            (['dfdx',] * dfdx.shape[0]) + (['dfdy',] * dfdy.shape[0]),
            dtype='<U6',
        )

        if self._method == 'bspline':
            xmax, ymax = np.max(x) + np.min(x), np.max(y) + np.min(y)
            self.f = Additive(
                [IdealGas, BSpline],
                [{'dof': self._dof}, {'extent': (0.0, xmax, 0.0, ymax)}],
            )
            self.f.fit(xx, yy, data, data_err, data_tags)
        elif self._method == 'poly':
            max_order = int(np.sqrt(2.0 * data.shape[0]) - 1.5)
            self.f = Additive(
                [IdealGas, Polynomial],
                [{'dof': self._dof}, {'order': max_order}],
            )
            self.f.fit(xx, yy, data, data_err, data_tags)
        else:
            raise RuntimeError('Internal `_method` field somehow invalid: '
                               + self._method)
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
            self._dof = float(params['dof'])
            if 'f' in params:
                self.f = Additive([IdealGas, BSpline], [dict(), dict()])
                self.f.load_params(params['f'])
            else:
                self.f = None
            self._method = 'bspline'
            self._fixed_preset = False
        elif params['method'] == 'poly':
            self._dof = float(params['dof'])
            if 'f' in params:
                self.f = Additive([IdealGas, Polynomial],
                                  [dict(), {'order': 2}])
                self.f.load_params(params['f'])
            else:
                self.f = None
            self._method = 'poly'
            self._fixed_preset = False
        elif params['method'] == 'iapws':
            self.f = IAPWS95()
            self._method = 'iapws'
            self._fixed_preset = True
        elif params['method'] == 'gerg':
            self.f = GERG2008()
            self._method = 'gerg'
            self._fixed_preset = True
        else:
            raise RuntimeError(f'method "{params['method']}" not recognized!')


    def save_to_dict(self):
        """Saves this free energy surface to a dict."""
        params = {'method': str(self._method)}
        if not self._fixed_preset:
            params['dof'] = float(self._dof)
            if self.f is not None:
                params['f'] = self.f.get_params()
        return params


    def _check_valid_and_convert_to_xy(self, V, T):
        if self.f is None:
            raise RuntimeError('method must be specified at construction!')
        return CRITICAL_VOLUME / V, CRITICAL_TEMPERATURE / T


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
        return K_B * T * y * self.f.dfdy(x, y)


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
        return K_B * ((y * self.f.dfdy(x, y)) - self.f.f(x, y))
    

    def pressure(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]] | float:
        """Returns the pressure `p = -∂F/∂V`, in units of bar, at the
        specified volume and temperature."""
        x, y = self._check_valid_and_convert_to_xy(V, T)
        return PRESSURE_CONVERSION * K_B * T * x * self.f.dfdx(x, y) / V


    def heat_capacity(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]] | float:
        """Returns the isochoric heat capacity `C = ∂U/∂T`, in units of eV/K,
        at the specified volume and temperature."""
        x, y = self._check_valid_and_convert_to_xy(V, T)
        return -K_B * (y**2) * self.f.ddfdyy(x, y)
    

    def compressibility(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]] | float:
        """Returns the isothermal compressibility `β = -(∂V/∂p) / V`, in units
        of bar⁻¹, at the specified volume and temperature."""
        x, y = self._check_valid_and_convert_to_xy(V, T)
        denom = ((2. * self.f.dfdx(x, y)) + (x * self.f.ddfdxx(x, y))) * x
        return V / (PRESSURE_CONVERSION * K_B * T * denom)


    def thermal_pressure_coefficient(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]] | float:
        """Returns the isochoric thermal pressure coefficient `γ = ∂p/∂T`, in
        units of bar/K, at the specified volume and temperature."""
        x, y = self._check_valid_and_convert_to_xy(V, T)
        numer = x * (self.f.dfdx(x, y) - (y * self.f.ddfdxy(x, y)))
        return PRESSURE_CONVERSION * K_B * numer / V


    def thermal_expansion_coefficient(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]] | float:
        """Returns the isobaric thermal expansion coefficient `α = (∂V/∂T)
        / V`, in units of K⁻¹, at the specified volume and temperature."""
        x, y = self._check_valid_and_convert_to_xy(V, T)
        numer = self.f.dfdx(x, y) - (y * self.f.ddfdxy(x, y))
        denom = (2. * self.f.dfdx(x, y)) + (x * self.f.ddfdxx(x, y))
        return numer / (T * denom)


    def isobaric_heat_capacity(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]] | float:
        """Returns the isobaric heat capacity, in units of eV/K, at the
        specified volume and temperature."""
        x, y = self._check_valid_and_convert_to_xy(V, T)
        part1 = (y**2) * self.f.ddfdyy(x, y)
        numer = self.f.dfdx(x, y) - (y * self.f.ddfdxy(x, y))
        denom = (2. * self.f.dfdx(x, y)) + (x * self.f.ddfdxx(x, y))
        return K_B * ((x * (numer**2) / denom) - part1)
    

    def isentropic_bulk_modulus(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]] | float:
        """Returns the isentropic bulk modulus `K = -V(∂p/∂V)`, in units of
        bar, at the specified volume and temperature."""
        x, y = self._check_valid_and_convert_to_xy(V, T)
        part1 = ((2. * self.f.dfdx(x, y)) + (x * self.f.ddfdxx(x, y))) * x
        numer = (self.f.dfdx(x, y) - (y * self.f.ddfdxy(x, y))) * x
        denom = (y**2) * self.f.ddfdyy(x, y)
        result = part1 - ((numer**2) / denom)
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
        target = p * CRITICAL_VOLUME / (PRESSURE_CONVERSION * K_B * T)
        x = 1. if init_guess is None else (CRITICAL_VOLUME / init_guess)
        y = CRITICAL_TEMPERATURE / T
        max_value = 1. / abs(tol)
        for _ in range(max_iter):
            numer = (x**2) * self.f.dfdx(x, y)
            denom = (2. * self.f.dfdx(x, y)) + (x * self.f.ddfdxx(x, y))
            delta = (numer - target) / (x * denom)
            if abs(delta) > (0.3 * x):
                x -= np.sign(delta) * 0.3 * x
            else:
                x -= delta
            if x < tol:
                return np.inf
            if x > max_value:
                return 0.0
            if abs(delta) < tol:
                return CRITICAL_VOLUME / x
        return np.nan
