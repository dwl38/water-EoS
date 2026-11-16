"""
Constructs the free energy surface `F(V, T)` in terms of volume `V` and
temperature `T` from a series of unbiased NVT simulations at varying
thermodynamic state points.
"""

import warnings
import numpy as np
from typing import Literal
from numpy._typing import _AnyShape
from .func.ideal_gas_plus_bspline import IdealGasPlusBSpline
from .func.iapws import IAPWS95
from .func.gerg import GERG2008
from .constants import (K_B, PRESSURE_CONVERSION, CRITICAL_VOLUME,
                        CRITICAL_TEMPERATURE)


class _FES:
    """Represents a free energy surface `F(V, T)`. The fit is performed in
    terms of the dimensionless free energy as a function of reduced density
    and reduced inverse temperature:
    ```
    x  =  V₀ / V ,
    y  =  T₀ / T ,
    F(V, T)  =  kT f(x, y) ,
    ```

    where `V₀`  and `T₀` are arbitrary reference volumes and temperatures
    respectively.

    Parameters
    ----------
    model : {'bspline', 'iapws', 'gerg'}, optional
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
            method: Literal['bspline', 'iapws', 'gerg'] = None,
            dof: int | float = 6,
        ):
        if method is None or method == 'bspline':
            self.f = None
            self._dof = float(dof)
            self._method = 'bspline'
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
        self._V_ref = CRITICAL_VOLUME
        self._T_ref = CRITICAL_TEMPERATURE
        self._U_offset = 0.0
    

    def fit_to_data(
            self,
            V: np.ndarray[tuple[int], np.dtype[float]],
            T: np.ndarray[tuple[int], np.dtype[float]],
            p: np.ndarray[tuple[int], np.dtype[float]],
            p_err: np.ndarray[tuple[int], np.dtype[float]],
            U: np.ndarray[tuple[int], np.dtype[float]],
            U_err: np.ndarray[tuple[int], np.dtype[float]],
            Cv: np.ndarray[tuple[int], np.dtype[float]],
            Cv_err: np.ndarray[tuple[int], np.dtype[float]],
            U_offset: float = 0.0,
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
        Cv : array_like of shape (N_data,), optional
            Isochoric heat capacities corresponding to the provided volumes
            and temperatures, in eV/K. Not necessary for fitting, but improves
            model curvature when provided.
        Cv_err : array_like of shape (N_data,), optional
            Uncertainties of isochoric heat capacities corresponding to the
            provided volumes and temperatures, in eV/K. Must be provided if
            `Cv` is provided.
        U_offset : float, optional
            Offset to subtract off internal energies during fitting.
        """
        if self._fixed_preset:
            self._V_ref = CRITICAL_VOLUME
            self._T_ref = CRITICAL_TEMPERATURE
            self._U_offset = 0.0
            warnings.warn('Cannot fit onto fixed functional form!',
                          UserWarning)
            return None
        
        inputs = np.broadcast_arrays(V, T, p, p_err, U, U_err, Cv, Cv_err)
        fit_V = inputs[0].astype(float).ravel()
        fit_T = inputs[1].astype(float).ravel()
        fit_p = inputs[2].astype(float).ravel() / PRESSURE_CONVERSION
        sigma_p = inputs[3].astype(float).ravel() / PRESSURE_CONVERSION
        fit_U = inputs[4].astype(float).ravel() - self._U_offset
        sigma_U = inputs[5].astype(float).ravel()
        fit_Cv = inputs[6].astype(float).ravel()
        sigma_Cv = inputs[7].astype(float).ravel()
        
        self._V_ref = float(min(
            2.0 * np.min(fit_V),
            1.0 / np.mean(1.0 / fit_V),
        ))
        self._T_ref = float(min(
            2.0 * np.min(fit_T),
            1.0 / np.mean(1.0 / fit_T),
        ))
        self._U_offset = float(U_offset)

        if Cv is None:

            x = CRITICAL_VOLUME / fit_V
            y = CRITICAL_TEMPERATURE / fit_T
            dfdx = fit_p * fit_V / (K_B * fit_T * x)
            dfdx_err = np.abs(sigma_p * fit_V / (K_B * fit_T * x))
            dfdy = fit_U / (K_B * CRITICAL_TEMPERATURE)
            dfdy_err = np.abs(sigma_U / (K_B * CRITICAL_TEMPERATURE))
            
            xx = np.concat((x, x), axis=0)
            yy = np.concat((y, y), axis=0)
            data = np.concat((dfdx, dfdy), axis=0)
            data_err = np.concat((dfdx_err, dfdy_err), axis=0)
            data_tags = np.asarray(
                (['dfdx',] * dfdx.shape[0]) + (['dfdy',] * dfdy.shape[0]),
                dtype='<U6',
            )

        x = self._V_ref / fit_V
        y = self._T_ref / fit_T

        dfdx = fit_p * fit_V / (K_B * fit_T * x)
        dfdx_err = np.abs(sigma_p * fit_V / (K_B * fit_T * x))

        dfdy = (fit_U - self._U_offset) / (K_B * self._T_ref)
        dfdy_err = np.abs(sigma_U / (K_B * self._T_ref))

        idx = np.logical_and(np.isfinite(fit_Cv), np.isfinite(sigma_Cv))
        ddfdyy = -fit_Cv[idx] / (K_B * (y[idx]**2))
        ddfdyy_err = np.abs(sigma_Cv[idx] / (K_B * (y[idx]**2)))
        
        xx = np.concat((x, x, x[idx]), axis=0)
        yy = np.concat((y, y, y[idx]), axis=0)
        data = np.concat((dfdx, dfdy, ddfdyy), axis=0)
        data_err = np.concat((dfdx_err, dfdy_err, ddfdyy_err), axis=0)
        data_tags = np.asarray(
            ((['dfdx',] * dfdx.shape[0]) + (['dfdy',] * dfdy.shape[0])
                + (['ddfdyy',] * ddfdyy.shape[0])), dtype='<U6',
        )

        N_points = x.shape[0]
        N_knots = min(round(np.sqrt(N_points)), 10)
        self.f = IdealGasPlusBSpline(dof=self._dof, n=N_knots,
                                     extent=(0.0, 2.0, 0.0, 2.0))
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
            self._dof = float(params['dof'])
            if 'f' in params:
                self.f = IdealGasPlusBSpline()
                self.f.load_params(params['f'])
            else:
                self.f = None
            self._method = 'bspline'
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
        self._V_ref = (CRITICAL_VOLUME if 'V_ref' not in params
                       else float(params['V_ref']))
        self._T_ref = (CRITICAL_TEMPERATURE if 'T_ref' not in params
                       else float(params['T_ref']))
        self._U_offset = (0.0 if 'U_offset' not in params
                          else float(params['U_offset']))


    def save_to_dict(self):
        """Saves this free energy surface to a dict."""
        params = {'method': str(self._method)}
        if not self._fixed_preset:
            params['dof'] = float(self._dof)
            if self.f is not None:
                params['V_ref'] = float(self._V_ref)
                params['T_ref'] = float(self._T_ref)
                params['U_offset'] = float(self._U_offset)
                params['f'] = self.f.get_params()
        return params


    def _check_valid_and_convert_to_xy(self, V, T):
        if self.f is None:
            raise RuntimeError('method must be specified at construction!')
        return self._V_ref / V, self._T_ref / T


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
        return (K_B * T * self.f.f(x, y)) + self._U_offset
        

    def internal_energy(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]] | float:
        """Returns the internal energy `U = F + TS`, in units of eV, at the
        specified volume and temperature."""
        x, y = self._check_valid_and_convert_to_xy(V, T)
        return (K_B * T * y * self.f.dfdy(x, y)) + self._U_offset


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
        target = p * self._V_ref / (PRESSURE_CONVERSION * K_B * T)
        x = 1. if init_guess is None else (self._V_ref / init_guess)
        y = self._T_ref / T
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
                return self._V_ref / x
        return np.nan
