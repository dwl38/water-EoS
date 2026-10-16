"""
Constructs the free energy surface `F(V, T)` in terms of volume `V` and
temperature `T` from a series of unbiased NVT simulations at varying
thermodynamic state points.
"""

import yaml
import numpy as np
from typing import Literal
from numpy._typing import _AnyShape
from ._fes import _FES


class FreeEnergySurface:
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
    N_models : int, optional
        Number of internal models for bootstrapping estimation of flexible
        model uncertainties. This parameter is ignored for fixed presets.
        Defaults to 100.
    """

    def __init__(
            self,
            method: Literal['bspline', 'poly', 'iapws', 'gerg'] = None,
            dof: int | float = 6,
            N_models: int = 100,
        ):
        if method is None or method == 'bspline':
            self._main_model = _FES('bspline', dof=dof)
            self._committee = None
            self._N_models = int(N_models)
            self._method = 'bspline'
            self._dof = float(dof)
        elif method == 'poly':
            self._main_model = _FES('poly', dof=dof)
            self._committee = None
            self._N_models = int(N_models)
            self._method = 'poly'
            self._dof = float(dof)
        elif method == 'iapws':
            self._main_model = _FES('iapws')
            self._committee = None
            self._N_models = 0
            self._method = 'iapws'
            self._dof = 6.0
        elif method == 'gerg':
            self._main_model = _FES('gerg')
            self._committee = None
            self._N_models = 0
            self._method = 'gerg'
            self._dof = 6.0
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
        self._main_model.fit_to_data(V, T, p, p_err, U, U_err)

        if self._N_models > 0:
            rng = np.random.default_rng()
            self._committee = list()
            for _ in range(self._N_models):
                new_p = p + rng.normal(scale=p_err)
                new_U = U + rng.normal(scale=U_err)
                model = _FES(self._method, self._dof)
                model.fit_to_data(V, T, new_p, p_err, new_U, U_err)
                self._committee.append(model)
        else:
            self._committee = None


    def load_from_file(self, file):
        """Loads this free energy surface from a YAML file.

        Parameters
        ----------
        file : file-like object, string, or pathlib.Path
            The file to load from.
        """
        with open(file, 'r') as fp:
            params = yaml.safe_load(fp)
        if 'method' not in params:
            raise IOError('"method" field missing from file!')
        if params['method'] in ('bspline', 'poly'):
            self._method = str(params['method'])
            self._N_models = int(params['N_models'])
            self._dof = float(params['dof'])
            self._main_model = _FES(self._method, self._dof)
            self._main_model.load_from_dict(params['main'])
            if 'committee' in params:
                self._committee = list()
                for p in params['committee']:
                    model = _FES(self._method, self._dof)
                    model.load_from_dict(p)
                    self._committee.append(p)
                assert len(self._committee) == self._N_models
            else:
                self._committee = None
        elif params['method'] in ('iapws', 'gerg'):
            self._method = str(params['method'])
            self._main_model = _FES(self._method)
            self._committee = None
            self._N_models = 0
            self._dof = 6.0
        else:
            raise RuntimeError(f'method "{params['method']}" not recognized!')


    def save_to_file(self, file):
        """Saves this free energy surface to a YAML file.

        Parameters
        ----------
        file : file, str, or pathlib.Path
            The file to save to.
        """
        params = {'method': str(self._method)}
        if self._method in ('bspline', 'poly'):
            params['N_models'] = int(self._N_models)
            params['dof'] = float(self._dof)
            params['main'] = self._main_model.save_to_dict()
            if self._committee is not None:
                params['committee'] = [model.save_to_dict()
                                       for model in self._committee]
        with open(file, 'w') as fp:
            yaml.safe_dump(params, fp)


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
        return self._main_model.free_energy(V, T)
        

    def free_energy_unc(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]] | float:
        """Returns the uncertainty of the free energy `F(T, V)`, in units of
        eV, at the specified volume and temperature.
        
        Note that the absolute value of free energy is meaningless, as the
        underlying partition function is arbitrary up to a multiplicative
        constant; only differences in free energies are meaningful."""
        if self._committee is None:
            return np.zeros_like(V)
        results = [model.free_energy(V, T) for model in self._committee]
        results.append(self._main_model.free_energy(V, T))
        return np.std(results, axis=0)


    def internal_energy(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]] | float:
        """Returns the internal energy `U = F + TS`, in units of eV, at the
        specified volume and temperature."""
        return self._main_model.internal_energy(V, T)


    def internal_energy_unc(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]] | float:
        """Returns the uncertainty of the internal energy `U = F + TS`, in
        units of eV, at the specified volume and temperature."""
        if self._committee is None:
            return np.zeros_like(V)
        results = [model.internal_energy(V, T) for model in self._committee]
        results.append(self._main_model.internal_energy(V, T))
        return np.std(results, axis=0)


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
        return self._main_model.entropy(V, T)


    def entropy_unc(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]] | float:
        """Returns the uncertainty of the entropy `S = -∂F/∂T`, in units of
        eV/K, at the specified volume and temperature.
        
        Note that the absolute value of entropy is meaningless, as the
        underlying partition function is arbitrary up to a multiplicative
        constant; only differences in entropies are meaningful."""
        if self._committee is None:
            return np.zeros_like(V)
        results = [model.entropy(V, T) for model in self._committee]
        results.append(self._main_model.entropy(V, T))
        return np.std(results, axis=0)
    

    def pressure(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]] | float:
        """Returns the pressure `p = -∂F/∂V`, in units of bar, at the
        specified volume and temperature."""
        return self._main_model.pressure(V, T)
    

    def pressure_unc(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]] | float:
        """Returns the uncertainty of the pressure `p = -∂F/∂V`, in units of
        bar, at the specified volume and temperature."""
        if self._committee is None:
            return np.zeros_like(V)
        results = [model.pressure(V, T) for model in self._committee]
        results.append(self._main_model.pressure(V, T))
        return np.std(results, axis=0)


    def heat_capacity(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]] | float:
        """Returns the isochoric heat capacity `C = ∂U/∂T`, in units of eV/K,
        at the specified volume and temperature."""
        return self._main_model.heat_capacity(V, T)


    def heat_capacity_unc(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]] | float:
        """Returns the uncertainty of the isochoric heat capacity `C = ∂U/∂T`,
        in units of eV/K, at the specified volume and temperature."""
        if self._committee is None:
            return np.zeros_like(V)
        results = [model.heat_capacity(V, T) for model in self._committee]
        results.append(self._main_model.heat_capacity(V, T))
        return np.std(results, axis=0)
    

    def compressibility(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]] | float:
        """Returns the isothermal compressibility `β = -(∂V/∂p) / V`, in units
        of bar⁻¹, at the specified volume and temperature."""
        return self._main_model.compressibility(V, T)
    

    def compressibility_unc(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]] | float:
        """Returns the uncertainty of the isothermal compressibility `β =
        -(∂V/∂p) / V`, in units of bar⁻¹, at the specified volume and
        temperature."""
        if self._committee is None:
            return np.zeros_like(V)
        results = [model.compressibility(V, T) for model in self._committee]
        results.append(self._main_model.compressibility(V, T))
        return np.std(results, axis=0)


    def thermal_pressure_coefficient(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]] | float:
        """Returns the isochoric thermal pressure coefficient `γ = ∂p/∂T`, in
        units of bar/K, at the specified volume and temperature."""
        return self._main_model.thermal_pressure_coefficient(V, T)


    def thermal_pressure_coefficient_unc(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]] | float:
        """Returns the uncertainty of the isochoric thermal pressure
        coefficient `γ = ∂p/∂T`, in units of bar/K, at the specified volume
        and temperature."""
        if self._committee is None:
            return np.zeros_like(V)
        results = [model.thermal_pressure_coefficient(V, T)
                   for model in self._committee]
        results.append(self._main_model.thermal_pressure_coefficient(V, T))
        return np.std(results, axis=0)


    def thermal_expansion_coefficient(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]] | float:
        """Returns the isobaric thermal expansion coefficient `α = (∂V/∂T)
        / V`, in units of K⁻¹, at the specified volume and temperature."""
        return self._main_model.thermal_expansion_coefficient(V, T)


    def thermal_expansion_coefficient_unc(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]] | float:
        """Returns the uncertainty of the isobaric thermal expansion
        coefficient `α = (∂V/∂T) / V`, in units of K⁻¹, at the specified
        volume and temperature."""
        if self._committee is None:
            return np.zeros_like(V)
        results = [model.thermal_expansion_coefficient(V, T)
                   for model in self._committee]
        results.append(self._main_model.thermal_expansion_coefficient(V, T))
        return np.std(results, axis=0)


    def isobaric_heat_capacity(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]] | float:
        """Returns the isobaric heat capacity, in units of eV/K, at the
        specified volume and temperature."""
        return self._main_model.isobaric_heat_capacity(V, T)


    def isobaric_heat_capacity_unc(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]] | float:
        """Returns the uncertainty of the isobaric heat capacity, in units of
        eV/K, at the specified volume and temperature."""
        if self._committee is None:
            return np.zeros_like(V)
        results = [model.isobaric_heat_capacity(V, T)
                   for model in self._committee]
        results.append(self._main_model.isobaric_heat_capacity(V, T))
        return np.std(results, axis=0)
    

    def isentropic_bulk_modulus(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]] | float:
        """Returns the isentropic bulk modulus `K = -V(∂p/∂V)`, in units of
        bar, at the specified volume and temperature."""
        return self._main_model.isentropic_bulk_modulus(V, T)
    

    def isentropic_bulk_modulus_unc(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]] | float:
        """Returns the uncertainty of the isentropic bulk modulus `K =
        -V(∂p/∂V)`, in units of bar, at the specified volume and temperature.
        """
        if self._committee is None:
            return np.zeros_like(V)
        results = [model.isentropic_bulk_modulus(V, T)
                   for model in self._committee]
        results.append(self._main_model.isentropic_bulk_modulus(V, T))
        return np.std(results, axis=0)
    

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

        Notes
        -----
        Once the volume is found, the corresponding uncertainty can be
        estimated from the pressure uncertainty and compressibility, e.g.:
        >>> vol = fes.volume_at_p(p, T)
        >>> p_unc = fes.pressure_unc(vol, T)
        >>> compressibility = fes.compressibility(vol, T)
        >>> vol_unc = abs(vol * compressibility * p_unc)
        """
        return self._main_model.volume_at_p(p, T, init_guess=init_guess,
                                            tol=tol, max_iter=max_iter)
