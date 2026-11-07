"""
Constructs the free energy surface `F(V, T)` in terms of volume `V` and
temperature `T` from a series of unbiased NVT simulations at varying
thermodynamic state points.
"""

import yaml
import warnings
import numpy as np
from typing import Literal
from numpy._typing import _AnyShape
from scipy.spatial import ConvexHull, QhullError
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
    extrapolate : bool, optional
        Whether or not to enable extrapolation, for flexible models; defaults
        to False. Evaluation of any observable outside the convex hull of the
        fitting data will return NaN if extrapolation is disabled. (Note that,
        even if `extrapolate` is set to True, the extrapolated observables
        are not physically meaningful for flexible models!) This parameter is
        ignored for fixed presets.

    Notes
    -----
    FreeEnergySurface includes a range of methods for calculating various
    thermodynamic observables; for example, the `pressure(V, T)` method
    calculates the pressures corresponding to the given volumes `V` and
    temperatures `T`. These methods share certain behaviors:

    - Each method is accompanied by another corresponding method, with
    `_unc` appended to its name, which reports the associated uncertainty
    based on statistical bootstrapping of an ensemble of models; for example,
    `pressure(V, T)` has `pressure_unc(V, T)`.

        - For fixed models, e.g. IAPWS-95 and GERG-2008, the reported
        uncertainty is always zero.

    - Regardless of the input shape, the output is always returned as a 1D
    ndarray, corresponding to flattened copies of the input `V` and `T`. This
    includes when `V` and `T` are provided as single floating point numbers.

    - For flexible models, if the `extrapolate` option is turned off (which is
    default behavior), input points lying outside of the convex hull of the
    provided fitting data will return NaN.

        - Otherwise, for fixed models, or for flexible models with the
        `extrapolate` option explicitly turned on, any input will always
        return a valid (but not necessarily meaningful) output.

    - Each method has a `lazy` option, which is enabled by default; this
    option triggers lazy evaluation, where only the nominal model is used for
    evaluation (thereby saving a factor of `N_models` in computational cost).
    Disabling lazy evaluation results in taking averages across the ensemble
    of models, which may be more accurate but costly. This option only affects
    flexible models. Uncertainty predictors do not have the `lazy` option, as
    they are always evaluated non-lazily.
    """

    def __init__(
            self,
            method: Literal['bspline', 'poly', 'iapws', 'gerg'] = None,
            dof: int | float = 6,
            N_models: int = 100,
            extrapolate: bool = False,
        ):
        if method is None or method == 'bspline':
            self._main_model = _FES('bspline', dof=dof)
            self._N_models = int(N_models)
            self._method = 'bspline'
            self._dof = float(dof)
            self._extrapolate = bool(extrapolate)
        elif method == 'poly':
            self._main_model = _FES('poly', dof=dof)
            self._N_models = int(N_models)
            self._method = 'poly'
            self._dof = float(dof)
            self._extrapolate = bool(extrapolate)
        elif method == 'iapws':
            self._main_model = _FES('iapws')
            self._N_models = 0
            self._method = 'iapws'
            self._dof = 6.0
            self._extrapolate = True
        elif method == 'gerg':
            self._main_model = _FES('gerg')
            self._N_models = 0
            self._method = 'gerg'
            self._dof = 6.0
            self._extrapolate = True
        else:
            raise RuntimeError(f'method "{method}" not recognized!')
        self._committee = None
        self._convex_hull_eqns = None
    

    def fit_to_data(
            self,
            V: np.ndarray[tuple[int], np.dtype[float]],
            T: np.ndarray[tuple[int], np.dtype[float]],
            p: np.ndarray[tuple[int], np.dtype[float]],
            p_err: np.ndarray[tuple[int], np.dtype[float]],
            U: np.ndarray[tuple[int], np.dtype[float]],
            U_err: np.ndarray[tuple[int], np.dtype[float]],
            Cv: np.ndarray[tuple[int], np.dtype[float]] = None,
            Cv_err: np.ndarray[tuple[int], np.dtype[float]] = None,
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
        """
        self._main_model.fit_to_data(V, T, p, p_err, U, U_err, Cv, Cv_err)

        if not self._extrapolate:
            V_cast, T_cast = np.broadcast_arrays(V, T)
            data = np.column_stack([V_cast.ravel(), T_cast.ravel()])
            try:
                self._convex_hull_eqns = ConvexHull(data).equations
            except QhullError:
                warnings.warn('Failed to construct convex hull, extrapolation'
                              ' defaulted to true', UserWarning)
                self._extrapolate = True

        if self._N_models > 0:
            rng = np.random.default_rng()
            self._committee = list()
            for _ in range(self._N_models):
                new_p = p + rng.normal(scale=p_err)
                new_U = U + rng.normal(scale=U_err)
                model = _FES(self._method, self._dof)
                if Cv is None:
                    model.fit_to_data(V, T, new_p, p_err, new_U, U_err)
                else:
                    new_Cv = Cv + rng.normal(scale=Cv_err)
                    model.fit_to_data(V, T, new_p, p_err, new_U, U_err,
                                      new_Cv, Cv_err)
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
                    self._committee.append(model)
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
        if 'extrapolate' in params:
            self._extrapolate = bool(params['extrapolate'])
            if 'convex_hull_eqns' in params:
                self._convex_hull_eqns = np.asarray(
                    params['convex_hull_eqns'], dtype=float,
                )
        else:
            self._extrapolate = True
            self._convex_hull_eqns = None


    def save_to_file(self, file):
        """Saves this free energy surface to a YAML file.

        Parameters
        ----------
        file : file, str, or pathlib.Path
            The file to save to.
        """
        params = {
            'method': str(self._method),
            'extrapolate': bool(self._extrapolate),
        }
        if self._method in ('bspline', 'poly'):
            params['N_models'] = int(self._N_models)
            params['dof'] = float(self._dof)
            params['main'] = self._main_model.save_to_dict()
            if self._committee is not None:
                params['committee'] = [model.save_to_dict()
                                       for model in self._committee]
        if not self._extrapolate and self._convex_hull_eqns is not None:
            params['convex_hull_eqns'] = self._convex_hull_eqns.tolist()
        with open(file, 'w') as fp:
            yaml.safe_dump(params, fp)


    def _mask_and_cast(self, V, T):
        V_cast, T_cast = np.broadcast_arrays(V, T)
        V_cast, T_cast = V_cast.ravel(), T_cast.ravel()
        if self._extrapolate or self._convex_hull_eqns is None:
            return np.full_like(V_cast, True), V_cast, T_cast
        points = np.column_stack([V_cast, T_cast])
        signed_dist = ((points @ self._convex_hull_eqns[:,:-1].T)
                       + self._convex_hull_eqns[:,-1])
        return np.all(signed_dist < 1.0e-12, axis=1), V_cast, T_cast


    def free_energy(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
            lazy : bool = True,
        ) -> np.ndarray[_AnyShape, np.dtype[float]]:
        """Returns the free energy `F(T, V)`, in units of eV, at the specified
        volume and temperature.
        
        Note that the absolute value of free energy is meaningless, as the
        underlying partition function is arbitrary up to a multiplicative
        constant; only differences in free energies are meaningful."""
        mask, V_cast, T_cast = self._mask_and_cast(V, T)
        if lazy or self._committee is None:
            return np.where(
                mask,
                self._main_model.free_energy(V_cast, T_cast),
                np.nan,
            )
        results = [model.free_energy(V_cast, T_cast)
                   for model in self._committee]
        results.append(self._main_model.free_energy(V_cast, T_cast))
        return np.where(mask, np.mean(results, axis=0), np.nan)
        

    def free_energy_unc(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]]:
        """Returns the uncertainty of the free energy `F(T, V)`, in units of
        eV, at the specified volume and temperature.
        
        Note that the absolute value of free energy is meaningless, as the
        underlying partition function is arbitrary up to a multiplicative
        constant; only differences in free energies are meaningful."""
        mask, V_cast, T_cast = self._mask_and_cast(V, T)
        if self._committee is None:
            return np.zeros_like(V_cast)
        results = [model.free_energy(V_cast, T_cast)
                   for model in self._committee]
        results.append(self._main_model.free_energy(V_cast, T_cast))
        return np.where(mask, np.std(results, axis=0), np.nan)


    def internal_energy(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
            lazy : bool = True,
        ) -> np.ndarray[_AnyShape, np.dtype[float]]:
        """Returns the internal energy `U = F + TS`, in units of eV, at the
        specified volume and temperature."""
        mask, V_cast, T_cast = self._mask_and_cast(V, T)
        if lazy or self._committee is None:
            return np.where(
                mask,
                self._main_model.internal_energy(V_cast, T_cast),
                np.nan,
            )
        results = [model.internal_energy(V_cast, T_cast)
                   for model in self._committee]
        results.append(self._main_model.internal_energy(V_cast, T_cast))
        return np.where(mask, np.mean(results, axis=0), np.nan)


    def internal_energy_unc(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]]:
        """Returns the uncertainty of the internal energy `U = F + TS`, in
        units of eV, at the specified volume and temperature."""
        mask, V_cast, T_cast = self._mask_and_cast(V, T)
        if self._committee is None:
            return np.zeros_like(V_cast)
        results = [model.internal_energy(V_cast, T_cast)
                   for model in self._committee]
        results.append(self._main_model.internal_energy(V_cast, T_cast))
        return np.where(mask, np.std(results, axis=0), np.nan)


    def entropy(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
            lazy : bool = True,
        ) -> np.ndarray[_AnyShape, np.dtype[float]]:
        """Returns the entropy `S = -∂F/∂T`, in units of eV/K, at the
        specified volume and temperature.
        
        Note that the absolute value of entropy is meaningless, as the
        underlying partition function is arbitrary up to a multiplicative
        constant; only differences in entropies are meaningful."""
        mask, V_cast, T_cast = self._mask_and_cast(V, T)
        if lazy or self._committee is None:
            return np.where(
                mask,
                self._main_model.entropy(V_cast, T_cast),
                np.nan,
            )
        results = [model.entropy(V_cast, T_cast)
                   for model in self._committee]
        results.append(self._main_model.entropy(V_cast, T_cast))
        return np.where(mask, np.mean(results, axis=0), np.nan)


    def entropy_unc(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]]:
        """Returns the uncertainty of the entropy `S = -∂F/∂T`, in units of
        eV/K, at the specified volume and temperature.
        
        Note that the absolute value of entropy is meaningless, as the
        underlying partition function is arbitrary up to a multiplicative
        constant; only differences in entropies are meaningful."""
        mask, V_cast, T_cast = self._mask_and_cast(V, T)
        if self._committee is None:
            return np.zeros_like(V_cast)
        results = [model.entropy(V_cast, T_cast)
                   for model in self._committee]
        results.append(self._main_model.entropy(V_cast, T_cast))
        return np.where(mask, np.std(results, axis=0), np.nan)
    

    def pressure(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
            lazy : bool = True,
        ) -> np.ndarray[_AnyShape, np.dtype[float]]:
        """Returns the pressure `p = -∂F/∂V`, in units of bar, at the
        specified volume and temperature."""
        mask, V_cast, T_cast = self._mask_and_cast(V, T)
        if lazy or self._committee is None:
            return np.where(
                mask,
                self._main_model.pressure(V_cast, T_cast),
                np.nan,
            )
        results = [model.pressure(V_cast, T_cast)
                   for model in self._committee]
        results.append(self._main_model.pressure(V_cast, T_cast))
        return np.where(mask, np.mean(results, axis=0), np.nan)
    

    def pressure_unc(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]]:
        """Returns the uncertainty of the pressure `p = -∂F/∂V`, in units of
        bar, at the specified volume and temperature."""
        mask, V_cast, T_cast = self._mask_and_cast(V, T)
        if self._committee is None:
            return np.zeros_like(V_cast)
        results = [model.pressure(V_cast, T_cast)
                   for model in self._committee]
        results.append(self._main_model.pressure(V_cast, T_cast))
        return np.where(mask, np.std(results, axis=0), np.nan)


    def heat_capacity(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
            lazy : bool = True,
        ) -> np.ndarray[_AnyShape, np.dtype[float]]:
        """Returns the isochoric heat capacity `C = ∂U/∂T`, in units of eV/K,
        at the specified volume and temperature."""
        mask, V_cast, T_cast = self._mask_and_cast(V, T)
        if lazy or self._committee is None:
            return np.where(
                mask,
                self._main_model.heat_capacity(V_cast, T_cast),
                np.nan,
            )
        results = [model.heat_capacity(V_cast, T_cast)
                   for model in self._committee]
        results.append(self._main_model.heat_capacity(V_cast, T_cast))
        return np.where(mask, np.mean(results, axis=0), np.nan)


    def heat_capacity_unc(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]]:
        """Returns the uncertainty of the isochoric heat capacity `C = ∂U/∂T`,
        in units of eV/K, at the specified volume and temperature."""
        mask, V_cast, T_cast = self._mask_and_cast(V, T)
        if self._committee is None:
            return np.zeros_like(V_cast)
        results = [model.heat_capacity(V_cast, T_cast)
                   for model in self._committee]
        results.append(self._main_model.heat_capacity(V_cast, T_cast))
        return np.where(mask, np.std(results, axis=0), np.nan)
    

    def compressibility(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
            lazy : bool = True,
        ) -> np.ndarray[_AnyShape, np.dtype[float]]:
        """Returns the isothermal compressibility `β = -(∂V/∂p) / V`, in units
        of bar⁻¹, at the specified volume and temperature."""
        mask, V_cast, T_cast = self._mask_and_cast(V, T)
        if lazy or self._committee is None:
            return np.where(
                mask,
                self._main_model.compressibility(V_cast, T_cast),
                np.nan,
            )
        results = [model.compressibility(V_cast, T_cast)
                   for model in self._committee]
        results.append(self._main_model.compressibility(V_cast, T_cast))
        return np.where(mask, np.mean(results, axis=0), np.nan)
    

    def compressibility_unc(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]]:
        """Returns the uncertainty of the isothermal compressibility `β =
        -(∂V/∂p) / V`, in units of bar⁻¹, at the specified volume and
        temperature."""
        mask, V_cast, T_cast = self._mask_and_cast(V, T)
        if self._committee is None:
            return np.zeros_like(V_cast)
        results = [model.compressibility(V_cast, T_cast)
                   for model in self._committee]
        results.append(self._main_model.compressibility(V_cast, T_cast))
        return np.where(mask, np.std(results, axis=0), np.nan)


    def thermal_pressure_coefficient(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
            lazy : bool = True,
        ) -> np.ndarray[_AnyShape, np.dtype[float]]:
        """Returns the isochoric thermal pressure coefficient `γ = ∂p/∂T`, in
        units of bar/K, at the specified volume and temperature."""
        mask, V_cast, T_cast = self._mask_and_cast(V, T)
        if lazy or self._committee is None:
            return np.where(
                mask,
                self._main_model.thermal_pressure_coefficient(V_cast, T_cast),
                np.nan,
            )
        results = [model.thermal_pressure_coefficient(V_cast, T_cast)
                   for model in self._committee]
        results.append(
            self._main_model.thermal_pressure_coefficient(V_cast, T_cast),
        )
        return np.where(mask, np.mean(results, axis=0), np.nan)


    def thermal_pressure_coefficient_unc(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]]:
        """Returns the uncertainty of the isochoric thermal pressure
        coefficient `γ = ∂p/∂T`, in units of bar/K, at the specified volume
        and temperature."""
        mask, V_cast, T_cast = self._mask_and_cast(V, T)
        if self._committee is None:
            return np.zeros_like(V_cast)
        results = [model.thermal_pressure_coefficient(V_cast, T_cast)
                   for model in self._committee]
        results.append(
            self._main_model.thermal_pressure_coefficient(V_cast, T_cast),
        )
        return np.where(mask, np.std(results, axis=0), np.nan)


    def thermal_expansion_coefficient(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
            lazy : bool = True,
        ) -> np.ndarray[_AnyShape, np.dtype[float]]:
        """Returns the isobaric thermal expansion coefficient `α = (∂V/∂T)
        / V`, in units of K⁻¹, at the specified volume and temperature."""
        mask, V_cast, T_cast = self._mask_and_cast(V, T)
        if lazy or self._committee is None:
            return np.where(
                mask,
                self._main_model.thermal_expansion_coefficient(V_cast, T_cast),
                np.nan,
            )
        results = [model.thermal_expansion_coefficient(V_cast, T_cast)
                   for model in self._committee]
        results.append(
            self._main_model.thermal_expansion_coefficient(V_cast, T_cast),
        )
        return np.where(mask, np.mean(results, axis=0), np.nan)


    def thermal_expansion_coefficient_unc(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]]:
        """Returns the uncertainty of the isobaric thermal expansion
        coefficient `α = (∂V/∂T) / V`, in units of K⁻¹, at the specified
        volume and temperature."""
        mask, V_cast, T_cast = self._mask_and_cast(V, T)
        if self._committee is None:
            return np.zeros_like(V_cast)
        results = [model.thermal_expansion_coefficient(V_cast, T_cast)
                   for model in self._committee]
        results.append(
            self._main_model.thermal_expansion_coefficient(V_cast, T_cast),
        )
        return np.where(mask, np.std(results, axis=0), np.nan)


    def isobaric_heat_capacity(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
            lazy : bool = True,
        ) -> np.ndarray[_AnyShape, np.dtype[float]]:
        """Returns the isobaric heat capacity, in units of eV/K, at the
        specified volume and temperature."""
        mask, V_cast, T_cast = self._mask_and_cast(V, T)
        if lazy or self._committee is None:
            return np.where(
                mask,
                self._main_model.isobaric_heat_capacity(V_cast, T_cast),
                np.nan,
            )
        results = [model.isobaric_heat_capacity(V_cast, T_cast)
                   for model in self._committee]
        results.append(
            self._main_model.isobaric_heat_capacity(V_cast, T_cast),
        )
        return np.where(mask, np.mean(results, axis=0), np.nan)


    def isobaric_heat_capacity_unc(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]]:
        """Returns the uncertainty of the isobaric heat capacity, in units of
        eV/K, at the specified volume and temperature."""
        mask, V_cast, T_cast = self._mask_and_cast(V, T)
        if self._committee is None:
            return np.zeros_like(V_cast)
        results = [model.isobaric_heat_capacity(V_cast, T_cast)
                   for model in self._committee]
        results.append(
            self._main_model.isobaric_heat_capacity(V_cast, T_cast),
        )
        return np.where(mask, np.std(results, axis=0), np.nan)
    

    def isentropic_bulk_modulus(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
            lazy : bool = True,
        ) -> np.ndarray[_AnyShape, np.dtype[float]]:
        """Returns the isentropic bulk modulus `K = -V(∂p/∂V)`, in units of
        bar, at the specified volume and temperature."""
        mask, V_cast, T_cast = self._mask_and_cast(V, T)
        if lazy or self._committee is None:
            return np.where(
                mask,
                self._main_model.isentropic_bulk_modulus(V_cast, T_cast),
                np.nan,
            )
        results = [model.isentropic_bulk_modulus(V_cast, T_cast)
                   for model in self._committee]
        results.append(
            self._main_model.isentropic_bulk_modulus(V_cast, T_cast),
        )
        return np.where(mask, np.mean(results, axis=0), np.nan)
    

    def isentropic_bulk_modulus_unc(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]]:
        """Returns the uncertainty of the isentropic bulk modulus `K =
        -V(∂p/∂V)`, in units of bar, at the specified volume and temperature.
        """
        mask, V_cast, T_cast = self._mask_and_cast(V, T)
        if self._committee is None:
            return np.zeros_like(V_cast)
        results = [model.isentropic_bulk_modulus(V_cast, T_cast)
                   for model in self._committee]
        results.append(
            self._main_model.isentropic_bulk_modulus(V_cast, T_cast),
        )
        return np.where(mask, np.std(results, axis=0), np.nan)
    

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
        result = self._main_model.volume_at_p(p, T, init_guess=init_guess,
                                              tol=tol, max_iter=max_iter)
        if self._extrapolate or self._convex_hull_eqns is None:
            return result
        point = np.asarray([result, T], dtype=float)
        signed_dist = ((point @ self._convex_hull_eqns[:,:-1].T)
                       + self._convex_hull_eqns[:,-1])
        if np.max(signed_dist) < 1.0e-12:
            return result
        return np.nan
