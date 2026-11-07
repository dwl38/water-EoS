"""
Constructs the free energy surface `F(V, T)` in terms of volume `V` and
temperature `T` from a series of unbiased NVT simulations at varying
thermodynamic state points.
"""

import numpy as np
from typing import Literal
from numpy._typing import _AnyShape


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
        ...
    

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
        ...


    def load_from_file(self, file):
        """Loads this free energy surface from a YAML file.

        Parameters
        ----------
        file : file-like object, string, or pathlib.Path
            The file to load from.
        """
        ...


    def save_to_file(self, file):
        """Saves this free energy surface to a YAML file.

        Parameters
        ----------
        file : file, str, or pathlib.Path
            The file to save to.
        """
        ...


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
        ...
        

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
        ...


    def internal_energy(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
            lazy : bool = True,
        ) -> np.ndarray[_AnyShape, np.dtype[float]]:
        """Returns the internal energy `U = F + TS`, in units of eV, at the
        specified volume and temperature."""
        ...


    def internal_energy_unc(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]]:
        """Returns the uncertainty of the internal energy `U = F + TS`, in
        units of eV, at the specified volume and temperature."""
        ...


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
        ...


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
        ...
    

    def pressure(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
            lazy : bool = True,
        ) -> np.ndarray[_AnyShape, np.dtype[float]]:
        """Returns the pressure `p = -∂F/∂V`, in units of bar, at the
        specified volume and temperature."""
        ...
    

    def pressure_unc(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]]:
        """Returns the uncertainty of the pressure `p = -∂F/∂V`, in units of
        bar, at the specified volume and temperature."""
        ...


    def heat_capacity(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
            lazy : bool = True,
        ) -> np.ndarray[_AnyShape, np.dtype[float]]:
        """Returns the isochoric heat capacity `C = ∂U/∂T`, in units of eV/K,
        at the specified volume and temperature."""
        ...


    def heat_capacity_unc(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]]:
        """Returns the uncertainty of the isochoric heat capacity `C = ∂U/∂T`,
        in units of eV/K, at the specified volume and temperature."""
        ...
    

    def compressibility(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
            lazy : bool = True,
        ) -> np.ndarray[_AnyShape, np.dtype[float]]:
        """Returns the isothermal compressibility `β = -(∂V/∂p) / V`, in units
        of bar⁻¹, at the specified volume and temperature."""
        ...
    

    def compressibility_unc(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]]:
        """Returns the uncertainty of the isothermal compressibility `β =
        -(∂V/∂p) / V`, in units of bar⁻¹, at the specified volume and
        temperature."""
        ...


    def thermal_pressure_coefficient(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
            lazy : bool = True,
        ) -> np.ndarray[_AnyShape, np.dtype[float]]:
        """Returns the isochoric thermal pressure coefficient `γ = ∂p/∂T`, in
        units of bar/K, at the specified volume and temperature."""
        ...


    def thermal_pressure_coefficient_unc(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]]:
        """Returns the uncertainty of the isochoric thermal pressure
        coefficient `γ = ∂p/∂T`, in units of bar/K, at the specified volume
        and temperature."""
        ...


    def thermal_expansion_coefficient(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
            lazy : bool = True,
        ) -> np.ndarray[_AnyShape, np.dtype[float]]:
        """Returns the isobaric thermal expansion coefficient `α = (∂V/∂T)
        / V`, in units of K⁻¹, at the specified volume and temperature."""
        ...


    def thermal_expansion_coefficient_unc(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]]:
        """Returns the uncertainty of the isobaric thermal expansion
        coefficient `α = (∂V/∂T) / V`, in units of K⁻¹, at the specified
        volume and temperature."""
        ...


    def isobaric_heat_capacity(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
            lazy : bool = True,
        ) -> np.ndarray[_AnyShape, np.dtype[float]]:
        """Returns the isobaric heat capacity, in units of eV/K, at the
        specified volume and temperature."""
        ...


    def isobaric_heat_capacity_unc(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]]:
        """Returns the uncertainty of the isobaric heat capacity, in units of
        eV/K, at the specified volume and temperature."""
        ...
    

    def isentropic_bulk_modulus(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
            lazy : bool = True,
        ) -> np.ndarray[_AnyShape, np.dtype[float]]:
        """Returns the isentropic bulk modulus `K = -V(∂p/∂V)`, in units of
        bar, at the specified volume and temperature."""
        ...
    

    def isentropic_bulk_modulus_unc(
            self,
            V: np.ndarray[_AnyShape, np.dtype[float]] | float,
            T: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]]:
        """Returns the uncertainty of the isentropic bulk modulus `K =
        -V(∂p/∂V)`, in units of bar, at the specified volume and temperature.
        """
        ...
    

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
        ...
