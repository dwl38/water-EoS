"""
Subpackage related to the generic implementation of twice-differentiable
parameterizable 2D real functions.

The base class of `ParameterizableFunc` is an abstract class (or "interface")
promising these behaviours:

- They represent a real function f(x, y), which is twice-differentiable with
continuous second derivatives;

- The value of the function f(x, y), and its first-order and second-order
partial derivatives, can be evaluated at any point, supplied as a pair of
either floating point numbers or `numpy` arrays;

- There is a `get_params()` method, which extracts the function's internal
representation in the form of a Python dictionary (which can then be further
written to a YAML file), and also a `load_params(...)` method, which allows
internal representation to be reconstructed consistently from the dictionary;

- There is a `fit(...)` method, which allows the internal representation of
the function to be best-fitted to some other known function g(x, y), whose
data is supplied as a sequence of independent noisy measurements of either
g(x, y), its first-order partial derivatives, its second-order partial
derivatives, or any mixture of those.

As such, the f(x, y) represented by the `ParameterizableFunc` is intended to
serve as an interpolant for the g(x, y) which it is fitted to. Subclasses then
represent specific implementations of an interpolation method, for example the
`Polynomial` subclass for a finite polynomial series or the `BSpline` subclass
for a piecewise polynomial spline.

Notes
-----
The generality of the interface allows for "hard-coded", i.e. fixed-form,
functions to be subclassed within `ParameterizableFunc`, which are *not* meant
to be used as interpolators. These are usually useful for testing purposes, or
to be able to insert known reference functions into the same evaluation
frameworks as the flexibly-fitted functions.

Loosely speaking, subclasses of `ParameterizableFunc` can be informally
divided into two categories of intention, namely "flexible" and "fixed-form":

- "Flexible" functions are meant to serve as best-fitted interpolations of a
source function g(x, y), while "fixed-form" functions are prescribed as fixed
references.

- Calling either the `fit(...)` or `load_params(...)` method on a "flexible"
function is an expected part of its normal usage, whereas calling either of
those two methods on a "fixed-form" function raises a UserWarning and performs
no action. On the flipside, a "fixed-form" function can be evaluated as soon
as it is instantiated, whereas a "flexible" function will first default to an
arbitrary choice of initial parameters when first instantiated, and will raise
a UserWarning upon evaluation if `fit(...)` or `load_params(...)` was not yet
performed (in which case the returned results are completely arbitrary).
"""

import warnings
import numpy as np
from abc import ABC, abstractmethod
from typing import Any
from numpy._typing import _AnyShape
from scipy.optimize import minimize_scalar


FITTING_TAGS = [
    'f',
    'dfdx',
    'dfdy',
    'ddfdxx',
    'ddfdxy',
    'ddfdyy',
]
"""The `fit(...)` method for `ParameterizableFunc` requires fitting data to be
tagged using strings drawn from `FITTING_TAGS`."""


_TYPE_REGISTRY: dict[str, type] = {}
"""Registry of all subclasses of `ParameterizableFunc`, keyed by their fully
qualified names, allowing classes to be recovered from plain strings."""


def _lookup_type(name: str, base: type) -> type:
    """Returns the registered subclass of `base` with qualified name `name`.

    Raises
    ------
    ValueError
        If no such class is registered, or it is not a subclass of `base`.
    """
    cls = _TYPE_REGISTRY.get(str(name))
    if cls is None:
        raise ValueError(f'Unrecognized function type: {name} (has its '
                         'module been imported?)')
    if not issubclass(cls, base):
        raise ValueError(f'{name} is not a subclass of {base.__name__}')
    return cls


class ParameterizableFunc(ABC):
    """Represents a real function f(x, y) on some rectangular domain, which is
    twice-differentiable with continuous second derivatives. The function may
    have a "flexible" internal representation, in which case it can be fitted
    to limited data from another source function to construct an interpolation
    of the source function; best-fitting is done via least-squares linear
    regression, possibly with regularization.

    Methods
    -------
    f(x, y) :
        Evaluate the function at the given inputs.
    dfdx(x, y) :
        Evaluate the function's partial x derivative at the given inputs.
    dfdy(x, y) :
        Evaluate the function's partial y derivative at the given inputs.
    ddfdxx(x, y) :
        Evaluate the function's second partial x derivative at the given
        inputs.
    ddfdxy(x, y) :
        Evaluate the function's second mixed derivative at the given inputs.
    ddfdyy(x, y) :
        Evaluate the function's second partial y derivative at the given
        inputs.
    get_params() :
        Extract the function's internal representation as a standard dict.
    load_params(p) :
        Load the function's internal representation from a standard dict.
    fit(x, y, data, data_err, data_tags) :
        Best-fit the function's internal representation to limited data from a
        source function, to construct an interpolation of the source function.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _TYPE_REGISTRY[f'{cls.__module__}.{cls.__qualname__}'] = cls

    @classmethod
    def _type_name(cls) -> str:
        """Returns the fully qualified name of this class."""
        return f'{cls.__module__}.{cls.__qualname__}'

    @abstractmethod
    def f(
            self,
            x: np.ndarray[_AnyShape, np.dtype[float]] | float,
            y: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]] | float:
        """Returns `f` at the given `(x, y)` points."""
        ...

    @abstractmethod
    def dfdx(
            self,
            x: np.ndarray[_AnyShape, np.dtype[float]] | float,
            y: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]] | float:
        """Returns `df/dx` at the given `(x, y)` points."""
        ...

    @abstractmethod
    def dfdy(
            self,
            x: np.ndarray[_AnyShape, np.dtype[float]] | float,
            y: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]] | float:
        """Returns `df/dy` at the given `(x, y)` points."""
        ...

    @abstractmethod
    def ddfdxx(
            self,
            x: np.ndarray[_AnyShape, np.dtype[float]] | float,
            y: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]] | float:
        """Returns `d²f/dx²` at the given `(x, y)` points."""
        ...

    @abstractmethod
    def ddfdxy(
            self,
            x: np.ndarray[_AnyShape, np.dtype[float]] | float,
            y: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]] | float:
        """Returns `d²f/dxdy` at the given `(x, y)` points."""
        ...

    @abstractmethod
    def ddfdyy(
            self,
            x: np.ndarray[_AnyShape, np.dtype[float]] | float,
            y: np.ndarray[_AnyShape, np.dtype[float]] | float,
        ) -> np.ndarray[_AnyShape, np.dtype[float]] | float:
        """Returns `d²f/dy²` at the given `(x, y)` points."""
        ...
    
    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Returns the internal representation parameters as a dict, such that
        the full state of the function can be restored using the
        `load_params(...)` method."""
        ...

    @abstractmethod
    def load_params(self, p: dict[str, Any]) -> None:
        """Loads the internal representation parameters from a dict, such that
        the full state of the function can be restored from the dict returned
        by the `get_params()` method."""
        ...

    @abstractmethod
    def _get_fitting_matrix(
            self,
            x: np.ndarray[tuple[int], np.dtype[float]],
            y: np.ndarray[tuple[int], np.dtype[float]],
            data_tags: np.ndarray[tuple[int], np.dtype[str]],
        ) -> np.ndarray[tuple[int, int], np.dtype[float]]:
        """Internal method required for `fit(...)`. Returns the fitting matrix
        `A` involved in the least-squares problem `A @ p = t`, where `p` is
        the vector of internal representation parameters, and `t` is the
        target data to fit to.
        
        Parameters
        ----------
        x : ndarray of floats of shape (N_data,)
            x-coordinates of the input data; guaranteed 1D ndarray.
        y : ndarray of floats of shape (N_data,)
            y-coordinates of the input data; guaranteed 1D ndarray.
        data_tags : ndarray of str of shape (N_data,)
            String tags denoting which type of data is provided in the target
            data vector `t`; guaranteed to be supplied as a 1D ndarray. Each
            tag will be one of "f", "dfdx", "dfdy", "ddfdxx", "ddfdxy", or
            "ddfdyy".

        Returns
        -------
        A_mat : ndarray of floats of shape (N_data, N_params)
            Fitting matrix `A` required to fit this model to the given data.

        Notes
        -----
        It is useful to perform any data-dependent reconfiguration of the
        function's internal parameters within this method, if necessary, as
        `_get_fitting_matrix(...)` is called exactly once at the start of the
        `fit(...)` method, and before both `_get_regularization_matrix()` and
        `_set_fitted_params(...)`.
        """
        ...

    def _get_regularization_matrix(
            self,
        ) -> np.ndarray[tuple[int, int], np.dtype[float]] | None:
        """Optional internal method used by `fit(...)` if available. Returns
        the regularization matrix `R` involved in the least-squares problem
        `R @ p = 0`, where `p` is the vector of internal representation
        parameters.

        Returns
        -------
        reg_mat : ndarray of floats of shape (N_reg, N_params)
            Regularization matrix `R` used to regularize this model. Will be
            ignored if None.
        """
        return None

    @abstractmethod
    def _set_fitted_params(
            self,
            params: np.ndarray[tuple[int], np.dtype[float]],
        ) -> None:
        """Internal method required for `fit(...)`. Once the least-squares
        problem `A @ p = b` is solved, the vector of internal representation
        parameters is loaded back into the model using this method.
        
        Parameters
        ----------
        params : ndarray of floats of shape (N_params,)
            Solution `p` of the least-squares fitting problem; guaranteed to
            be supplied as a 1D ndarray.
        
        Notes
        -----
        The ordering of `params` should be consistent with the columns of
        `A_mat` returned by the `_get_fitting_matrix(...)` method.
        """
        ...
    
    def fit(
            self,
            x: np.ndarray[tuple[int], np.dtype[float]],
            y: np.ndarray[tuple[int], np.dtype[float]],
            data: np.ndarray[tuple[int], np.dtype[float]],
            data_err: np.ndarray[tuple[int], np.dtype[float]],
            data_tags: np.ndarray[tuple[int], np.dtype[str]] = 'f',
        ) -> None:
        """Fits this function to a set of known values.

        Parameters
        ----------
        x : array_like of floats of shape (N_data,)
            x-coordinates of the input data.
        y : array_like of floats of shape (N_data,)
            y-coordinates of the input data.
        data : array_like of floats of shape (N_data,)
            Known values at the given data points to be fitted to.
        data_err : array_like of floats of shape (N_data,)
            Uncertainties of the given data points; assumed to be independent,
            and numerically equal to the standard deviation. The least-squares
            fitting problem is normalized using inverse variance weightage, so
            it is important that `data_err` should contain realistic positive
            values.
        data_tags : array_like of str of shape (N_data,), optional
            String tags denoting which type of data was provided in `data`.
            Each tag must be one of "f", "dfdx", "dfdy", "ddfdxx", "ddfdxy",
            or "ddfdyy". Defaults to "f" for all points.
        """
        
        inputs = np.broadcast_arrays(x, y, data, data_err, data_tags)
        fit_x = inputs[0].astype(float).ravel()
        fit_y = inputs[1].astype(float).ravel()
        fit_data = inputs[2].astype(float).ravel()
        fit_data_err = inputs[3].astype(float).ravel()
        fit_tags = inputs[4].astype('<U6').ravel()
        unique_tags = np.unique(fit_tags)
        bad_tags = np.logical_not(np.isin(unique_tags, FITTING_TAGS))
        if np.any(fit_data_err <= 0.0):
            raise ValueError('data_err must be strictly positive!')
        if np.any(bad_tags):
            raise ValueError(f'Unrecognized tags: {unique_tags[bad_tags]}')
        
        A_mat = self._get_fitting_matrix(fit_x, fit_y, fit_tags)
        reg_mat = self._get_regularization_matrix()
        b_vec_norm = fit_data / fit_data_err
        A_mat_norm = np.diag(1.0 / fit_data_err) @ A_mat

        if A_mat is None:
            warnings.warn('Cannot fit onto fixed functional form!',
                          UserWarning)
            return None

        if reg_mat is None:

            best_fit = np.linalg.lstsq(A_mat_norm, b_vec_norm)[0]
            self._set_fitted_params(best_fit)

        else:

            N_reg, N_params = reg_mat.shape
            assert A_mat_norm.shape[1] == N_params

            def _try_fit(regularizer):
                A_mat_full = np.concat((A_mat_norm, regularizer * reg_mat),
                                       axis=0)
                b_vec_full = np.concat((b_vec_norm, np.zeros(N_reg)), axis=0)
                try:
                    best_fit = np.linalg.lstsq(A_mat_full, b_vec_full)[0]
                except np.linalg.LinAlgError:
                    return 1000.0
                chi_sq = np.mean((b_vec_norm - (A_mat_norm @ best_fit))**2)
                return (np.log(chi_sq))**2
            
            minimal = minimize_scalar(_try_fit)
            A_mat_full = np.concat((A_mat_norm, minimal.x * reg_mat), axis=0)
            b_vec_full = np.concat((b_vec_norm, np.zeros(N_reg)), axis=0)
            best_fit = np.linalg.lstsq(A_mat_full, b_vec_full)[0]
            self._set_fitted_params(best_fit)

