"""
An ParameterizableFunc assembled by adding multiple ParameterizableFuncs
together.
"""

import numpy as np
from collections.abc import Iterable
from . import ParameterizableFunc, _lookup_type


class Additive(ParameterizableFunc):
    """This class provides a method to add multiple ParameterizableFuncs to
    create a single ParameterizableFuncs, whose parameters can be jointly
    fitted.

    Parameters
    ----------
    types : iterable of subclasses of ParameterizableFunc
        Iterable of subclasses of ParameterizableFunc to add together.
    kwargs : iterable of dict
        Iterable of dictionaries, containing keyword arguments to be passed to
        each ParameterizableFunc's constructor.
    """

    def __init__(
            self,
            types: Iterable[type[ParameterizableFunc]],
            kwargs: Iterable[dict],
        ):
        self._types = list()
        self._kwargs = list()
        self._models = list()
        for t, kw in zip(types, kwargs):
            if not (isinstance(t, type) and
                    issubclass(t, ParameterizableFunc)):
                raise TypeError('type must be a subclass of '
                                'ParameterizableFunc')
            self._types.append(t)
            self._kwargs.append(dict(kw))
            self._models.append(t(**self._kwargs[-1]))


    def f(self, x, y):
        return np.sum([model.f(x, y) for model in self._models], axis=0)


    def dfdx(self, x, y):
        return np.sum([model.dfdx(x, y) for model in self._models], axis=0)


    def dfdy(self, x, y):
        return np.sum([model.dfdy(x, y) for model in self._models], axis=0)


    def ddfdxx(self, x, y):
        return np.sum([model.ddfdxx(x, y) for model in self._models], axis=0)
    

    def ddfdxy(self, x, y):
        return np.sum([model.ddfdxy(x, y) for model in self._models], axis=0)
    

    def ddfdyy(self, x, y):
        return np.sum([model.ddfdyy(x, y) for model in self._models], axis=0)


    def get_params(self):
        return {
            'types': [t._type_name() for t in self._types],
            'kwargs': [dict(kw) for kw in self._kwargs],
            'params': [model.get_params() for model in self._models],
        }
    

    def load_params(self, p):
        self._types = list()
        self._kwargs = list()
        self._models = list()
        for t, kw, params in zip(p['types'], p['kwargs'], p['params']):
            self._types.append(_lookup_type(t, ParameterizableFunc))
            self._kwargs.append(dict(kw))
            self._models.append(self._types[-1](**self._kwargs[-1]))
            self._models[-1].load_params(params)


    def _get_fitting_matrix(self, x, y, data_tags):
        result = np.zeros((x.shape[0], 0), dtype=float)
        self._param_idxs = list()
        k = 0
        for model in self._models:
            submat = model._get_fitting_matrix(x, y, data_tags)
            if submat is None:
                self._param_idxs.append(None)
            else:
                result = np.concat((result, submat), axis=1)
                self._param_idxs.append(slice(k, k + submat.shape[1]))
                k += submat.shape[1]
        if result.shape[1] > 0:
            return result
        return None


    def _get_zero_offset(self, x, y, data_tags):
        return np.sum([model._get_zero_offset(x, y, data_tags)
                       for model in self._models], axis=0)


    def _get_regularization_matrix(self):
        result = np.zeros((0, 0), dtype=float)
        for model, idxs in zip(self._models, self._param_idxs):
            if idxs is None:
                continue
            submat = model._get_regularization_matrix()
            old_N_reg, old_N_params = result.shape
            if submat is None:
                new_N_params = idxs.stop - idxs.start
                result = np.concat(
                    (result, np.zeros((old_N_reg, new_N_params))),
                    axis=1,
                )
            else:
                new_N_reg, new_N_params = submat.shape
                result = np.block([
                    [result, np.zeros((old_N_reg, new_N_params))],
                    [np.zeros((new_N_reg, old_N_params)), submat]
                ])
        if result.shape[1] > 0:
            return result
        return None


    def _set_fitted_params(self, params):
        for model, idxs in zip(self._models, self._param_idxs):
            if idxs is not None:
                model._set_fitted_params(params[idxs])

        