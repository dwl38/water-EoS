import re
import yaml
import warnings
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from ._fes import _FES


def _make_observable(name):
    fn = getattr(_FES, name)

    def method(self, V, T, lazy=True):
        return self._evaluate(fn, V, T, lazy)

    def method_unc(self, V, T):
        return self._evaluate_unc(fn, V, T)

    method.__name__ = method.__qualname__ = name
    method.__doc__ = fn.__doc__

    method_unc.__name__ = method_unc.__qualname__ = name + '_unc'
    method_unc.__doc__ = re.sub(
        r'^Returns the ',
        'Returns the uncertainty of the ',
        fn.__doc__,
    )

    return method, method_unc


class FreeEnergySurface:

    def __init__(self, method=None, dof=6, N_models=100, extrapolate=False):
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
    

    def fit_to_data(self, V, T, p, p_err, U, U_err, Cv=None, Cv_err=None):

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


    def _evaluate(self, fn, V, T, lazy):
        mask, V_cast, T_cast = self._mask_and_cast(V, T)
        if lazy or self._committee is None:
            return np.where(mask, fn(self._main_model, V_cast, T_cast),
                            np.nan)
        results = [fn(model, V_cast, T_cast) for model in self._committee]
        results.append(fn(self._main_model, V_cast, T_cast))
        return np.where(mask, np.mean(results, axis=0), np.nan)


    def _evaluate_unc(self, fn, V, T):
        mask, V_cast, T_cast = self._mask_and_cast(V, T)
        if self._committee is None:
            return np.zeros_like(V_cast)
        results = [fn(model, V_cast, T_cast) for model in self._committee]
        results.append(fn(self._main_model, V_cast, T_cast))
        return np.where(mask, np.std(results, axis=0), np.nan)


    free_energy, free_energy_unc = _make_observable('free_energy')
    internal_energy, internal_energy_unc = _make_observable('internal_energy')
    entropy, entropy_unc = _make_observable('entropy')
    pressure, pressure_unc = _make_observable('pressure')
    heat_capacity, heat_capacity_unc = _make_observable('heat_capacity')
    compressibility, compressibility_unc = _make_observable('compressibility')
    (thermal_pressure_coefficient,
     thermal_pressure_coefficient_unc) = _make_observable(
        'thermal_pressure_coefficient',
    )
    (thermal_expansion_coefficient,
     thermal_expansion_coefficient_unc) = _make_observable(
        'thermal_expansion_coefficient',
    )
    isobaric_heat_capacity, isobaric_heat_capacity_unc = _make_observable(
        'isobaric_heat_capacity',
    )
    isentropic_bulk_modulus, isentropic_bulk_modulus_unc = _make_observable(
        'isentropic_bulk_modulus',
    )
    

    def volume_at_p(self, p, T, *, init_guess=None, tol=1.e-12,
                    max_iter=10000):
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
