import pytest
import numpy as np
from fast_water_EoS.func.additive import Additive
from fast_water_EoS.func.bspline import BSpline
from fast_water_EoS.func.ideal_gas import IdealGas
from fast_water_EoS.func.ideal_gas_plus_bspline import IdealGasPlusBSpline


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng()


def test_against_additive(rng: np.random.Generator):

    PLOT_X_MIN, PLOT_X_MAX = 0.5, 1.5
    SAMPLE_SIZE = 200

    plot_x = np.linspace(PLOT_X_MIN, PLOT_X_MAX, 201)
    plot_y = np.linspace(PLOT_X_MIN, PLOT_X_MAX, 201)
    plot_x, plot_y = np.meshgrid(plot_x, plot_y)
    plot_x, plot_y = plot_x.ravel(), plot_y.ravel()

    sample_x = rng.uniform(PLOT_X_MIN, PLOT_X_MAX, SAMPLE_SIZE)
    sample_y = rng.uniform(PLOT_X_MIN, PLOT_X_MAX, SAMPLE_SIZE)
    sample_f = rng.normal(0.0, 1.0, SAMPLE_SIZE)

    ref_model = Additive([IdealGas, BSpline], [{'dof': 6.0}, dict()])
    ref_model.fit(sample_x, sample_y, sample_f, 1.0, 'f')

    test_model = IdealGasPlusBSpline()
    test_model.fit(sample_x, sample_y, sample_f, 1.0, 'f')

    ref_f = ref_model.f(plot_x, plot_y)
    test_f = test_model.f(plot_x, plot_y)
    assert np.allclose(ref_f, test_f)

    ref_f = ref_model.dfdx(plot_x, plot_y)
    test_f = test_model.dfdx(plot_x, plot_y)
    assert np.allclose(ref_f, test_f)

    ref_f = ref_model.dfdy(plot_x, plot_y)
    test_f = test_model.dfdy(plot_x, plot_y)
    assert np.allclose(ref_f, test_f)

    ref_f = ref_model.ddfdxx(plot_x, plot_y)
    test_f = test_model.ddfdxx(plot_x, plot_y)
    assert np.allclose(ref_f, test_f)

    ref_f = ref_model.ddfdxy(plot_x, plot_y)
    test_f = test_model.ddfdxy(plot_x, plot_y)
    assert np.allclose(ref_f, test_f)

    ref_f = ref_model.ddfdyy(plot_x, plot_y)
    test_f = test_model.ddfdyy(plot_x, plot_y)
    assert np.allclose(ref_f, test_f)
