import pytest
import numpy as np
from fast_water_EoS.stats import statistical_inefficiency, mean_and_error, variance_and_error, covariance_and_error


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


def ar1(rng: np.random.Generator, n: int, phi: float) -> np.ndarray:
    # Autoregressive model of order 1; statistical inefficiency theoretically should be (1 + phi) / (1 - phi)
    noise = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = noise[0] / np.sqrt(1.0 - phi**2)
    for i in range(1, n):
        x[i] = (phi * x[i-1]) + noise[i]
    return x


def correlated_pair(rng: np.random.Generator, n: int, phi: float, rho: float):
    # Two autoregressive series with nonzero covariance
    a = ar1(rng, n, phi)
    b = (rho * a) + (np.sqrt(1.0 - rho**2) * ar1(rng, n, phi))
    return a, b


# -------------------------------------------------------------------------------------------------
# statistical_inefficiency


def test_stat_ineff_constant_series():
    assert statistical_inefficiency(np.full(100, 3.7)) == 1.0


def test_stat_ineff_single_point():
    assert statistical_inefficiency(np.array([2.0])) == 1.0


def test_stat_ineff_white_noise(rng):
    g = statistical_inefficiency(rng.standard_normal(20000))
    assert g >= 1.0
    assert g == pytest.approx(1.0, abs=0.15)


def test_stat_ineff_ar1(rng):
    phi = 0.9
    g = statistical_inefficiency(ar1(rng, 50000, phi))
    assert g == pytest.approx((1 + phi) / (1 - phi), rel=0.25)


def test_stat_ineff_increases_with_correlation(rng):
    gs = [statistical_inefficiency(ar1(rng, 20000, phi)) for phi in (0.0, 0.5, 0.9)]
    assert gs[0] < gs[1] < gs[2]


def test_stat_ineff_flattens_and_accepts_lists(rng):
    x = ar1(rng, 1000, 0.7)
    expected = statistical_inefficiency(x)
    assert statistical_inefficiency(x.reshape(10, 100)) == expected
    assert statistical_inefficiency(x.tolist()) == expected


def test_stat_ineff_does_not_mutate_input(rng):
    x = ar1(rng, 500, 0.7)
    x_copy = x.copy()
    statistical_inefficiency(x)
    np.testing.assert_array_equal(x, x_copy)


# -------------------------------------------------------------------------------------------------
# mean_and_error


def test_mean_and_error_uncorrelated_matches_standard_error(rng):
    x = rng.normal(5.0, 2.0, size=300)
    mean, err = mean_and_error(x, adjust_for_correlation=False)
    assert mean == pytest.approx(np.mean(x), rel=1e-12)
    assert err == pytest.approx(np.std(x, ddof=1) / np.sqrt(len(x)), rel=1e-12)
    assert isinstance(mean, float) and isinstance(err, float)


def test_mean_and_error_weights(rng):
    x = rng.normal(size=100)
    w = rng.uniform(0.5, 2.0, size=100)
    mean, _ = mean_and_error(x, weights=w, adjust_for_correlation=False)
    assert mean == pytest.approx(np.average(x, weights=w), rel=1e-12)

    # Uniform weights must reproduce the unweighted result
    uniform = mean_and_error(x, weights=np.full(100, 3.0), adjust_for_correlation=False)
    plain = mean_and_error(x, adjust_for_correlation=False)
    assert uniform == pytest.approx(plain, rel=1e-12)


def test_mean_and_error_sigmas_inverse_variance(rng):
    x = rng.normal(size=50)
    s = rng.uniform(0.5, 3.0, size=50)
    mean, err = mean_and_error(x, sigmas=s, adjust_for_correlation=False)
    assert mean == pytest.approx(np.average(x, weights=1 / s**2), rel=1e-12)
    assert err == pytest.approx(1.0 / np.sqrt(np.sum(1 / s**2)), rel=1e-12)


def test_mean_and_error_scalar_sigma_broadcasts(rng):
    x = rng.normal(size=64)
    mean, err = mean_and_error(x, sigmas=0.4, adjust_for_correlation=False)
    assert mean == pytest.approx(np.mean(x), rel=1e-12)
    assert err == pytest.approx(0.4 / np.sqrt(64), rel=1e-12)


def test_mean_and_error_weights_and_sigmas(rng):
    x = rng.normal(size=40)
    w = rng.uniform(0.5, 2.0, size=40)
    s = rng.uniform(0.5, 2.0, size=40)
    mean, err = mean_and_error(x, weights=w, sigmas=s, adjust_for_correlation=False)
    w_norm = w / w.sum()
    assert mean == pytest.approx(np.dot(w_norm, x), rel=1e-12)
    assert err == pytest.approx(np.sqrt(np.sum(w_norm**2 * s**2)), rel=1e-12)


def test_mean_and_error_correlation_white_noise(rng):
    x = rng.standard_normal(5000)
    _, naive = mean_and_error(x, adjust_for_correlation=False)
    _, adjusted = mean_and_error(x, adjust_for_correlation=True)
    assert adjusted >= naive
    assert (adjusted / naive) < 1.2


def test_mean_and_error_correlation_matches_stat_ineff(rng):
    x = ar1(rng, 20000, 0.9)
    _, naive = mean_and_error(x, adjust_for_correlation=False)
    _, adjusted = mean_and_error(x, adjust_for_correlation=True)
    g = statistical_inefficiency(x)
    assert g > 5.0
    assert adjusted > naive
    assert adjusted / naive == pytest.approx(np.sqrt(g), rel=1e-6)


def test_mean_and_error_coverage_on_correlated_data(rng):
    # About 68% of one-sigma intervals should contain the true mean (0)
    n_trials, hits = 200, 0
    for _ in range(n_trials):
        mean, err = mean_and_error(ar1(rng, 1000, 0.5))
        hits += abs(mean) <= err
    assert 0.55 <= hits / n_trials <= 0.80


def test_mean_and_error_single_point_warns():
    with pytest.warns(RuntimeWarning):
        mean, err = mean_and_error(np.array([4.2]))
    assert mean == 4.2
    assert err == 0.0


def test_mean_and_error_flattens_input(rng):
    x = rng.normal(size=(10, 20))
    assert mean_and_error(x) == mean_and_error(x.ravel())


# ---------------------------------------------------------------------------
# variance_and_error


def test_variance_matches_numpy(rng):
    x = ar1(rng, 400, 0.5)
    for adjust in (True, False):
        var, err = variance_and_error(x, adjust_for_correlation=adjust)
        assert var == pytest.approx(np.var(x, ddof=1), rel=1e-12)
        assert np.isfinite(err) and err > 0.0


def test_variance_single_point_warns():
    with pytest.warns(RuntimeWarning, match='one data point'):
        result = variance_and_error(np.array([1.0]))
    assert result == (0.0, 0.0)


@pytest.mark.parametrize('n', [2, 3])
def test_variance_fewer_than_four_points_warns(n):
    x = np.array([1.0, 2.5, 4.0])[:n]
    with pytest.warns(RuntimeWarning, match='Fewer than four'):
        var, err = variance_and_error(x)
    assert var == pytest.approx(np.var(x, ddof=1))
    assert err == 0.0


def test_variance_error_uncorrelated_gaussian(rng):
    n = 2000
    x = rng.standard_normal(n)
    var, err = variance_and_error(x, adjust_for_correlation=False)
    # Var[s^2] = 2 sigma^4 / (n - 1) for Gaussian samples
    assert err == pytest.approx(var * np.sqrt(2.0 / (n - 1)), rel=0.2)


def test_variance_error_larger_with_correlation_adjustment(rng):
    x = ar1(rng, 5000, 0.9)
    _, naive = variance_and_error(x, adjust_for_correlation=False)
    _, adjusted = variance_and_error(x, adjust_for_correlation=True)
    assert adjusted > naive


def test_variance_too_few_blocks_warns(rng):
    x = ar1(rng, 30, 0.9)
    with pytest.warns(RuntimeWarning, match='Too few blocks'):
        var, err = variance_and_error(x)
    assert np.isfinite(var) and np.isfinite(err) and err > 0.0


def test_variance_uneven_blocks(rng):
    # Length not divisible by the number of blocks exercises the remainder branch
    x = rng.standard_normal(1003)
    var, err = variance_and_error(x, adjust_for_correlation=True)
    assert var == pytest.approx(np.var(x, ddof=1), rel=1e-12)
    assert np.isfinite(err) and err > 0.0


def test_variance_constant_data():
    var, err = variance_and_error(np.full(100, 2.0))
    assert var == 0.0
    assert err == 0.0


def test_variance_shift_and_scale(rng):
    x = ar1(rng, 1000, 0.6)
    var, err = variance_and_error(x)

    var_shift, err_shift = variance_and_error(x + 1000.0)
    assert var_shift == pytest.approx(var, rel=1e-6)
    assert err_shift == pytest.approx(err, rel=1e-6)

    a = 3.0
    var_scale, err_scale = variance_and_error(a * x)
    assert var_scale == pytest.approx(a**2 * var, rel=1e-9)
    assert err_scale == pytest.approx(a**2 * err, rel=1e-9)


# ---------------------------------------------------------------------------
# covariance_and_error


def test_covariance_matches_numpy(rng):
    a, b = correlated_pair(rng, 400, 0.5, 0.6)
    for adjust in (True, False):
        cov, err = covariance_and_error(a, b, adjust_for_correlation=adjust)
        assert cov == pytest.approx(np.cov(a, b)[0, 1], rel=1e-12)
        assert np.isfinite(err) and err > 0.0
        assert isinstance(cov, float) and isinstance(err, float)


def test_covariance_reduces_to_variance(rng):
    x = ar1(rng, 600, 0.6)
    for adjust in (True, False):
        cov, cov_err = covariance_and_error(x, x, adjust_for_correlation=adjust)
        var, var_err = variance_and_error(x, adjust_for_correlation=adjust)
        assert cov == pytest.approx(var, rel=1e-10)
        assert cov_err == pytest.approx(var_err, rel=1e-10)


def test_covariance_symmetric(rng):
    a, b = correlated_pair(rng, 500, 0.7, 0.5)
    assert covariance_and_error(a, b) == pytest.approx(covariance_and_error(b, a), rel=1e-12)


def test_covariance_error_uncorrelated_gaussian(rng):
    n, rho = 3000, 0.6
    a = rng.standard_normal(n)
    b = (rho * a) + (np.sqrt(1.0 - rho**2) * rng.standard_normal(n))
    cov, err = covariance_and_error(a, b, adjust_for_correlation=False)
    assert cov == pytest.approx(rho, abs=4.0 * err)
    # Var[s_ab] = (1 + rho^2) / (n - 1) for unit-variance bivariate Gaussian samples
    assert err == pytest.approx(np.sqrt((1.0 + rho**2) / (n - 1)), rel=0.25)


def test_covariance_independent_series_consistent_with_zero(rng):
    a = rng.standard_normal(2000)
    b = rng.standard_normal(2000)
    cov, err = covariance_and_error(a, b, adjust_for_correlation=False)
    assert abs(cov) < 4.0 * err


def test_covariance_error_larger_with_correlation_adjustment(rng):
    a, b = correlated_pair(rng, 5000, 0.9, 0.6)
    _, naive = covariance_and_error(a, b, adjust_for_correlation=False)
    _, adjusted = covariance_and_error(a, b, adjust_for_correlation=True)
    assert adjusted > naive


def test_covariance_single_point_warns():
    with pytest.warns(RuntimeWarning, match='one data point'):
        result = covariance_and_error(np.array([1.0]), np.array([2.0]))
    assert result == (0.0, 0.0)


@pytest.mark.parametrize('n', [2, 3])
def test_covariance_fewer_than_four_points_warns(n):
    a = np.array([1.0, 2.5, 4.0])[:n]
    b = np.array([0.5, -1.0, 2.0])[:n]
    with pytest.warns(RuntimeWarning, match='Fewer than four'):
        cov, err = covariance_and_error(a, b)
    assert cov == pytest.approx(np.cov(a, b)[0, 1])
    assert err == 0.0


def test_covariance_too_few_blocks_warns(rng):
    a, b = correlated_pair(rng, 30, 0.9, 0.5)
    with pytest.warns(RuntimeWarning, match='Too few blocks'):
        cov, err = covariance_and_error(a, b)
    assert np.isfinite(cov) and np.isfinite(err) and err > 0.0


def test_covariance_uneven_blocks(rng):
    # Length not divisible by the number of blocks exercises the remainder branch
    a = rng.standard_normal(1003)
    b = a + rng.standard_normal(1003)
    cov, err = covariance_and_error(a, b, adjust_for_correlation=True)
    assert cov == pytest.approx(np.cov(a, b)[0, 1], rel=1e-12)
    assert np.isfinite(err) and err > 0.0


def test_covariance_empty_raises():
    with pytest.raises(ValueError, match='Empty'):
        covariance_and_error(np.array([]), np.array([]))


@pytest.mark.parametrize('nA, nB', [(10, 9), (100, 1), (1, 100), (0, 5)])
def test_covariance_length_mismatch_raises(nA, nB):
    with pytest.raises(ValueError, match='mismatch'):
        covariance_and_error(np.zeros(nA), np.zeros(nB))


def test_covariance_constant_data(rng):
    cov, err = covariance_and_error(np.full(100, 2.0), rng.standard_normal(100))
    assert cov == pytest.approx(0.0, abs=1e-12)
    assert np.isfinite(err)


def test_covariance_shift_and_scale(rng):
    a, b = correlated_pair(rng, 1000, 0.6, 0.5)
    cov, err = covariance_and_error(a, b)

    cov_shift, err_shift = covariance_and_error(a + 1000.0, b - 500.0)
    assert cov_shift == pytest.approx(cov, rel=1e-6)
    assert err_shift == pytest.approx(err, rel=1e-6)

    sa, sb = 3.0, -2.0
    cov_scale, err_scale = covariance_and_error(sa * a, sb * b)
    assert cov_scale == pytest.approx(sa * sb * cov, rel=1e-9)
    assert err_scale == pytest.approx(abs(sa * sb) * err, rel=1e-9)


def test_covariance_flattens_and_accepts_lists(rng):
    a, b = correlated_pair(rng, 1000, 0.6, 0.5)
    expected = covariance_and_error(a, b)
    assert covariance_and_error(a.reshape(10, 100), b.reshape(100, 10)) == expected
    assert covariance_and_error(a.tolist(), b.tolist()) == expected


def test_covariance_does_not_mutate_input(rng):
    a, b = correlated_pair(rng, 500, 0.7, 0.5)
    a_copy, b_copy = a.copy(), b.copy()
    covariance_and_error(a, b)
    np.testing.assert_array_equal(a, a_copy)
    np.testing.assert_array_equal(b, b_copy)
