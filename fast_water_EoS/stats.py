"""
A collection of methods useful for studying the statistics of a time series
`A(t)` over uniform intervals, where samples may or may not be correlated.
"""

import warnings
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import fftconvolve


def mean_and_error(
        data: np.ndarray,
        weights: np.ndarray = None,
        sigmas: np.ndarray = None,
        adjust_for_correlation: bool = True,
    ) -> tuple[float, float]:
    """Finds the sample mean, and standard error of the sample mean (i.e. the
    half-width of the one sigma confidence interval), given a series of data.
    The data is assumed to be taken from a time series of uniform interval,
    for the purposes of correlation adjustment.

    Parameters
    ----------
    data : array_like of shape (N_data,)
        Data to calculate the sample mean over. It is expected to be a 1D
        array of floats representing a uniform-interval time series; any other
        shape will be flattened. The assumption of uniform time interval is
        not relevant if `adjust_for_correlation` is False.
    weights : array_like of shape (N_data,), optional
        Weights to use, if a weighted mean is desired. Must be the same shape
        as `data`, or broadcastable to the same shape. Defaults to 1 (i.e.
        unweighted mean) if both `weights` and `sigma` are unspecified, or to
        `1/sigma**2` (i.e. inverse-variance weighted mean) if `sigma` is
        specified but `weights` is left unspecified.
    sigmas : array_like of shape (N_data,), optional
        Expected standard deviations (square root of variance) of the provided
        samples. Must be the same shape as `data`, or broadcastable to the
        same shape. If not specified, the samples are assumed to have the same
        variance, which is estimated from the sample variance.
    adjust_for_correlation : bool, optional
        Whether or not to perform the correlation correction. If deactivated,
        the provided samples are assumed to be fully uncorrelated, and the
        reported error will be the naive "standard error of sample mean" based
        on the samples' variances. If activated, which is the default
        behavior, the reported standard error will be magnified based on the
        samples' statistical inefficiency.
    
    Returns
    -------
    mean : float
        The sample mean.
    error : float
        The standard error of the sample mean.

    Notes
    -----
    For a time series `A(t)`, the time-average `<A>` and its error can be
    found using the default settings:
    >>> A_mean, A_mean_err = mean_and_error(At, None, None, True)
    """
    samples = np.asarray(data, dtype=float).ravel()
    N_data = samples.shape[0]
    if N_data < 1:
        raise ValueError('mean_and_error: Empty array')
    if N_data == 1:
        warnings.warn(
            'mean_and_error: Only one data point to average over',
            RuntimeWarning,
        )
        return float(np.mean(samples)), 0.0
    
    if sigmas is None:
        s = np.ones(N_data)
    else:
        s = np.broadcast_to(np.asarray(sigmas, dtype=float), samples.shape)

    if weights is None:
        w_raw = 1.0 / (s**2)
    else:
        w_raw = np.broadcast_to(
            np.asarray(weights, dtype=float),
            samples.shape,
        )
    w_norm = w_raw / np.sum(w_raw)

    mean = np.dot(w_norm, samples)
    residuals = samples - mean

    if sigmas is None:
        v1 = np.sum(w_raw)
        v2 = np.sum(w_raw**2)
        correction = 1.0 - (v2 / (v1**2))
        sample_variance = float(np.dot(w_norm, residuals**2) / correction)
        variance = sample_variance * float(np.sum(w_norm**2))
    else:
        variance = float(np.dot(w_norm**2, s**2))

    if not adjust_for_correlation:
        return float(mean), float(np.sqrt(variance))

    # adjust_for_correlation is True: use the moving block bootstrap
    # Refer to "Politis, White (2004). Automatic block-length selection for
    # the dependent bootstrap. Econom. Rev. 23(1), 53-70."

    r = residuals - np.mean(residuals)
    autocorr = fftconvolve(r, r[::-1], mode='full')[(N_data - 1):]
    if autocorr[0] <= np.finfo(float).eps:
        return float(mean), 0.0
    autocorr /= autocorr[0]
    tolerance = 2.0 * np.sqrt(np.log10(N_data) / N_data)
    cutoff_idx = np.argmax(
        sliding_window_view(
            np.abs(autocorr) < tolerance,
            max(5, int(np.sqrt(np.log10(N_data)))),
        ).all(axis=1),
    )
    if cutoff_idx == 0:
        return float(mean), float(np.sqrt(variance))
    truncate = 2 * cutoff_idx
    k = np.arange(truncate)
    flat_top = 2.0 * (1.0 - ((k + 1) / truncate))
    flat_top[:cutoff_idx] = 1.0
    G_hat = 2.0 * np.sum(flat_top * (k + 1) * autocorr[1:(truncate + 1)])
    g_hat = 1.0 + (2.0 * np.sum(flat_top * autocorr[1:(truncate + 1)]))
    block_length = int(round(np.power(1.5 * N_data * ((G_hat / g_hat)**2),
                                      1.0 / 3.0)))
    if block_length < 2:
        return float(mean), float(np.sqrt(variance))
    if block_length > (0.5 * N_data):
        warnings.warn(
            'mean_and_error: Data too strongly correlated',
            RuntimeWarning,
        )
        g = statistical_inefficiency(r)
        return float(mean), float(np.sqrt(variance * g))
    cumulative = np.concat(([0.0,], np.cumsum(r)))
    block_means = ((cumulative[block_length:] - cumulative[:-block_length])
                   / block_length)
    block_variance = block_length * np.var(block_means)
    
    return float(mean), float(np.sqrt(block_variance / N_data))


def statistical_inefficiency(
        data: np.ndarray,
    ) -> float:
    """Finds the statistical inefficiency of a time series of uniform interval
    (i.e. the effective correlation length between consecutive data points).
    The statistical efficiency also represents the minimum subsampling
    interval required to extract effectively independent samples from the
    data.

    Parameters
    ----------
    data : array_like of shape (N_data,)
        Data to calculate the statistical inefficiency of. It is expected to
        be a 1D array of floats representing a uniform-interval time series;
        any other shape will be flattened.
    
    Returns
    -------
    stat_ineff : float
        The statistical inefficiency; it is bounded from below by 1, but
        potentially unbounded from above.
    """
    samples = np.asarray(data, dtype=float).ravel()
    N_data = samples.shape[0]
    if N_data < 1:
        raise ValueError('statistical_inefficiency: Empty array')
    r = samples - np.mean(samples)
    autocorr = fftconvolve(r, r[::-1], mode='full')[(N_data - 1):]
    if autocorr[0] <= np.finfo(float).eps:
        return 1.0
    autocorr /= autocorr[0]
    tolerance = 2.0 * np.sqrt(np.log10(N_data) / N_data)
    cutoff_idx = np.argmax(
        sliding_window_view(
            np.abs(autocorr) < tolerance,
            max(5, int(np.sqrt(np.log10(N_data)))),
        ).all(axis=1),
    )
    if cutoff_idx > 0:
        autocorr = autocorr[:cutoff_idx]
        lags = np.arange(1, cutoff_idx)
        sum_autocorrs = np.sum((1.0 - (lags / N_data)) * autocorr[1:])
        stat_ineff = float(1.0 + (2.0 * sum_autocorrs))
        return max(stat_ineff, 1.0)
    return 1.0


def variance_and_error(
        data: np.ndarray,
        adjust_for_correlation: bool = True,
    ) -> tuple[float, float]:
    """Finds the variance, and standard error of the variance, given a series
    of data. The standard error is estimated using a jackknife.

    Parameters
    ----------
    data : array_like of shape (N_data,)
        Data to calculate the variance over. It is expected to be a 1D array
        of floats representing a uniform-interval time series; any other shape
        will be flattened. The assumption of uniform time interval is not
        relevant if `adjust_for_correlation` is False.
    adjust_for_correlation : bool, optional
        Whether or not to perform the correlation correction. If deactivated,
        the provided samples are assumed to be fully uncorrelated, and the
        reported error will be based on simple jackknifing of the samples. If
        activated, which is the default behavior, the jackknifing procedure
        will be adjusted so as to subsample blocks longer than the statistical
        inefficiency.
    
    Returns
    -------
    var : float
        The sample variance.
    var_error : float
        The standard error of the sample variance.

    Notes
    -----
    For a time series `A(t)`, the time-averaged fluctuation width `<ΔA²>` and
    its error can be found using the default settings:
    >>> A_fluc, A_fluc_err = variance_and_error(At, True)
    """
    samples = np.asarray(data, dtype=float).ravel()
    N_data = samples.shape[0]
    if N_data < 1:
        raise ValueError('variance_and_error: Empty array')
    if N_data == 1:
        warnings.warn(
            'variance_and_error: Only one data point',
            RuntimeWarning,
        )
        return 0.0, 0.0
    if N_data < 4:
        warnings.warn(
            'variance_and_error: Fewer than four data points',
            RuntimeWarning,
        )
        return float(np.var(samples, ddof=1)), 0.0
    
    var_full = np.var(samples, ddof=1)

    if adjust_for_correlation:
        target_block_length = round(5.0 * statistical_inefficiency(samples))
        N_blocks = N_data // int(target_block_length)
        if N_blocks < 4:
            warnings.warn(
                'variance_and_error: Too few blocks found',
                RuntimeWarning,
            )
            N_blocks = 4
        block_length = N_data // N_blocks
        remainder = N_data % N_blocks
        idxs, i = list(), 0
        for _ in range(remainder):
            idxs.append(slice(i, i + block_length + 1, 1))
            i += block_length + 1
        for _ in range(N_blocks - remainder):
            idxs.append(slice(i, i + block_length, 1))
            i += block_length
    else:
        N_blocks = N_data
        idxs = [slice(i, i + 1, 1) for i in range(N_data)]
    
    jackknifes = [np.var(np.delete(samples, idx), ddof=1) for idx in idxs]
    var_err = np.sqrt((N_blocks - 1) * np.var(jackknifes))

    return float(var_full), float(var_err)


def covariance_and_error(
        dataA: np.ndarray,
        dataB: np.ndarray,
        adjust_for_correlation: bool = True,
    ) -> tuple[float, float]:
    """Finds the covariance, and standard error of the covariance, given two
    series of data of the same length. The standard error is estimated using a
    jackknife.

    Parameters
    ----------
    dataA : array_like of shape (N_data,)
        Data series A to calculate the covariance over. It is expected to be a
        1D array of floats representing a uniform-interval time series; any
        other shape will be flattened. The assumption of uniform time interval
        is not relevant if `adjust_for_correlation` is False.
    dataB : array_like of shape (N_data,)
        Data series B to calculate the covariance over. It is expected to be a
        1D array of floats representing a uniform-interval time series, drawn
        from the same instants as `dataA`; any other shape will be flattened.
        Must have the same length as `dataA`.
    adjust_for_correlation : bool, optional
        Whether or not to perform the correlation correction. If deactivated,
        the provided samples are assumed to be fully uncorrelated, and the
        reported error will be based on simple jackknifing of the samples. If
        activated, which is the default behavior, the jackknifing procedure
        will be adjusted so as to subsample blocks longer than the statistical
        inefficiency.
    
    Returns
    -------
    covar : float
        The samples' covariance.
    covar_error : float
        The standard error of the samples' covariance.

    Notes
    -----
    For two time series `A(t)` and `B(t)`, the time-averaged co-fluctuation
    width `<ΔA⋅ΔB>` and its error can be found using the default settings:
    >>> AB_fluc, AB_fluc_err = covariance_and_error(At, Bt, True)
    """
    sampleA = np.asarray(dataA, dtype=float).ravel()
    sampleB = np.asarray(dataB, dtype=float).ravel()
    if sampleA.shape != sampleB.shape:
        raise ValueError(
            'covariance_and_error: Length mismatch between dataA and dataB '
            f'({sampleA.shape[0]} vs {sampleB.shape[0]})'
        )
    N_data = sampleA.shape[0]
    if N_data < 1:
        raise ValueError('covariance_and_error: Empty arrays')
    if N_data == 1:
        warnings.warn(
            'covariance_and_error: Only one data point',
            RuntimeWarning,
        )
        return 0.0, 0.0
    
    def make_covar(At, Bt):
        meanA = np.mean(At)
        meanB = np.mean(Bt)
        return np.sum((At - meanA) * (Bt - meanB)) / (At.shape[0] - 1)
    
    if N_data < 4:
        warnings.warn(
            'covariance_and_error: Fewer than four data points',
            RuntimeWarning,
        )
        return float(make_covar(sampleA, sampleB)), 0.0

    covar_full = make_covar(sampleA, sampleB)

    if adjust_for_correlation:
        target_block_length = max(
            round(5.0 * statistical_inefficiency(sampleA)),
            round(5.0 * statistical_inefficiency(sampleB)),
        )
        N_blocks = N_data // int(target_block_length)
        if N_blocks < 4:
            warnings.warn(
                'covariance_and_error: Too few blocks found',
                RuntimeWarning,
            )
            N_blocks = 4
        block_length = N_data // N_blocks
        remainder = N_data % N_blocks
        idxs, i = list(), 0
        for _ in range(remainder):
            idxs.append(slice(i, i + block_length + 1, 1))
            i += block_length + 1
        for _ in range(N_blocks - remainder):
            idxs.append(slice(i, i + block_length, 1))
            i += block_length
    else:
        N_blocks = N_data
        idxs = [slice(i, i + 1, 1) for i in range(N_data)]
    
    jackknifes = [
        make_covar(np.delete(sampleA, idx), np.delete(sampleB, idx))
        for idx in idxs
    ]
    covar_err = np.sqrt((N_blocks - 1) * np.var(jackknifes))

    return float(covar_full), float(covar_err)
