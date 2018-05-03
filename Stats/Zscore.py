"""Zscore.py, author=LoganF
A script that serves as a wrapper for scipy's zscore to allow it to be compatible with TimeSeriesX
"""
from scipy.stats import zscore
from ptsa.data.TimeSeriesX import TimeSeriesX


def zscore(timeseries, dim, inplace=True):
    """Computes a Zscore using scipy without converting timeseriesx into a np.array

    ------
    INPUTS:
    timeseries: TimeSeriesX
    dim: str, dim over which to zscore
    inplace: bool, default False, whether or not to modify the timeseries in palce or return a copy

    ------
    OUTPUTS:
    zscored TimeSeriesX
    """

    dims_to_axis = {dims: k for k, dims in enumerate(timeseries.dims)}
    axis = dims_to_axis[dim]
    raw = zscore(timeseries, axis)

    if inplace:
        timeseries.data = raw
        return timeseries

    copy = TimeSeriesX.create(
        data=raw, dims=timeseries.dims, coords=timeseries.coords, samplerate=timeseries.samplerate)
    return copy
