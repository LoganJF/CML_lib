# -*- coding: utf-8 -*-
"""RollingAverage.py, author: LoganF

A script to compute rolling averages (sliding time windows) along a time series, the main function for importing is
sliding_mean_fast which will reference the other functions appropriately. To use sliding_mean_fast the user must
install the package bottleneck (pip install bottleneck).


Example usuage:

from RollingAverage import sliding_mean_fast
# Where timeseries is a TimeSeriesX object containing your data...
averaged = sliding_mean_fast(timeseries, windows=.5, desired_step = .01, dim='time')
"""
from ptsa.data.TimeSeriesX import TimeSeriesX

try:
    import bottleneck as bn
except ImportError:
    print('Please install the package bottleneck to use sliding_mean_fast, sliding_mean_slow should still work though!')

from copy import deepcopy
import numpy as np
import math


def find_nearest(array, value, return_index_not_value=True, is_sorted=True):
    """Given an array and a value, returns either the index or value of the nearest match

    Parameters
    ----------
    array: np.array, array of values to check for matches
    value: int/float, value to find the closest match to
    return_index_not_value: bool, whether to return the index(True) or the value (False)
        of the found match
    is_sorted: bool, whether the array is sorted in order of values
    Returns
    -------
    Either the index or value of the nearest match
    """
    if is_sorted:
        idx = np.searchsorted(array, value, side="left")

        if ((idx > 0) and (idx == len(array)) or (math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx]))):

            if not return_index_not_value:
                return array[idx - 1]

            if return_index_not_value:
                return idx - 1

        else:

            if not return_index_not_value:
                return array[idx]

            if return_index_not_value:
                return idx

    elif not is_sorted:
        idx = (np.abs(array - value)).argmin()

        if not return_index_not_value:
            return array[idx]

        if return_index_not_value:
            return idx


def round_down(n):
    """"Rounds number n to nearest 100th place

    Parameters
    ----------
    n: int/float, a number to round down

    Returns
    -------
    n: float, n rounded to nearest 100th place
    """
    return float(int(n / 100) * 100)


def sliding_mean_slow(ts, window=.5, desired_step=.01):
    """Computes a sliding mean across inputted timeseries ts
    Parameters
    -------
    INPUTS:
    ts: TimeSeriesX, TimeSeries over which to be averaged
    desired_window: float, the window length in seconds to be averaged over
                    by default value is half a second (.5)
    desired_step: float, the time in seconds over which to
                  increase the window each step, by default value is 10ms
                  (.01)
    -------
    OUTPUTS:
    ts: TimeSeriesX, averaged into windows
    """
    # E.g. if ts is -2, +2, this will be 350 points from -2 to 1.5
    # points = np.arange(ts['time'][0].data, int(ts['time'][-1].data)-window, desired_step)
    # points = np.arange(-2, 1.5, desired_step)
    ts = ts.transpose('frequency', 'bipolar_pairs', 'events', 'time')
    points = np.arange(ts['time'][0].data,
                       ts['time'][-1].data - window,
                       desired_step)
    # Go through each time point and find the nearest point, return the index
    data = []
    for time in points:
        start = find_nearest(array=ts.time.data, value=time, return_index_not_value=True)
        stop = find_nearest(array=ts.time.data, value=time + window, return_index_not_value=True)

        # Slice the timeseries from the nearest start and stop values
        data.append(ts[:, :, :, start:stop].mean('time'))
    timeseries = TimeSeriesX.concat(data, 'time')
    # ts = ts.transpose('frequency', 'bipolar_pairs', 'events', 'time')
    # Make the time axis represent the midpoint of the window.
    points = points + (window / 2.)
    timeseries['time'] = points
    timeseries = timeseries.transpose('frequency', 'bipolar_pairs', 'events', 'time')
    return timeseries


def sliding_mean_fast(ts, window=.5, desired_step=.01, dim='time'):
    """Creates averaged epochs from timeseries in an incredibly fast manner

    Created April 10 2018, changed order of creating slicer, desired_window and copy
    and implemented nearest hundredth place rounding

    April 16: implemented calling of sliding_mean_slow because resampling took longer than the slow-down

    #### NOTES ####  CREATES A COPY OF TIMESERIES AND RETURN THE COPY INSTEAD OF MODIFING THE INPUT!!!
    ------
    INPUTS:
    ts: TimeSeriesX, TimeSeries over which to be averaged
    desired_window: float, the window length in seconds to be averaged over
                    by default value is half a second (.5)
    desired_step: float, the time in seconds over which to
                  increase the window each step, by default value is 10ms
                  (.01)
    dim: str, the dimension over which to average
    ------
    OUTPUTS:
    time_series: TimeSeriesX, the time series with averaged windows


    ------
    Notes
    %timeit deepcopy(ts)
    10 loops, best of 3: 167 ms per loop
    %timeit TimeSeriesX.create(data=ts.data, dims=ts.dims, samplerate=ts.samplerate)
    10000 loops, best of 3: 94.4 µs per loop
    """
    ts = ts.transpose('frequency', 'bipolar_pairs', 'events', 'time')

    dims_to_axis = {dim: k for k, dim in enumerate(ts.dims)}
    axis = dims_to_axis[dim]

    # Set the sample rate
    sr = np.round(ts.samplerate.data)  # Changed to round to avoid 499.997 errors
    # Check if we need to resample the timeseries
    slicer = sr * desired_step
    if int(slicer) != slicer:  # Sanity check!
        print('Calling non-optimized moving mean due to sample rate')
        # return sliding_mean_slow(ts=ts, window=window, desired_step=desired_step)
        print('For now we won\'t run the slow mean....')
        return
        # sr = round_down(sr)
        # ts = ts.resampled(sr)
        # slicer = int(sr * desired_step)

    # Set the window step in num_samples
    desired_window = int(window * sr)

    # Create a new timeseriesX object so we don't over-write the old one
    copy = TimeSeriesX.create(data=ts.data,
                              dims=ts.dims,
                              samplerate=ts.samplerate)

    # Get the sliding average as a numpy array
    print('Calling optimized moving mean')
    sliding_average = bn.move_mean(copy, desired_window, axis)

    # Reset the data to the sliding averages
    copy.data = sliding_average
    # copy['frequency']
    for dim in copy.dims:
        copy[dim] = ts[dim]

    # So we don't start with nans that don't matter
    copy = copy.sel(time=copy.time >= copy.time[desired_window])
    # return copy
    copy['bipolar_pairs'] = ts['bipolar_pairs']
    copy['events'] = ts['events']

    # Code technically slides over every possible point
    # So we need to return slices along time dim
    copy = copy[:, :, :, ::int(slicer)]
    # Reset the time axis so it corresponds correlect with the midpoint
    # Of the averaged window
    start_time = ts['time'][0].data + (window / 2.)
    end_time = ts['time'][-1].data - (window / 2.)
    copy['time'].data = np.arange(start_time, end_time, desired_step)

    return copy  # [:, :, :, ::int(slicer)]


def reorder_dims(darray, dim1, dim2):
    """Interchange two dimensions of a DataArray in a similar way as numpy's swap_axes

    Parameters
    ---------
    darray: Xarray.DataArray/TimeSeriesX
    dim1: str, the first dim we want to swap
    dim2: str, the second dim we want to swap

    Returns
    ---------
    transposed darray

    Example Usage
    -------------
    # Swap the timeseries object's first dim and time dim
    check_first_dim = timeseries.dims[0]
    timeseries_swapped_time_dim = reorder_dims(timeseries, dim1=check_first_dim, dim2='time')

    See
    -------
    https://github.com/pydata/xarray/issues/1081, rafa-guedes comment on Aug 21, 2017
    """
    dims = list(darray.dims)
    assert set([dim1, dim2]).issubset(dims), 'dim1 and dim2 must be existing dimensions in darray'
    ind1, ind2 = dims.index(dim1), dims.index(dim2)
    dims[ind2], dims[ind1] = dims[ind1], dims[ind2]
    return darray.transpose(*dims)


def xarray_rolling_average(timeseries, desired_window, desired_step):
    """Computes a rolling average analogous to sliding_mean_fast using xarray.rolling given a desired window, and step

    Parameters
    ----------
    timeseries: TimeSeriesX, time series we which to average into temporal windows.
                Assumes time dim is explicitly labeled time
    desired_window: float, time in seconds of the desired window (e.g. 500ms/.5s is .5)
    desired_step: float, time in seconds of the desired step between windows (e.g. 10ms is .01)

    Returns
    -------
    timeseries_rolled: TimeSeriesX, time series we inputted but with time dim now representing the midpoint of that time
    point averaged across desired_window seconds.
    """
    # Transpose it so that time is on the first axis
    first_dim = timeseries.dims[0]
    timeseries_transposed = reorder_dims(darray=timeseries,
                                         dim1=first_dim,
                                         dim2='time')
    # Use sample rate to determine indexing for windows and steps
    samplerate = 1 / timeseries_transposed['samplerate'].data
    timeseries_window_length = int(desired_window / samplerate)
    timeseries_step_length = int(desired_step / samplerate)
    # Using default xarray.DataArray.rolling to compute a mean
    rolling_average = timeseries_transposed.rolling(time=timeseries_window_length,
                                                    center=True).mean()
    # Data is only valid from half the window after start until half before end; e.g. removing nans
    timeseries_rolled = rolling_average[timeseries_window_length / 2:-timeseries_window_length / 2]

    # Return windows that correspond with our desired step
    timeseries_rolled = timeseries_rolled[::timeseries_step_length]
    # timeseries_rolled = reorder_dims(darray=timeseries_rolled,
    # dim1='time',
    # dim2=first_dim)
    return timeseries_rolled.transpose('frequency', 'bipolar_pairs', 'events', 'time')