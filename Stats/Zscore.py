"""Zscore.py, author=LoganF
A script that serves as a wrapper for scipy's zscore to allow it to be compatible with TimeSeriesX DO NOT USE UNTIL
DEBUG curr_sess issue!!!
"""
from scipy.stats import zscore
from ptsa.data.TimeSeriesX import TimeSeriesX
import numpy as np

def scipy_zscore(timeseries, dim='events'):
    """Z-scores Data using scipy by default along events axis, does so for multiple sessions
    Parameters
    ----------
    timeseries: TimeSeriesX,
    dim: str, by default 'events',
            dimension to z-score over
    """
    if len(np.unique(timeseries['events'].data['session'])) > 1:
        raise
    # Convert dim to axis for use in scipy
    dim_to_axis = dict(zip(timeseries.dims, xrange(len(timeseries.dims))))
    axis = dim_to_axis[dim]

    # Go through each session and z-score relative to itself.
    z_data = []
    for sess in np.unique(timeseries['events'].data['session']):
        curr_sess = timeseries.sel(events=timeseries['events'].data['session'] == sess)
        curr_sess.data = zscore(curr_sess, axis)
        z_data.append(curr_sess)

    return TimeSeriesX.concat(z_data, 'events')