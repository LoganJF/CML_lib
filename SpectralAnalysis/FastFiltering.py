"""FastFiltering.py, author=LoganF, updated May 3 2018
A script containing various functions that allow for easy and fast frequency filtering with TimeSeriesX objects
Specifically, we make a butterworth bandpass filter, a highpass filter, a hilbert filter and various functions to
be used in conjunction with these filters
"""

import numpy as np
from ptsa.data.TimeSeriesX import TimeSeriesX
from scipy.signal import butter, lfilter, hilbert, filtfilt
from scipy.ndimage.filters import gaussian_filter1d


# -------> Scipy butherworth bandpass filter
def _butter_bandpass(lowcut, highcut, fs, order=5):
    """Helper function for butter_bandpass_filter
    ------
    INPUTS:
    lowcut: frequency to start pass
    highcut: frequency to stop pass
    fs: sampling frequency of the series
    order: the order of the filter, by default 5
    -----
    OUTPUTS:
    b, a : ndarray, ndarray
           Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
    """
    # from scipy.signal import butter
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Applies a bandpass filter on the signal
    ------
    INPUTS:
    data: np.array, TimeSeriesX; data on which to apply the filter over
    lowcut: frequency to start pass
    highcut: frequency to stop pass
    fs: sampling frequency of the series
    order: the order of the filter, by default 5
    """
    # from scipy.signal import lfilter
    b, a = _butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)

    # If we input a TimeSeriesX we should output a TimeSeriesX
    if type(data) == TimeSeriesX:
        # Create a copy of timeseries
        copy = TimeSeriesX.create(data=y,
                                  dims=data.dims,
                                  coords=data.coords,
                                  samplerate=data.samplerate)
        return copy
    # If it's not a TimeSeriesX Just return an array
    return y


# -------> Scipy butterworth highpass filter (low stop)

def _butter_highpass(cutoff, fs, order=4):
    """Helper function for butter_highpass_filter
    ------
    INPUTS:
    cutoff: frequency cut out
    fs: sampling frequency of the series
    order: the order of the filter, by default 4
    -----
    OUTPUTS:
    b, a : ndarray, ndarray
           Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=4):
    """Applies a high pass (low stop) filter on the signal
    This is used to remove slow drifting signals <.1 in the data that contribute to noise
    ------
    INPUTS:
    data: np.array, TimeSeriesX; data on which to apply the filter over
    cutoff: frequency to cutoff
    fs: sampling frequency of the series
    order: the order of the filter, by default 4
    """
    b, a = _butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    # If we input a TimeSeriesX we should output a TimeSeriesX
    if type(data) == TimeSeriesX:
        # Create a copy of timeseries
        copy = TimeSeriesX.create(data=y,
                                  dims=data.dims,
                                  coords=data.coords,
                                  samplerate=data.samplerate)
        return copy
    # If it's not a TimeSeriesX Just return an array
    return y


# ------> Hilbert related functions
def Hilbert(signal):
    """Applies a hilbert transform to the signal

    ### IMPORTANT NOTES####
    This function adds zero padding to speed up the processing of the fast
    fourier transformation(FFT) if the length of the signal passed is
    not a power of two (for example a 49999 lengthed signal
    will take orders of magnitude longer than a 50000 lengthed signal to
    compute a Fast fourier transformation on )

    ------
    INPUTS:
    signal: array like or TimeSeriesX.
    ------
    OUTPUTS:

        results: array, numpy array of shape signal that is hilbert filtered
        signal: TimeSeriesX, a TimeSeries object
    """
    # import numpy as np
    # from scipy.signal import hilbert
    # from ptsa.data.TimeSeriesX import TimeSeriesX

    if len(signal) == 1:
        padding = np.zeros(int(2 ** np.ceil(np.log2(len(signal)))) - len(signal))
        tohilbert = np.hstack((signal, padding))
        result = hilbert(tohilbert)
        result = result[0:len(signal)]
        return result

    results = []
    for sig in signal:
        padding = np.zeros(int(2 ** np.ceil(np.log2(len(sig)))) - len(sig))
        tohilbert = np.hstack((sig, padding))
        result = hilbert(tohilbert)
        result = result[0:len(sig)]
        results.append(result)

    results = np.concatenate(results).reshape(signal.shape)

    if type(signal) == TimeSeriesX:
        # Create a copy of timeseries
        copy = TimeSeriesX.create(data=results,
                                  dims=signal.dims,
                                  coords=signal.coords,
                                  samplerate=signal.samplerate)
        return copy
        # BELOW WILL SOMEHOW EFFECT GLOBAL SPACE INPUT???
        # signal.data = results
        # return signal
    return results


def get_amplitude_envelope(signal):
    """Returns that instantaneous amplitude evenlope of the analytic signal from the Hilbert transformation
    ------
    INPUTS:
    signal: array like or TimeSeriesX.
    """
    return np.abs(Hilbert(signal))


def gaussian_smooth(data, sampling_frequency=None, sigma=0.004, axis=-1, truncate=8):
    '''1D convolution of the data with a Gaussian.

    The standard deviation of the gaussian is in the units of the sampling
    frequency. The function is just a wrapper around scipy's
    `gaussian_filter1d`, The support is truncated at 8 by default, instead
    of 4 in `gaussian_filter1d`

    Parameters
    ----------
    data : array_like
    sigma : float
    sampling_frequency : int
    axis : int, optional
    truncate : int, optional

    Returns
    -------
    smoothed_data : array_like

    See: https://rhino2.psych.upenn.edu:8000/user/loganf/edit/ripple_detection/ripple_detection/core.py#
    '''
    # from scipy.ndimage.filters import gaussian_filter1d

    if ((sampling_frequency is None) & (type(data) == TimeSeriesX)):
        sampling_frequency = float(ts['samplerate'])

    filtered = gaussian_filter1d(data, sigma * sampling_frequency,
                                 truncate=truncate, axis=axis, mode='constant')

    if type(data) == TimeSeriesX:
        # Create a copy of timeseries
        copy = TimeSeriesX.create(data=filtered,
                                  dims=data.dims,
                                  coords=data.coords,
                                  samplerate=data.samplerate)
        return copy

    return filtered