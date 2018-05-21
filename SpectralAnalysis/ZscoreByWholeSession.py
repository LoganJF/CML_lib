"""ZscoreByWholeSession.py, Author=LoganF
The Purpose of this script is to allow a user to generate z-scores using the entire session's.
# -------> Example import and use to z-score a timeseries by it's whole session
import sys
sys.path.append('/home2/loganf/SecondYear/CML_lib') # Path to CML_lib
from ZscoreByWholeSession import Zscore_WholeSession
z_ts = Zscore_WholeSession(timeseries=timeseries_data,
                           step_in_seconds=60,
                           random_range=10,
                           morlet_cycles=7,
                           start_time=-2.,
                           end_time=2.)
mean, std = z_ts.mean('bipolar_pairs'), z_ts.std('bipolar_pairs')
timeseries_data.data -= mean.data[:,:,None,None]
timeseries_data.data /= std.data[:,:,None,None]
"""
#from ptsa.data.readers import EEGReader
#from ptsa.data.filters import MonopolarToBipolarMapper, ButterworthFilter, DataChopper

from ptsa.data.readers import BaseEventReader, EEGReader, JsonIndexReader, TalReader
from ptsa.data.filters import MonopolarToBipolarMapper, ButterworthFilter, DataChopper
from time import time

import numpy as np
from copy import deepcopy

#sys.path.append('/home2/loganf/SecondYear/CML_lib/')
#from GetData import get_subs
#from SpectralAnalysis.MentalChronometry import Subject
from RetrievalCreationHelper import RetrievalEventCreator


try:
    from ptsa.data.filters.MorletWaveletFilterCpp import MorletWaveletFilterCpp as MorletWaveletFilter
except ImportError:
    from ptsa.data.filters.MorletWaveletFilter import MorletWaveletFilter

# ------> Function to import
# -------------> BETTER FASTER WAYS NOT INVOLVING DATACHOPPER
def get_behavioral_events_zscoring(subject, experiment, session, desired_step_in_sec, jitter):
    """

    Parameters
    ----------
    subject
    experiment
    session
    desired_step_in_sec
    jitter

    Returns
    -------

    """
    subject_instance = RetrievalEventCreator(subject=subject, experiment=experiment, session=session,
                                             inclusion_time_before=2000, inclusion_time_after=1500)
    subject_instance.initialize_recall_events()
    behavioral_events = subject_instance.events

    start_mstime = np.min(behavioral_events['mstime'])
    end_mstime = np.max(behavioral_events['mstime'])

    offsets = np.arange(np.min(behavioral_events['mstime']),
                        np.max(behavioral_events['mstime']),
                        desired_step_in_sec * 1000)  # Second to ms conversion
    randomize = np.random.randint(-jitter, jitter, len(offsets)) * 1000  # Second to ms conversion
    # offsets -= (np.random.randint(-jitter, jitter, len(offsets)) * 1000) # Randomize the offsets
    offsets -= randomize  # Randomize the offsets

    trial_field = 'trial' if 'trial' in behavioral_events.dtype.names else 'list'
    item_field = 'item_name' if 'item_name' in behavioral_events.dtype.names else 'item'
    item_number_field = 'item_num' if 'item_num' in behavioral_events.dtype.names else 'itemno'

    z_events = np.zeros(len(offsets), dtype=behavioral_events.dtype).view(np.recarray)
    z_events['mstime'] = offsets
    z_events['eegfile'] = behavioral_events['eegfile'][0]
    z_events['subject'] = behavioral_events['subject'][0]
    z_events[trial_field] = -999
    z_events['serialpos'] = -999
    z_events['type'] = 'Z-SCORE'
    z_events[item_field] = 'Z'
    z_events[item_number_field] = -999
    z_events['recalled'] = -999
    z_events['intrusion'] = -999
    z_events['rectime'] = -999
    z_events['protocol'] = behavioral_events['protocol'][0]
    z_events['session'] = behavioral_events['session'][0]
    z_events['match'] = -999
    z_events['timebefore'] = -999
    z_events['timeafter'] = -999
    z_events['montage'] = behavioral_events['montage'][0]
    z_events['experiment'] = behavioral_events['experiment'][0]

    all_evs = np.append(behavioral_events, z_events)
    all_evs.sort(order='mstime')
    for i, event in enumerate(all_evs):
        if all_evs[i]['type'] == 'Z-SCORE':
            diff_seconds = (all_evs[i]['mstime'] - all_evs[i - 1]['mstime']) / 1000.
            n_samples_since_prev = diff_seconds * subject_instance.sample_rate
            all_evs[i]['eegoffset'] = all_evs[i - 1]['eegoffset'] + n_samples_since_prev

    z_events = all_evs[all_evs['type'] == 'Z-SCORE']
    return z_events.view(np.recarray)


def get_zscore_eeg(subject, experiment, session, desired_step_in_sec, jitter):
    """

    Parameters
    ----------
    subject
    experiment
    session
    desired_step_in_sec
    jitter

    Returns
    -------

    """
    events = get_behavioral_events_zscoring(subject=subject,
                                            experiment=experiment,
                                            session=session,
                                            desired_step_in_sec=desired_step_in_sec,
                                            jitter=jitter)
    # Get electrode info
    jr = JsonIndexReader('/protocols/r1.json')
    pairs_path = jr.get_value('pairs', subject=subject, experiment=experiment)
    tal_reader = TalReader(filename=pairs_path)
    mp = tal_reader.get_monopolar_channels()
    bp = tal_reader.get_bipolar_pairs()

    # Load eeg from start to end, include buffer
    eeg_reader = EEGReader(events=events, channels=mp, start_time=-2.,
                           end_time=2., buffer_time=1.)
    eeg = eeg_reader.read()

    m2b = MonopolarToBipolarMapper(time_series=eeg, bipolar_pairs=bp)
    eeg = m2b.filter()

    return eeg


def get_zscored_wavelets(subject, experiment, session, desired_step_in_sec, jitter, frequencies, morlet_cycles):
    """

    Parameters
    ----------
    subject
    experiment
    session
    desired_step_in_sec
    jitter
    frequencies
    morlet_cycles

    Returns
    -------

    """
    s = time()
    eeg = get_zscore_eeg(subject=subject,
                         experiment=experiment,
                         session=session,
                         desired_step_in_sec=desired_step_in_sec,
                         jitter=jitter)

    line_noise = [48., 52.] if 'FR' in subject else [58., 62.]

    # Apply butter filter to remove line noise
    b_filter = ButterworthFilter(time_series=eeg,
                                 freq_range=line_noise,
                                 filt_type='stop',
                                 order=4)
    data = b_filter.filter()

    # Apply morlet wavelet
    wf = MorletWaveletFilter(time_series=data,
                             freqs=frequencies,
                             output='power',
                             # frequency_dim_pos=0,
                             verbose=True,
                             width=morlet_cycles)

    z_pow_wavelet, phase_wavelet = wf.filter()
    # Remove the buffer
    z_pow_wavelet = z_pow_wavelet.remove_buffer(1.0)

    # Log transform the data
    np.log10(z_pow_wavelet.data, out=z_pow_wavelet.data);
    print('Total Z-score Whole Session Time: ', time() - s)
    return z_pow_wavelet[:, :, :, len(z_pow_wavelet['time']) / 2]  # Return only middle point

# ----> Old functions
def Zscore_WholeSession(timeseries, step_in_seconds, random_range, morlet_cycles, start_time=-2., end_time=2.):
    """Given a TimeSeriesX corresponding to only a single session of a subject, returns a TimeSeriesX containing
    randomly spaced points every step_in_seconds seconds throughout the entire session. randomly spaced every ~60 for a
    session

    Parameters
    ----------
    timeseries: TimeSeriesX, dims = (frequency, events, bipolar_pairs, time)
                timeseries object with events corresponding to a single session of subject data, must have
                a valid dim 'frequency' containing an array of valid frequencies that we would like to convolve.
    step_in_seconds: int, number of seconds between each point we sample from to construct the mean/std from
    random_range: int, amount of time to use in the jitter +/- surrounding step_in_seconds
    morlet_cycles: int, number of cycles (width) to use in the morlet wavelet convolution
    start_time: float, by default -2., the time relative to each offset to start loading eeg (doesn't really matter)
    end_time: float, by default 2., the time relative to each offset to stop loading eeg (doesn't really matter)

    Returns
    -------
    z_pow_wavelet: TimeSeriesX dims ('time', 'offsets', 'events'), whereby user can get mean and std using:
                   mean, std = z_pow_wavelet.mean('events'), z_pow_wavelet.std('events')
    """
    # Loads all the eeg
    eeg = get_continuous_eeg_from_timeseries(timeseries)
    frequencies = timeseries['frequency'].data
    # Get randomly spaced time points given the user inputs
    offsets = randomly_spaced_time_points(eeg=eeg,
                                          random_range=random_range,
                                          step_in_seconds=step_in_seconds)

    # Chop the data into segments
    dc = DataChopper(start_offsets=offsets,
                     session_data=eeg,
                     start_time=start_time,
                     end_time=end_time,
                     buffer_time=1.0)
    chopped = dc.filter()
    # Make a copy of the bp_data since it's going to be over-written
    bp_data = deepcopy(eeg.bipolar_pairs.data)
    del(eeg) # Delete eeg in order to save memory
    # Germany (Europe) = 50Hz, America = 60 Hz
    subject = timeseries['events'][0].data['subject']
    line_noise = [48., 52.] if 'FR' in subject else [58., 62.]

    # Apply butter filter to remove line noise
    b_filter = ButterworthFilter(time_series=chopped,
                                 freq_range=line_noise,
                                 filt_type='stop',
                                 order=4)
    data = b_filter.filter()

    # Apply morlet wavelet
    wf = MorletWaveletFilter(time_series=data,
                             freqs=frequencies,
                             output='power',
                             #frequency_dim_pos=0,
                             verbose=True,
                             width=morlet_cycles)

    z_pow_wavelet, phase_wavelet = wf.filter()
    z_pow_wavelet = z_pow_wavelet.rename({'start_offsets': 'events'})
    # Remove the buffer
    z_pow_wavelet = z_pow_wavelet.remove_buffer(1.0)

    # Log transform the data
    np.log10(z_pow_wavelet.data, out=z_pow_wavelet.data);

    # We only want one time point corresponding to each offset.
    # pow_wavelet = pow_wavelet.sel(time=pow_wavelet.samplerate.data*end_time)

    # We need to reset the bipolar pairs b/c chopper removes them
    z_pow_wavelet['bipolar_pairs'].data = bp_data
    # mean = pow_wavelet.mean(dim='events')
    # std = pow_wavelet.std(dim='events')
    return z_pow_wavelet[:,:,:,len(z_pow_wavelet['time'])/2] # Return only middle point


def get_continuous_eeg_from_timeseries(timeseries):
    """Returns continuous bipolar eeg representing the entire session given an inputted TimeSeriesX containing only a
    single session worth of data

    Parameters
    ----------
    timeseries: TimeSeriesX, dims = (frequency, events, bipolar_pairs, time)
                timeseries object with events corresponding to a single session of subject data, must have
                a valid dim 'frequency' containing an array of valid frequencies that we would like to convolve.

    Returns
    -------
    bp_eeg: TimeSeriesX: TimeSeriesX of the whole session, bipolar referenced
    """

    """Given a timeseries with valid behavioral events will return eeg corresponding to the continuous session"""
    if not 'bipolar_pairs' in timeseries.dims:
        print('Please insert a TimeSeriesX Object with dimension bipolar_pairs')
        return
    # Get unique monopolar channels and bipolar pairs from the timeseries
    mp = np.unique((timeseries['bipolar_pairs'].data['ch0'],
                    timeseries['bipolar_pairs'].data['ch1'])).view(np.recarray)
    bp = timeseries['bipolar_pairs'].data.view(np.recarray)
    # Get the dataroot to load into the session reader
    dataroot = timeseries['events'].data[0]['eegfile']
    session_reader = EEGReader(session_dataroot=dataroot, channels=mp)
    eeg = session_reader.read()
    # Convert to bipolar
    m2b = MonopolarToBipolarMapper(time_series=eeg, bipolar_pairs=bp)
    bp_eeg = m2b.filter()
    bp_eeg['bipolar_pairs'].data = bp
    return bp_eeg


def randomly_spaced_time_points(eeg, random_range=10, step_in_seconds=60):
    """Return randomly selected offset points for an inputted array
    ------
    INPUTS:
    eeg: TimeSeriesX, a continuous session worth of eeg
    random_range: int, used to construct the +/- jitter
    step_in_seconds: int, number of seconds along which we step
    ------
    OUTPUTS:
    offsets: np.array, represents the offsets along which one can chop a
                       continuous eeg recording
    """
    # Find points where the session is invalid (e.g. the eeg is 0)
    # invalid_points_start = np.where(eeg[0,0]==0)[0][0]

    offsets = np.arange(eeg.offsets.data[0],
                        eeg.offsets.data[-1],  # Avoid loading in 0 points at the end...
                        eeg.samplerate.data * step_in_seconds)

    # Randomize it such that it can vary from 50 seconds to 70 seconds
    randomize = np.random.randint(-random_range,
                                  random_range,
                                  len(offsets)) * 1000
    offsets = offsets - randomize
    # Remove the first two and last two points to avoid 0's in times before sess starts and after it ends!
    return offsets[2:-2]


# Possible alternative way of handling zeros in randomly_spaced_time_points instead of just droping them?
def replaceZeroes(ts):
    """Replace zeros in a timeseries with the minimum value in the timeseries
    ------
    INPUTS
    ts: TimeSeriesX
    """
    data = ts.data
    min_nonzero = np.min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    ts.data = data
    return ts