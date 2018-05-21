# General Imports
import sys, os
from matplotlib import pyplot as plt
from copy import deepcopy
# Add some dimension to your life, np=arr, pd=labeled arr, xr=labeled N-D arr
import numpy as np  # I say numpy like 'lumpy', no I don't mean num-pie

# Pizza ptsa stuff
from ptsa.data.TimeSeriesX import TimeSeriesX

try:
    from ptsa.data.filters.MorletWaveletFilterCpp import MorletWaveletFilterCpp as MorletWaveletFilter
except:
    from ptsa.data.filters.MorletWaveletFilter import MorletWaveletFilter

# Stats
from scipy.stats import ttest_ind, zscore, ttest_rel

# Relative imports form toolbox
#sys.path.append('/home2/loganf/SecondYear/CML_lib/')
from SpectralAnalysis.RollingAverage import sliding_mean_fast


# --------> Functions to import for post_processing z-scored averaged bands/epochs
def BurkeMethodology(timeseries, lobar=True, minimum_num_events=20):
    """Applies Burke post-processing steps from raw wavelets to epoched z-tstats
    timeseries: TimeSeriesX; freq wavelet X bp X events X time"""
    # Average Frequency into bands
    ts_bands_time = average_timeseries_frequency(timeseries)
    # Average time into windows of 200ms stepped every 20 ms
    ts_bands_windows = sliding_mean_fast(ts_bands_time, window=.5, desired_step=.01)
    # Z-score the data by session
    z_ts_bp = scipy_zscore(ts_bands_windows, dim='events')
    # Average bp into lobar or gyrus rois
    average_ts_func = average_timeseries_lobar if lobar else average_timeseries_rois
    z_ts_rois = average_ts_func(z_ts_bp)
    # Independent t-test
    recs, bl, incl = split_rec_bl(z_ts_rois, minimum_num_events)
    if incl:
        tstats = independent_ttest(recs, bl)
        return tstats
    return

def LoganMethodology(timeseries, lobar=True, minimum_num_events=20):
    """Applies Logan post-processing steps from raw wavelets to epoched z-tstats
    timeseries: TimeSeriesX; freq wavelet X bp X events X time"""
    # Average Frequency into bands
    ts_bands_time = average_timeseries_frequency(timeseries)
    # Average time into windows of 200ms stepped every 20 ms
    ts_bands_windows = sliding_mean_fast(ts_bands_time, window=.2, desired_step=.02)
    # Z-score the data by session
    z_ts_bp = scipy_zscore(ts_bands_windows, dim='events')
    # Average bp into lobar or gyrus rois
    average_ts_func = average_timeseries_lobar if lobar else average_timeseries_rois
    z_ts_rois = average_ts_func(z_ts_bp)
    # Independent t-test
    recs, bl, incl = split_rec_bl(z_ts_rois, minimum_num_events)
    if incl:
        tstats = independent_ttest(recs, bl)
        return tstats
    return


# ---------> Post Processing Helper functions
def average_timeseries_rois(timeseries):
    ts = deepcopy(timeseries)
    subject = ts['events'].data[0]['subject']
    experiment = ts['events'].data[0]['experiment']

    brain_atlas = get_logan_tal(subject=subject, experiment='FR1')

    # Over-ride TimeSeries channels
    rois = brain_atlas['ind']
    lobar_roi = brain_atlas['lobe']
    gyrus_none = np.where(rois == None)
    rois[gyrus_none] = lobar_roi[gyrus_none]
    # Make it hemisphere + rois
    rois = brain_atlas['hemi'] + ' ' + rois
    ts['bipolar_pairs'].data = rois
    # Only select good channels and then average into rois
    ts = ts.sel(bipolar_pairs=~brain_atlas['bad ch'])
    return ts.groupby('bipolar_pairs').mean('bipolar_pairs')


def average_timeseries_lobar(timeseries):
    ts = deepcopy(timeseries)
    subject = ts['events'].data[0]['subject']
    experiment = ts['events'].data[0]['experiment']

    brain_atlas = get_logan_tal(subject=subject, experiment='FR1')

    # Over-ride TimeSeries channels
    rois = brain_atlas['hemi'] + ' ' + brain_atlas['lobe']
    ts['bipolar_pairs'].data = rois
    # Only select good channels and then average into rois
    ts = ts.sel(bipolar_pairs=~brain_atlas['bad ch'])
    return ts.groupby('bipolar_pairs').mean('bipolar_pairs')


def average_timeseries_frequency(timeseries):
    ts = deepcopy(timeseries)
    freq_names = ['theta' if x else 'hfa' for x in ts['frequency'] <= 8]
    ts['frequency'] = freq_names
    return ts.groupby('frequency').mean('frequency')


def split_rec_bl(ts, inclusion_minimum_number=20):
    """Returns out recalls, baselines, and a boolean on whether or not to included the data"""
    recs = ts.sel(events=ts['events'].data['type'] == 'REC_WORD')
    bl = ts.sel(events=ts['events'].data['type'] != 'REC_WORD')
    included = False
    if ((len(recs['events'].data) >= inclusion_minimum_number)
            & (len(bl['events'].data) >= inclusion_minimum_number)):
        included = True
    return recs, bl, included


# For FR1 and catFR1
def get_logan_tal(subject, experiment='FR1'):
    load_path = '/scratch/loganf/subject_brain_atlas/{}/{}_tal_indiv.npy'
    if os.path.exists(load_path.format(experiment, subject)):
        return np.load(load_path.format(experiment, subject))
    print('Warning could not find {}'.format(load_path.format(experiment, subject)))
    return


# For pyFR
def get_matlab_tal_rois(subject, experiment='pyFR'):
    """Returns hemisphere and lobe for subject, with all electrodes even bad ones included"""
    mp, bp, tal = get_sub_tal(subject=subject, experiment=experiment, exclude_bad=False)
    # Location of hippocampus electrodes...
    hpc_locs = np.where(tal.Loc5 == 'Hippocampus')
    # Hemisphere information e.g. left/right
    hemi = np.array(map(lambda s: s.split('Cerebrum')[0], tal.Loc1))
    # Lobar information, e.g. frontal lobe
    lobe = tal.Loc2
    # Reset lobar information of hpc electrodes to Hpc
    lobe[hpc_locs] = 'Hippocampus'
    # Return array of info like Left Hippocampus Left Frontal etc.
    roi_names = np.array(map(''.join, zip(hemi, lobe)))
    return roi_names


def scipy_zscore(timeseries, dim='events'):
    """Z-scores Data using scipy by default along events axis, does so for multiple sessions
    Parameters
    ----------
    timeseries: TimeSeriesX, SINGLE SESSION ONLY
    dim: str, by default 'events',
            dimension to z-score over
    """
    # Convert dim to axis for use in scipy
    ts = deepcopy(timeseries)
    dim_to_axis = dict(zip(ts.dims, xrange(len(ts.dims))))
    axis = dim_to_axis[dim]
    ts.data = zscore(ts, axis)
    return ts
    """ Below does not work inside the function but it does outside...?
    dim_to_axis = dict(zip(timeseries.dims, xrange(len(timeseries.dims))))
    axis = dim_to_axis[dim]

    # Go through each session and z-score relative to itself.
    z_data = []
    for sess in np.unique(timeseries['events'].data['session']):
        curr_sess = timeseries.sel(events=timeseries['events'].data['session'] == sess)
        curr_sess.data = zscore(curr_sess, axis)
        z_data.append(curr_sess)

    return TimeSeriesX.concat(z_data, 'events')
    """


def paired_ttest(ts_a, ts_b, dim='events', nan_policy='propagate'):
    """ Serves as a wrapper around scipy.stats.ttest_rel
    Calculates the T-test on TWO RELATED samples of scores, ts_a and ts_b.

    This is a two-sided test for the null hypothesis that 2 related or
    repeated samples have identical average (expected) values.

    Parameters
    ----------
    ts_a, ts_b : TimeSeriesX
        The TimeSeriesX must have the same shape.
    dim : str, optional, by default 'events
        dimensions along which to compute test.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan. 'propagate' returns nan,
        'raise' throws an error, 'omit' performs the calculations ignoring nan
        values. Default is 'propagate'.

    Returns
    -------

    References
    ----------
    http://en.wikipedia.org/wiki/T-test#Dependent_t-test
    """
    # Axis -> Dim functionality
    dim_to_axis = dict(zip(ts_a.dims, xrange(len(ts_a.dims))))
    axis = dim_to_axis[dim]

    # Run the t-test
    t, p = ttest_rel(a=ts_a.data,
                     b=ts_b.data,
                     axis=axis,
                     nan_policy=nan_policy)

    # Create a copy timeseries to serve as a shell for the t-stats
    tstats = ts_a.create(data=np.array(ts_a),
                         dims=ts_a.dims,
                         coords=ts_a.coords,
                         samplerate=ts_a['samplerate'].data,
                         attrs=ts_a.attrs,
                         name=ts_a.name).mean('events')

    # Reset data and add nice formatting convience
    tstats.data = t
    tstats.name = 'Recall-Deliberation t-statistics'
    # tstats = tstats.drop('events')
    if 'subject' not in tstats.coords.keys():
        tstats['subject'] = ts_a['events'].data['subject'][0]

    return tstats


def independent_ttest(ts_a, ts_b, dim='events', equal_var=True, nan_policy='propagate'):
    """ Serves as a wrapper around scipy.stats.ttest_ind
    Calculates the T-test for the means of *two independent* samples of scores.

    This is a two-sided test for the null hypothesis that 2 independent samples
    have identical average (expected) values. This test assumes that the
    populations have identical variances by default.

    Parameters
    ----------
    ts_a, ts_b : TimeSeriesX
        The TimeSeriesX must have the same shape.
    dim : str, optional, by default 'events
        dimensions along which to compute test.
    equal_var : bool, optional
        If True (default), perform a standard independent 2 sample test
        that assumes equal population variances [1]_.
        If False, perform Welch's t-test, which does not assume equal
        population variance [2]_.

        .. versionadded:: 0.11.0
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan. 'propagate' returns nan,
        'raise' throws an error, 'omit' performs the calculations ignoring nan
        values. Default is 'propagate'.


    Returns
    -------
    statistic : float or array
        The calculated t-statistic.
    pvalue : float or array
        The two-tailed p-value.

    Notes
    -----
    We can use this test, if we observe two independent samples from
    the same or different population, e.g. exam scores of boys and
    girls or of two ethnic groups. The test measures whether the
    average (expected) value differs significantly across samples. If
    we observe a large p-value, for example larger than 0.05 or 0.1,
    then we cannot reject the null hypothesis of identical average scores.
    If the p-value is smaller than the threshold, e.g. 1%, 5% or 10%,
    then we reject the null hypothesis of equal averages.
    """
    # Axis -> Dim functionality
    dim_to_axis = dict(zip(ts_a.dims, xrange(len(ts_a.dims))))
    axis = dim_to_axis[dim]

    # Run the t-test
    t, p = ttest_ind(a=ts_a.data,
                     b=ts_b.data,
                     axis=axis,
                     equal_var=equal_var,
                     nan_policy=nan_policy)

    # Create a copy timeseries to serve as a shell for the t-stats
    tstats = ts_a.create(data=np.array(ts_a),
                         dims=ts_a.dims,
                         coords=ts_a.coords,
                         samplerate=ts_a['samplerate'].data,
                         attrs=ts_a.attrs,
                         name=ts_a.name).mean('events')

    # Reset data and add nice formatting convience
    tstats.data = t
    tstats.name = 'Recall-Deliberation t-statistics'
    # tstats = tstats.drop('events')
    if 'subject' not in tstats.coords.keys():
        tstats['subject'] = ts_a['events'].data['subject'][0]

    return tstats


def make_indiv_subject_tstat_roi_plots(tstats):
    """Given a subject's tstatistics makes a plot of all rois"""
    num_rois = len(tstats['bipolar_pairs'])
    # x_axis = np.arange(-1150, 150, 20)
    x_axis = tstats['time'] * 1000
    fig, ax = plt.subplots(num_rois, figsize=(6, 24), sharex=True)

    for i in xrange(num_rois):
        ax[i].plot(x_axis, tstats[0, i], label='HFA (65-95 Hz)', color="#EC4E20")
        ax[i].plot(x_axis, tstats[1, i], label='theta (3-8 Hz)', color="#399AE7")
        ax[i].set_title(str(tstats[1, i]['bipolar_pairs'].data))
        ax[i].set_ylabel('Z-Power \n(T-stat)')

        ymin, ymax = ax[i].get_ylim()
        ax[i].vlines(x=0, ymin=ymin, ymax=ymax, linestyle='--', color='black')
        xmin, xmax = ax[i].get_xlim()
        ax[i].hlines(y=0, xmin=xmin, xmax=xmax, linestyle='--', color='black')

    # plt.xlim(-1150, 130)
    plt.tight_layout()
    plt.xlabel('Time (ms)', fontsize=16)

    return


# -------> Not using
def get_correct_time_axis(timeseries_time, window, desired_step):
    start_time = timeseries_time['time'][0].data + (window / 2.)
    end_time = timeseries_time['time'][-1].data - (window / 2.)
    timeseries_time['time'].data = np.arange(start_time, end_time, desired_step)
    return np.arange(start_time, end_time, desired_step)