"""ZscoreWholeSession.py, author=LoganF
A script used to Z-score data using a subject's entire session's worth of data.
"""
# General Imports
import os, sys, math
from glob import glob  # Glob.glob is more fun to say
from time import time

# Add some dimension to your life, np=arr, pd=labeled arr, xr=labeled N-D arr
import numpy as np  # I say numpy like 'lumpy', no I don't mean num-pie
import pandas as pd  # "But matlab makes it so much easier to view things"
import xarray as xr  # N-D labels!

# Pizza ptsa stuff
from ptsa.data.filters import DataChopper
from ptsa.data.TimeSeriesX import TimeSeriesX
from ptsa.data.common import TypeValTuple, PropertiedObject
from ptsa.data.filters import MonopolarToBipolarMapper, ButterworthFilter

try:
    from ptsa.data.filters.MorletWaveletFilterCpp import MorletWaveletFilterCpp as MorletWaveletFilter
except:
    from ptsa.data.filters.MorletWaveletFilter import MorletWaveletFilter

try:
    from ptsa.data.readers import BaseEventReader, TalReader, EEGReader
    from ptsa.data.readers.IndexReader import JsonIndexReader
except ImportError:
    from ptsa.data.readers import JsonIndexReader, EEGReader

sys.path.append('/home2/loganf/SecondYear/Functions/CML/')
from Utility.FrequencyCreator import logspace


def get_events(subject, experiment, session):
    jr = JsonIndexReader('/protocols/r1.json')
    f = jr.aggregate_values('task_events', subject=subject,
                            experiment=experiment, session=session)
    print('Setting events....')
    events = BaseEventReader(filename=list(f)[0]).read()
    return events


def set_brain_atlas(subject, experiment):
    """Loads a brain atlas (essentially modified tal) for the subject"""
    path = '/scratch/loganf/subject_brain_atlas/{}/{}_tal_indiv.npy'
    if os.path.exists(path.format(experiment, subject)):
        atlas = np.load(path.format(experiment, subject))

        mp = np.unique(np.concatenate(list(zip(atlas.channel_1, atlas.channel_2)))).astype('<U6')
        bp = np.array(list(zip(atlas.channel_1, atlas.channel_2)), dtype=[('ch0', '<U6'), ('ch1', '<U6')])
        return mp.view(np.recarray), bp.view(np.recarray), atlas


def get_processed_events(subject, experiment):  # , session):
    path = '/scratch/loganf/25_subject_debug/{}/events/{}.npy'.format(experiment, subject)
    # path='/scratch/loganf/RegionTiming/Events/{}_{}_{}.npy'.format(subject,experiment,session)
    evs = np.load(path)
    return evs.view(np.recarray)


def load_whole_session_eeg(dataroot, mp, bp):
    session_reader = EEGReader(session_dataroot=dataroot, channels=mp,
                               start_time=-2.,
                               end_time=2.,
                               buffer_time=1.)
    eeg = session_reader.read()
    m2b = MonopolarToBipolarMapper(time_series=eeg, bipolar_pairs=bp)
    eeg = m2b.filter()
    eeg['bipolar_pairs'].data = bp
    return eeg


def randomly_spaced_timepoints(eeg, start_time, end_time,
                               random_range=10, step_in_seconds=60):
    """Return randomly selected offset points for an inputted array
    ------
    INPUTS:
    eeg: TimeSeriesX, a continous session worth of eeg
    start_time: float, the time relative to each offset to start recording
    end_time: float, the time relative to each offset to stop recording
    random_range: int, used to construct the +/- jitter
    step_in_seconds: int, number of seconds along which we step
    ------
    OUTPUTS:
    offsets: np.array, represents the offsets along which one can chop a
                       continous eeg recording
    """
    offsets = np.arange(eeg.offsets.data[0],
                        eeg.offsets.data[-1],
                        eeg.samplerate.data * step_in_seconds)

    # Randomize it such that it can vary from 50 seconds to 70 seconds
    randomize = np.random.randint(-random_range,
                                  random_range,
                                  len(offsets)) * 1000
    offsets = offsets - randomize

    # SANITY CHECK TO ENSURE RANDOMIZATION DID NOT OVERSHOOT THE INDEX!!!!!
    cannot_be_larger_than = eeg.time.size - (end_time * eeg.samplerate.data) - 1
    cannot_be_smaller_than = end_time * 1000

    # If we'll throw an error by loading before the session starts redo
    # First randomization point
    while offsets[0] <= cannot_be_smaller_than:
        print('offset {} cannot be smaller than {}'.format(
            offsets[0], cannot_be_smaller_than))
        print('Sanity Checking: redoing randomization of failed offset')
        offsets[0] = np.random.randint(-random_range, random_range) * 1000
    # Handling this by just dropping it seems...non ideal
    while offsets[-1] > cannot_be_larger_than - 2:
        offsets = offsets[:-1]

    return offsets


def split_hfa_theta(timeseries, average=True):
    theta = timeseries.sel(frequency=(timeseries.frequency >= 3)
                                     & (timeseries.frequency <= 8))
    hfa = timeseries.sel(frequency=(timeseries.frequency >= 65))

    if average:
        return theta.mean('frequency'), hfa.mean('frequency')

    return theta, hfa


def z_score_from_whole_session(subject, experiment, session, width=7,
                               freqs=None, random_range=10, step_in_seconds=60, average_bands=True):
    if freqs is None:
        freqs = logspace(3, 95, 30)
        freqs = freqs[(freqs <= 8) | (freqs >= 65)]
    print('Morlet wavelet frequencies = {}'.format(freqs))
    print('Morlet wavelet cycles= {}'.format(width))

    evs = get_processed_events(subject, experiment)
    # evs = evs[evs['session']==session]
    dataroot = evs['eegfile'][0]
    mp, bp, atlas = set_brain_atlas(subject, experiment)

    # subject, experiment, session, _, _ = os.path.basename(dataroot).split('_')

    eeg = load_whole_session_eeg(dataroot, mp, bp)

    offsets = randomly_spaced_timepoints(
        eeg, -2., 2., random_range=random_range, step_in_seconds=step_in_seconds)

    dc = DataChopper(start_offsets=offsets, session_data=eeg,
                     start_time=-2., end_time=2., buffer_time=1.0)
    chopped = dc.filter()
    chopped['bipolar_pairs'] = bp

    # Germany (All of Europe?) = 50Hz, America = 60 Hz
    line_noise = [58., 62.] if not 'FR' in subject else [48., 52.]

    # Apply butter filter to remove line noise
    b_filter = ButterworthFilter(time_series=chopped,
                                 freq_range=line_noise,
                                 filt_type='stop',
                                 order=4)

    data = b_filter.filter()

    wf = MorletWaveletFilter(
        time_series=data, freqs=freqs, output='power', width=width, verbose=True)

    pow_wavelet, phase_wavelet = wf.filter()
    pow_wavelet = pow_wavelet.rename({'start_offsets': 'events'})

    pow_wavelet['buffer_time'] = 1.
    pow_wavelet = pow_wavelet.remove_buffer(pow_wavelet.buffer_time)

    # Log transform the data
    np.log10(pow_wavelet.data, out=pow_wavelet.data);

    # Get at time point zero
    pow_wavelet = pow_wavelet.sel(time=len(pow_wavelet['time'].data) / 2)

    pow_wavelet['frequency'].data = freqs
    if average_bands:
        band_names = map(lambda x: 'theta' if x else 'hfa', pow_wavelet.frequency.data <= 8)
        pow_wavelet['frequency'] = band_names
        ts = pow_wavelet.groupby('frequency').mean('frequency')
        return ts

    elif not average_bands:
        return pow_wavelet


        # theta, hfa = split_hfa_theta(pow_wavelet)
        # ts = TimeSeriesX.concat([theta,hfa], 'frequency')
        # ts['frequency'] = np.array(['theta', 'hfa'])

        # save_path = '/scratch/loganf/RegionTiming/Zscores/{}_{}/{}_{}_{}'.format(
        # step_in_seconds, random_range, subject, experiment, session)
        # ts.to_hdf(save_path)
        # return ts


class ZScoreMethods(object):
    """An object to run all raw z-scoring baseline criteria for MentalChronometry Analysis"""

    def __init__(self, subject, experiment, session=None, save=True,
                 outdir='/scratch/loganf/RegionTimingZscores/', main=False):
        # Init relevant passed args
        self.subject, self.experiment = subject, experiment
        self.session, self.save, self.outdir = session, save, outdir
        self.save_path = self.outdir + '{}/{}/z_scores/{}/{}'

        # Get events and electrodes
        self.set_events()
        self.set_electrodes()

        # Set freqs, only care about hfa and theta right now
        freqs = logspace(3, 95, 30)
        self.freqs = freqs[((freqs >= 3) & (freqs <= 8)) | (freqs >= 65)]

        # Create a directory structure for the subject
        self.path_creator()

        if main:
            self.main()
        return

    # -----------> Get Events


    def set_events(self):
        """Sets all events"""
        self.events = get_sub_events(self.subject,
                                     self.experiment)
        self.events = self.events[self.events['list'] > 0]
        return

    def set_distracter_events(self):
        """Sets attribute z_events to instance whereby z_events corresponds to distracter period"""
        events = self.events
        self.events = self.events[self.events['list'] > 0]
        distracters = events[(events['type'] == 'DISTRACT_START')
                             | (events['type'] == 'DISTRACT_END')]
        distracters = distracters.view(np.recarray)
        self.z_events = distracters
        return

    def set_countdown_events(self):
        """Sets attribute z_events to instance whereby z_events corresponds to countdown period"""
        events = self.events
        self.events = self.events[self.events['list'] > 0]
        countdown = events[(events['type'] == 'COUNTDOWN_START')
                           | (events['type'] == 'COUNTDOWN_END')]
        countdown = countdown.view(np.recarray)
        self.z_events = countdown
        return

    # -----------> Get Electrodes


    def set_electrodes(self):
        """Returns talirach using NEW METHOD and JsonIndexReader!
        NOTES: DOES NOT FILTER ANY BAD CHANNELS OUT use electrode categories_reader function for this"""
        jr = JsonIndexReader('/protocols/r1.json')
        pairs_path = jr.get_value('pairs', subject=self.subject, experiment=self.experiment)
        tal_reader = TalReader(filename=pairs_path)
        self.mp = tal_reader.get_monopolar_channels()
        self.bp = tal_reader.get_bipolar_pairs()
        self.tals = tal_reader.read()
        return

    # -----------> Get EEG


    def set_distracter_eeg(self, bipolar=True):
        """Sets attribute eeg to instance whereby eeg corresponds to distracter period"""
        self.set_distracter_events()
        evs = self.z_events

        all_z_scores = []

        # We have to do this tacky business because they have variable
        # Start-stop length, so we'll just run the analysis on each list level
        for sess in np.unique(evs['session']):
            sess_data = evs[(evs['session'] == sess)]
            for trial in np.unique(sess_data['list']):
                # Trial level data
                data = sess_data[sess_data['list'] == trial]
                # When to stop recording from start
                end = float(np.diff(data['mstime'])[0] / 1000)
                # Remove Ends since we're not looking at them
                data = data[data['type'] == u'DISTRACT_START']

                # Get eeg
                eeg_reader = EEGReader(events=data, channels=self.mp,
                                       start_time=0., end_time=end, buffer_time=1.0)
                eeg = eeg_reader.read()

                # bipolar transformation
                if bipolar:
                    m2b = MonopolarToBipolarMapper(time_series=eeg,
                                                   bipolar_pairs=self.bp)
                    eeg = m2b.filter()
                    eeg['bipolar_pairs'].data = self.bp

                # Remove line noise, morlet wavelet and log trans data
                power, _ = self.morlet(eeg)
                # print(power['frequency'])

                # Average into theta and hfa then concat the data together
                theta = power.sel(frequency=(power.frequency >= 3) & (power.frequency <= 8)).mean(dim='frequency')
                hfa = power.sel(frequency=(power.frequency >= 65)).mean(dim='frequency')
                data = TimeSeriesX.concat([theta, hfa], 'frequency')
                data['frequency'] = np.array(['theta', 'hfa'])

                # Get data's mean and std, concat together and add to list
                mean = data.mean('time')
                std = data.std('time')
                z_data = TimeSeriesX.concat([mean, std], 'z_score')
                z_data['z_score'] = np.array(['mean', 'std'])
                all_z_scores.append(z_data)

        # Make a timeseries object out of all the data
        all_z_scores = TimeSeriesX.concat(all_z_scores, 'events')
        self.eeg = all_z_scores
        if self.save:
            self.eeg.to_hdf(self.save_path.format(self.subject, self.experiment, 'math', self.subject))
        return

    def set_countdown_eeg(self, bipolar=True):
        """Sets attribute eeg to instance whereby eeg corresponds to countdown period"""
        self.set_countdown_events()
        evs = self.z_events
        events = evs[evs['type'] == 'COUNTDOWN_START']
        end_time = np.diff(evs.mstime)[0] / 1000.
        # Load each countdown start from the start until the end
        eeg_reader = EEGReader(events=events, channels=self.mp,
                               start_time=0., end_time=end_time, buffer_time=1.0)
        self.eeg = eeg_reader.read()
        if bipolar:
            m2b = MonopolarToBipolarMapper(time_series=self.eeg, bipolar_pairs=self.bp)
            self.eeg = m2b.filter()

        # Remove line noise, morlet wavelet and log trans data
        power, _ = self.morlet(self.eeg)
        # print(power['frequency'])

        # Average into theta and hfa then concat the data together
        theta = power.sel(frequency=(power.frequency >= 3) & (power.frequency <= 8)).mean(dim='frequency')
        hfa = power.sel(frequency=(power.frequency >= 65)).mean(dim='frequency')
        data = TimeSeriesX.concat([theta, hfa], 'frequency')
        data['frequency'] = np.array(['theta', 'hfa'])

        # Get data's mean and std, concat together and save
        mean = data.mean('time')
        std = data.std('time')
        z_data = TimeSeriesX.concat([mean, std], 'z_score')
        z_data['z_score'] = np.array(['mean', 'std'])
        self.eeg = z_data

        if self.save:
            self.eeg.to_hdf(self.save_path.format(self.subject, self.experiment, 'countdown', self.subject))
        return

    def set_whole_session(self, session=None, bipolar=True):
        """Sets a whole session worth of data, functionality currently absent"""
        evs = self.events
        if session == None:
            session = self.session
        dataroot = str(evs[evs['session'] == session][0]['eegfile'])
        # Load in entire session of eeg
        session_reader = EEGReader(session_dataroot=dataroot,
                                   channels=self.mp)
        self.eeg = session_reader.read()

        if bipolar:
            m2b = MonopolarToBipolarMapper(time_series=self.eeg,
                                           bipolar_pairs=self.bp)
            self.eeg = m2b.filter()
            self.eeg['bipolar_pairs'].data = self.bp
        return

    def whole_session(self, step_in_seconds=5, random_range=10):
        """
        Construct a Z-scores of whole session every step_in_seconds seconds,
        With random of +/- random_range seconds
        """
        subj, exp, freqs = self.subject, self.experiment, self.freqs
        exp = self.experiment
        jr = JsonIndexReader('/protocols/r1.json')
        sessions = np.array(list(jr.aggregate_values('sessions', experiment=exp, subject=subj)), dtype=int)

        # Butterworth -> Morlet wavelet filter
        for sess in sessions:
            try:

                save_path = self.outdir + '{}/{}/z_scores/{}s/sess_{}'.format(
                    self.subject, self.experiment, str(step_in_seconds), sess)

                # if os.path.exists(save_path.format(
                # self.subject, self.experiment,str(step_in_seconds), sess)):
                # load = save_path.format(self.subject, self.experiment, str(step_in_seconds), sess)
                if os.path.exists(save_path):
                    test_load = TimeSeriesX.from_hdf(save_path)
                    check = np.array([test_load.shape[0], test_load.shape[1]])
                    """Sanity check to ensure that z_score (mean, std) and frequency (hfa, theta) are correct len
                    if it is then the data doesn't need to be reprocessed, otherwise it does."""
                    if (all(check == 2) & ('z_score' in test_load.dims)):  # Means that they have z_score by freq
                        continue
                # print(subj, sess)
                s = time()
                whole_sess_zscore = BurkeZscoreNormalization(subject=subj, experiment=exp,
                                                             start_time=-.5, end_time=.5,
                                                             session=sess, freqs=freqs,
                                                             replace_zeroes=False, random_range=random_range,
                                                             step_in_seconds=step_in_seconds)
                # Mean across theta and HFA then recompose the data shape
                # print(whole_sess_zscore['frequency'])

                theta = whole_sess_zscore.sel(frequency=((whole_sess_zscore.frequency.data >= 3)
                                                         & (whole_sess_zscore.frequency.data <= 8))
                                              ).mean(dim='frequency')
                theta['frequency'] = 'theta'

                hfa = whole_sess_zscore.sel(frequency=whole_sess_zscore.frequency.data >= 65
                                            ).mean(dim='frequency')
                hfa['frequency'] = 'hfa'

                whole_sess_zscore = TimeSeriesX.concat([theta, hfa], dim='frequency')
                data = whole_sess_zscore[:, :, :, len(whole_sess_zscore.time) / 2]  # .mean('time')

                # Save only the mean, median, and std
                mean = data.mean('events')
                median = data.median('events')
                std = data.std('events')
                z_data = TimeSeriesX.concat((mean, median, std), dim='z_score')
                z_data['z_score'] = np.array(['mean', 'median', 'std'])
                # whole_sess_zscore.sel(bipolar_pairs=ts.bipolar_pairs.data)
                if self.save:
                    z_data.to_hdf(save_path)  # .format(
                    # self.subject, self.experiment,str(step_in_seconds), sess))
                # return whole_sess_zscore
                print(time() - s)
            except MemoryError:
                continue
            except Exception as e:
                print(e)
                pass
        return

    # -----------> General Utility


    def morlet(self, data, verbose=True):
        """Applies a butterworth filter, morlet wavelet filter, and log trans on inputted data"""
        subject, freqs = self.subject, self.freqs

        # Germany (All of Europe?) = 50Hz, America = 60 Hz
        line_noise = [58., 62.] if not 'FR' in subject else [48., 52.]

        # Apply butter filter to remove line noise
        b_filter = ButterworthFilter(time_series=data, freq_range=line_noise,
                                     filt_type='stop', order=4)
        data = b_filter.filter()
        # Morlet-Wavelet gets the frequencies' power and phase try using complied if not default to non-compiled
        try:
            from ptsa.data.filters.MorletWaveletFilterCpp import MorletWaveletFilterCpp
            wf = MorletWaveletFilterCpp(time_series=data,
                                        freqs=freqs,
                                        output='both',
                                        width=4)
            pow_wavelet, phase_wavelet = wf.filter()

        except:
            wf = MorletWaveletFilter(time_series=data,
                                     freqs=freqs,
                                     output='both',
                                     width=4)

            pow_wavelet, phase_wavelet = wf.filter()

        # Remove the buffer space and log transform the data
        # pow_wavelet = pow_wavelet.remove_buffer(duration=1.)
        pow_wavelet = np.log10(pow_wavelet)
        pow_wavelet = pow_wavelet.remove_buffer(duration=1.)
        phase_wavelet = phase_wavelet.remove_buffer(duration=1.)
        return pow_wavelet, phase_wavelet

    def path_creator(self):
        """Creates relevant directories in scratch folder"""

        def make_path(path):
            if not os.path.exists(path):
                os.makedirs(path)

        subject, experiment, outdir = self.subject, self.experiment, self.outdir

        root = self.outdir + self.subject + '/'
        make_path(root)
        root = root + self.experiment + '/'
        proc = root + 'processed/'
        raw = root + 'raw/'
        all_recs = raw + 'all_recs/'
        delibs = raw + 'deliberation/'
        z_scores = root + 'z_scores/'
        z_math = z_scores + 'math/'
        z_count = z_scores + 'countdown/'
        z_5 = z_scores + '5s/'
        z_10 = z_scores + '10s/'
        z_60 = z_scores + '60s/'

        proc_all = proc + 'all_recs/'
        proc_delib = proc + 'deliberation/'

        # Iterate over all of them and create the path
        my_list = [root, proc, raw, all_recs, delibs, z_scores, z_10,
                   z_math, z_count, z_5, z_60,
                   proc_all, proc_delib]

        for p in my_list:
            make_path(p)

    def main(self):
        """Run all five kind of z-scoring"""
        self.whole_session(step_in_seconds=5, random_range=10)
        self.whole_session(step_in_seconds=60, random_range=10)
        self.whole_session(step_in_seconds=10, random_range=5)
        self.set_countdown_eeg()
        self.set_distracter_eeg()
        return