"""Documentation in progress: 2-14-18 - LF
A SCRIPT USED TO LAUNCH THE REGIONAL TIMING COMPARASION FOR LOGANS MENTAL CHRONOMETRY ANALYSIS!

TODO: Make sure all the imports still work given name changes in API...
"""
# General Imports
import sys, os

# Add some dimension to your life, np=arr, pd=labeled arr, xr=labeled N-D arr
import numpy as np  # I say numpy like 'lumpy', no I don't mean num-pie
import pandas as pd  # For Undergrads that say "But matlab makes it so much easier to view things"
import xarray as xr  # N-D labels!

# Pizza ptsa stuff
from ptsa.data.TimeSeriesX import TimeSeriesX
from ptsa.data.readers import BaseEventReader, EEGReader, JsonIndexReader, TalReader
from ptsa.data.filters import MonopolarToBipolarMapper, ButterworthFilter, DataChopper

try:
    from ptsa.data.filters.MorletWaveletFilterCpp import MorletWaveletFilterCpp as MorletWaveletFilter
except:
    from ptsa.data.filters.MorletWaveletFilter import MorletWaveletFilter

# Stats
from scipy.stats import ttest_ind

# Import to allow for event creation
# sys.path.append('/home2/loganf/SecondYear/MentalChronometry_April_2018/')
from RetrievalCreation import RetrievalEventCreator

# Toolbox analysis rests upon
#sys.path.append('/home2/loganf/SecondYear/Functions/CML/')
from SpectralAnalysis.RollingAverage import sliding_mean_fast
from Utility.FrequencyCreator import logspace
from SpectralAnalysis.FastFiltering import butter_highpass_filter


class Subject(object):
    def __init__(self, subject, experiment, session, save=True, verbose=True,
                 outdir='/scratch/loganf/RegionTesting/', width=4, samplerate=500,
                 eeg_start=-1.25, eeg_end=1.25, eeg_buffer=1., freqs=None):
        """Initializes an object used as a blueprint for raw analysis
        ------
        INPUTS:
        subject: str, subject ID; subject IDs can be gathered from CML
                 get_subs function
        experiment: str; the experiment to analyze; experiments can be gathered
                    from CML get_exps function
        session: int; session of data to examine
        save: bool; whether to save the data; by default True
        outdir: str; the output directory to save to if save is True; by default my scratch dir
        width: int; width of the wavelet; by default 4
        samplerate:  int; see notes; by default 500
        eeg_start: float, start of eeg
        eeg_end: float, end of eeg
        eeg_buffer: float, buffer for convolution
        freqs: np.array; frequencies on with to apply the morlet wavelet;
               by default 30 logspaced freqs from 3-180 [only 15 examined ]
        ------
        OUTPUTS: da object
        """

        self.subject, self.experiment = subject, experiment
        self.session, self.save, self.outdir = session, save, outdir
        freqs = logspace(3, 180, 30) if freqs is None else freqs
        self.freqs = freqs[((freqs >= 3) & (freqs <= 8.1)) | (freqs >= 65)]
        self.width, self.samplerate, self.verbose = width, samplerate, verbose
        self.eeg_start, self.eeg_end, self.eeg_buffer = eeg_start, eeg_end, eeg_buffer
        return

    def set_possible_sessions(self):
        """Sets an attribute possible_session that is an array of all possible sessions for the subject"""
        if self.experiment == 'pyFR':
            evs = self.get_pyFR_events(self.subject)
            sessions = np.unique(evs['session'])

        else:
            jr = JsonIndexReader('/protocols/r1.json')
            # Get all possible sessions
            sessions = jr.aggregate_values(
                'sessions', subject=self.subject, experiment=self.experiment)

        # type(sessions) == set of strings; make them an array of intergers
        self.possible_sessions = np.array(sorted(map(int, sessions)))
        return

    def set_events(self):
        jr = JsonIndexReader('/protocols/r1.json')
        f = jr.aggregate_values('task_events', subject=self.subject,
                                experiment=self.experiment, session=self.session)
        if self.verbose: print('Setting all events....')
        self.events = BaseEventReader(filename=list(f)[0]).read()
        return

    def set_json_tals(self):
        """Sets the default monopolar, bipolar, and tal using json"""
        pairs_path = JsonIndexReader('/protocols/r1.json').get_value(
            'pairs', subject=self.subject, experiment=self.experiment)
        tal_reader = TalReader(filename=pairs_path)
        self.mp = tal_reader.get_monopolar_channels()
        self.bp = tal_reader.get_bipolar_pairs()
        self.json_tal = tal_reader.read()
        return

    def set_mlab_tals(self):
        """Sets the default monopolar, bipolar, and tals using matlab structures"""
        tal_path = '/data/eeg/{}/tal/{}_talLocs_database_bipol.mat'.format(self.subject, self.subject)
        tal_reader = TalReader(filename=tal_path)
        self.mlab_tal = tal_reader.read()
        self.mp = tal_reader.get_monopolar_channels()
        self.bp = tal_reader.get_bipolar_pairs()
        return

    def set_logan_tals(self):
        tal_path = '/scratch/loganf/subject_brain_atlas/{}/{}_tal_indiv.npy'
        tal_path = tal_path.format(self.experiment, self.subject)
        if os.path.exists(tal_path):
            self.tal = np.load(tal_path)
        else:
            self.set_json_tals()
            self.tal = self.json_tal
            if self.verbose:
                print('Setting json_tal to .tal due to failure in loading Logan tals')
        return

    def set_eeg(self, events=None, bipolar=True):
        """Sets bp eeg for subject, by default uses set_events, otherwise pass in events to override"""

        # Load eeg from start to end, include buffer for edge contamination
        default = 'passed events'
        if events is None:
            events = self.events
            default = 'default events'

        eeg_reader = EEGReader(events=events, channels=self.mp, start_time=self.eeg_start,
                               end_time=self.eeg_end, buffer_time=self.eeg_buffer)
        if self.verbose:
            print('Setting eeg for {}....'.format(default))
        self.eeg = eeg_reader.read()
        if bipolar:
            self.eeg = self.bipolar_filter(data=self.eeg, bipolar_pairs=self.bp)
        return

    def morlet(self, data=None, verbose=True, output='power'):
        """Applies a butterworth filter, morlet wavelet filter, and log trans on inputted data
        output: str, can be power, phase, or both, by default just power
        """
        # By default use the attribute eeg for the morlet wavelet otherwise the passed data
        data = self.eeg if data is None else data
        # Apply butter filter to remove line noise
        data = self.linenoise_filter(data)

        # Morlet-Wavelet gets the frequencies' power and phase
        if self.verbose:
            print('Starting Morlet Wavelet....')
        wf = MorletWaveletFilter(time_series=data,
                                 freqs=self.freqs,
                                 output=output,
                                 width=self.width)

        pow_wavelet, phase_wavelet = wf.filter()

        # Remove the buffer space and log transform the data
        if self.verbose:
            print('Starting Log transform....')
        np.log10(pow_wavelet.data, out=pow_wavelet.data);
        # pow_wavelet = np.log10(pow_wavelet)


        if self.verbose:
            print('Removing the buffer')

        pow_wavelet = pow_wavelet.remove_buffer(duration=self.eeg_buffer)
        self.power = pow_wavelet

        if output == 'both':
            phase_wavelet = phase_wavelet.remove_buffer(duration=self.eeg_buffer)
            self.phase = phase_wavelet
        return

    # --------> Helper Functions!
    def highpass_filter(self, data, cutoff=.1, order=4):
        """Applies a highpass filter on the data using a butterworth to remove slow drifting eeg signal
        --------
        INPUTS:
        data: np.array, TimeSeriesX; data on which to apply the filter over
        cutoff: float, by default .1, frequency to cutoff
        order: the order of the filter, by default 4
        -------
        OUTPUTS:
        y: timeseriesX with slow signal frequencies < cutoff removed
        """
        y = butter_highpass_filter(data=data,
                                   cutoff=cutoff,
                                   fs=float(data['samplerate']),
                                   order=order)
        return y

    @staticmethod
    def bipolar_filter(data, bipolar_pairs, verbose=True):
        """Filter monopolar data to bipolar data"""
        m2b = MonopolarToBipolarMapper(time_series=data, bipolar_pairs=bipolar_pairs)
        if verbose: print('Converting monopolar channels to bipolar pairs....')
        return m2b.filter()

    def linenoise_filter(self, data):
        """Apply butter filter to remove line noise Europe = 50Hz, America = 60 Hz"""
        line_noise = [58., 62.] if not 'FR' in self.subject else [48., 52.]
        # Apply butter filter to remove line noise
        b_filter = ButterworthFilter(time_series=data, freq_range=line_noise,
                                     filt_type='stop', order=4)
        if self.verbose:
            print('Removing {} linenoise....'.format(line_noise))
        return b_filter.filter()

    @staticmethod
    def get_pyFR_events(subject, experiment='pyFR'):
        """ Utility to get pyFR events

        Parameters
        ----------
        subject: str, subject ID, e.g. 'BW001'
        experiment: str, by default pyFR, experiment ID

        Returns
        -------
        behavioral_events: np.array, behavioral events OF ALL SESSIONS
        """
        """ Utility to get pyFR events

        :param subject: str, subject id
        :param experiment:
        :return:
        """
        event_path = '/data/events/{}/{}_events.mat'.format(experiment, subject)
        base_e_reader = BaseEventReader(filename=event_path, eliminate_events_with_no_eeg=True)
        behavioral_events = base_e_reader.read()
        return behavioral_events


class RegionTimingAnalysis_BurkeReplication(Subject):
    """An object used as a blueprint for the raw analysis of Logan's Mental Chronometry

    NOTES:
    if the subject's default sampling rate does not allow for an even division of 200 500ms windows
    from -1.25s to 1.25s then the subject's sampling rate AFTER applying a morlet wavelet will be
    resampled to 500ms before applying the mean. Else you'll get a correct but different number of
    time windows like 238 etc.

    """

    def __init__(self, **kwargs):
        """REQUIRED: subject = , experiment = , session =, """
        super(RegionTimingAnalysis_BurkeReplication, self).__init__(**kwargs)
        self.width = 7
        self.eeg_start = -2.
        self.eeg_end = 2.
        freqs = logspace(3, 95, 30)
        self.freqs = freqs[((freqs >= 3) & (freqs <= 8)) | (freqs >= 65)]

    def return_matched_events(self, set_to_events=False):
        """Return matched events for the subject"""
        if self.verbose:
            print('Creating Matched Events....')

        event_creator = RetrievalEventCreator(
            subject=self.subject, experiment=self.experiment, session=self.session,
            inclusion_time_before=2000, inclusion_time_after=1500, eeg_length=4000)

        events = event_creator.matched_events

        if set_to_events:
            self.events = events

        print(pd.Series(events['type']).value_counts())

        return events

    def moving_mean(self, window=.5, step=.01):
        """Applies a moving mean"""
        if self.verbose:
            print('Creating a moving averaged window....')
        self.power = sliding_mean_fast(self.power, window, step, 'time')
        self.power['frequency'] = self.freqs
        return

    def raw_analysis(self):
        """Run this function to call all the relevant steps in the analysis"""
        # Call relevant method and attributes

        # Set all possible sessions
        self.set_possible_sessions()

        # Load default monopolar, bipolar, and tal atlas
        self.set_json_tals()

        # Load Logan's modified tal atlas
        self.set_logan_tals()

        # Load all behavioral events
        self.set_events()
        self.matched_events = self.return_matched_events(set_to_events=False)
        self.matched_events = self.matched_events[self.matched_events['session'] == self.session]
        self.kind = 'delib'  # This will be used to save!

        # Set eeg and do morlet wavelet etc.
        self.set_eeg(self.matched_events, bipolar=True)
        self.morlet(data=self.eeg)
        self.moving_mean(window=.5, step=.01)
        self.eeg = self.power  # Rename so we don't erase it w/ below

        # --------> Average than z-score
        # First average into bands of theta and high frequency activity
        if self.verbose:
            print('Averaging into theta and HFA wavelet bands')

        band_names = map(lambda x: 'theta' if x else 'hfa', self.eeg.frequency.data <= 8.1)
        self.eeg['frequency'] = band_names
        self.eeg = self.eeg.groupby('frequency').mean('frequency')

        # Get whole session for z-score, apply a morlet wavelet
        self.set_whole_session(random_range=10, step_in_seconds=60)
        self.morlet(data=self.sess_data)

        # Get the randomly selected time points at exactly time 0, then apply z-score
        z_sess_data = self.power[:, :, :, len(self.power['time']) / 2]

        # Now average the z_score into bands
        z_sess_data['frequency'] = band_names
        z_sess_data = z_sess_data.groupby('frequency').mean('frequency')

        z_sess_mean = z_sess_data.mean('start_offsets')
        z_sess_std = z_sess_data.std('start_offsets')

        if self.verbose:
            print('Z-scoring the timeseries....')

        # Z-score the time series
        z_eeg = (self.eeg - z_sess_mean.data[:, :, None, None]) / z_sess_std.data[:, :, None, None]
        del (self.eeg)

        self.eeg = z_eeg
        return

        # ----> Z-scoring

    def set_whole_session(self, random_range=10, step_in_seconds=60):
        """Sets chopped bipolar data"""
        self._set_whole_session()
        if self.verbose: print('Randomizing offsets....')
        chopped = self.chop_whole_session(
            data=self.sess_data, random_range=random_range, step_in_seconds=step_in_seconds)
        chopped['channels'] = self.mp
        self.sess_data = self.bipolar_filter(data=chopped, bipolar_pairs=self.bp)

    def _set_whole_session(self):
        """Sets whole session of mp data for subject WHOLE DATA!!!! to attribute self.sess_data"""
        session_dataroot = self.events['eegfile'][0]
        sess_reader = EEGReader(session_dataroot=session_dataroot, channels=self.mp)
        if self.verbose: print('Setting whole session....')
        self.sess_data = sess_reader.read()

    @staticmethod
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
        offsets = np.arange(eeg.offsets.data[0], eeg.offsets.data[-1],
                            eeg.samplerate.data * step_in_seconds)

        # Randomize it such that it can vary from 50 seconds to 70 seconds
        randomize = np.random.randint(-random_range, random_range, len(offsets)) * 1000
        offsets = offsets - randomize

        # SANITY CHECK TO ENSURE RANDOMIZATION DID NOT OVERSHOOT THE INDEX!!!!!
        cannot_be_larger_than = eeg.time.size - (end_time * eeg.samplerate.data) - 1
        cannot_be_smaller_than = end_time * 1000

        # If we throw an error by loading before the session starts redo 1st point
        while offsets[0] <= cannot_be_smaller_than:
            print('offset {} cannot be smaller than {}'.format(offsets[0], cannot_be_smaller_than))
            print('Sanity Checking: redoing randomization of failed offset')
            offsets[0] = np.random.randint(-random_range, random_range) * 1000
        # Handling this by just dropping it seems...non ideal
        while offsets[-1] > cannot_be_larger_than - 2:
            offsets = offsets[:-1]

        return offsets

    # @staticmethod
    def chop_whole_session(self, data, random_range=10, step_in_seconds=60):
        """Uses randomly_spaced_timepoints and ptsa's DataChopper to chop data"""
        # Pizza ptsa stuff
        # Chop the data
        offsets = self.randomly_spaced_timepoints(
            eeg=data, start_time=-.5, end_time=.5,
            random_range=random_range, step_in_seconds=step_in_seconds)

        dc = DataChopper(start_offsets=offsets, session_data=data,
                         start_time=-.5, end_time=.5, buffer_time=1.0)
        chopped = dc.filter()
        return chopped

    def apply_zscore(self):
        """Applies a z-score using the sessions mean and std dev"""
        if self.verbose: print('Applying Z-score....')
        theta, hfa = self.split_hfa_theta(timeseries=self.eeg, average=True)
        theta_bl, hfa_bl = self.split_hfa_theta(timeseries=self.power[:, :, :, len(self.power[-1]) / 2], average=True)
        theta_mean, theta_std = theta_bl.mean('start_offsets'), theta_bl.std('start_offsets')
        hfa_mean, hfa_std = hfa_bl.mean('start_offsets'), hfa_bl.std('start_offsets')
        hfa = (hfa - hfa_mean.data[:, None, None]) / hfa_std.data[:, None, None]
        theta = (theta - theta_mean.data[:, None, None]) / theta_std.data[:, None, None]
        return theta, hfa

    # -----> Utlity
    def remove_bad_channels(self, data):
        if self.verbose: print('Removing bad channels, resetting names to lobar information....')
        # Select only good channels
        data = data.sel(bipolar_pairs=~self.tal['bad ch'])
        # Get hemi/lobe name for each ch that's not bad
        name = np.array([a['hemi'] + ' ' + a['lobe']
                         if a['lobe'] != u'' else u'discard'
                         for a in self.tal])[~self.tal['bad ch']]
        data['bipolar_pairs'] = name
        # Since we label any ch w/o lobe info as discard we'll then need to remove these
        good_channels = np.in1d(name, np.intersect1d(name, u'discard')) == False
        data = data.sel(bipolar_pairs=good_channels)
        return data

    @staticmethod
    def split_hfa_theta(timeseries, average=True):
        theta = timeseries.sel(frequency=(timeseries.frequency >= 3)
                                         & (timeseries.frequency <= 8.1))
        hfa = timeseries.sel(frequency=(timeseries.frequency >= 65))

        if average:
            return theta.mean('frequency'), hfa.mean('frequency')

        return theta, hfa

    @staticmethod
    def split_recalls_baselines(timeseries):
        recalls = timeseries.sel(events=timeseries.events.data['type'] == 'REC_WORD')
        baselines = timeseries.sel(events=timeseries.events.data['type'] != 'REC_WORD')
        return recalls, baselines

    # -------> Statistics
    @staticmethod
    def ttest(data):
        """Returns a timeseriesX after computing the independent within subj t-statistic"""
        shell = data[:, 0, :]
        print('Computing within subject t-statistic....')
        recs = data.sel(events=data['events'].data['type'] == 'REC_WORD')
        delib = data.sel(events=data['events'].data['type'] != 'REC_WORD')
        t, p = ttest_ind(recs, delib, 1)
        shell.data, shell['name'] = t, 't-statistic'
        return shell

    def average_roi(data, minimum=2):
        df = pd.DataFrame(pd.Series(data.bipolar_pairs).value_counts())
        valid_rois = df[df > minimum].dropna().index
        data = data.groupby('bipolar_pairs').mean('bipolar_pairs')
        data = data.sel(bipolar_pairs=valid_rois)
        return data