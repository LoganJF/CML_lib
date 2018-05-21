"""Documentation in progress: 5-21-18 - LF

A SCRIPT USED TO LAUNCH THE REGIONAL TIMING COMPARASION FOR LOGANS MENTAL CHRONOMETRY ANALYSIS!


UPDATE 5-21-18:
Updated script so that now it will check a monopoalr channel and see if it's entirely zero,
if so it will remove that channel and any bipolar pair referenced to that channel as being
a possible valid channel


TODO: Make sure all the imports still work given name changes in API...
"""
# General Imports
import os
from matplotlib import pyplot as plt

# Add some dimension to your life, np=arr
import numpy as np  # I say numpy like 'lumpy', no I don't mean num-pie

# Stats
from scipy.stats import zscore

# Pizza ptsa stuff
from ptsa.data.TimeSeriesX import TimeSeriesX
from ptsa.data.readers import BaseEventReader, EEGReader, JsonIndexReader, TalReader
from ptsa.data.filters import MonopolarToBipolarMapper, ButterworthFilter
try:
    from ptsa.data.filters.MorletWaveletFilterCpp import MorletWaveletFilterCpp as MorletWaveletFilter
except:
    from ptsa.data.filters.MorletWaveletFilter import MorletWaveletFilter

# Relative imports from toolbox
#import sys
#sys.path.append('/home2/loganf/SecondYear/CML_lib/')
from RetrievalCreationHelper import create_matched_events #as create_retrieval_and_matched_deliberation
from SpectralAnalysis.RollingAverage import sliding_mean_fast
from Utility.FrequencyCreator import logspace
from SpectralAnalysis.FastFiltering import butter_highpass_filter


class Subject(object):
    jr_scalp = JsonIndexReader('/protocols/ltp.json')
    jr = JsonIndexReader('/protocols/r1.json')

    def __init__(self, subject, experiment, session,
                 eeg_start=-1.25, eeg_end=.25, eeg_buffer=1.,
                 bipolar=True, width=5, freqs=None, resampled_rate=500.,
                 verbose=True,
                 save=True, outdir='/scratch/loganf/MentalChronometry_May2018/'):
        """Initializes an object used as a blueprint for raw analysis
        ------
        INPUTS:
        subject: str, subject ID; subject IDs can be gathered from CML
                 get_subs function
        experiment: str; the experiment to analyze; experiments can be gathered
                    from CML get_exps function
        session: int; session of data to examine
        eeg_start: float, start of eeg
        eeg_end: float, end of eeg
        eeg_buffer: float, buffer for convolution
        bipolar: bool; default True,
                 whether to load the subject's bipolar or monopolar montage
        width: int; width of the wavelet; by default 5
        freqs: np.array; frequencies on with to apply the morlet wavelet;
               by default 30 logspaced freqs from 3-95 [only 13 examined ]
        resampled_rate: frequency at which we want to downsample the signal to after
                        computing power wavelets
        verbose: bool; whether to print out the steps of the code along the way
        save: bool; whether to save the data; by default True
        outdir: str; the output directory to save to if save is True; by default my scratch dir
        ------
        OUTPUTS: da object
        """
        # Initialize passed parameters/defaults
        self.subject=subject
        self.experiment = experiment
        self.session=session
        self.save = save
        self.outdir= outdir
        freqs = logspace(3, 95, 30) if freqs is None else freqs
        self.freqs = freqs[((freqs >= 3) & (freqs <= 8.1)) | (freqs >= 65)]
        self.resampled_rate = resampled_rate
        self.width = width
        self.verbose = verbose
        self.bipolar = bipolar
        self.eeg_start = eeg_start
        self.eeg_end = eeg_end
        self.eeg_buffer = eeg_buffer

        # Initialize relevant parameters we'll set as None
        self.possible_sessions = None
        self.events = None
        self.matched_events = None
        self.encoding_events = None
        self.mp = None
        self.bp = None
        self.json_tal = None
        self.mlab_tal = None
        self.tal = None
        self.eeg = None
        self.power = None
        self.phase = None

        # These parameters will be over-riden by whatever inputs user uses in
        # set_matched_retrieval_deliberation_events
        self.rec_inclusion_before = None
        self.rec_inclusion_after = None
        self.remove_before_recall = None
        self.remove_after_recall = None
        self.match_tolerance = None
        self.bad_channels = None

        return

    ############ Methods to set appropriate values on the instance ############

    # ------> Figure out how many sessions there are

    def set_possible_sessions(self):
        """Sets an attribute possible_session that is an array of all possible sessions for the subject"""
        # Get all possible sessions

        if self.experiment == 'pyFR':
            event_path = '/data/events/pyFR/{}_events.mat'.format(self.subject)
            base_e_reader = BaseEventReader(filename=event_path,
                                            eliminate_events_with_no_eeg=True)
            evs = base_e_reader.read()
            sessions = np.unique(evs['session'])

        # ltpFR2
        if self.experiment in self.jr_scalp.experiments():
            sessions = list(self.jr_scalp.aggregate_values(
                'sessions', subject=self.subject, experiment=self.experiment))

        # RAM
        if self.experiment in self.jr.experiments():
            # Get all possible sessions
            sessions = list(self.jr.aggregate_values(
                'sessions', subject=self.subject, experiment=self.experiment))

        # type(sessions) == set of strings; make them an array of intergers
        self.possible_sessions = np.array(sorted(map(int, sessions)))

        if self.session not in self.possible_sessions:
            self.session = self.possible_sessions[0]
        return

    # ------> Set appropriate events
    def set_events(self):
        """Sets the behavioral events of a subject to the attribute events

        Returns
        -------
        Attribute self.event
        """
        # Handling for pyFR subjects
        if self.experiment == 'pyFR':
            event_path = '/data/events/pyFR/{}_events.mat'.format(self.subject)
            base_e_reader = BaseEventReader(filename=event_path,
                                            eliminate_events_with_no_eeg=True)
            events = base_e_reader.read()
            self.events = events[events['session'] == self.session]
            if self.verbose:
                print('Setting all events....')
            return

        # Handling for ltpFR2 subjects
        if self.experiment == 'ltpFR2':
            jr = JsonIndexReader('/protocols/ltp.json')

        # Handling for RAM subjects
        else:
            jr = JsonIndexReader('/protocols/r1.json')

        f = jr.aggregate_values('task_events', subject=self.subject,
                                experiment=self.experiment, session=self.session)
        if self.verbose:
            print('Setting all events....')
        self.events = BaseEventReader(filename=list(f)[0]).read()
        return

    def set_matched_retrieval_deliberation_events(self, rec_min_free_before, rec_min_free_after,
                                                  remove_before_recall=2000,
                                                  remove_after_recall=1500,
                                                  match_tol=2000):
        """Sets matched recall/deliberation behavioral events to .matched_events

        Parameters
        ----------
        rec_min_free_before: int, time in ms recall must be free prior to vocalization onset in order to be counted as a
                             included recall
        rec_min_free_after: int, time in ms recall must be free after vocalization onset in order to be counted as a
                            included recall
        remove_around_recall: int, by default, 3000, time in ms to remove as a valid deliberation period before and after
                              a vocalization
        match_tol: int, by default 2000, time in ms to tolerate as a possible recall deliberation match

        Creates
        -------
        Attribute matched_events
        """
        evs = create_matched_events(subject=self.subject,
                                    experiment=self.experiment,
                                    session=int(self.session),
                                    rec_inclusion_before=rec_min_free_before,
                                    rec_inclusion_after=rec_min_free_after,
                                    remove_before_recall=remove_before_recall,
                                    remove_after_recall=remove_after_recall,
                                    recall_eeg_start=int(self.eeg_start * 1000),
                                    recall_eeg_end=int(self.eeg_end * 1000),
                                    match_tolerance=match_tol,
                                    verbose=self.verbose,
                                    goodness_fit_check=self.verbose)
        # If verbose is True it prints out the steps used by RetrievalEventCreator object
        self.matched_events = evs.view(np.recarray)

        # Set to attributes
        self.rec_inclusion_before = rec_min_free_before
        self.rec_inclusion_after = rec_min_free_after
        self.remove_before_recall = remove_before_recall
        self.remove_after_recall = remove_after_recall
        self.match_tolerance = match_tol

        return

    # ------> Set appropriate brain atlas information
    def set_json_tals(self):
        """Sets the default monopolar, bipolar, and tal using json"""
        pairs_path = JsonIndexReader('/protocols/r1.json').get_value(
            'pairs', subject=self.subject, experiment=self.experiment)
        tal_reader = TalReader(filename=pairs_path)
        if self.verbose:
            print('Setting Json mp, bp, and tal')
        self.mp = tal_reader.get_monopolar_channels()
        self.bp = tal_reader.get_bipolar_pairs()
        self.json_tal = tal_reader.read()
        return

    def set_mlab_tals(self):
        """Sets the default monopolar, bipolar, and tals using matlab structures"""
        tal_path = '/data/eeg/{}/tal/{}_talLocs_database_bipol.mat'.format(self.subject, self.subject)
        tal_reader = TalReader(filename=tal_path)
        if self.verbose:
            print('Setting matlab mp, bp, and tal')
        self.mlab_tal = tal_reader.read()
        self.mp = tal_reader.get_monopolar_channels()
        self.bp = tal_reader.get_bipolar_pairs()
        return

    def set_logan_tals(self):
        tal_path = '/scratch/loganf/subject_brain_atlas/{}/{}_tal_indiv.npy'
        tal_path = tal_path.format(self.experiment, self.subject)
        if os.path.exists(tal_path):
            self.tal = np.load(tal_path)
            if self.verbose:
                print('Setting Logan brain atlas to tal')
        else:
            print('Could not find {}'.format(tal_path))
            self.set_json_tals()
            self.tal = self.json_tal
            if self.verbose:
                print('Setting json_tal to .tal due to failure in loading Logan tals')
        return

    # ------> Set subject's EEG
    def set_eeg(self, events=None):
        """Sets bp eeg for subject, by default uses set_events, otherwise pass in events to override"""

        # Load eeg from start to end, include buffer for edge contamination
        default = 'passed events'
        if events is None:  # If no events are passed by default we'll use .events
            events = self.events
            default = 'default events'
        # If channels haven't been loaded already then load them!
        if self.mp is None:
            if self.experiment != 'pyFR':
                self.set_json_tals()
                try:
                    self.set_logan_tals()
                except:
                    pass
            else:
                self.set_mlab_tals()

        eeg_reader = EEGReader(events=events, channels=self.mp, start_time=self.eeg_start,
                               end_time=self.eeg_end, buffer_time=self.eeg_buffer)
        if self.verbose:
            print('Setting eeg for {}....'.format(default))

        self.eeg = eeg_reader.read()

        # Remove line-noise, slow drift, and chs of all zeros
        self.remove_artifacts(data=self.eeg)

        if self.bipolar:
            self.eeg = self.bipolar_filter(data=self.eeg, bipolar_pairs=self.bp)

        return

    # ------> Artifact removal
    def remove_artifacts(self, data=None):
        """Removes line noise and slow drifting signal from eeg data
        --------
        INPUTS:
        data: TimeSeriesX; by default none and self.eeg is used,
              data to remove artifacts from
        -------
        OUTPUTS:
        data: inputted data with artifact removed
        """
        # By default use the attribute eeg for the morlet wavelet otherwise the passed data
        data = self.eeg if data is None else data

        # ----> Find any bipolar_pair channels that are entirely zero!
        cleaned_data, bad_channels = self.find_and_remove_zero_bipolarpairs_from_eeg(data)

        if self.verbose:
            print("Setting attribute bad_channels")
        self.bad_channels = bad_channels

        self.update_channel_info_after_removal_bad_channels()

        # ----> Apply butterworth filter to remove line noise Europe = 50Hz, America = 60 Hz
        line_noise = [58., 62.] if not 'FR' in self.subject else [48., 52.]
        b_filter = ButterworthFilter(time_series=cleaned_data,
                                     freq_range=line_noise,
                                     filt_type='stop',
                                     order=4)
        if self.verbose:
            print('Removing {} linenoise....'.format(line_noise))

        cleaned_data = b_filter.filter()

        # ----> Apply high pass filter to remove slow drift from signal

        data = butter_highpass_filter(data=data,
                                      cutoff=.1,
                                      fs=float(data['samplerate']),
                                      order=4)
        if self.verbose:
            print('Removing slow drifting <.1Hz Signal....')
            #print('Not removing frequency < .1Hz....')
        return cleaned_data

    @staticmethod
    def find_and_remove_zero_bipolarpairs_from_eeg(eeg):
        """Remove any channels with zeros in eeg across all time and all events
        eeg: TimeSeriesX: dims=bipolar_pairs, events, time!
        """
        ch_dim = None
        if 'bipolar_pairs' in eeg.dims:
            ch_dim = 'bipolar_pairs'

        elif 'channels' in eeg.dims:
            ch_dim = 'channels'

        if ch_dim is None:
            print('Uh... check the dim corresponding to your eeg channels?')
            print(eeg.dims)
            return eeg, None


        bad_channels = np.all(np.all(eeg.data==0,-1)==True, -1)
        bad_bipolar_pairs = eeg[ch_dim].data[bad_channels]
        good_bipolar_pairs = eeg[ch_dim].data[~bad_channels]

        if any(bad_channels):
            print("Removing channels of entirely zeros...")
            print('Removed channels {}'.format(bad_bipolar_pairs))

        # Monopolar vs bipolar handling
        if ch_dim == 'bipolar_pairs':
            good_eeg = eeg.sel(bipolar_pairs=good_bipolar_pairs)
            return good_eeg, bad_bipolar_pairs

        elif ch_dim == 'channels':
            good_eeg = eeg.sel(channels=good_bipolar_pairs)
            return good_eeg, bad_bipolar_pairs

    def update_channel_info_after_removal_bad_channels(self):
        """A function that finds intersections between all channels and bad channels returning only good ones"""

        # Find good mp
        locs_good_mp = np.where(np.in1d(self.mp, np.intersect1d(self.mp, self.bad_channels))==False)
        valid_mp = self.mp[locs_good_mp].view(np.recarray)

        # Find good bp
        bp0 = self.bp['ch0']
        locs_good0 = np.where(np.in1d(bp0, np.intersect1d(bp0, self.bad_channels))==False)
        valid_bps = self.bp[locs_good0].view(np.recarray)

        bp1 = valid_bps['ch1']
        locs_good1 = np.where(np.in1d(bp1, np.intersect1d(bp1, self.bad_channels))==False)
        valid_bps = valid_bps[locs_good1]

        # Inform user of the update
        if self.verbose:
            print('Updating instance bp/mp attributes...')
            print('number of initial bipolar pairs: ', len(self.bp))
            print('number of bad monopolar channels: ', len(self.bad_channels))
            print('number of valid bipolar pairs:', len(valid_bps))

        self.mp = valid_mp
        self.bp = valid_bps

        # Update the tal attribute to reflect the changes
        logan_tal_0 = self.tal['channel_1']
        valid_bp_0 = self.bp['ch0']
        valid_bp_1 = self.bp['ch1']
        keep0 = np.where(np.in1d(logan_tal_0,np.intersect1d(logan_tal_0, valid_bp_0)))
        brain_atlas = self.tal[keep0]
        logan_tal_1 = brain_atlas['channel_2']
        keep1 = np.where(np.in1d(logan_tal_1, np.intersect1d(logan_tal_1, valid_bp_1)))
        brain_atlas = brain_atlas[keep1]
        self.tal = brain_atlas

        if self.verbose:
            print('Updated instance tal attribute')


        return

    # ------> Convolution/Spectral Analysis
    def morlet(self, data=None, output='power'):
        """Applies a butterworth filter, morlet wavelet filter, and log trans on inputted data
        output: str, can be power, phase, or both, by default just power
        """
        # By default use the attribute eeg for the morlet wavelet otherwise the passed data
        data = self.eeg if data is None else data

        # Remove artifact from the signal
        #data = self.remove_artifacts(data=data)

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

        #self.power = pow_wavelet

        if output == 'both':
            phase_wavelet = phase_wavelet.remove_buffer(duration=self.eeg_buffer)
            self.phase = phase_wavelet

        # -----> Resample the data to 500Hz
        if self.resampled_rate == pow_wavelet.samplerate.data:
            time_axis_rs_hz = np.arange(self.eeg_start, self.eeg_end, 1/self.resampled_rate)
            pow_wavelet['time'].data = time_axis_rs_hz
            self.power = pow_wavelet


        elif self.resampled_rate != pow_wavelet.samplerate.data:
            pow_wavelet_resampled = pow_wavelet.resampled(resampled_rate=self.resampled_rate)
            time_axis_rs_hz = np.arange(self.eeg_start, self.eeg_end, 1/self.resampled_rate)
            pow_wavelet_resampled['time'].data = time_axis_rs_hz
            self.power = pow_wavelet_resampled

        return

    # ------> Helper functions
    @staticmethod
    def bipolar_filter(data, bipolar_pairs, verbose=True):
        """Filter monopolar data to bipolar data"""
        m2b = MonopolarToBipolarMapper(time_series=data, bipolar_pairs=bipolar_pairs)
        if verbose:
            print('Converting monopolar channels to bipolar pairs....')
        return m2b.filter()

    @staticmethod
    def scipy_zscore(timeseries, dim='events'):
        """Z-scores Data using scipy by default along events axis, does so for multiple sessions
        Parameters
        ----------
        timeseries: TimeSeriesX,
        dim: str, by default 'events',
                dimension to z-score over
        """
        # Convert dim to axis for use in scipy
        dim_to_axis = dict(zip(timeseries.dims, xrange(len(timeseries.dims))))
        axis = dim_to_axis[dim]

        zvalues = zscore(timeseries.data, axis=axis)
        timeseries.data = zvalues
        timeseries['name'] = 'Scipy Z-Scored'
        return timeseries


    def moving_mean(self, data=None, window=.2, step=.02):
        """Applies a moving mean"""
        if self.verbose:
            print('Creating a moving averaged window....')
        if data is None:
            data = self.power

        averaged_data = sliding_mean_fast(data,
                                          window=window,
                                          desired_step=step,
                                          dim='time')
        #averaged_data['frequency'] = self.freqs
        return averaged_data

    def path_creator(self):
        """Create experimental folders using the save dir"""
        subject, experiment, outdir = self.subject, self.experiment, self.outdir

        root = self.outdir + self.subject + '/'
        if not os.path.exists(root):
            os.makedirs(root)

        root = root + self.experiment + '/'
        self.outdir = root
        #figs = root + 'figures/'
        #evs = root + 'events/'
        proc = root + 'processed/'
        raw = root + 'raw/'
        #all_recs = raw + 'all_recs/'
        delibs = raw + 'deliberation/'
        z_scores = root + 'z_scores/'
        #z_math = z_scores + 'math/'
        #z_count = z_scores + 'countdown/'
        z_5 = z_scores + '5s/'
        z_60 = z_scores + '60s/'
        proc_all = proc + 'all_recs/'
        proc_delib = proc + 'deliberation/'
        final = root + 'final/'

        my_list = [root, proc, raw, delibs, z_scores,
                   #root, figs, proc, evs, raw, all_recs, delibs, z_scores,
                   #z_math, z_count,
                   z_5, z_60,
                   proc_all, proc_delib,
                   final]

        for path in my_list:
            if not os.path.exists(path):
                os.makedirs(path)