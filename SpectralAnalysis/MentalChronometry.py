"""Documentation in progress: 5-12-18 - LF

A SCRIPT USED TO LAUNCH THE REGIONAL TIMING COMPARASION FOR LOGANS MENTAL CHRONOMETRY ANALYSIS!

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
from RetrievalCreationHelper import create_matched_events as create_retrieval_and_matched_deliberation
from SpectralAnalysis.RollingAverage import sliding_mean_fast
from Utility.FrequencyCreator import logspace
from SpectralAnalysis.FastFiltering import butter_highpass_filter


class Subject(object):
    jr_scalp = JsonIndexReader('/protocols/ltp.json')
    jr = JsonIndexReader('/protocols/r1.json')

    def __init__(self, subject, experiment, session,
                 eeg_start=-1.25, eeg_end=.25, eeg_buffer=1.,
                 bipolar=True, width=5, freqs=None, verbose=True,
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
        verbose: bool; whether to print out the steps of the code along the way
        save: bool; whether to save the data; by default True
        outdir: str; the output directory to save to if save is True; by default my scratch dir
        ------
        OUTPUTS: da object
        """

        self.subject, self.experiment = subject, experiment
        self.session, self.save, self.outdir = session, save, outdir
        freqs = logspace(3, 95, 30) if freqs is None else freqs
        self.freqs = freqs[((freqs >= 3) & (freqs <= 8.1)) | (freqs >= 65)]
        self.width, self.verbose, self.bipolar = width, verbose, bipolar
        self.eeg_start, self.eeg_end, self.eeg_buffer = eeg_start, eeg_end, eeg_buffer

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

        return

    ############ Methods to set appropriate values on the instance ############

    # ------> Figure out how many sessions there are
    def _set_possible_sessions(self):
        """Sets an attribute possible_session that is an array of all possible sessions for the subject"""
        if self.experiment == 'pyFR':
            event_path = '/data/events/pyFR/{}_events.mat'.format(self.subject)
            base_e_reader = BaseEventReader(filename=event_path,
                                            eliminate_events_with_no_eeg=True)
            evs = base_e_reader.read()
            sessions = np.unique(evs['session'])
        if self.experiment == 'ltpFR2':
            jr = JsonIndexReader('/protocols/ltp.json')
            sessions = jr.aggregate_values(
                'sessions', subject=self.subject, experiment=self.experiment)
        else:
            jr = JsonIndexReader('/protocols/r1.json')
            # Get all possible sessions
            sessions = jr.aggregate_values(
                'sessions', subject=self.subject, experiment=self.experiment)

        # type(sessions) == set of strings; make them an array of intergers
        self.possible_sessions = np.array(sorted(map(int, sessions)))

        if self.session not in self.possible_sessions:
            self.session = self.possible_sessions[0]
        return

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
        evs = create_retrieval_and_matched_deliberation(subject=self.subject,
                                                        experiment=self.experiment,
                                                        session=int(self.session),
                                                        rec_inclusion_before=rec_min_free_before,
                                                        rec_inclusion_after=rec_min_free_after,
                                                        remove_before_recall=remove_before_recall,
                                                        remove_after_recall=remove_after_recall,
                                                        recall_eeg_start=int(self.eeg_start * 1000),
                                                        recall_eeg_end=int(self.eeg_end * 1000),
                                                        match_tolerance=match_tol,
                                                        verbose=self.verbose)
        # If verbose is True it prints out the steps used by RetrievalEventCreator object
        self.matched_events = evs

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

        eeg_reader = EEGReader(events=events, channels=self.mp, start_time=self.eeg_start,
                               end_time=self.eeg_end, buffer_time=self.eeg_buffer)
        if self.verbose:
            print('Setting eeg for {}....'.format(default))

        self.eeg = eeg_reader.read()

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

        # ----> Apply butterworth filter to remove line noise Europe = 50Hz, America = 60 Hz
        line_noise = [58., 62.] if not 'FR' in self.subject else [48., 52.]
        b_filter = ButterworthFilter(time_series=data,
                                     freq_range=line_noise,
                                     filt_type='stop',
                                     order=4)
        if self.verbose:
            print('Removing {} linenoise....'.format(line_noise))

        data = b_filter.filter()

        # ----> Apply high pass filter to remove slow drift from signal
        data = butter_highpass_filter(data=data,
                                      cutoff=.1,
                                      fs=float(data['samplerate']),
                                      order=4)
        if self.verbose:
            print('Removing <.1Hz Signal....')

        return data

    # ------> Convolution/Spectral Analysis
    def morlet(self, data=None, output='power'):
        """Applies a butterworth filter, morlet wavelet filter, and log trans on inputted data
        output: str, can be power, phase, or both, by default just power
        """
        # By default use the attribute eeg for the morlet wavelet otherwise the passed data
        data = self.eeg if data is None else data

        # Remove artifact from the signal
        data = self.remove_artifacts(data=data)

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

        # Go through each session and z-score relative to itself.
        z_data = []
        for sess in np.unique(timeseries['events'].data['session']):
            curr_sess = timeseries.sel(events=timeseries['events'].data['session'] == sess)
            curr_sess.data = zscore(curr_sess, axis)
            z_data.append(curr_sess)
        z_data = TimeSeriesX.concat(z_data, 'events')
        z_data['name'] = 'Scipy Z-Scored'
        return z_data

    def moving_mean(self, window=.2, step=.02):
        """Applies a moving mean"""
        if self.verbose:
            print('Creating a moving averaged window....')
        self.power = sliding_mean_fast(self.power, window, step, 'time')
        self.power['frequency'] = self.freqs
        return

    def path_creator(self):
        """Create experimental folders using the save dir"""
        subject, experiment, outdir = self.subject, self.experiment, self.outdir

        root = self.outdir + self.subject + '/'
        if not os.path.exists(root):
            os.makedirs(root)

        root = root + self.experiment + '/'
        self.outdir = root
        figs = root + 'figures/'
        evs = root + 'events/'
        proc = root + 'processed/'
        raw = root + 'raw/'
        all_recs = raw + 'all_recs/'
        delibs = raw + 'deliberation/'
        z_scores = root + 'z_scores/'
        z_math = z_scores + 'math/'
        z_count = z_scores + 'countdown/'
        z_5 = z_scores + '5s/'
        z_60 = z_scores + '60s/'
        proc_all = proc + 'all_recs/'
        proc_delib = proc + 'deliberation/'
        final = root + 'final/'

        my_list = [root, figs, proc, evs, raw, all_recs, delibs, z_scores,
                   z_math, z_count, z_5, z_60, proc_all, proc_delib, final]

        for path in my_list:
            if not os.path.exists(path):
                os.makedirs(path)

    # Function for plotting out retrieval period for the session of interest
    def plot_recalls_and_baselines(self, save_path=None):
        """Generates a plot of recalls and baselines match selection

        Returns
        -------
        pretty plot
        """

        def return_asrange(arr, timebefore, timeafter):
            if len(arr) == 1:
                return np.arange(arr - timebefore, arr + timeafter)
            if len(arr) > 1:
                return np.concatenate([np.arange(x - timebefore, x + timeafter) for x in arr])
            else:
                return None

        ########## Pre-plotting preamble code ##########
        # -------> Set relevant parameters for code to work from attributes of the instance
        matched_events = self.matched_events
        events = self.events
        eeg_length = (self.eeg_end - self.eeg_start) * 1000.
        trial_field = 'trial' if 'trial' in matched_events.dtype.names else 'list'
        events = events[events[trial_field] > 0]
        recs = matched_events[matched_events['type'] == 'REC_WORD']
        bls = matched_events[matched_events['type'] == 'REC_BASE']
        subj, sess, exp = self.subject, self.session, self.experiment

        if self.experiment.lower() == 'pyfr':
            recall_length = 45000
        elif self.experiment.lower() == 'ltpfr2':
            recall_length = 75000
        else:
            recall_length = 30000

        # ---------> Figure out included events that failed match, matched events, excluded recalls
        rec_indx = np.where(events['type'] == 'REC_WORD')
        all_recalls = events[rec_indx]
        matched_recalls = matched_events[matched_events['type'] == 'REC_WORD']

        # Indices where there is an unmatched recall
        unmatched_recall_idx = np.where(np.in1d(all_recalls['mstime'],
                                                np.intersect1d(all_recalls['mstime'],
                                                               matched_recalls['mstime'])) == False)
        unmatched_recalls = all_recalls[unmatched_recall_idx]

        # For the timebefore the first point, set zero, otherwise it's the difference
        timebefore = np.append(np.array([0]),  # [rec_indx] to get only recalls
                               np.diff(events['mstime']))[rec_indx]
        # For the timeafter event set difference of next point, for last it's zero
        timeafter = np.append(np.diff(events['mstime']),  # [rec_indx] to get only recalls
                              np.array([0]))[rec_indx]
        # Only need to look at unmatched recalls
        timebefore, timeafter = timebefore[unmatched_recall_idx], timeafter[unmatched_recall_idx]
        # Locs where unmatched recalls are valid (e.g. code could not match them ): )
        valid = ((timebefore >= self.rec_inclusion_before) & (timeafter >= self.rec_inclusion_after))
        excluded_recalls = unmatched_recalls[~valid]
        did_not_match_incl_recalls = unmatched_recalls[valid]
        vocalizations = events[events['type'] == 'REC_WORD_VV']

        ############ Make a plot of the retrieval phase  #############
        fig, ax = plt.subplots(figsize=(12, 4))
        # So that we only have one event for each kind in the legend use these booleans
        did_rec_legend = False
        did_bl_legend = False
        did_ex_rec_legend = False
        did_um_rec_legend = False
        did_voc_legend = False
        did_intrusion_legend = False

        for trial in np.unique(all_recalls[trial_field]):

            # ------------> PLOT INCLUDED RECALLS
            rec = recs[recs[trial_field] == trial]['rectime']
            rec = return_asrange(rec, eeg_length / 2., eeg_length / 2.)
            rec = recs[recs[trial_field] == trial]['rectime']
            rec = return_asrange(rec, eeg_length / 2., eeg_length / 2.)
            if rec is not None:  # is not None is b/c of output of return_asrange if len(arr) == 0
                ax.scatter(rec, np.zeros(len(rec)) + trial, c='red', marker='*', alpha=.5, s=.1)
                if not did_rec_legend:
                    ax.scatter(rec[0], 0 + trial, c='red', marker='*', label='included recall', alpha=.5, s=1)
                    did_rec_legend = True

            # ------------> PLOT MATCHED DELIBERATION
            bl = bls[bls[trial_field] == trial]['rectime']
            bl = return_asrange(bl, eeg_length / 2., eeg_length / 2.)
            if bl is not None:
                ax.scatter(bl, np.zeros(len(bl)) + trial, c='b', marker='o', alpha=1, s=.1)
                if not did_bl_legend:
                    ax.scatter(bl[0], 0 + trial, c='b', marker='o', label='Deliberation', alpha=1, s=1)
                    did_bl_legend = True

            # ------------> PLOT EXCLUDED RECALLS (Intrusions and not included recalls)
            ex_rec = excluded_recalls[excluded_recalls[trial_field] == trial]
            if len(ex_rec) > 0:
                intrusions = ex_rec[ex_rec['intrusion'] != 0]['rectime']
                if len(intrusions) > 0:
                    ax.scatter(intrusions, np.zeros(len(intrusions)) + trial, c='purple', marker='<',
                               alpha=1)  # , s=.1)
                    if not did_intrusion_legend:
                        ax.scatter(intrusions[0], 0 + trial, c='purple', marker='<', label='Intrusion',
                                   alpha=1)  # , s=.1)
                        did_intrusion_legend = True

                ex_rec = ex_rec[ex_rec['intrusion'] == 0]['rectime']
                if len(ex_rec) > 0:
                    ax.scatter(ex_rec, np.zeros(len(ex_rec)) + trial, c='g', marker='X', alpha=1)  # , s=.1)
                    if not did_ex_rec_legend:
                        ax.scatter(ex_rec[0], 0 + trial, c='g', marker='X', label='Excluded Recall', alpha=1)  # , s=.1)
                        did_ex_rec_legend = True

            # ------------> PLOT UNMATCHED INCLUDED RECALLS
            um_rec = did_not_match_incl_recalls[did_not_match_incl_recalls[trial_field] == trial]  # ['rectime']
            if len(um_rec) > 0:
                intrusions = um_rec[um_rec['intrusion'] != 0]['rectime']
                if len(intrusions) > 0:
                    ax.scatter(intrusions, np.zeros(len(intrusions)) + trial, c='purple', marker='<',
                               alpha=1)  # , s=.1)
                    if not did_intrusion_legend:
                        ax.scatter(intrusions[0], 0 + trial, c='purple', marker='<', label='Intrusion',
                                   alpha=1)  # , s=.1)
                        did_intrusion_legend = True
                um_rec = um_rec[um_rec['intrusion'] == 0]['rectime']
                if len(um_rec) > 0:
                    ax.scatter(um_rec, np.zeros(len(um_rec)) + trial, c='pink', marker='H', alpha=1)  # , s=.1)
                    if not did_um_rec_legend:
                        ax.scatter(um_rec[0], 0 + trial, c='pink', marker='H', label='Failed Match Recall',
                                   alpha=1)  # , s=.1)
                        did_um_rec_legend = True

            # ------------> PLOT VOCALIZATIONS
            vocs = vocalizations[vocalizations[trial_field] == trial]['rectime']
            if len(vocs) > 0:
                ax.scatter(vocs, np.zeros(len(vocs)) + trial, c='orange', marker='v', alpha=1)
                if not did_voc_legend:
                    ax.scatter(vocs[0], 0 + trial, c='orange', marker='v', label='Vocalization', alpha=1)
                    did_voc_legend = True

        ########## Make the plot look pretty ##########

        # --------> Format axis, ticks and labels
        # Format x axis to 1-30 step 1 (time in s to ms conversion)
        plt.xlabel('Time (s)', fontsize=16)
        plt.xlim(0, recall_length)
        ax.set_xticks(ticks=np.arange(0, recall_length + 1000, 1000))
        ax.set_xticklabels(labels=np.arange(0, (recall_length / 1000) + 1, 1))
        # Format y axis to number of trial, make red if no matched recall in trial else black
        ax.set_yticks(ticks=np.unique(events[trial_field]))
        ax.tick_params(axis='y', colors='black')
        plt.ylabel('Trial Number', fontsize=16)
        plt.ylim(min(np.unique(events[trial_field])) - .5,
                 max(np.unique(events[trial_field])) + .5)
        no_match_trials = np.where(np.in1d(np.unique(events[trial_field]),
                                           np.intersect1d(np.unique(events[trial_field]),
                                                          np.unique(matched_events[trial_field]))) == False)
        ticklabels = ax.get_yticklabels()
        for index, label in enumerate(ticklabels):
            if index in no_match_trials[0]:
                label.set_color('r')  # Make any trial without a included recall have a red y-axis tick

        # --------> Format rest of the plot
        plt.grid(True)
        ax.set_axisbelow(True)  # Make sure it appears behind our plot
        gridlines = ax.get_xgridlines() + ax.get_ygridlines()
        for line in gridlines:  # Make the grid a dashed line
            line.set_linestyle('-.')
        plt.title('{} {} Session {}'.format(subj, exp, sess), fontsize=22)
        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1), ncol=1, fancybox=True)

        # Save data (Wonky shit below is so that the legend doesn't get cut off while saving)
        if self.save:
            if save_path is None:
                save_dir = os.path.join(self.outdir, 'figures')
                save_path = save_dir + '/retrievalperiod_{}.pdf'.format(sess)
            fig.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
            fig.savefig(save_path.replace('pdf', 'png'), bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.show()