"""RetrievalCreationHelper.py, author = LoganF, updated May 16, 2018

A script used to create retrieval events and their corresponding "deliberation" matches. A deliberation match being
a point in time in a different list where a subject is actively trying, but failing to recall a word.

Works using data formatted for the Kahana Computational Memory Lab.

The code has been built for functionality with all RAM data, scalp ltpFR2, and pyFR. However, the author
only tested out the functionality on the FR1, catFR1, ltpFR2, and pyFR data.

For help please see docstrings (function_name? or help(function_name) or contact LoganF
For errors please contact LoganF

Example usage and import to use:

from RetrievalCreationHelper import create_matched_events

#------> Example FR1 usage
events = create_matched_events(subject='R1111M', experiment='FR1', session=0,
                               rec_inclusion_before = 2000, rec_inclusion_after = 1000,
                               remove_before_recall = 2000, remove_after_recall = 2000,
                               recall_eeg_start = -1250, recall_eeg_end = 250,
                               match_tolerance = 2000, verbose=True)

#------> Example ltpFR2
events = create_matched_events(subject='LTP093', experiment='ltpFR2', session=1,
                               rec_inclusion_before = 3000, rec_inclusion_after = 1000,
                               remove_before_recall = 3000, remove_after_recall = 3000,
                               recall_eeg_start = -1500, recall_eeg_end = 500,
                               match_tolerance = 3000)
"""
# General imports
from copy import deepcopy
from collections import OrderedDict
from IPython.display import display
from matplotlib import pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
import warnings
warnings.filterwarnings("ignore")

# Add some dimension to your life, np=arr, pd=labeled arr, xr=labeled N-D arr
import numpy as np  # I say numpy like 'lumpy', no I don't mean num-pie
import pandas as pd  # For Undergrads that say "But matlab makes it so much easier to view things"

# Pizza ptsa stuff
from ptsa.data.readers import BaseEventReader, JsonIndexReader

# Stats
from scipy.stats import ks_2samp

# Helper Function for behavioral event manipulation from CML_lib
from Utility.NumpyFunctions import append_fields

# ----------> Main function to import
def create_matched_events(subject, experiment, session,
                          rec_inclusion_before, rec_inclusion_after,
                          recall_eeg_start, recall_eeg_end, match_tolerance,
                          remove_before_recall, remove_after_recall,
                          desired_duration=None, verbose=False, goodness_fit_check=False):
    """Creates behavioral events for recall and matched-in-time deliberation points

    Parameters
    ----------
    :subject:
            str; subject id, e.g. 'R1111M'
    :experiment:
        str, experiment id, e.g. 'FR1'
        valid intracranial experiments:
            ['FR1', 'FR2', 'FR3', 'FR5', 'FR6', 'PAL1', 'PAL2', 'PAL3', 'PAL5',
             'PS1', 'PS2', 'PS2.1', 'PS3', 'PS4_FR', 'PS4_catFR', 'PS5_catFR',
             'TH1', 'TH3', 'THR', 'THR1', 'YC1', 'YC2', 'catFR1', 'catFR2',
             'catFR3', 'catFR5', 'catFR6', 'pyFR']
        valid scalp experiments:
            ['ltpFR2']
    :session:
        int, session to analyze
    :rec_inclusion_before:
        int,  time in ms before each recall that must be free
        from other events (vocalizations, recalls, stimuli, etc.) to count
        as a valid recall
    :rec_inclusion_after:
        int, time in ms after each recall that must be free
        from other events (vocalizations, recalls, stimuli, etc.) to count
        as a valid recall
    :remove_before_recall:
        int, time in ms to remove before a recall/vocalization
        used to know when a point is "valid" for a baseline
    :remove_after_recall:
        int, time in ms to remove after a recall/vocalization
        used to know when a point is "valid" for a baseline
    :recall_eeg_start:
        int, time in ms of eeg that we would start at
        relative to recall onset, note the sign (e.g. -1500 vs 1500) does not matter
    :recall_eeg_end:
        int, time in ms of eeg that we would stop at relative to recall onset
    :match_tolerance:
        int, time in ms that a deliberation may deviate from the
        retrieval of an item and still count as a match.
    :desired_duration:
        ###FOR NOW DO NOT USE THIS PARAMETER!!!! TODO: ADD IN FULL FUNCTIONALITY###
        int, default = None, time in ms of eeg that we would like the deliberation
        period to last for. If None, the code assumes the desired duration is calculated
        using recall_eeg_start and recall_eeg_end
    :verbose:
        bool, by default False, whether or not to print out the steps of the code along the way
    :goodness_fit_check:
        bool, by default False, whether or not to display information regarding the goodness of
        fit of the code

    Returns
    ----------
    :matched_events:
        np.recarray, array of behavioral events corresponding to included recalls with matches
        and their corresponding matches

    Example Usage
    ---------

    #------> Example FR1 usage
    events = create_matched_events(subject='R1111M', experiment='FR1', session=0,
                                   rec_inclusion_before = 2000, rec_inclusion_after = 1000,
                                   remove_before_recall = 2000, remove_after_recall = 2000,
                                   recall_eeg_start = -1250, recall_eeg_end = 250,
                                   match_tolerance = 2000, verbose=True)

    #------> Example ltpFR2
    events = create_matched_events(subject='LTP093', experiment='ltpFR2', session=1,
                                   rec_inclusion_before = 3000, rec_inclusion_after = 1000,
                                   remove_before_recall = 3000, remove_after_recall = 3000,
                                   recall_eeg_start = -1500, recall_eeg_end = 500,
                                   match_tolerance = 3000, goodness_fit_check=True)

    Notes
    -------
    Code will do a exact match in time first, afterwards will do a "tolerated" match. Any recalls that
    are not matched are dropped. If there are multiple possibles matches (either exact or tolerated) for a
    recall then the code will select the match that is closest in trial number to the recall.
    """
    subject_instance = DeliberationEventCreator(subject=subject,
                                                experiment=experiment,
                                                session=session,
                                                rec_inclusion_before=rec_inclusion_before,
                                                rec_inclusion_after=rec_inclusion_after,
                                                recall_eeg_start=recall_eeg_start,
                                                recall_eeg_end=recall_eeg_end,
                                                match_tolerance=match_tolerance,
                                                remove_before_recall=remove_before_recall,
                                                remove_after_recall=remove_after_recall,
                                                desired_duration=None,  # UNTIL IMPLEMENTED SET AS NONE!
                                                verbose=verbose)

    matched_events = subject_instance.create_matched_recarray()
    if goodness_fit_check:
        subject_instance.display_goodness_of_matching_info(plot_retrieval_period=True)
    return matched_events


# -------> Create an object to handle returning consistent event structures under the hood
class RetrievalEventCreator(object):
    """An object used to create recall behavioral events that are formatted in a consistent way regardless of the CML
    experiment.

    PARAMETERS
    -------
    INPUTS:
        :subject:
            str; subject id, e.g. 'R1111M'
        :experiment:
            str, experiment id, e.g. 'FR1'
            valid intracranial experiments:
                ['FR1', 'FR2', 'FR3', 'FR5', 'FR6', 'PAL1', 'PAL2', 'PAL3', 'PAL5',
                 'PS1', 'PS2', 'PS2.1', 'PS3', 'PS4_FR', 'PS4_catFR', 'PS5_catFR',
                 'TH1', 'TH3', 'THR', 'THR1', 'YC1', 'YC2', 'catFR1', 'catFR2',
                 'catFR3', 'catFR5', 'catFR6', 'pyFR']
            valid scalp experiments:
                ['ltpFR2']
        :session:
            int, session to analyze
        :inclusion_time_before:
            int, by default 1500, time in ms before each recall that must be free
            from other events (vocalizations, recalls, stimuli, etc.) to count
            as a valid recall
        :inclusion_time_after:
            int, by default 250, time in ms after each recall that must be free
            from other events (vocalizations, recalls, stimuli, etc.) to count
            as a valid recall
        :verbose:
            bool, by default False, whether or not to print out steps along the way


    EXAMPLE USAGE
    --------------
    old_ieeg_data = RetrievalEventCreator(subject='BW022', experiment='pyFR',
                                 session=2, inclusion_time_before=1500,
                                inclusion_time_after=500, verbose=True)
    # Set all the attributes of the object, which can then be used.
    old_ieeg_data.initialize_recall_events()


    self = RetrievalEventCreator(subject='R1111M', experiment='FR1', session=2,
                                 inclusion_time_before=1500, inclusion_time_after=500,
                                 verbose=True)
    """
    # Shared by the class
    jr = JsonIndexReader('/protocols/r1.json')
    jr_scalp = JsonIndexReader('/protocols/ltp.json')

    # -----> Initialize the instance
    def __init__(self, subject, experiment, session,
                 inclusion_time_before, inclusion_time_after,
                 verbose=False):

        # Initialize passed arguments
        self.subject = subject
        self.experiment = experiment
        self.session = session
        self.inclusion_time_before = inclusion_time_before
        self.inclusion_time_after = inclusion_time_after
        self.verbose = verbose
        # Sanity check!
        if (type(self.session) == unicode) | (type(self.session) == str):

            if self.verbose:
                print('Please use int for session Value, converting session to int')
                self.session = int(self.session)

        # So here we need to set the time of the recall, which varies by experiment
        self.rectime = 45000 if self.experiment == 'pyFR' else 30000
        if self.experiment == 'ltpFR2':  # Add this for scalp-functionality
            self.rectime = 75000

        # Initialize attributes we'll want to construct initially as None
        self.event_path = None
        self.events = None
        self.sample_rate = None
        self.montage = None
        self.possible_sessions = None
        self.trials = None
        self.included_recalls = None
        self.mean_rec = None
        return

    # ----------> FUNCTION TO CONSTRUCT STUFF
    def initialize_recall_events(self):
        """Main code to run through the steps in the code"""
        self.set_possible_sessions()
        self.set_events()
        self.events = self.add_fields_timebefore_and_timeafter(self.events)
        if self.verbose:
            print('Added fields "timebefore" and "timeafter" to attribute events')
        self.set_samplerate_params()
        self.set_valid_trials()

        # ----------> Check the formatting of the events to ensure that they have correct info
        has_recall_start_events = self.check_events_for_recall_starts(self.events)
        if not has_recall_start_events:
            print('Could not find REC_START events, creating REC_START events')
            self.events = self.create_REC_START_events()

        has_recall_end_events = self.check_events_for_recall_ends(self.events)
        if not has_recall_end_events:
            if self.verbose:
                print('Could not find REC_END events, creating REC_END events')
            self.events = self.create_REC_END_events(events=self.events,
                                                     time_of_the_recall_period=self.rectime)
            if self.verbose:
                print('Warning, Only set valid mstime for REC_END events')

        self.set_included_recalls()
        return

    # -------------> Methods to operate upon the instance
    def set_possible_sessions(self):
        """sets the values of all possible session values (array) to attribute possible_sessions

        Sets
        ------
        Attributes self.possible_sessions
        """

        # If ltpFR2
        if self.experiment in self.jr_scalp.experiments():
            sessions = self.jr_scalp.aggregate_values('sessions',
                                                      subject=self.subject,
                                                      experiment=self.experiment)
        # If RAM:
        if self.experiment in self.jr.experiments():
            # find out the montage for this session
            montage = list(self.jr.aggregate_values('montage',
                                                    subject=self.subject,
                                                    experiment=self.experiment,
                                                    session=self.session))[0]
            self.montage = montage
            # Find out all possible sessions with the montage
            sessions = self.jr.aggregate_values('sessions',
                                                subject=self.subject,
                                                experiment=self.experiment,
                                                montage=montage)
        # if pyFR
        if self.experiment == 'pyFR':
            evs = self.get_pyFR_events(self.subject)
            sessions = np.unique(evs['session'])

        # Replaced np.array((map(int, (sessions)))) for py3 functionality
        self.possible_sessions = np.array(list(map(int, (sessions))))

        # If the user chose a session not in the possible sessions then by default
        # We will defer to the first session of the possible sessions
        """5/18/18: Changing default behavior to raise an error if session inputted is not valid"""

        if self.session not in self.possible_sessions:
            raise DoneGoofed_InvalidSession(self.session, self.possible_sessions)
            #if self.verbose:
                #print('Could not find session {} in possible sessions: {}'.format(self.session, self.possible_sessions))
                #print('Over-riding attribute session input {} with {}'.format(self.session,
                                                                              #self.possible_sessions[0]))

            #self.session = self.possible_sessions[0]

        if self.verbose:
            print('Set Attribute possible_sessions')
        return

    def set_behavioral_event_path(self):
        """Sets the behavioral events path to attribute event_path

        Sets
        -------
        Attributes event_path
        """
        # -----------> Set the behavioral event path

        # If they want to do scalp reference scalp json protocol [ltpFR2]
        if self.experiment in self.jr_scalp.experiments():
            event_path = list(self.jr_scalp.aggregate_values('task_events',
                                                             subject=self.subject,
                                                             experiment=self.experiment,
                                                             session=self.session))[0]

        # If they're doing RAM data (catFR1, FR1)
        elif self.experiment in self.jr.experiments():
            event_path = list(self.jr.aggregate_values('task_events',
                                                       subject=self.subject,
                                                       experiment=self.experiment,
                                                       session=self.session))[0]
        # If they're doing pyFR
        elif self.experiment == 'pyFR':  # THIS WILL INITIALLY HAVE ALL SESSIONS!
            event_path = '/data/events/{}/{}_events.mat'.format(self.experiment, self.subject)

        # TO DO: Add in ltpFR1 functionality
        # event_path = '/data/eeg/scalp/ltp/ltpFR/behavioral/events/events_all_LTP065.mat'


        # If Logan is unsure where the data is
        else:
            if self.verbose:
                print('Unclear where the path of the data is is...')
                print('Is {} a valid experiment?'.format(self.experiment))
                print('Not creating attribute event_path')
            return

        # Set the event path to the attribute event_path
        self.event_path = event_path

        if self.verbose:
            print('Set attribute event_path')

        return

    def set_events(self):
        """sets all behavioral events to attributes events

        Sets
        -------
        Attributes self.events
        """
        if self.event_path is None:
            self.set_behavioral_event_path()

        # ---------> Read the events
        self.events = BaseEventReader(filename=self.event_path,
                                      eliminate_events_with_no_eeg=True).read()

        # This only really matters for pyFR for everything else it shouldn't do anything
        self.events = self.events[self.events['session'] == self.session]

        if self.verbose:
            print('Set attribute events')

        # Remove practice events if there are any
        trial_field = 'trial' if 'trial' in self.events.dtype.names else 'list'
        self.events = self.events[self.events[trial_field] >= 0]

        if self.verbose:
            print('Removing practice events')

        self.events = append_fields(self.events, [('match', '<i8')])
        if self.verbose:
            print('Adding field "match" to attribute events')

        if self.mean_rec is None:
            self.mean_rec = np.mean(self.events[self.events['type']=='WORD']['recalled'])*100
            if self.verbose:
                print('Setting Attribute mean_rec')
                print('Mean Recall in session: {}%'.format(self.mean_rec))
        return

    def set_valid_trials(self):
        """Sets the attribute trials, an array of the unique number of trials for this session

        Sets
        -------
        Attribute self.trials
        """
        trial_field = 'trial' if self.experiment in self.jr_scalp.experiments() else 'list'
        self.trials = np.unique(self.events[trial_field])

        if self.verbose:
            print('Set attribute trials')
        return

    def set_samplerate_params(self):
        """Sets the attribute sample_rate to the instance

        Sets
        ------
        Attribute self.sample_rate

        Notes
        -------
        Codes assumes that for ltpFR2 subjects with IDs < 331, the sample rate is 500, for subjects >= 331
        it assumes a sampling rate of 2048.
        For ieeg subjects it will load the eeg can check manually what the sample_rate is

        This code also will not work on ltpFR1, this functionality needs to be updated in.
        """
        scalp = True if self.experiment in self.jr_scalp.experiments() else False

        if scalp:
            if int(self.subject.split('LTP')[-1]) < 331:
                self.sample_rate = 500.

            elif int(self.subject.split('LTP')[-1]) >= 331:
                self.sample_rate = 2048.

        if not scalp:
            if self.events is None:
                self.set_behavioral_events()

            # Use recall's mstime and eegoffset to determine the sample rate
            # Seriously, use the recall event not an arbitary type event or errors
            recalls = self.events[self.events['type'] == 'REC_WORD']

            if recalls.shape[0] < 2:
                raise DoneGoofed_NoRecalls(recalls.shape[0])

            diff_in_seconds = (recalls['mstime'][1] - recalls['mstime'][0]) / 1000.
            diff_in_samples = recalls['eegoffset'][1] - recalls['eegoffset'][0]

            # Round is because we want 499.997 etc. to be 500
            self.sample_rate = np.round(diff_in_samples / diff_in_seconds)

        if self.verbose:
            print('Set attribute sample_rate')
        return

    def set_included_recalls(self, events=None):
        """Sets all included recalls to attribute self.recalls

        Parameters
        ----------
        events: np.array, by default None and self.events is used, behavioral events of the subject

        Sets
        -------
        Attributes self.recalls
        """
        if events is None:
            events = self.events

        recalls = events[(events['type'] == 'REC_WORD')
                         & (events['intrusion'] == 0)
                         & (events['timebefore'] > self.inclusion_time_before)
                         & (events['timeafter'] > self.inclusion_time_after)]

        recalls['match'] = np.arange(len(recalls))
        self.included_recalls = recalls

        if self.verbose:
            print('Set attribute included_recalls')

        if len(self.included_recalls) == 0:
            print('No recalls detected for {} session {}'.format(self.subject, self.session))

        return

    # ----------> Staticmethods
    @staticmethod
    def add_fields_timebefore_and_timeafter(events):
        """Adds fields timebefore and timeafter to behavioral events

        Parameters
        ----------
        events: np.array, behavioral events to add the field to

        Returns
        -------
        events: np.array, behavioral events now with the fields timebefore and timeafter added
        """
        events = append_fields(events, [('timebefore', '<i8'), ('timeafter', '<i8')])

        # For the timebefore the first point, set zero, otherwise it's the difference
        events['timebefore'] = np.append(np.array([0]), np.diff(events['mstime']))
        # For the timeafter event set difference of next point, for last it's zero
        events['timeafter'] = np.append(np.diff(events['mstime']), np.array([0]))

        return events

    # -------> Utility methods to allow data that isn't formatted like RAM to work with this code
    @staticmethod
    def check_events_for_recall_ends(events):
        """Check the inputted behavior events to see if they have events corresponding to the end of recall
        Returns True if they have it, False if they don't

        Parameters
        ----------
        events: np.array, behavioral events

        Returns
        -------
        True if there are REC_END type events in events otherwise returns False
        """

        return True if len(events[events['type'] == 'REC_END']) > 0 else False

    @staticmethod
    def check_events_for_recall_starts(events):
        """Check the inputted behavior events to see if they have events corresponding to the start of recall
        Returns True if they have it, False if they don't

        Parameters
        ----------
        events: np.array, behavioral events

        Returns
        -------
        True if there are REC_END type events in events otherwise returns False
        """

        return True if len(events[events['type'] == 'REC_START']) > 0 else False

    @staticmethod
    def create_REC_END_events(events, time_of_the_recall_period=45000):
        """Creates events that corresponding to the end of the recall field

        Parameters
        ----------
        events: np.array, behavioral events of a subject
        time_of_the_recall_period: int, by default 45000, duration in ms of the recall period

        Returns
        -------
        all_events: np.array, behavioral events of a subject with added events for the end of recall


        Notes
        ------
        This will only work on events with REC_START fields, also assumes that there's no break taken
        once the recall period starts...
        Also only correctly sets mstime field everything else is copied from rec_start events
        """
        # To avoid altering the events make a copy first
        rec_stops = deepcopy(events[events['type'] == 'REC_START'])
        # Reset the fields
        rec_stops['type'] = 'REC_END'
        rec_stops['mstime'] += time_of_the_recall_period
        all_events = np.concatenate((events, rec_stops)).view(np.recarray)
        all_events.sort(order='mstime')

        return all_events

    def create_REC_START_events(self):
        """Creates events that corresponding to the start of the recall field

        Parameters
        ----------
        self: the instance of the object

        Returns
        -------
        events: np.array, behavioral events of a subject with added events for the start of recall


        Notes
        ------
        This will work on behavioral events with REC_START events
        This part was very quickly remade to further improve pyFR functionality...
        """
        # Set fields that vary from experiment to experiment
        trial_field = 'trial' if 'trial' in self.events.dtype.names else 'list'
        item_field = 'item_name' if 'item_name' in self.events.dtype.names else 'item'
        item_number_field = 'item_num' if 'item_num' in self.events.dtype.names else 'itemno'

        if self.sample_rate is None:
            self.set_samplerate_params()

        events = self.events
        self.set_valid_trials()

        all_rec_starts = []

        for i, trial in enumerate(self.trials):
            recs = events[(events['type'] == 'REC_WORD') & (events[trial_field] == trial)]
            if len(recs) > 0:
                recs = recs[0]
                rec_starts = deepcopy(recs)
            else:
                continue

            # -------> Set the correct behavioral fields

            # Take the absolute time of the first recall in the trial and minus the relative time since rec start
            rec_starts['mstime'] = recs['mstime'] - recs['rectime']

            # This is the difference in time between the recall start and first recall in seconds
            diff_time_seconds = recs['rectime'] / 1000.

            # Find the number of samples between the first recall and the start of the recall
            n_samples = diff_time_seconds * self.sample_rate

            # The int/round non-sense is because n_samples can be something like 3006.5
            rec_starts['eegoffset'] = int(recs['eegoffset'] - round(n_samples))
            rec_starts[trial_field] = recs[trial_field]
            rec_starts['intrusion'] = -999
            rec_starts['rectime'] = -999
            rec_starts['type'] = 'REC_START'
            rec_starts['serialpos'] = -999
            rec_starts[item_field] = 'X'
            rec_starts[item_number_field] = -999
            # In case we run it on FR1...
            rec_starts['recalled'] = -999
            try:
                rec_starts['timebefore'] = 0
                rec_starts['timeafter'] = 0
                # rec_starts['recserialpos'] = 0
            except:
                pass

            all_rec_starts.append(rec_starts)

        all_rec_starts = np.array(all_rec_starts).view(np.recarray)

        events = np.concatenate((all_rec_starts, events)).view(np.recarray)
        events.sort(order='mstime')

        return events

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
        event_path = '/data/events/{}/{}_events.mat'.format(experiment, subject)
        base_e_reader = BaseEventReader(filename=event_path, eliminate_events_with_no_eeg=True)
        behavioral_events = base_e_reader.read()
        return behavioral_events

# ------> Create an object to create matched in time periods of deliberation
class DeliberationEventCreator(RetrievalEventCreator):
    """An object used to create recall behavioral events that are formatted in a consistent way regardless of the CML
       experiment.

       PARAMETERS
       -------
       INPUTS:
           :subject:
               str; subject id, e.g. 'R1111M'
           :experiment:
               str, experiment id, e.g. 'FR1'
               valid intracranial experiments:
                   ['FR1', 'FR2', 'FR3', 'FR5', 'FR6', 'PAL1', 'PAL2', 'PAL3', 'PAL5',
                    'PS1', 'PS2', 'PS2.1', 'PS3', 'PS4_FR', 'PS4_catFR', 'PS5_catFR',
                    'TH1', 'TH3', 'THR', 'THR1', 'YC1', 'YC2', 'catFR1', 'catFR2',
                    'catFR3', 'catFR5', 'catFR6', 'pyFR']
               valid scalp experiments:
                   ['ltpFR2']
           :session:
               int, session to analyze
           :rec_inclusion_before:
               int, time in ms before each recall that must be free from other events
               (vocalizations, recalls, stimuli, etc.) to count as a valid recall
           :rec_inclusion_after:
               int, time in ms after each recall that must be free from other events
               (vocalizations, recalls, stimuli, etc.) to count as a valid recall
           :recall_eeg_start:
               int, time in ms prior to each recall's vocalization onset that we want
               to start looking at the eeg at
           :match_tolerance:
               int, time in ms relative to a recall's retrieval period (e.g. from
               recall_eeg_start up until vocalization onset) that a match is allowed
               to vary by while still counting as a match
           :remove_before_recall:
               int, time in ms to remove before a recall (or vocalization) from being
               a valid deliberation period
           :remove_after_recall:
               int, time in ms to remove after a recall (or vocalization) from being
               a valid deliberation period
           ####### TODO: UPDATE TO INCLUDED FUNCTIONALITY #######
           :desired_duration:
               int, by default set as None and assumed to be the total eeg length,
               Not currently implemented besides default setting. Do not use until then
           ############################
           :verbose:
               bool, by default False, whether or not to print out steps along the way,

       EXAMPLE USAGE
       --------------
       # ------> Example RAM data
       subject = DeliberationEventCreator(subject='R1111M', experiment='FR1', session=0,
                                          rec_inclusion_before = 2000, rec_inclusion_after = 1000,
                                          remove_before_recall = 2000, remove_after_recall = 2000,
                                          recall_eeg_start = -1250, recall_eeg_end = 250,
                                          match_tolerance = 2000, verbose=True)
       events = subject.create_matched_recarray()


       # -------> Example ltpFR2
       scalp_sub = DeliberationEventCreator(subject='LTP093', experiment='ltpFR2', session=1,
                                            rec_inclusion_before = 2000, rec_inclusion_after = 1000,
                                            remove_before_recall = 3000, remove_after_recall = 3000,
                                            recall_eeg_start = -1250, recall_eeg_end = 250,
                                            match_tolerance = 2000, verbose=True)
       scalp_events = scalp_sub.create_matched_recarray()


       Notes
       -------
       Code will do a exact match in time first, afterwards will do a "tolerated" match. Any recalls that
       are not matched are dropped. If there are multiple possibles matches (either exact or tolerated) for a
       recall then the code will select the match that is closest in trial number to the recall.

    """
    def __init__(self, subject, experiment, session,
                 rec_inclusion_before, rec_inclusion_after,
                 recall_eeg_start, recall_eeg_end, match_tolerance,
                 remove_before_recall, remove_after_recall,
                 desired_duration=None, verbose=False):

        # Inheritance (Sets all attributes of RetrievalEventCreator)
        super(DeliberationEventCreator, self).__init__(subject=subject,
                                                       experiment=experiment,
                                                       session=session,
                                                       inclusion_time_before=rec_inclusion_before,
                                                       inclusion_time_after=rec_inclusion_after,
                                                       verbose=verbose)

        if (type(self.session) == unicode) | (type(self.session) == str):
            self.session = int(self.session)

        # Initialize the relevant RetrievalEventCreator attributes
        self.initialize_recall_events()

        # This seems silly to have four attributes refer to two things?
        self.rec_inclusion_before = rec_inclusion_before
        self.rec_inclusion_after = rec_inclusion_after

        # Intialize relevant passed arguments and relevant DeliberationEventCreator attributes
        # Just to handle everything consistently regardless of user input...
        if np.sign(recall_eeg_start) == -1:
            self.recall_eeg_start = -1 * recall_eeg_start
        else:
            self.recall_eeg_start = recall_eeg_start

        self.recall_eeg_end = recall_eeg_end
        self.match_tolerance = match_tolerance
        self.remove_before_recall = remove_before_recall
        self.remove_after_recall = remove_after_recall
        self.desired_duration = desired_duration
        if self.desired_duration is None:
            self.desired_duration = np.abs(self.recall_eeg_start) + np.abs(self.recall_eeg_end)
        self.trial_field = 'trial' if 'trial' in self.events.dtype.names else 'list'
        # Create attributes we'll set later initially as None
        self.baseline_array = None
        self.matches = None
        self.ordered_recalls = None
        self.matched_events = None

    def set_valid_baseline_intervals(self):
        """Sets to attribute baseline_array an array of 1 and 0 (num_unique_trials x 30000) where 1 is a valid time point

        Parameters
        -----------
        INPUTS EXTRACTED FROM INSTANCE:
            behavioral_events: np.array, behavioral events of a subject for one session of data.
            recall_period: int, by default 30000,
                time in ms of recall period (scalp = 750000, pyFR = 450000, RAM = 300000)
            desired_bl_duration: int, by default 3000,
                the desired time in ms we want to match over (e.g. if looking at recall from -2000 to 0, this
                should be 2000)
            remove_before_recall: int, by default 1500,
                time in ms to exclude before each recall/vocalization as invalid
            remove_after_recall: int, by default 1500,
                time in ms to exclude after each recall/vocalization as invalid
        Sets
        -------
        Attributes self.baseline_array

        """
        # Remove any practice events
        behavioral_events = self.events[self.events[self.trial_field] >= 0]
        # trials = np.unique(behavioral_events[self.trial_field])

        # Create an array of ones of shape trials X recall_period (in ms)
        baseline_array = np.ones((self.rectime * len(self.trials))).reshape(len(self.trials), self.rectime)
        valid, invalid = 1, 0

        # Convert any invalid point in the baseline to zero
        for index, trial in enumerate(self.trials):
            # Get the events of the trial
            trial_events = behavioral_events[behavioral_events[self.trial_field] == trial]

            # Get recall period start and stop points
            starts = trial_events[trial_events['type'] == 'REC_START']
            stops = trial_events[trial_events['type'] == 'REC_END']

            # If we don't have any recall periods then we will invalide the whole thing
            if ((starts.shape[0] == 0) & (stops.shape[0] == 0)):
                print('No recall period detected, trial: ', trial)
                baseline_array[index] = invalid
                continue
            # --------> Find Recalls or vocalizations
            possible_recalls = trial_events[
                (trial_events['type'] == 'REC_WORD') | (trial_events['type'] == 'REC_WORD_VV')]

            # -----> Use Recall rectimes to construct ranges of invalid times before and after them
            if len(possible_recalls['rectime']) == 1:  # If only one recall in the list
                invalid_points = np.arange(possible_recalls['rectime'] - self.remove_before_recall,
                                           possible_recalls['rectime'] + self.remove_after_recall)
            elif len(possible_recalls['rectime']) > 1:  # If multiple recalls in the list
                # TODO: Replace with np.apply or broadcasting of some kind?
                invalid_points = np.concatenate([np.arange(x - self.remove_before_recall, x + self.remove_after_recall)
                                                 for x in possible_recalls['rectime']])
            else:  # Get rid of any trials where we can't find any invalid points.
                baseline_array[index] = invalid
                continue

            # Ensure the points to be invalidated are within the boundaries of the recall period
            invalid_points = invalid_points[np.where(invalid_points >= 0)]
            invalid_points = invalid_points[np.where(invalid_points < self.rectime)]
            invalid_points = (np.unique(invalid_points),)  # ((),) similiar to np.where output

            # Removes initial recall contamination (-remove_before_recall,+remove_after_recall)
            baseline_array[index][invalid_points] = invalid

        self.baseline_array = baseline_array
        return

    def order_recalls_by_num_exact_matches(self):
        """Orders included_recalls array by least to most number of exact matches to create attribute ordered_recalls
        Creates
        -------
        Attribute ordered_recalls

        Notes
        ---------
        ordered_recalls[0] has the least number of matches and ordered_recalls[-1] has the most
        """
        # Desired start and stop points of each included recall
        recs_desired_starts = self.included_recalls['rectime'] - self.recall_eeg_start
        recs_desired_stops = self.included_recalls['rectime'] + self.recall_eeg_end

        # Store matches here
        exactly_matched = []

        # Go through each recall, find perfect matches
        for (start, stop) in zip(recs_desired_starts, recs_desired_stops):
            has_match_in_trial = np.all(self.baseline_array[:, start:stop] == 1, 1)
            exactly_matched.append(self.trials[has_match_in_trial])

        # Sort the recalls by ordering events from least to most number of exact matches
        index_match = np.array([[i, len(x)] for i, x in enumerate(np.array(exactly_matched))])
        sorted_order = pd.DataFrame(index_match, columns=['rec_index', 'num_matches']).sort_values('num_matches')
        ordered_recalls = np.array([self.included_recalls[i] for i in sorted_order['rec_index'].index]).view(
            np.recarray)

        self.ordered_recalls = ordered_recalls
        return

    def match_accumulator(self):
        """Accumulates matches between included recalls and baseline array, upon selection of a match invalidates it for other recalls

        Code will first go through each recall (ordered from least to most number of matches)  and try to select an exact match in time
        in another trial/list. If it cannot, after completeion of all exact matches the code will go through and try to find a tolerated
        match, that is a period in time that is within the instance's match_tolrance relative to the retrieval phase (eeg_rec_start up
        until vocalization onset)

        Modifies
        --------
        Attribute baseline_array

        Sets
        ----
        Attribute matches
        """
        if self.baseline_array is None:
            self.set_valid_baseline_intervals()
        if self.ordered_recalls is None:
            self.order_recalls_by_num_exact_matches()

        # This is used to keep track of the trial vs row indexing issues
        trial_to_row_mapper = dict(zip(self.trials, np.arange(len(self.trials))))

        # Check if user made it negative or positive
        # if np.sign(self.recall_eeg_start) == -1:
        # self.recall_eeg_start *= -1

        recs_desired_starts = self.ordered_recalls['rectime'] - self.recall_eeg_start
        recs_desired_stops = self.ordered_recalls['rectime'] + self.recall_eeg_end
        ################ EXACT MATCH ACCUMULATION ################
        # -----> Go through each recall, find perfect matches, if multiple select one closest to trial
        valid, invalid = 1, 0  # Explicit > implicit
        matches = OrderedDict()  # Store matches here
        if self.verbose:
            print('Starting Exact Matching Procedure...')
        for index, (start, stop) in enumerate(zip(recs_desired_starts, recs_desired_stops)):
            if index not in matches:
                matches[index] = []
            # Match across all trials from -recall_eeg_start to + recall_eeg_end
            ## TODO: Add in here to modify an option for putting in a different duration
            has_match_in_trial = np.all(self.baseline_array[:, start:stop] == valid, 1)
            trial_matches = self.trials[has_match_in_trial]

            # If there aren't any perfect matches just continue
            if len(trial_matches) == 0:
                if self.verbose:
                    print('Could not perfectly match recall index {}'.format(index))
                continue

            # Select the match that's closest to the recall's trial
            recalls_trial_num = self.ordered_recalls[index][self.trial_field]
            idx_closest_trial = np.abs(trial_matches - recalls_trial_num).argmin()
            selection = trial_matches[idx_closest_trial]

            if index in matches:
                matches[index].append((selection, start, stop))

            # Void the selection so other recalls cannot use it as valid
            self.baseline_array[trial_to_row_mapper[selection], start:stop] = invalid

        ################ TOLERATED MATCH ACCUMULATION ################
        # -------> Go through each recall, find tolerated matches
        if self.verbose:
            print('Starting Tolerated Matching Procedure...')
        for index, (start, stop) in enumerate(zip(recs_desired_starts, recs_desired_stops)):
            if matches[index] != []:
                continue  # Don't redo already matched recalls
            # Tolerance is defined around recall_eeg_start up until volcalization onset
            before_start_within_tol = start - self.match_tolerance
            after_start_within_tol = start + self.match_tolerance + self.recall_eeg_start

            # -----> Sanity check: cannot be before or after recall period
            if before_start_within_tol < 0:
                before_start_within_tol = 0
            if after_start_within_tol > self.baseline_array.shape[-1]:
                after_start_within_tol = self.baseline_array.shape[-1]

            # ------> Find out where there are valid tolerated points
            recalls_trial_num = self.ordered_recalls[index][self.trial_field]
            # Only need to check between tolerated points
            relevant_bl_times = self.baseline_array[:, before_start_within_tol:after_start_within_tol]
            # Use convolution of a kernel of ones for the desired duration to figure out where there are valid periods
            kernel = np.ones(self.desired_duration, dtype=int)
            sliding_sum = np.apply_along_axis(np.convolve, axis=1, arr=relevant_bl_times,
                                              v=kernel, mode='valid')

            valid_rows, valid_time_sliding_sum = np.where(sliding_sum == self.desired_duration)
            # Convert row to trial number through indexing the valid rows
            valid_trials = self.trials[(np.unique(valid_rows),)]
            if len(valid_trials) == 0:
                if self.verbose:
                    print('Could not match recall index {}'.format(index))
                continue

            # Find the closest trial
            idx_closest_trial = np.abs(valid_trials - recalls_trial_num).argmin()
            # Row in baseline_array vs actually recording the correct trial number
            selected_row = valid_rows[idx_closest_trial]
            selected_trial = valid_trials[idx_closest_trial]
            valid_first_point = valid_time_sliding_sum[0]
            # Essentially a conversion between convolution window and mstime
            valid_start = before_start_within_tol + valid_first_point
            valid_stop = valid_start + self.desired_duration  # b/c sliding mean slides to the right
            if index in matches:
                matches[index].append((valid_trials[idx_closest_trial], valid_start, valid_stop))
                # Void the selection so other recalls cannot use it is valid
                self.baseline_array[selected_row, valid_start:valid_stop] = invalid
        self.matches = matches

        return

    def create_matched_recarray(self):
        """Constructs a recarray of ordered_recalls and their matched deliberation points

        Returns
        -------
        behavioral_events: np.recarray, array of included recalls and matched deliberation periods
        """
        # if np.sign(self.recall_eeg_start) == -1:
        # self.recall_eeg_start *= -1

        if self.matches is None:
            self.match_accumulator()

        rec_start = self.events[self.events['type'] == 'REC_START']
        # rec_end = self.events[self.events['type']=='REC_END']
        # Non-sense is due to inconsistent fields
        trial_field = 'trial' if 'trial' in self.ordered_recalls.dtype.names else 'list'
        item_field = 'item_name' if 'item_name' in self.ordered_recalls.dtype.names else 'item'
        item_number_field = 'item_num' if 'item_num' in self.ordered_recalls.dtype.names else 'itemno'

        valid_recalls, valid_deliberation = [], []
        # Use the matches dictionary to construct a recarray
        for k, v in enumerate(self.matches):
            if self.matches[v] == []:
                print('Code could not successfully match recall index {}, dropping recall index {}'.format(k, k))
                continue

            valid_recalls.append(self.ordered_recalls[k])

            trial, rel_start, rel_stop = self.matches[v][0]  # [0] b/c tuple
            trial_rec_start_events = rec_start[rec_start[trial_field] == trial]

            bl = deepcopy(self.ordered_recalls[k])
            bl['type'] = 'REC_BASE'
            bl[trial_field] = trial
            bl[item_field] = 'N/A'
            bl[item_number_field] = -999
            bl['timebefore'] = -999
            bl['timeafter'] = -999
            bl['eegoffset'] = -999
            # Since we'll want to use EEGReader for both at once the below adjustment should work to allow it to do so
            # Changed from - to + because of *= -1 at start of code; bl['rectime'] = rel_start - recall_eeg_start
            bl['rectime'] = rel_start + self.recall_eeg_start
            bl['mstime'] = trial_rec_start_events['mstime'] + bl['rectime']
            bl['eegoffset'] = trial_rec_start_events['eegoffset'] + (self.sample_rate * (bl['rectime'] / 1000.))
            valid_deliberation.append(bl)

        valid_recalls = np.array(valid_recalls).view(np.recarray)
        valid_deliberation = np.array(valid_deliberation).view(np.recarray)

        behavioral_events = np.concatenate((valid_deliberation, valid_recalls)).view(np.recarray)
        behavioral_events.sort(order='match')
        self.matched_events = behavioral_events

        if self.verbose:
            print('Set attribute matched_events')
            #self.display_goodness_of_matching_info(plot_retrieval_period=True)
        return behavioral_events

    def display_goodness_of_matching_info(self, plot_retrieval_period=True):
        """Function for verifying the output of the code is correct, prints out KS distribution statistics, mean rectime, and an optional plot
        Parameters
        ---------
        plot_retrieval_period:
            bool, whether or not to plot the data visually as well
        """
        if self.verbose:
            print('\n\n\n\n\n\nDeterming Goodness of Fit...\n\n\n\n\n\n')
        evs = self.matched_events
        recalls = evs[evs['type'] == 'REC_WORD']['rectime']
        deliberations = evs[evs['type'] == 'REC_BASE']['rectime']

        print('Kolmogorov-Smirnov statistic: ', ks_2samp(recalls, deliberations))

        info_mean_time = '\nMean recall time (ms) = {}\nMean deliberation time (ms) = {}\nMean difference in time (ms) = {}'
        print(info_mean_time.format(recalls.mean(), deliberations.mean(), recalls.mean() - deliberations.mean()))


        ############### Preamble to plotting ###############
        # Here we must format events for plotting
        all_recalls = self.events[self.events['type'] == 'REC_WORD']

        # -------> Vocalizations
        vocalizations = self.events[self.events['type'] == 'REC_WORD_VV']
        vocalizations['type'] = 'Vocalization'

        # -------> Included recalls successfully matched
        matched_recs = self.matched_events[self.matched_events['type'] == 'REC_WORD']
        matched_recs['type'] = 'Successfully Matched Included Recall'

        # -------> Matched deliberation points
        matched_delib = self.matched_events[self.matched_events['type'] == 'REC_BASE']
        matched_delib['type'] = 'Successfully Matched Deliberation'

        # Filter out included recalls from all recalls array
        rec_match_intersection = np.intersect1d(all_recalls['mstime'], matched_recs['mstime'])
        rec_match_intersection = np.in1d(all_recalls['mstime'], rec_match_intersection)
        not_matched_idx = np.where(rec_match_intersection == False)
        not_matched_recs = all_recalls[not_matched_idx]

        # -------> Included recalls unsuccessfully matched
        failed_to_match_incl_recs = not_matched_recs[(not_matched_recs['timebefore'] > self.rec_inclusion_before)
                                                     & (not_matched_recs['timeafter'] > self.rec_inclusion_after)
                                                     & (not_matched_recs['intrusion'] == 0)]
        failed_to_match_incl_recs['type'] = 'Unsuccessfully Matched Included Recall'

        # -------> Intrusions
        intrusions = not_matched_recs[not_matched_recs['intrusion'] != 0]  # 0 indicates it is not an intrusion
        eli = intrusions[intrusions == -999]  # -999 used to indicate extra list intrusions
        eli['type'] = 'Extra List Intrusion'
        pli = intrusions[intrusions != -999]  # The number here indicates number of list back the intrusion is from
        pli['type'] = 'Prior List Intrusion'

        # -------> Excluded Recalls
        excluded_recs = not_matched_recs[((not_matched_recs['timebefore'] < self.rec_inclusion_before)
                                          | (not_matched_recs['timeafter'] < self.rec_inclusion_after))
                                         & (not_matched_recs['intrusion'] == 0)]
        excluded_recs['type'] = 'Excluded Recall'

        # -------> Combine them all into a single array
        retrieval_events = deepcopy(matched_recs).view(np.recarray)
        # This non-sense avoids any ValueErrors from concating empty things
        for ev_type in [vocalizations, matched_delib, failed_to_match_incl_recs, pli, eli, excluded_recs]:
            if ev_type.shape[0] == 0:
                continue
            retrieval_events = np.append(retrieval_events, ev_type)
        retrieval_events = retrieval_events.view(np.recarray)

        # If this isn't true there's a problem
        try:
            assert (len(retrieval_events) - len(matched_recs) - len(vocalizations)) == len(all_recalls)
        except:
            print('(len(retrieval_events) - len(matched_recs) - len(vocalizations)) == len(all_recalls):')
            print((len(retrieval_events) - len(matched_recs) - len(vocalizations)), len(all_recalls))

        ############# PLOT HIST OF DISTRIBUTION  #############
        num_incl_recs = len(retrieval_events[(retrieval_events['type'] == 'Unsuccessfully Matched Included Recall')
                                             | (retrieval_events['type'] == 'Successfully Matched Included Recall')])
        percentage_matched = float(
            len(retrieval_events[retrieval_events['type'] == 'Successfully Matched Included Recall']))
        percentage_matched = (percentage_matched / num_incl_recs) * 100
        print('1). How many Recalls are excluded due to match failure?\n')
        print('{}% of {} included recalls were successfully matched'.format(percentage_matched, num_incl_recs))
        print('\n2). How close does the distribution of Deliberation times match distribution of Recall times?\n')
        delib_rel_time = pd.DataFrame(matched_delib['rectime'], columns=['Deliberation'])
        rec_rel_time = pd.DataFrame(matched_recs['rectime'], columns=['Included Recalls'])
        excluded_recalls = pd.DataFrame(excluded_recs['rectime'], columns=['Excluded Recalls'])
        df = rec_rel_time.join(delib_rel_time)
        display(df.T)
        df = df.join(excluded_recalls)
        #display(df.T)
        colors = ["#EC4E20", "#FF9505", "#399AE7"]
        fig, ax = plt.subplots(figsize=(12,6))

        # .dropna() in case there are uneven amounts of each
        sns.distplot(df['Included Recalls'].dropna(), rug=True,
                     label='Included Recalls',ax=ax, color="#EC4E20")
        sns.distplot(df['Deliberation'].dropna(), rug=True,
                     label='Deliberations',ax=ax, color="#399AE7")
        sns.distplot(df['Excluded Recalls'].dropna(), rug=True,
                     label='Excluded Recalls',ax=ax, color="#FF9505")
        plt.legend()
        plt.title('Matched Included Recalls, Deliberation and Excluded Recalls', fontsize=22)
        ax.set_xlabel('Time Relative to Retrieval Phase (ms)', fontsize=18)
        ax.set_ylabel('Frequency', fontsize=18)
        ax.set_xlim(0, self.rectime)
        plt.tight_layout()
        plt.show()

        ############### Make a Plot All Retrieval Periods Using The Data Above ###############
        # -----> Create dictionaries to hold the values for colors/markers used in plot
        kind_to_color_d = {'Excluded Recall': "#FF9505",#'green',
                           'Prior List Intrusion': 'purple',
                           'Extra List Intrusion': 'black',
                           'Successfully Matched Deliberation': "#399AE7",
                           'Successfully Matched Included Recall': "#EC4E20",
                           'Unsuccessfully Matched Included Recall': 'pink',
                           'Vocalization': 'green'}#'orange'}

        kind_to_marker_d = {'Excluded Recall': 'X',
                            'Prior List Intrusion': '<',
                            'Extra List Intrusion': '>',
                            'Successfully Matched Deliberation': 'o',
                            'Successfully Matched Included Recall': '*',
                            'Unsuccessfully Matched Included Recall': 'H',
                            'Vocalization': 'v'}

        fig, ax = plt.subplots(figsize=(12, 6))

        # -----> Go through each event and make a plot
        for ev in retrieval_events:
            ax.scatter(ev['rectime'],
                       ev[self.trial_field],
                       color=kind_to_color_d[ev['type']],
                       marker=kind_to_marker_d[ev['type']],
                       label=ev['type'])

        # ------> Format x axis
        plt.xlim(0, self.rectime)
        ax.set_xticks(ticks=np.arange(0, self.rectime + 1000, 1000))
        ax.set_xticklabels(labels=np.arange(0, int((self.rectime / 1000.) + 1), 1))
        plt.xlabel('Time (s)', fontsize=16)

        # -----> Format y axis
        ax.set_yticks(ticks=self.trials)
        ax.tick_params(axis='y', colors='black')
        plt.ylabel('Trial Number', fontsize=16)
        plt.ylim(np.min(self.trials) - .5, np.max(self.trials) + .5)
        ticklabels = ax.get_yticklabels()
        has_recalls_in_trial = np.unique(all_recalls[self.trial_field])
        for index, label in enumerate(ticklabels):
            if self.trials[index] not in has_recalls_in_trial:
                label.set_color('r')  # Make any trial without a recall have a red tick

        # --------> Format rest of the plot
        plt.grid(True)
        ax.set_axisbelow(True)  # Make sure it appears behind our plot
        gridlines = ax.get_xgridlines() + ax.get_ygridlines()
        for line in gridlines:  # Make the grid a dashed line
            line.set_linestyle('-.')

        handles, labels = ax.get_legend_handles_labels()
        handle_list, label_list = [], []
        for handle, label in zip(handles, labels):
            if label not in label_list:
                handle_list.append(handle)
                label_list.append(label)
        lgd = ax.legend(handle_list, label_list, loc='upper left', bbox_to_anchor=(1, 1), ncol=1, fancybox=True)
        title = '{} {} Session {} Matched Retrieval Events'.format(self.subject, self.experiment, self.session)
        plt.title(title, fontsize=22)
        plt.show()


class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class DoneGoofed_InvalidSession(Error):
    """Exception raised for errors in the input, here if they didn't enter a valid session.

    Attributes:
        session -- input session of thes
        possible_sessions -- valid sessions of the subject
    """

    def __init__(self, session, possible_sessions):
        warning = 'DoneGoofed Session Error: Ah Shucks sorry to say but it looks like ya done goofed...'
        warning2 = 'I could not find session {}. Have you considered using a valid session instead? Try: \n{}'
        self.warning = warning2.format(session, possible_sessions)
        print(warning)
        print(self.warning)

class DoneGoofed_NoRecalls(Error):
    """Exception raised for errors in the input, here if they didn't enter a valid session.

    Attributes:
        session -- input session of thes
        possible_sessions -- valid sessions of the subject
    """

    def __init__(self, recalls):
        warning = 'DoneGoofed IndexError: Subject has {} valid recalls cannot index for sample rate setting'
        self.warning = warning.format(recalls)
        print(self.warning)