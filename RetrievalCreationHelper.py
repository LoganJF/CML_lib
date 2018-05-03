"""RetrievalCreationHelper.py, author = LoganF, updated May 2, 2018

A script used to create retrieval events and their corresponding "deliberation" matches. A deliberation match being
a point in time in a different list where a subject is actively trying, but failing to recall a word.
Works using data formatted for the Kahana Computational Memory Lab.
The code has been built for functionality with all RAM data, scalp ltpFR2, and pyFR. However, the author
only tested out the functionality on the RAM and pyFR data.

For help please see docstrings or contact LoganF, for errors please contact LoganF

Example usage and import to use:

from RetrievalCreationHelper import create_retrieval_and_matched_deliberation

#------> Example FR1
events = create_retrieval_and_matched_deliberation(subject='R1111M', experiment='FR1', session=0,
                                                   rec_inclusion_before = 2000, rec_inclusion_after = 1000,
                                                   remove_before_recall = 2000, remove_after_recall = 2000,
                                                   recall_eeg_start = -1250, recall_eeg_end = 250,
                                                   match_tolerance = 2000)

#------> Example ltpFR2
events = create_retrieval_and_matched_deliberation(subject='LTP093', experiment='ltpFR2', session=1,
                                                   rec_inclusion_before = 3000, rec_inclusion_after = 1000,
                                                   remove_before_recall = 3000, remove_after_recall = 3000,
                                                   recall_eeg_start = -1500, recall_eeg_end = 500,
                                                   match_tolerance = 3000)
"""
# --------> General Imports
import sys
# Add some dimension to your life, np=arr, pd=labeled arr, xr=labeled N-D arr
import numpy as np  # I say numpy like 'lumpy', no I don't mean num-pie
import pandas as pd  # Make it pretty
from ptsa.data.readers import JsonIndexReader, BaseEventReader
from copy import deepcopy
# Check if python 3 or two
if int(sys.version[0]) == 3:
    xrange = range
    print('Why_are_you_using_python3.jpeg')


# ------------> Relevant import, main function, the bee's knees, what people should use etc.
def create_retrieval_and_matched_deliberation(subject, experiment, session,
                                              rec_inclusion_before=2000, rec_inclusion_after=1000,
                                              remove_before_recall=2000, remove_after_recall=2000,
                                              recall_eeg_start=-1250, recall_eeg_end=250,
                                              match_tolerance=2000, verbose=False):
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
        int, by default 2000, time in ms before each recall that must be free
        from other events (vocalizations, recalls, stimuli, etc.) to count
        as a valid recall
    :rec_inclusion_after:
        int, by default 1000, time in ms after each recall that must be free
        from other events (vocalizations, recalls, stimuli, etc.) to count
        as a valid recall
    :remove_before_recall:
        int, by default 2000, time in ms to remove before a recall/vocalization
        used to know when a point is "valid" for a baseline
    :remove_after_recall:
        int, by default 2000, time in ms to remove after a recall/vocalization
        used to know when a point is "valid" for a baseline
    :recall_eeg_start:
        int, default = -1250, time in ms of eeg that we would start at
        relative to recall onset, note the sign (e.g. -1500 vs 1500) does not matter
    :recall_eeg_end:
        int, default = 500, time in ms of eeg that we would stop at relative to recall onset
    :match_tolerance:
        int, by default = 2000, time in ms that a deliberation may deviate from the
        retrieval of an item and still count as a match.
    :verbose:
        bool, by default False, whether or not to print out the steps of the code along the way

    Returns
    ----------
    :matched_events:
        np.recarray, array of behavioral events corresponding to included recalls with matches
        and their corresponding matches

    Notes
    -------
    Code will do a perfect match first, afterwards will do a "tolerated" match. Any recalls that
    are not matched are dropped
    """
    # Create an instance surrounding the subject's retrieval events
    self = RetrievalEventCreator(subject=subject, experiment=experiment, session=int(session),
                                 inclusion_time_before=rec_inclusion_before,
                                 inclusion_time_after=rec_inclusion_after,
                                 verbose=verbose)
    # Initialize the relevant parameters/attributes
    self.initialize_recall_events()

    # Inform user of the number of "included" vs total recalls
    print('\nStarting match event creation for {}'.format(self.subject))
    num_evs = len(self.events[(self.events['type'] == 'REC_WORD') & (self.events['intrusion'] == 0)])
    num_incl = len(self.included_recalls)
    print('\n{}/{} recalls included\n'.format(num_incl, num_evs))

    # Create an array of valid deliberation points
    baseline_array = get_valid_continuous_baseline_intervals_session(behavioral_events=self.events,
                                                                     recall_period=self.rectime,
                                                                     desired_bl_duration=(
                                                                     recall_eeg_end - recall_eeg_start),
                                                                     remove_before_recall=remove_before_recall,
                                                                     remove_after_recall=remove_after_recall)

    # Match each recall exactly in time
    perfect_matches = get_perfect_matches_in_lists(recalls=self.included_recalls,
                                                   baseline_arr=baseline_array,
                                                   trials=self.trials,
                                                   recall_eeg_start=recall_eeg_start,
                                                   recall_eeg_end=recall_eeg_end)

    # Sort the recalls by ordering them from most to least
    sorted_order = sort_recalls_by_num_matches(matches=perfect_matches)
    ordered_recalls = np.array([self.included_recalls[i] for i in sorted_order['rec_index'].index]).view(np.recarray)

    # Iterate through the ordered recalls and select exact matches, removing them from the baseline array
    matches, baseline_array = perfect_match_accumulator(ordered_recalls=ordered_recalls,
                                                        baseline_array=baseline_array,
                                                        trials=self.trials,
                                                        recall_eeg_start=recall_eeg_start,
                                                        recall_eeg_end=recall_eeg_end)

    # Inform the user of the number of included recalls that were perfectly matched
    number_matched = sum([1 for k, v in enumerate(matches) if matches[v] != []])
    percentage_exact_matched = 100 * (float(number_matched) / len(matches))
    print('\n{}% of recalls were exactly matched\n'.format(percentage_exact_matched))

    # Iterate through the ordered recalls and select tolerated matches, removing them from the baseline array
    matches, baseline_array = within_tolerance_match_accumulator(ordered_recalls=ordered_recalls,
                                                                 baseline_array=baseline_array,
                                                                 trials=self.trials,
                                                                 matches=matches,
                                                                 match_tolerance=match_tolerance,
                                                                 recall_eeg_start=recall_eeg_start,
                                                                 recall_eeg_end=recall_eeg_end)

    # Inform the user of the number of included recalls that were matched
    number_matched = sum([1 for k, v in enumerate(matches) if matches[v] != []])
    percentage_matched = 100 * (float(number_matched) / len(matches))
    print('\n{}% of recalls were matched\n'.format(percentage_matched))

    # Finally, use the information above to create a np.recarray containing included recalls and matched deliberations
    matched_events = create_matched_recarray(matches=matches,
                                             ordered_recalls=ordered_recalls,
                                             subject_instance=self,
                                             recall_eeg_start=-1250)

    return matched_events


# -------> UTILITY OBJECTS FOR INFORMING USER OF ERRORS
class BaseException(Exception):
    """Base Exception Object for handling"""

    def __init__(self, *args):
        self.args = args
        self._set_error_message()

    def _set_error_message(self):
        self.message = self.args[0] if self.args else None


class FieldAlreadyExistsException(BaseException):
    """Utility for handling excepts of appending fields"""

    def __init__(self, msg):
        self.msg = msg
        warning = '{}\nSo, the field is already in the array, returning inputted array'
        super(FieldAlreadyExistsException, self).__init__('ValueError: {}'.format(msg))
        print(warning.format(self.message))


# -------> UTILITY FUNCTIONS FOR BEHAVIORAL EVENT MANIPULATIONS
def append_fields(old_array, list_of_tuples_field_type):
    """Return a new array that is like "old_array", but has additional fields.

    The contents of "old_array" are copied over to the appropriate fields in
    the new array, whereas the new fields are uninitialized.  The
    arguments are not modified.

    *This is necessary to do use than using the np.lib.recfunction.append_fields
    function b/c the json loaded events use a dictionary for stim_params in the events*
    -----
    INPUTS
    old_array: a structured numpy array, the behavioral events from ptsa
    list_of_tuples_field_type: a numpy type description of the new fields
    -----
    OUTPUTS
    new_array: a structured numpy array, a copy of old_array with the new fields
    ------
    EXAMPLE USE
    >>> events = BaseEventReader(filename = logans_file_path).read()
    >>> events = append_field_workaround(events, [('inclusion', '<i8'), ('foobar', float)])
    >>> sa = np.array([(1, 'Foo'), (2, 'Bar')], \
                         dtype=[('id', int), ('name', 'S3')])
    >>> sa.dtype.descr == np.dtype([('id', int), ('name', 'S3')])
    True
    """
    if old_array.dtype.fields is None:
        raise ValueError("'old_array' must be a structured numpy array")
    new_dtype = old_array.dtype.descr + list_of_tuples_field_type

    # Try to add the new field to the array, should work if it's not already a field
    try:
        new_array = np.empty(old_array.shape, dtype=new_dtype).view(np.recarray)
        for name in old_array.dtype.names:
            new_array[name] = old_array[name]
        return new_array
    # If user accidentally tried to add a field already there, then return the old array
    except ValueError as e:
        print(sys.exc_info()[0])
        error = FieldAlreadyExistsException(e)
        return old_array


# -------> Object to create retrieval events, crux of the functions below it.
class RetrievalEventCreator(object):
    """An object used to create recall behavioral events

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
    # Set all the attributes of the object.
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
                 inclusion_time_before=1500, inclusion_time_after=250,
                 verbose=False):

        # Initialize passed arguments
        self.subject = subject
        self.experiment = experiment
        self.session = session
        self.inclusion_time_before = inclusion_time_before
        self.inclusion_time_after = inclusion_time_after
        self.verbose = verbose

        # So here we need to set the time of the recall, which varies by experiment
        self.rectime = 45000 if self.experiment == 'pyFR' else 30000
        if self.experiment == 'ltpFR2':  # Add this for scalp-functionality
            self.rectime = 75000

        # Initialize attributes we'll want to construct initially as None
        self.event_path = None
        self.events = None
        self.sample_rate = None
        self.possible_sessions = None
        self.trials = None
        self.included_recalls = None
        return

    # ----------> FUNCTION TO CONSTRUCT STUFF
    def initialize_recall_events(self):
        """Main code to run through the steps in the code"""
        self.set_possible_sessions()
        self.set_events()
        self.events = self.add_fields_timebefore_and_timeafter(self.events)
        self.set_samplerate_params()
        self.set_valid_trials()

        # ----------> Check the formatting of the events to ensure that they have correct info
        has_recall_start_events = self.check_events_for_recall_starts(self.events)
        if not has_recall_start_events:
            self.events = self.create_REC_START_events()

        has_recall_end_events = self.check_events_for_recall_ends(self.events)
        if not has_recall_end_events:
            self.events = self.create_REC_END_events(events=self.events,
                                                     time_of_the_recall_period=self.rectime)
            if self.verbose:
                print('Warning, Only set valid mstime for REC_END events...')

        self.set_included_recalls()
        return

    # -------------> Methods to operate upon the instance
    def set_possible_sessions(self):
        """sets the values of all possible session values (array) to attribute possible_sessions

        Returns
        ------
        Attributes self.possible_sessions
        """

        # If ltpFR2
        if self.experiment in self.jr_scalp.experiments():
            sessions = self.jr_scalp.aggregate_values('sessions',
                                                      subject=self.subject,
                                                      experiment=self.experiment)
        # If RAM
        if self.experiment in self.jr.experiments():
            sessions = self.jr.aggregate_values('sessions',
                                                subject=self.subject,
                                                experiment=self.experiment)
        # if pyFR
        if self.experiment == 'pyFR':
            evs = self.get_pyFR_events(self.subject)
            sessions = np.unique(evs['session'])

        # Replaced np.array((map(int, (sessions)))) for py3 functionality
        self.possible_sessions = np.array(list(map(int, (sessions))))

        # If the user chose a session not in the possible sessions then by default
        # We will defer to the first session of the possible sessions
        if self.session not in self.possible_sessions:
            if self.verbose:
                print('Could not find session {} in {}'.format(self.session, self.possible_sessions))
                print('Over-riding attribute session input {} with {}'.format(self.session,
                                                                              self.possible_sessions[0]))
            self.session = self.possible_sessions[0]

        if self.verbose:
            print('Set Attribute possible_sessions')
        return

    def set_behavioral_event_path(self):
        """Sets the behavioral events path to attribute event_path

        Creates
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

        Returns
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

        return

    def set_valid_trials(self):
        """Sets the attribute trials, an array of the unique number of trials for this session

        Returns
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

        Returns
        ------
        Attribute self.sample_rate

        Notes
        -------
        Codes assumes that for ltpFR2 subjects with IDs < 331, the sample rate is 500, for subjects >= 331
        it assumes a sampling rate of 2048.
        For ieeg subjects it will load the eeg can check manually what the sample_rate is

        This code also will not work on ltpFR1, this functionality needs to be updatd in.
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

            # eeg = get_sub_eeg(subject=self.subject, experiment=self.experiment, events=self.events[0:1])
            # self.sample_rate = float(eeg.samplerate.data)

            diff_in_seconds = (self.events['mstime'][4] - self.events['mstime'][3]) / 1000.
            diff_in_samples = self.events['eegoffset'][4] - self.events['eegoffset'][3]
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

        Returns
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
        """
        from copy import deepcopy
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
        """
        from copy import deepcopy
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
                #rec_starts['recserialpos'] = 0
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
        """ Utility to get pyFR events

        :param subject: str, subject id
        :param experiment:
        :return:
        """
        event_path = '/data/events/{}/{}_events.mat'.format(experiment, subject)
        base_e_reader = BaseEventReader(filename=event_path, eliminate_events_with_no_eeg=True)
        behavioral_events = base_e_reader.read()
        return behavioral_events


# ---------> Helper Functions to create deliberation periods!
def remove_practice_events(events):
    """ Removes practice events from the events so they are not analyzed

    Parameters
    ----------
    events: np.array like; behavioral events of a session

    Returns
    -------
    events: np.array like; behavioral events of a session with practice events removed
    """
    # Added next line to enable functionality with scalp data
    trial_field = 'trial' if 'trial' in events.dtype.names else 'list'
    events = events[events[trial_field] >= 0]
    return events


def return_recall_starts_stops(events):
    """Splits inputted events into two arrays corresponding to recall starts [0] and recall ends [1]

    Parameters
    ----------
    events: np.array like; behavioral events of a session

    Returns
    -------
    start, stop: behavioral events representing start and end times for a recall
    """
    start = events[events['type'] == 'REC_START']
    stop = events[events['type'] == 'REC_END']
    return start, stop


def iterate_events_by_trial(session_events, trial):
    """Returns a copy of inputted session_events with only those in the corresponding inputted trial

    Parameters
    ----------
    session_events: np.array like; behavioral events of a session
    trial: int, the trial REMEMBER THAT TRIALS START AT 1 PYTHON INDEXES AT 0!

    Returns
    -------
    trial_evs: np.recarray; behavioral events belonging the inputted trial
    """
    # Added next line to enable functionality with scalp data
    from copy import deepcopy
    trial_field = 'trial' if 'trial' in session_events.dtype.names else 'list'
    trial_evs = deepcopy(session_events[session_events[trial_field] == trial])
    return trial_evs.view(np.recarray)


def inside_recall_period(trial_events, start_events, stop_events):
    """Returns trial_events with time points between the start and end point of recall for a given trial

    Parameters
    ----------
    trial_events: np.array like; behavioral events of a trial
    start_events: np.array like; behavioral event corresponding to trial recall start
    stop_events: np.array like; behavioral event corresponding to trial recall stop

    Returns
    -------
    trial_events with time points between the start and end point of recall for a given trial
    """
    return trial_events[(trial_events['mstime'] > start_events['mstime'])
                        & (trial_events['mstime'] < stop_events['mstime'])]


def within_boundaries(arr, recall_period_start=0, recall_period_end=30000):
    """Determines whether an inputted arr is within the boundaries of the recall position

    Parameters
    ----------
    arr: np.array; arr of values corresponding to indices
    recall_period_start: int, by default 0; the time in ms where the recall period starts
    recall_period_end: int, by default 30000; the time in ms where the recall period ends

    Returns
    -------
    valid points given the recall boundary
    """
    arr = arr[np.where(arr >= recall_period_start)]
    return arr[np.where(arr < recall_period_end)]


def return_asrange(arr, timebefore, timeafter):
    """Returns data as a range, essentially np.arange by an array of arrays...

    Parameters
    ----------
    arr: np.array, a list of time values
    time_before: int, the time before the event to include in the range
    time_after: int, the time after the event to include in the range

    Returns
    -------
    arrays of np.arange if the code worked, None if it failed
    """

    if len(arr) == 1:
        return np.arange(arr - timebefore, arr + timeafter)
    if len(arr) > 1:
        return np.concatenate([np.arange(x - timebefore, x + timeafter) for x in arr])
    else:
        return None


def find_consecutive_data(data, stepsize=1):
    """Splits Data into a list of arrays where there is more than a single step of 1

    Parameters
    ----------
    data: np.array of zeroes and ones, the data to split
    step_size: int, by default 1

    Returns
    -------
    list of split data
    """
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


def sort_recalls_by_num_matches(matches):
    """ Sorts the recall by the number of matches they have (from least to most)


    ##### LOGAN FIX THIS ITS WAY TOO VAUGE, WHICH OBJECT? ######


    Parameters
    ----------
    matches: matches generated using object, maps index of event to lists of possible matches

    Returns
    -------
    sorted values of recalls ordered from least to most matches
    """
    index_match = np.array([[i, len(x)] for i, x in enumerate(np.array(matches))])
    df = pd.DataFrame(index_match, columns=['rec_index', 'num_matches'])
    return df.sort_values('num_matches')


def remove_too_small(baseline_arr, trial, desired_duration=3000):
    """Returns sections of baseline_arr that are too small (less than desired duration) or valid baselines

    Parameters
    ----------
    baseline_arr: np.array, ones and zeros, shape = (num_unique_trials x 30000)
    trial: int, the index along the baseline_arr to edit
    desired_duration: int, by default 3000, time in ms that the duration must be

    Returns
    -------
    remove_small: boolean; True if there are locations that are too small and should be removed, else False
    too_small_locs: tuple or list; tuple locations of where it's valid or empty list if no values are valid


    Notes
    ------
    Baseline_arr is an array of 0's and 1's, what this code does is it calculates the difference between
    elements i and i+1, and if that difference is not zero for a duration of desired_duration then it will remove it
    because it is too small

    """
    valid_points = np.where(baseline_arr[trial] == 1)
    list_of_consecutive_vp = find_consecutive_data(valid_points[0])

    too_small_locs = [valid_baseline for valid_baseline in list_of_consecutive_vp
                      if len(valid_baseline) < desired_duration]

    remove_small = False if len(too_small_locs) == 0 else True

    if remove_small:
        too_small_locs = (np.concatenate(too_small_locs),)

    return remove_small, too_small_locs


# ---------> Functions to create deliberation periods!
def get_valid_continuous_baseline_intervals_session(behavioral_events,
                                                    recall_period=30000, desired_bl_duration=3000,
                                                    remove_before_recall=1500, remove_after_recall=1500):
    """Creates an array of ones and zeros (num_unique_trials x 30000) where one is a valid time point

    Parameters
    ----------
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

    Returns
    -------
    baseline_arr: np.array, an array of ones and zeros where there's a valid time point

    Develeopment notes
    ---------
    In the for loop the reasons for using index vs trial is from differences in
    starting iteration with zero in python vs whatever arbitary trial is in
    the event structure, and how I choose to format baseline_arr
    """
    # So that this will also work on scalp data since they have inconsistent fields...
    trial_field = 'trial' if 'trial' in behavioral_events.dtype.names else 'list'

    behavioral_events = remove_practice_events(behavioral_events)
    trials = np.unique(behavioral_events[trial_field])

    # Create an array of ones of shape n_trials X recall_period (in ms)
    baseline_arr = np.ones((recall_period * len(trials))).reshape(len(trials), recall_period)
    valid, invalid = 1, 0

    # Convert any invalid point in the baseline to zero
    for index, trial in enumerate(trials):

        # Get the events of the trial
        trial_events = iterate_events_by_trial(behavioral_events, trial)
        # Get recall period start and stop points
        starts, stops = return_recall_starts_stops(trial_events)
        # If we don't have any recall periods then we will invalide the whole thing
        if ((starts.shape[0] == 0) & (stops.shape[0] == 0)):
            print('No recall period detected, trial: ', trial)
            baseline_arr[index] = invalid
            continue

        # Recalls or vocalizations
        possible_recalls = inside_recall_period(trial_events, starts, stops)
        invalid_points = return_asrange(possible_recalls['rectime'],
                                        remove_before_recall,
                                        remove_after_recall)
        # Get rid of any trials where we can't find any invalid points.
        if invalid_points is None:
            baseline_arr[index] = invalid
            continue

        # Check the points are within the boundaries of the recall period
        invalid_points = within_boundaries(invalid_points)
        invalid_points = (np.unique(invalid_points),)  # ((),) similiar to np.where

        # Removes initial recall contamination (-remove_before_recall,+remove_after_recall)
        baseline_arr[index][invalid_points] = invalid

        # Remove any valid baselines that are too small (less than desired_bl_duration)
        remove_small, too_small_locs = remove_too_small(baseline_arr,
                                                        index,
                                                        desired_bl_duration)
        if remove_small:
            baseline_arr[index][too_small_locs] = invalid

    return baseline_arr


def get_perfect_matches_in_lists(recalls, baseline_arr, trials, recall_eeg_start=1500, recall_eeg_end=500):
    """Returns possible "perfect" recall-baseline match selections from other lists

    Parameters
    ----------
    recalls: np.array, behavioral events of the recalls of interest
    baseline_arr: np.array, array of ones and zeros for the baseline created using
              get_valid_continuous_baseline_intervals_session
    trials: np.array, array of ints for each valid trial
    recall_eeg_start: int, default = 1500, time in ms of eeg that we would start at
                    relative to recall onset, note the sign (e.g. -1500 vs 1500) does not matter
    recall_eeg_end: int, default = 500, time in ms of eeg that we would stop at
                    relative to recall onset
    Returns
    -------
    perfectly_matched: "perfect" matches, rows are identical to recalls array, columns are valid trials
    """

    # Check if user made it negative or positive
    if np.sign(recall_eeg_start) == -1:
        recall_eeg_start *= -1

    recs_desired_starts = recalls['rectime'] - recall_eeg_start
    recs_desired_stops = recalls['rectime'] + recall_eeg_end

    # Store matches here
    perfectly_matched = []

    # Go through each recall, find perfect matches
    for (start, stop) in zip(recs_desired_starts, recs_desired_stops):
        # Match from -recall_eeg_start to + recall_eeg_end
        has_perf_match_in_list = np.unique(np.where(baseline_arr[:, start:stop] == 1)[0])
        perfectly_matched.append(has_perf_match_in_list)

    return perfectly_matched


def perfect_match_accumulator(ordered_recalls, baseline_array, trials, recall_eeg_start=1500, recall_eeg_end=500):
    """Accumulates exact matches between recalls and baseline array, upon selection of a match invalidates it for other recalls

    Parameters
    ----------
    ordered_recalls: np.array, behavioral events of the recalls of interest, single session of data only,
                     SORTED FROM LEAST TO MOST EXACT MATCHES
    baseline_array: np.array, array of ones and zeros for the baseline created using
                    get_valid_continuous_baseline_intervals_session
    trials: np.array, array of ints for each valid trial
    recall_eeg_start: int, default = 1500, time in ms of eeg that we would start at
                    relative to recall onset, note the sign (e.g. -1500 vs 1500) does not matter
    recall_eeg_end: int, default = 500, time in ms of eeg that we would stop at
                    relative to recall onset
    Returns
    -------
    matches: dict; where keys are rows in ordered_recall array
             and values are [(trial_num, relative_start, relative_stop)] corresponding
             to the given recall
    baseline_array: np.array, array of ones and zeros for the baseline created using
                    get_valid_continuous_baseline_intervals_session. Selected matches are invalid
    """
    # __develop_mode = True for building/debugging
    __develop_mode = False

    # Check if user made it negative or positive
    if np.sign(recall_eeg_start) == -1:
        recall_eeg_start *= -1

    recs_desired_starts = ordered_recalls['rectime'] - recall_eeg_start
    recs_desired_stops = ordered_recalls['rectime'] + recall_eeg_end
    # Store matches here
    matches = {}

    # Go through each recall, find perfect matches, if multiple randomly select one
    for index, (start, stop) in enumerate(zip(recs_desired_starts, recs_desired_stops)):
        if index not in matches:
            matches[index] = []

        # Match from -recall_eeg_start to + recall_eeg_end
        has_match_in_list = np.all(baseline_array[:, start:stop] == 1, 1)
        list_matches = np.where(has_match_in_list == True)[0]

        # If there aren't any perfect matches just continue
        if len(list_matches) == 0:
            print('Could not perfectly match recall index {}'.format(index))
            continue

        # If there are multiple perfect matches, randomly select one of them
        random_selection = np.random.choice(list_matches)

        if index in matches:
            matches[index].append((trials[random_selection], start, stop))

        if __develop_mode:
            print('before modifying {} {} {}'.format(random_selection, start, stop))
            print(baseline_array[random_selection, start:stop])

        # Void the selection so other recalls cannot think it is valid
        baseline_array[random_selection, start:stop] = 0.

        if __develop_mode:
            print('after modifying {} {} {}'.format(random_selection, start, stop))
            print(baseline_array[random_selection, start:stop])

    return matches, baseline_array


def within_tolerance_match_accumulator(ordered_recalls, baseline_array, trials, matches, match_tolerance=2000,
                                       recall_eeg_start=-1500, recall_eeg_end=500):
    """Accumulates matches within a tolerance between recalls and baseline array, upon selection of a match invalidates it for other recalls

    Parameters
    ----------
    ordered_recalls: np.array, behavioral events of the recalls of interest, single session of data only,
                     SORTED FROM LEAST TO MOST EXACT MATCHES
    baseline_array: np.array, array of ones and zeros for the baseline created using
                    get_valid_continuous_baseline_intervals_session
    trials: np.array, array of ints for each valid trial
    matches: dict; where keys are rows in ordered_recall array
             and values are [(trial_num, relative_start, relative_stop)] corresponding
             to the given recall. Created using perfect_match_accumulator
    match_tolerance: int, by default = 2000, time in ms that a deliberation may deviate from the
                     retrieval of an item and still count as a match.
    recall_eeg_start: int, default = 1500, time in ms of eeg that we would start at
                    relative to recall onset, note the sign (e.g. -1500 vs 1500) does not matter
    recall_eeg_end: int, default = 500, time in ms of eeg that we would stop at
                    relative to recall onset
    Returns
    -------
    matches: dict; where keys are rows in ordered_recall array
             and values are [(trial_num, relative_start, relative_stop)] corresponding
             to the given recall
    baseline_array: np.array, array of ones and zeros for the baseline created using
                    get_valid_continuous_baseline_intervals_session. Selected matches are invalid
    """
    # __develop_mode = True for building/debugging
    __develop_mode = False

    if np.sign(recall_eeg_start) == -1:
        recall_eeg_start *= -1

    recs_desired_starts = ordered_recalls['rectime'] - recall_eeg_start
    recs_desired_stops = ordered_recalls['rectime'] + recall_eeg_end
    eeg_len = recall_eeg_end + recall_eeg_start

    # Go through each recall, find perfect matches
    for index, (start, stop) in enumerate(zip(recs_desired_starts, recs_desired_stops)):

        # Don't do already matched recalls
        if matches[index] != []:
            continue

        # Tolerance is around recall_eeg_start up until volcalization onset
        before_start_within_tol = start - match_tolerance
        after_start_within_tol = start + match_tolerance + recall_eeg_start

        # -----> Sanity check: cannot be before or after recall period
        if before_start_within_tol < 0:
            before_start_within_tol = 0

        if after_start_within_tol > baseline_array.shape[-1]:
            after_start_within_tol = baseline_array.shape[-1]

        # Go through from tolerance before recall up until tolerance after, try to find a match
        found_match = False
        while before_start_within_tol < after_start_within_tol:
            if found_match:
                break

            start_slice = before_start_within_tol
            end_slice = start_slice + eeg_len

            has_match_in_list = np.all(baseline_array[:, start_slice:end_slice] == 1, 1)

            # If there are any matches, randomly select one then void those locations
            if np.any(has_match_in_list):
                found_match = True
                list_matches = np.where(has_match_in_list == True)[0]
                random_selection = np.random.choice(list_matches)

                if index in matches:
                    matches[index].append((trials[random_selection], start_slice, end_slice))

                if __develop_mode:
                    print('before modifying {} {} {}'.format(random_selection, start_slice, end_slice))
                    print(baseline_array[random_selection, start_slice:end_slice])

                # Void the selection so other recalls cannot think it is valid
                baseline_array[random_selection, start_slice:end_slice] = 0.

                if __develop_mode:
                    print('after modifying {} {} {}'.format(random_selection, start_slice, end_slice))
                    print(baseline_array[random_selection, start_slice:end_slice])

            before_start_within_tol += 1

        if not found_match:
            print('Could not match recall index {}'.format(index))

    return matches, baseline_array


def create_matched_recarray(matches, ordered_recalls, subject_instance, recall_eeg_start=-1250):
    """Constructs a recarray of ordered_recalls and their matched deliberation points

    Parameters
    ----------
    matches: dict; where keys are rows in ordered_recall array
             and values are [(trial_num, relative_start, relative_stop)] corresponding
             to the given recall. Created using perfect_match_accumulator, and
             within_tolerance_match_accumulator
    ordered_recalls: np.array, behavioral events of the recalls of interest, single session of data only,
                     SORTED FROM LEAST TO MOST EXACT MATCHES
    subject_instance: RetrievalEventCreator, instance of the subject after running method
                      .initialize_recall_events
    recall_eeg_start: int, default = 1500, time in ms of eeg that we would start at
                    relative to recall onset, note the sign (e.g. -1500 vs 1500) does not matter
    Returns
    -------
    behavioral_events: np.recarray, array of included recalls and matched deliberation periods
    """
    if np.sign(recall_eeg_start) == -1:
        recall_eeg_start *= -1

    rec_start, rec_end = return_recall_starts_stops(subject_instance.events)
    sample_rate = subject_instance.sample_rate

    # Non-sense is due to inconsistent fields
    trial_field = 'trial' if 'trial' in ordered_recalls.dtype.names else 'list'
    item_field = 'item_name' if 'item_name' in ordered_recalls.dtype.names else 'item'
    item_number_field = 'item_num' if 'item_num' in ordered_recalls.dtype.names else 'itemno'

    valid_recalls, valid_deliberation = [], []
    # Use the matches dictionary to construct a recarray
    for k, v in enumerate(matches):
        if matches[v] == []:
            continue

        valid_recalls.append(ordered_recalls[k])

        trial, rel_start, rel_stop = matches[v][0]
        trial_rec_start_events = rec_start[rec_start[trial_field] == trial]

        bl = deepcopy(ordered_recalls[k])
        bl['type'] = 'REC_BASE'
        bl[trial_field] = trial
        bl[item_field] = 'N/A'
        bl[item_number_field] = -999
        bl['timebefore'] = -999
        bl['timeafter'] = -999
        bl['eegoffset'] = -999
        # Since we'll want to use EEGReader for both at once the below adjustment should work
        bl['rectime'] = rel_start - recall_eeg_start
        bl['mstime'] = trial_rec_start_events['mstime'] + bl['rectime']
        bl['eegoffset'] = trial_rec_start_events['eegoffset'] + (sample_rate * (bl['rectime'] / 1000.))
        valid_deliberation.append(bl)

    valid_recalls = np.array(valid_recalls).view(np.recarray)
    valid_deliberation = np.array(valid_deliberation).view(np.recarray)

    behavioral_events = np.concatenate((valid_deliberation, valid_recalls)).view(np.recarray)
    behavioral_events.sort(order='match')

    return behavioral_events