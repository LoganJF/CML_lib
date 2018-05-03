"""GetData.py, author=LoganF
A script used to extract various kinds of data from the CML datasets"""
from glob import glob
import numpy as np
from ptsa.data.filters import MonopolarToBipolarMapper

try:
    from ptsa.data.readers import EEGReader, BaseEventReader
    from ptsa.data.readers.IndexReader import JsonIndexReader
except ImportError:
    from ptsa.data.readers import JsonIndexReader, EEGReader, BaseEventReader
import sys
# sys.path.append('/home2/loganf/SecondYear/Functions/CML/Utility')
from Utility.GetDataHelper import GetTalirach, GetEvents

def get_subs(experiment):
    """return the subjects of a given experiment, by default it will return json over matlab if both are available
    ------
    INPUTS:
    experiment: str, the kind of experiment to input, for RAM subjects use json reader field for json files and
                matlab folders for matlab files
                e.g. 'FR1' will get you json 'RAM_FR1' will get you matlab"""

    try:  # If they're using rhino
        jr = JsonIndexReader('/protocols/r1.json')
        json_experiments = jr.experiments()
        path = '/data/events/{}/*_events.mat'
    except FileNotFoundError:  # if they're not using rhino
        jr = JsonIndexReader('/Volumes/rhino/protocols/r1.json')
        json_experiments = jr.experiments()
        path = '/Volumes/rhino/data/events/{}/*_events.mat'

    matlab_experiments = np.unique([x.split('/')[3] for x in glob(path.format('*'))])

    if experiment in json_experiments:
        subs = jr.subjects(experiment=experiment)
        return np.array(subs)

    elif experiment in matlab_experiments:
        path = path.format(experiment)
        subs = [filepath.split('/')[-1].split('_ev')[0] for filepath in glob(path)]
        return np.array(subs)
    else:
        print('json experiments:\n', json_experiments)
        print('\n')
        print('matlab experiments:\n', matlab_experiments)
        print('\n')
        raise ValueError('Experiment not in matlab format or json format, sorry')


def get_sub_events(subject, experiment):
    """ Returns a single subjects event structures using BaseEventReader from ptsa
    -----
    INPUTS:
    subjects: str, subject_id e.g. 'R1111M'
    experiment: str, experiment, e.g. 'FR1', 'pyFR' the prefix of 'RAM_'
                in front of any ram experiment, e.g. 'RAM_FR1' will load the
                matlab path

    """
    instance = GetEvents(subject=subject, experiment=experiment)
    return instance.events


def get_json_events(subject, experiment, session='all', verbose=True):
    """Returns json event structure, np.recarray
    -----
    INPUTS:
    subject: str, e.g. 'R1111M', 'LTP093', etc.
    experiment: str, e.g. 'FR1', 'ltpFR2', etc.
    session: int/str/ 'all'; sessions to return
    verbose: bool, debugging purposes and useflow
    """
    # If scalp they're loading scalp data
    if experiment.lower() == 'ltpfr2':
        experiment = 'ltpFR2'  # Necessary due to inflexibility in capitilization in jsonreader!!
        jr = JsonIndexReader('/protocols/ltp.json')

        # Fix any default uasuage 'oops' by user
        if session == 'all':  # Can only load one session of scalp at a time!
            filepaths = jr.get_value('task_events', subject=subject, experiment=experiment, session=1)
            print('please enter an int for session not "all", returning session 1 by default')
            events = BaseEventReader(filename=filepaths).read()
            return events

        # Correct use of function for scalp, check for str float int handling.
        if ((type(session) == int) or (type(session) == str)):
            filepaths = jr.get_value('task_events', subject=subject,
                                     experiment=experiment, session=str(session))
            events = BaseEventReader(filename=filepaths).read()
            return events

    # If they're loading intracranial data
    jr = JsonIndexReader('/protocols/r1.json')
    if experiment not in jr.experiments():  # Check it's there first
        print('Something went wrong, perhaps {} is not in {}'.format(subject, experiment))
        print('These are the subjects in {}:\n{}'.format(experiment, jr.subjects(experiment=experiment)))
        raise ValueError('Something went wrong, perhaps {} is not in {}'.format(experiment, jr.experiments()))

    # Intracranial below; only reason for split is due to huge amount of time it takes to load all sessions of scalp
    if session == 'all':
        if verbose:
            print('returning all session for {}'.format(subject))
        filepaths = jr.aggregate_values('task_events', subject=subject, experiment=experiment)
        events = [BaseEventReader(filename=f).read() for f in filepaths]

        try:
            events = np.concatenate(events).view(np.recarray)
            return events
        except ValueError:  # Inform if they're not in the experiment
            print('ValueError: Something went wrong, perhaps {} is not in {}'.format(self.subject, self.experiment))
            return events
        except TypeError:  # Inform if there are different dtypes and try to correct it
            print('Theres probably a bug with json reading....check if the dtypes are consistent')
            return fix_data_fields(events, self.verbose)
        except:
            print("No clue what went wrong!!!")
            return events

    if ((type(session) == int) or (type(session) == str)):
        filepaths = jr.get_value('task_events', subject=subject,
                                 experiment=experiment, session=str(session))
        events = BaseEventReader(filename=filepaths).read()
        return events


def get_sub_tal(subject, experiment, exclude_bad=True, bipolar=True):
    """Returns indexes for monopolar channel and bipolar pairs as well as Talirach structure for a subject
    -----
    INPUTS:
    :subject: str, the subject's ID number
    :experiment: str, the experiment to examine
    :exclude_pad: boolean, whether to discard electrodes indicated as bad, True by default
    :bipolar: boolean, True to load bipolar talirach False to load monopolar
    -----
    OUTPUTS:
    :monopolar_channels: array like, indexes used to create a subject's monopolar eeg
    :bipolar_pairs: array like, indexes used to create a subject's bipolar eeg
    :talirach_stuctures: rec.array like, detailed electrode localizaition information of the subject

    ------
    EXAMPLE USES:
    mp, bp, tal = get_sub_tal(subject='R1111M', experiment='FR1')
    mp, bp, tal = get_sub_tal(subject='BW022', experiment='pyFR', exclude_bad=True)
    """
    instance = GetTalirach(subject=subject, experiment=experiment)
    if exclude_bad:
        try:
            monopolar_channels, bipolar_pairs, talirach_stuctures = instance.get_good_montage(bipolar_reference=bipolar)
            return monopolar_channels, bipolar_pairs, talirach_stuctures
        except UnboundLocalError:
            print('Failure to remove bad channels, all channels will be loaded')
        except IOError:  # If there's not a matlab file
            return instance._get_json_mp_bp_tal()
            # monopolar_channels, bipolar_pairs = instance.get_good_trodes()
            # return monopolar_channels, bipolar_pairs
    try:  # If they don't want to exclude bad chs give the matlab,
        return instance.monopolar_channels, instance.bipolar_pairs, instance.talirach_structures
    except IOError:
        return instance._get_json_mp_bp_tal()


def get_sub_eeg(subject, experiment, events=None, reference='bipolar',
                eeg_start=-2., eeg_end=2., eeg_buffer=1., exclude_bad=True):
    """Returns a subjects EEG using ptsa's EEGReader
     -----
    INPUTS
    subjects: str, subject_id e.g. 'R1111M'
    experiment: str, experiment, e.g. 'FR1', 'pyFR' the prefix of 'RAM_'
                in front of any ram experiment, e.g. 'RAM_FR1' will load the
                matlab path
    events: np.recarray like, by default none will load all eeg, if passed will only
            load eeg corresponding to the events array
    reference: str, what kind of referencing system to load, by default 'bipolar',
               accepted arguments: 'bipolar', 'monopolar'
    eeg_start: float, by default -2., the time point relative to the event to start loading
    eeg_end: float, by default 2., the time point relative to the event to stop loading
    eeg_buffer: float, by default 1., the time to add on each side of the data that can be removed
                after convolution
    exclude_bad: bool, by default true; True if you'd like to only look at "valid" channels,
                       False if you want to return all channels
    """
    evs = get_sub_events(subject=subject, experiment=experiment) if events is None else events

    mp, bp, tal = get_sub_tal(subject=subject, experiment=experiment, exclude_bad=exclude_bad)

    # Load eeg from start to end, include buffer
    eeg_reader = EEGReader(events=evs, channels=mp, start_time=eeg_start,
                           end_time=eeg_end, buffer_time=eeg_buffer)

    eeg = eeg_reader.read()

    if reference.lower() == 'bipolar':
        m2b = MonopolarToBipolarMapper(time_series=eeg, bipolar_pairs=bp)
        eeg = m2b.filter()
        return eeg

    if reference.lower() == 'monopolar':
        return eeg


def get_intrusions(base_events, seperate=False):
    """Return intrusion events from an inputted event array
    ------
    INPUTS:
    base_events: np.recarray like, events to return intrusions from
    seperate: boolean, by default false, set to true if you would like
              to seperate prior list intrusions from extra list intrusions
              into two seperate arrays
    ------
    OUTPUTS (seperate=False):
    intrusion_events: np.recarray like, all intrusion events
    ------
    OUTPUTS (seperate=True):
    prior_list: np.recarray like, prior list intrusions (true recalls,
                incorrect context/list)
    extra_list: np.recarray like, extra list intrusions (false recalls,
                words were never shown)
    -------
    EXAMPLE USES
    intrusions = get_intrusions(evs)
    priors,extras = get_intrusions(base_events = events, seperate=True)

    -------
    RELEVANT CODING DETAILS:
    In the CML lab, we indicate intrusion status with the field 'intrusion',
    extra-list intrusions are indiciated with the field being -1, correct recalls
    are 0, events where intrusion doesn't make sense (for example, during encoding,
    distractor tasks etc.) are -999, and prior list intrusions are indicated with
    numbers ranging from 1-24 with the number representing the number of lists prior
    the word was actually presented in.
    """
    # Create index to remove correct recalls and non-recalls
    intrusion_index = np.logical_and(base_events['intrusion'] != -999,
                                     base_events['intrusion'] != 0)
    intrustion_events = base_events[intrusion_index]
    # Seperate intrusions into prior list and extra lists
    if seperate:
        prior_list = intrustion_events[intrustion_events['intrusion'] != -1]
        extra_list = intrustion_events[intrustion_events['intrusion'] == -1]
        return prior_list, extra_list
    """
    TO DO: ask christoph about if below would be prefered syntax, it ~doubles
    performance time (78us->163us) for a ~2500 sized array
    prior_list = intrustion_events[intrustion_events['intrusion'] != -1]
    extra_list = intrustion_events[intrustion_events['intrusion'] == -1]
    return (prior_list, extra_list) if seperate else intrustion_events
    """

    return intrustion_events