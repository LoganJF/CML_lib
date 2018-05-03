"""GetDataHelper.py, author=Loganf
A helper script used to help the script GetData.py
"""
try:
    from ptsa.data.readers.IndexReader import JsonIndexReader
    from ptsa.data.readers import TalReader
    from ptsa.data.readers import BaseEventReader
except ImportError:
    from ptsa.data.readers import JsonIndexReader, TalReader, BaseEventReader
import os, sys
import numpy as np
from glob import glob


class GetTalirach():
    def __init__(self, subject, experiment):
        self.subject = subject
        self.experiment = experiment
        self.set_mp_bp_tal()

    def set_mp_bp_tal(self, kind='matlab', bipolar=True):
        try:
            if kind.lower() == 'matlab':
                m, b, t = self._get_matlab_mp_bp_tal(bipolar=bipolar)
                self.monopolar_channels, self.bipolar_pairs, self.talirach_structures = m, b, t
                return
        except:
            m, b, t = self._get_json_mp_bp_tal()
            self.monopolar_channels, self.bipolar_pairs, self.talirach_structures = m, b, t
        return

    def _get_json_mp_bp_tal(self):
        """Returns mp,bp and by default (tals=True) talirach"""
        jr = JsonIndexReader('/protocols/r1.json')
        pairs_path = jr.get_value('pairs', subject=self.subject, experiment=self.experiment)
        tal_reader = TalReader(filename=pairs_path)
        mp = tal_reader.get_monopolar_channels()
        bp = tal_reader.get_bipolar_pairs()
        tals = tal_reader.read()
        return mp, bp, tals

    def _get_matlab_mp_bp_tal(self, bipolar=True):

        path = os.path.join('/data/eeg/', self.subject, 'tal', self.subject + '_talLocs_database_{}pol.mat')
        if bipolar:
            path = path.format('bi')
            tal_reader = TalReader(filename=path)
        else:
            path = path.format('mono')
            tal_reader = TalReader(filename=path, struct_name="talStruct")
        mp = tal_reader.get_monopolar_channels()
        bp = tal_reader.get_bipolar_pairs()
        tals = tal_reader.read()
        return mp, bp, tals

    def _exclude_bad_channels(self, bipolar=True):
        path = os.path.join('/data/eeg/', self.subject, 'tal', self.subject + '_talLocs_database_{}pol.mat')
        tal_path = path.format('bi') if bipolar else path.format('mono')

        monopolar_channels = [tal_reader.get_monopolar_channels() for tal_reader in
                              [TalReader(filename=tal_path)]]
        bipolar_pairs = [tal_reader.get_bipolar_pairs() for tal_reader in [TalReader(filename=tal_path)]]

        # Exclude bad channels
        bp = []
        mp = []
        for talPathi, bipolar_pairsi, monopolar_channelsi in zip(
                tal_path, bipolar_pairs, monopolar_channels):
            try:
                gf = open(os.path.dirname(tal_path) + '/good_leads.txt', 'r')
            except IOError:
                print('No good talirach files were found for {} this needs to be corrected'.format(self.subject))
            goodleads = gf.read().split('\n')
            gf.close

            if os.path.isfile(os.path.dirname(tal_path) + '/bad_leads.txt'):
                bf = open(os.path.dirname(tal_path) + '/bad_leads.txt', 'r')
                badleads = bf.read().split('\n')
                bf.close
            else:
                badleads = []
            subPairs = np.array([pairi for pairi in bipolar_pairsi
                                 if ( str(pairi[0]).lstrip('0') in goodleads)
                                 and ( str(pairi[1]).lstrip('0') in goodleads)
                                 and ( str(pairi[0]).lstrip('0') not in badleads)
                                 and ( str(pairi[1]).lstrip('0') not in badleads)])
            subPairs = np.rec.array(subPairs)
            subChans = np.array([chani for chani in monopolar_channelsi
                                 if (str(chani).lstrip('0') in goodleads)
                                 and (str(chani).lstrip('0') not in badleads)])
            bp.append(subPairs)
            mp.append(subChans)
        return mp[0], bp[0]

    def _get_good_tal(self, return_all=False, bipolar=True):
        """
        BREAKS ON MONOPOLAR!!!
        """
        path = os.path.join('/data/eeg/', self.subject, 'tal', self.subject + '_talLocs_database_{}pol.mat')
        tal_path = path.format('bi') if bipolar else path.format('mono')

        mp, bp = self._exclude_bad_channels()
        tal_reader = TalReader(filename=tal_path) if bipolar else TalReader(filename=tal_path, struct_name="talStruct")
        tal_structs = tal_reader.read()

        """Get only good pairs; int conversion necessary to convert [001,002]
        into [1,2] for comparasion match"""
        # return mp, bp, tal_structs
        channels_indx = []
        if bipolar:
            for i, ch in enumerate(tal_structs.channel):
                for x, pair in enumerate(bp):
                    if (int(pair[0]) in ch) & (int(pair[1]) in ch):
                        channels_indx.append(i)
        if not bipolar:
            for i, pair in enumerate(mp):
                if (int(pair) in tal_structs.channel):
                    channels_indx.append(i)
        # Get good tals from all tals using index
        good_tal = np.array([tal for indx, tal in enumerate(tal_structs) if indx
                             in channels_indx]).view(np.recarray)
        return mp, bp, good_tal if return_all else good_tal

    def set_good_montage(self, bipolar_reference=True):
        m, b, t = self._get_good_tal(return_all=True, bipolar=bipolar_reference)
        self.monopolar_channels, self.bipolar_pairs, self.talirach_structures = m, b, t

    def get_good_montage(self, bipolar_reference=True):
        return self._get_good_tal(return_all=True, bipolar=bipolar_reference)

class GetEvents():
    """TO DO: ADD IN A STRING CONVERSION SO THAT THE UPPERCASE/LOWERCASE DOESNT MATTER
    E.g. input.lower()...."""
    json_experiments = JsonIndexReader('/protocols/r1.json').experiments()
    path = '/data/events/*/*_events.mat'
    mlab_experiments = np.unique([x.split('/')[3] for x in glob(path)])

    def __init__(self, subject, experiment, overide_json=False, verbose=True):
        self.subject = subject
        self.experiment = experiment
        self.verbose = verbose
        self._check_old_or_new()
        if overide_json:
            self.events = self._get_matlab_events()

    def _check_old_or_new(self):
        """Check if subject can be loaded from matlab/json. By default preferentially loads json"""
        boolean = self.experiment in self.json_experiments
        self.events = self._get_matlab_events() if not boolean else self._get_json_events()

    def _get_matlab_events(self):
        path = '/data/events/{}/{}_events.mat'.format(self.experiment, self.subject)
        if os.path.exists(path):
            events = BaseEventReader(filename=path).read().view(np.recarray)
            return events

    def _get_json_events(self):
        jr = JsonIndexReader('/protocols/r1.json')
        filepaths = jr.aggregate_values('task_events', subject=self.subject,
                                        experiment=self.experiment)
        events = [BaseEventReader(filename=f).read() for f in filepaths]

        try:
            events = np.concatenate(events).view(np.recarray)
            return events
        except ValueError:  # Inform if they're not in the experiment
            print('ValueError: Something went wrong, perhaps {} is not in {}'.format(self.subject, self.experiment))
            return events
        except TypeError:  # Inform if there are different dtypes and try to correct it
            print('Theres probably a bug with json reading....check if the dtypes are consistent')
            if type(events) == list:
                return fix_data_fields(events, self.verbose)
        except:
            print("No clue what went wrong!!!")
            return events

# --------> HELPER FUNCTIONS
def fix_data_fields(events, verbose=True):
    """Fixes any data discrepancy by deleting extra fields ADDED FEB 5th to fix issues"""
    import numpy as np

    def remove_field_name(arr, name):
        a = np.copy(arr)
        names = list(a.dtype.names)
        if name in names:
            names.remove(name)
        b = a[names]
        return b.view(np.recarray)

    # Find numpy of unique dtypes
    dtypes = np.unique([ev.dtype for ev in events])
    # Find minimum and maximum len of the dtypes
    min_dtype = min(dtypes, key=len)
    max_dtype = max(dtypes, key=len)
    # Find the value we need to remove
    missing_dtype = np.setxor1d(min_dtype.names, max_dtype.names)

    bad_indicies = []
    for index, evs in enumerate(events):
        if len(evs.dtype) == len(max_dtype):
            for value in missing_dtype:
                evs = remove_field_name(evs, value)
            events[index] = evs
            bad_indicies.append(index)
    events = np.concatenate(events).view(np.recarray)
    if verbose:
        print("WARNING: There appears to be an issue with indicies {}".format(bad_indicies))
        print("Going to remove the additional fields at {} and return the events!".format(bad_indicies))
        return events