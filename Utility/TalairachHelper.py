"""TalairachHelper.py, author=LoganF
General purpose script for helpful interactions with the Talairach"""

import numpy as np
import os
import sys
from ptsa.data.readers import TalReader, JsonIndexReader
from ElectrodeCategoryReader import electrode_categories_reader

# Functions to load talirach, see below for removing bad channels
def get_tal_new(subject, experiment):
    """Returns talirach using NEW METHOD and JsonIndexReader!

    Parameters
    ----------
    subject
    experiment

    Returns
    -------

    """
    jr = JsonIndexReader('/protocols/r1.json')
    pairs_path = jr.get_value('pairs', subject=subject, experiment=experiment)
    tal_reader = TalReader(filename=pairs_path)
    return tal_reader.read()


def get_tal_old(subject, experiment, bipolar=True):
    """Returns talirach using OLD METHOD and .mat FILES!

    Parameters
    ----------
    subject
    experiment
    bipolar

    Returns
    -------

    """
    path = os.path.join('/data/eeg/', subject, 'tal', subject + '_talLocs_database_{}pol.mat')
    if bipolar:
        tal_reader = TalReader(filename=path.format('bi'))
    else:
        # struct_name is a bit obnoxious to need to pass into TalReader~
        tal_reader = TalReader(filename=path.format('mono'), struct_name="talStruct")
    return tal_reader.read()


# Converting data from one format to another
def tal_ch_to_string(tals):
    """Utility function to return the strings necessary for using EEGReader from talirach structure
    ------
    INPUTS:
    tals: np.recarray, talirach structure returned by ptsa's TalReader or CML's get_sub_tal
    ------
    OUTPUTS:
    mp_string_chs: np.array; monopolar channels represented as strings, e.g. ('001', '002'...)
    bp_string_chs:np.array; bipolar channels represented as strings, e.g. ('001-002'...)
    """
    bp_string_chs = []
    mp_string_chs = []
    for ch in tals['channel']:
        ch1, ch2 = ('000' + str(ch[0]))[-3:], ('000' + str(ch[1]))[-3:]
        bp_string_chs.append(np.array((ch1, ch2), dtype=[('ch0', 'S3'), ('ch1', 'S3')]))
        mp_string_chs.append(np.array(ch1, dtype='|S3'))
        mp_string_chs.append(np.array(ch2, dtype='|S3'))
    bp_string_chs = np.array(bp_string_chs).view(np.recarray)
    mp_string_chs = np.unique(mp_string_chs).view(np.recarray)
    return mp_string_chs, bp_string_chs



# -----> OLD?

# Use to remove bad channels
def excluded_electrodes(subject):
    """Returns out a dictionary containing values associated with bad channels"""

    check = {'bad electrodes', 'bad electrodes:', 'broken leads', 'broken leads:'}
    groups = electrode_categories_reader(subject)
    bad_channels = {v: groups[v] for k, v in enumerate(groups) if v in check}
    return bad_channels


# Load Data only good channels
def get_good_tal_new(subject, experiment='FR1'):
    """Returns out a mp,bp, and tal using the new json format that corresponds to only correct chs"""
    # Get new talarach structure, channels
    new_tal = get_tal_new(subject, experiment)
    monopolar, bipolar = tal_ch_to_string(new_tal)
    try:
        # Get electrodes we need to remove, raw
        ex = excluded_electrodes(subject)
        # Get the channels to remove, filtered by bp matches
        if len(ex) > 0:
            remove = [x for x in new_tal.tagName
                      if ((x.split('-')[0] in ex[ex.keys()[0]])
                          or (x.split('-')[1] in ex[ex.keys()[0]]))
                      ]  # Check looks weird b/c bipolar formatting

            # Get the locations of where to remove bad channels
            locs = np.where(np.in1d(new_tal.tagName,
                                    np.intersect1d(np.array(remove),
                                                   new_tal.tagName)) == False)
            # Return the filtered structures
            tal = new_tal[locs]
            bp = bipolar[locs]
            mp = np.unique(np.concatenate([[x[0], x[1]] for x in bp]))
            return mp, bp, tal
        else:

            return monopolar, bipolar, new_tal
    except:
        print('failure, {}'.format(subject))
        return monopolar, bipolar, new_tal


# ----> Getting the dictionary to work...

# Return a dictionary of all atlases
def get_atlases(subj, exp='FR1'):
    """Return a dictionary of all possible atlas dictionarys for a subject"""
    _, _, tal_new = get_good_tal_new(subj, exp)
    ind, stein = [], []

    atlas_d = np.unique(np.concatenate([tal['atlases'].keys() for tal in tal_new]))
    atlas_d = {v: [] for v in atlas_d}

    for index, tal in enumerate(tal_new):
        for atlas in atlas_d:
            if atlas in tal['atlases'].keys():
                atlas_d[atlas].append((tal['atlases'][atlas]['region']))
            else:
                atlas_d[atlas].append('')

    return atlas_d


# Return an array of associated names where it's hemi + Region
def get_new_tal_names_default(subj, exp='FR1', add_hemi=True):
    """Return the default stein or ind atlas region location, adds in hemisphere"""
    _, _, tal_new_correct = get_good_tal_new(subj, exp)
    tal_new = get_tal_new(subj, exp)
    removed_chs = np.where(np.in1d(tal_new, (np.intersect1d(tal_new, tal_new_correct))) == False)

    talairach = []
    hemi = []
    for index, tal in enumerate(tal_new):
        hemi.append(np.sign(tal['atlases']['ind']['x']))

        if 'stein' in tal['atlases'].keys():
            talairach.append(tal['atlases']['stein']['region'])
            continue

        if tal['atlases']['ind']['region'] is not None:
            talairach.append(tal['atlases']['ind']['region'])
            continue

        else:
            talairach.append('unmarked')
    if add_hemi:
        hemi = np.array(['Left ' if h == -1 else 'Right ' for h in hemi])
        for i, t in enumerate(talairach):
            if ((t[:4] != 'Left') and (t[:5] != 'Right')):
                talairach[i] = hemi[i] + t
            if i in removed_chs[0]:  # [0] b/c tuple
                talairach[i] = 'BAD ' + talairach[i]
    return np.array(talairach)


def return_lobe_loc(subj, exp='FR1', remove_nones=False, mask_nones=True, freesurf_loc=2):
    """Returns a dictionary of the associated name for the lobe
    -----
    INPUTS
    subj
    exp
    remove_nones: bool, default=False, whether to remove the nones from the dictionary
    mask_nones: bool, default=True, whether to replace None key with a 'unmarked' key
    ------
    OUTPUTS
    lobe_match: dict, dictionary where keys are new tala region entries and values are only tala loc field"""
    # Get matlab associated tal
    tal_old = get_tal_old(subj, exp)
    # Get json associated tal using either stein or ind region
    new_regions = get_new_tal_names_default(subj, exp)

    freesurf_loc = 'Loc' + str(freesurf_loc)
    # Get match between regions and lobe; Using a set will
    # Remove duplicates and you create a dict from it afterwards
    lobe_match = dict(set(zip(new_regions, tal_old[freesurf_loc])))

    # if we want a mask on the lobe match replace None key with 'unmarked' key
    if (mask_nones) and (None in lobe_match.keys()):
        lobe_match['unmarked'] = lobe_match[None]
        lobe_match.pop(None)

    # If remove_nones is True remove them
    if (remove_nones) and (None in lobe_match.keys()):
        lobe_match.pop(None)
    return lobe_match