"""A script used to generate usable BrainAtlases for each subject from the TalReader and electrode category text"""

import os
import numpy as np
from ElectrodeCategoryReader import electrode_categories_reader, get_elec_cat

def load_sub_tal(subject):
    """Loads subject tal from logans scratch directory rather than ptsa"""
    load_path = '/scratch/loganf/electrode_information/{}/{}.npy'
    mp = np.load(load_path.format(subject, 'mp'))
    bp = np.load(load_path.format(subject, 'bp'))
    tal = np.load(load_path.format(subject, 'tal'))
    return mp, bp, tal


def get_lobe(arr):
    """Adds in lobe information"""
    region_to_lobe_d = {  # Always frontal!
        u'caudalmiddlefrontal': u'Frontal Lobe',
        u'DLPFC': u'Frontal Lobe',
        u'Caudal Middle Frontal Cortex': u'Frontal Lobe',
        u'rostralmiddlefrontal': u'Frontal Lobe',
        u'lateralorbitofrontal': u'Frontal Lobe',
        u'medialorbitofrontal': u'Frontal Lobe',
        u'parstriangularis': u'Frontal Lobe',
        u'parsorbitalis': u'Frontal Lobe',
        u'parsopercularis': u'Frontal Lobe',
        u'paracentral': u'Frontal Lobe',
        u'precentral': u'Frontal Lobe',
        u'Precentral Gyrus': u'Frontal Lobe',
        u'superiorfrontal': u'Frontal Lobe',
        u'Superior Frontal Gyrus': u'Frontal Lobe',
        u'frontalpole': u'Frontal Lobe',
        u'rostralanteriorcingulate': u'Frontal Lobe',
        u'caudalanteriorcingulate': u'Frontal Lobe',
        u'ACg': u'Frontal Lobe',

        # Always Temporal!
        u'inferiortemporal': u'Temporal Lobe',
        u'middletemporal': u'Temporal Lobe',
        u'Middle Temporal Gyrus': u'Temporal Lobe',
        u'superiortemporal': u'Temporal Lobe',
        u'parahippocampal': u'Temporal Lobe',
        u'temporalpole': u'Temporal Lobe',
        u'bankssts': u'Temporal Lobe',
        u'transversetemporal': u'Temporal Lobe',
        u'MTL WM': u'Temporal Lobe',
        u'STG': u'Temporal Lobe',
        u'TC': u'Temporal Lobe',
        u'fusiform': u'Temporal Lobe',

        # Always Parietal!
        u'inferiorparietal': u'Parietal Lobe',
        u'superiorparietal': u'Parietal Lobe',
        u'supramarginal': u'Parietal Lobe',
        u'Supramarginal Gyrus': u'Parietal Lobe',
        u'precuneus': u'Parietal Lobe',
        u'Precuneus': u'Parietal Lobe',
        u'postcentral': u'Parietal Lobe',
        u'posteriorcingulate': u'Parietal Lobe',
        u'isthmuscingulate': u'Parietal Lobe',
        u'PCg': u'Parietal Lobe',

        # Always Occipital!
        u'lateraloccipital': u'Occipital Lobe',
        u'lingual': u'Occipital Lobe',
        u'cuneus': u'Occipital Lobe',
        u'pericalcarine': u'Occipital Lobe',

        # Medial Temporal Lobe
        u'EC': u'Medial Temporal Lobe',
        u'Amy': u'Medial Temporal Lobe',
        u'CA1': u'Medial Temporal Lobe',
        u'CA2': u'Medial Temporal Lobe',
        u'CA3': u'Medial Temporal Lobe',
        u'DG': u'Medial Temporal Lobe',
        u'PHC': u'Medial Temporal Lobe',
        u'PRC': u'Medial Temporal Lobe',
        u'Sub': u'Medial Temporal Lobe',
        u'entorhinal': u'Medial Temporal Lobe',
        u'Hippocampus': u'Medial Temporal Lobe',

        # Wild cards
        None: 'None',
        u'insula': u'insula',
        u'MCg': u'MCg'
    }

    hemi = np.array([region_to_lobe_d[x] for x in [s['locTag']
                                                   if s['locTag'] != 'unmarked' else s['ind']
                                                   for s in arr]
                     ])

    arr['hemi'] = hemi
    return arr


def get_region_level_label(subject):
    """Returns a region level label [e.g. Left inferiorfrontal, Right hippocampus...]
    and corresponding valid bipolar_pairs for a subject
    """

    # Default paths for loading subjects bp and electrode information
    channel_paths = '/scratch/loganf/electrode_information/{}/{}.npy'.format(subject, 'bp')
    atlas_path = '/scratch/loganf/subject_brain_atlas/{}.npy'.format(subject)

    if os.path.exists(atlas_path):
        atlas = np.load(atlas_path)
    if os.path.exists(channel_paths):
        bp = np.load(channel_paths)
    else:
        print('something wrong with the path! Check out subject {}'.format(subject))
        return

    # Check that they have the relevant fields....
    has_stein = True if 'stein' in atlas['atlases'].dtype.names else False
    has_ind = True if 'ind' in atlas['atlases'].dtype.names else False
    has_avg = True if 'avg' in atlas['atlases'].dtype.names else False
    has_tal = True if 'tal' in atlas['atlases'].dtype.names else False

    # Values for hippocampal electrodes, used in a comparasion
    # Just makes a list of Left CA1, Right CA1...etc
    hpc = np.concatenate([[['Left ' + x], ['Right ' + x]]
                          for x in ['CA1', 'CA2', 'CA3', 'DG']])

    # The atlas we're going to use, and the roi names
    atlas_choice, names = 'none', None

    # Check the atlases and get the region name
    if has_ind:
        names = atlas['atlases']['ind']['region']
        atlas_choice = 'ind'

    elif has_tal:
        names = atlas['atlases']['tal']['region']
        atlas_choice = 'tal'

    elif has_avg:
        names = atlas['atlases']['avg']['region']
        atlas_choice = 'avg'

    # We're going to use the stein localization to determine if they are HPC electrodes....
    if has_stein:
        # If there's a stein atlas then we'll relabel the hpc!
        hpc_locs = np.where(np.in1d(atlas['atlases']['stein']['region'],
                                    np.intersect1d(atlas['atlases']['stein']['region'],
                                                   hpc)))
        names[hpc_locs] = 'hippocampus'

        # If there are still nones and there's a stein then try to replace it....
        has_nones = np.where(names == u'None')
        rename_if_valid = np.array([x if x != u'nan' else u'None'
                                    for x in atlas['atlases']['stein']['region'][has_nones]])
        names[has_nones] = rename_if_valid

    # Remove any bad chanenls identified from electrode cat txt file
    my_channels = names[~atlas['bad ch']]
    # Remove any 'bad channels' labeled as 'None' b/c I have no idea where they are
    remove = np.where(my_channels != u'None')

    # Get the hemisphere, remove bad channels from electrode cat file
    hemi = atlas['hemi']
    my_hemis = hemi[~atlas['bad ch']]
    # Put the hemi and roi together then return the names and valid bipolar pairs
    names = [' '.join(x) for x in zip(my_hemis[remove], my_channels[remove])]
    valid_bp = bp[~atlas['bad ch']][remove]

    return np.array(names).view(np.recarray), valid_bp

# OLD ATLAS REDO!
def build_atlas(subject, experiment):
    """Builds an atlas with relevant fields for data analysis
    np.dtype([('tagName', 'O'),('ind', 'O'), ('locTag', 'O'), ('hemi', 'O'), ('channel_1', 'O'),
                         ('channel_2', 'O'), ('tag_1', 'O'), ('tag_2', 'O'),
                         ('x', np.float32), ('y', np.float32), ('z', np.float32),
                         ('SOZ', bool), ('IS', bool), ('brain lesion', bool), ('bad ch', bool)])

    """
    tal = get_tal_new(subject, experiment)
    # columns = ['ind', 'locTag', 'x','y','z', 'hemi', 'lobe', 'SOZ', 'IS', 'brain lesion', 'bad ch']
    # Create empty array
    my_dtype = np.dtype([('tagName', 'O'), ('ind', 'O'), ('locTag', 'O'), ('hemi', 'O'), ('channel_1', 'O'),
                         ('channel_2', 'O'), ('tag_1', 'O'), ('tag_2', 'O'),
                         ('x', np.float32), ('y', np.float32), ('z', np.float32),
                         ('SOZ', bool), ('IS', bool), ('brain lesion', bool), ('bad ch', bool)])

    arr = np.zeros((len(tal)), dtype=my_dtype)
    arr = arr.view(np.recarray)

    # Get bad channels, seizure onset zone etc.
    e_cat_reader = get_elec_cat(subject)
    hpc = ['CA1', 'CA2', 'CA3', 'DG', 'Sub']  # Use to relabel

    for index, t in enumerate(tal):
        # Get tagname
        arr[index]['tagName'] = t['tagName']
        arr[index]['tag_1'], arr[index]['tag_2'] = arr[index]['tagName'].split('-')

        # Add indiv surface region
        ind = 'unmarked'
        if 'ind' in t['atlases'].keys():
            ind = t['atlases']['ind']['region']

        arr[index]['ind'] = ind

        # Add stein loctags
        stein = 'unmarked'
        if 'stein' in t['atlases'].keys():
            stein = t['atlases']['stein']['region']
            stein = stein.split('Left ')[-1].split('Right ')[-1]
            if stein in hpc:
                stein = 'Hippocampus'
        arr[index]['locTag'] = stein

        # Add xyz coords
        try:
            x, y, z = t['atlases']['ind']['x'], t['atlases']['ind']['y'], t['atlases']['ind']['z']
        except KeyError:
            x, y, z = None, None, None
        arr[index]['x'] = x
        arr[index]['y'] = y
        arr[index]['z'] = z

        # Add hemi
        arr[index]['hemi'] = 'Right' if np.sign(x) == 1.0 else 'Left'

        # Get channel 1 and 2, render as str so more relevant
        arr[index]['channel_1'] = ('000' + str(t['channel_1']))[-3:]
        arr[index]['channel_2'] = ('000' + str(t['channel_2']))[-3:]

        # Add in bad ch soz etc
        if e_cat_reader is not None:
            for key in e_cat_reader.keys():
                if ((arr[index]['tag_1'] in e_cat_reader[key])
                    or (arr[index]['tag_2'] in e_cat_reader[key])):
                    arr[index][key] = True

    # Add in lobe details
    arr = get_lobe(arr)

    return arr
