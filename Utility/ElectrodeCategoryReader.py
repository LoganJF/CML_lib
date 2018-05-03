"""ElectrodeCategoryReader.py, author = LoganF, updated May 3 2018

A script used to parse relevant information from the electrode category text file, wherever it may be.
"""
import os
import numpy as np


def electrode_categories_reader(subject):
    """Returns a dictionary mapping categories to electrode from the electrode_categories.txt file

    Parameters
    ----------
    :subject:
        str, subject ID for subjects in the RAM project. E.g. 'R1111M'

    Returns
    -------
    :groups:
        dict, dictionary mapping relevant field values (bad channel, SOZ, etc. to the corresponding channels

    --------
    NOTES: This function is only required because there's so much inconsistency in where and how
    the data corresponding to bad electrodes are stored
        """
    # Used to indicate relevant strings in the text files
    relevant = {'seizure onset zone', 'seizure onset zones', 'seizure onset',
                'interictal', 'interictal spiking', 'interictal spikes', 'ictal onset',
                'ictal onset:', 'interictal spiking:',
                'brain lesions', 'brain lesions:',
                'octal onset zone',
                'bad electrodes', 'bad electrodes:', 'broken leads', 'broken leads:'}

    # Open the file, which may exist in multiple places
    fn = '/data/eeg/{}/docs/electrode_categories.txt'.format(subject)

    if not os.path.exists(fn):
        print('{} does not exist'.format(fn))
        fn = '/scratch/pwanda/electrode_categories/{}_electrode_categories.txt'.format(subject)
    if not os.path.exists(fn):  # Check spot one, if not go to paul's scratch directory
        # Because literally it's ACTUALLY SAVED ON PAUL'S SCRATCH DIRECTORY
        fn = '/scratch/pwanda/electrode_categories/electrode_categories_{}.txt'.format(subject)

    try:
        with open(fn, 'r') as f:
            ch_info = f.read().split('\n')

    except IOError:
        print('File {} not found'.format(fn))
        return

    # This will be used to initalize a before after kind of check to sort the groups
    prev = ch_info[0]
    count = 0
    groups = {}  # Save the groups here

    for index, current in enumerate(ch_info[2:]):
        """We skip to two because all files start with line one being subject followed by
        another line of '', if the user wishes to access the information feel free to modify
        below"""
        # Blank spaces used to seperate, if we encountered one count increases
        if (current == ''):
            count += 1
            continue  # Ensures '' isn't appened to dict[group_name]

        # Check if the line is relevant if so add a blank list to the dict
        if current.lower() in relevant:
            count = 0
            group_name = current.lower()
            groups[group_name] = []
            # Sanity check to ensure that they had a '' after the relevant
            if ch_info[2:][index + 1] != '':  # Skipping two for same reason
                count += 1
            continue  # Ensures that group_name isn't later appended to dict[group_name]

        # Triggered when inside of a group e.g. they're channel names
        if (count == 1) and (current != ''):  # indicates start of group values
            groups[group_name].append(current)

        prev = current
    return groups


def get_elec_cat(subject):
    """ Return electrode categories from relevant textfile; ensures that the fields are consistent regardless of the
    actual field the RA entered into the textfile

    Parameters
    ----------
    :subject:
        str, subject ID for subjects in the RAM project. E.g. 'R1111M'

    Returns
    -------
    :e_cat_reader:
        dict, dictionary mapping relevant field values (bad channel, SOZ, etc. to the corresponding channels
    """
    """Cleans up the electrode_cat_reader function so that we get consistent data fields"""
    convert = {'seizure onset zone': 'SOZ', 'seizure onset zones': 'SOZ',
               'seizure onset': 'SOZ',  # Epilepsy onset

               # Interictal activity
               'interictal': 'IS', 'interictal spiking': 'IS',
               'interictal spikes': 'IS', 'ictal onset': 'IS',
               'ictal onset:': 'IS', 'interictal spiking:': 'IS',
               'octal onset zone': 'IS',

               # Lesioned Tissue
               'brain lesions': 'brain lesion', 'brain lesions:': 'brain lesion',

               # Bad channels
               'bad electrodes': 'bad ch', 'bad electrodes:': 'bad ch',
               'broken leads': 'bad ch', 'broken leads:': 'bad ch'}

    e_cat_reader = electrode_categories_reader(subject)
    if e_cat_reader is not None:
        e_cat_reader = {convert[v]: np.array([s.upper() for s in e_cat_reader[v]])
                        for k, v in enumerate(e_cat_reader)}
    return e_cat_reader


if __name__ == "__main__":
    # Code below is an example of use case
    subject = 'R1111M'
    categories = get_elec_cat(subject)
    print(categories)
