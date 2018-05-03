"""IndependentTtest.py, author=LoganF
A script used to serve as a wrapper around scipy.stats.ttest_ind, allowing for it to be used in conjunction with
TimeSeriesX objects without forcing the user to discard the TimeSeriesX
"""
from ptsa.data import TimeSeriesX
import numpy as np
import pandas as pd
import sys, os

sys.path.append('/home2/loganf/SecondYear/Functions/Logans_EEGToolbox/')
from GetTalirach import get_sub_tal
from scipy.stats import ttest_ind


def seperate_rec_baseline(ts):
    rec = ts.sel(events=ts.events.data['type'] == 'REC_WORD')
    bl = ts.sel(events=ts.events.data['type'] == 'REC_BASE')
    return rec, bl


def _within_subj_ttest(ts, dim='events'):
    # Allows for someone to enter in dims instead of axis
    dims_to_axis = {dim: k for k, dim in enumerate(ts.dims)}
    axis = dims_to_axis[dim]

    # Seperate recalled from deliberation periods
    rec, delib = seperate_rec_baseline(ts)

    # Unpaired t-test
    t, p = ttest_ind(rec, delib, axis=axis)
    ts = rec.mean(dim=dim)  # Just so there's a shape match
    ts.data = t  # Reset the data
    ts.name = 't-statistics'  # Just so we know it was done
    return ts


def regions_included(names, minimum_number=3):
    # Create a dataframe
    df = pd.DataFrame(pd.Series(names).value_counts())
    # Replace with nan for any roi with less than three bp
    drop = df[df >= minimum_number]

    # Check if they have a hippocampus
    # Replace the nan with the number since any hpc counts
    if 'L Hippocampus' in df.index.values:
        drop[drop.index == 'L Hippocampus'] = df[df.index == 'L Hippocampus']
    if 'R Hippocampus' in df.index.values:
        drop[drop.index == 'R Hippocampus'] = df[df.index == 'R Hippocampus']

    # Drop excluded pairs
    roi_selection = drop.dropna().index.values

    return roi_selection


def get_valid_rois_names(tstats, subject, experiment):
    m, b, tals = get_sub_tal(subject, experiment,
                             exclude_bad=True, bipolar=True)

    bp = np.concatenate(tstats.bipolar_pairs.data.flatten().tolist())
    bp = np.array([int(x) for x in bp])
    bp = bp.reshape(bp.shape[0] / 2, 2)

    names = []
    for i, tal in enumerate(tals):
        if tal.Loc5 != 'Hippocampus':
            # Gyrus
            # name=str(j.Loc1[0]) +' ' + str(j.Loc3)
            # Lobe
            name = str(tal.Loc1[0]) + ' ' + str(tal.Loc2)

        if tal.Loc5 == 'Hippocampus':
            name = str(tal.Loc1[0]) + ' ' + str(tal.Loc5)

        if tal.channel in bp:
            names.append(name)

    names = np.array(names)

    return names


def within_subject_ttest(ts, dim='events', minimum_number=5):
    """minimum number corresponds to minimum number of bp per region subj must have"""
    # FR1, catfr1
    if 'exp_version' in ts['events'].dtype.names:
        # Use eegfile to generate subject and experiment values
        clever_split = ts['events'].data['eegfile'][0].split('/')[-1].split('_')
        subject, experiment = clever_split[0], clever_split[1]
    else:  # pyFR
        subject = ts['events'].data['subject'][0]
        experiment = 'pyFR'

    # Do a ttest within the subject
    tstats = _within_subj_ttest(ts, 'events')

    # Get corresponding lobe-names for each bp pair, reset bipolar_pairs to this
    names = get_valid_rois_names(tstats, subject, experiment)
    tstats['bipolar_pairs'] = names

    # Average across ALL regions
    tstats = tstats.groupby('bipolar_pairs').mean(dim='bipolar_pairs')

    # Figure out which regions do not meet our criteria for inclusion
    valid_rois = regions_included(names, minimum_number)

    # Return only the regions that are valid
    tstats = tstats.sel(bipolar_pairs=valid_rois)

    return tstats