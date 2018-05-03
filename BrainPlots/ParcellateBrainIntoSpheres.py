from time import time
import numpy as np
from scipy.spatial.distance import euclidean
from os import path as op
from nibabel.freesurfer.io import read_geometry
from six import b
from scipy import sparse
from scipy.sparse import coo_matrix, csr_matrix
import nibabel as nib
from itertools import product
import os, sys

sys.path.append('/home2/loganf/SecondYear/Functions/CML/')
from GetData import get_sub_tal


def Save_Data(dataframe, savedfilename):
    """
    The purpose of this code is to be able to quickly save any kind of data
    using pickle.
    """
    import pickle
    try:
        with open((savedfilename), 'wb') as handle:
            pickle.dump(dataframe, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('saved successfully, file save as:\n' + savedfilename)
    except:
        print('failed to save')


def get_avg_surface_spheres_ranges(radius=10, step=3):
    """ Generates a sphere w/ radius radius in mm stepped every step in mm using average subject brain

    Parameters
    ------INPUTS------
    radius: int, by default 10, the radius in mm of the average surface spheres to generate
    step: int, by default 3, the step in mm of the average surface spheres to generate
    ------OUTPUTS-----
    x_range, y_range, z_range: np.array, range of valid values with discrete steps.
    """
    # Format path for local mounted vs local access
    basedir = '/data/eeg/freesurfer/subjects/'
    if not os.path.exists(basedir):
        basedir = os.path.join('/Volumes/rhino', basedir)

        # Get cortical surface of average patient right and left hemi
    rh = nib.freesurfer.read_geometry(os.path.join(basedir, 'average/surf/rh.pial'))
    lh = nib.freesurfer.read_geometry(os.path.join(basedir, 'average/surf/lh.pial'))
    surface = np.concatenate((lh[0], rh[0]))

    # Get minimum and maximum x value, use parameters to make range
    x_min, x_max = np.min(surface[:, 0]), np.max(surface[:, 0])
    x_range = np.arange(x_min + radius, x_max - radius, step)

    # Get minimum and maximum y value, use parameters to make range
    y_min, y_max = np.min(surface[:, 1]), np.max(surface[:, 1])
    y_range = np.arange(y_min + radius, y_max - radius, step)

    # Get minimum and maximum z value, use parameters to make range
    z_min, z_max = np.min(surface[:, 2]), np.max(surface[:, 2])
    z_range = np.arange(z_min + radius, z_max - radius, step)

    return x_range, y_range, z_range


def get_avg_surface_spheres(subject, radius=10, step=3, save=False,
                            save_path='/scratch/loganf/spheres/{}_FR1'):
    """Returns a dictionary where each key is str(x)_str(y)_str(z) in average subject space
    and values are corresponding bipolar channel names (['061', '062', etc.]) within radius mm
    of the atlas brain

    Parameters
    ------INPUTS------
    subject: str, subject_id,
    radius: int, by default 10, the radius in mm of the average surface spheres to generate
    step: int, by default 3, the step in mm of the average surface spheres to generate
    save: bool, by default False, whether or not to save
    save_path: str, output directory to save in
    ------OUTPUTS------
    subject_sphere_d: dictionary of matches
    """
    # Load subject brain atlas and get electrode coordinates
    bp_path = '/scratch/loganf/electrode_information/{}/bp.npy'.format(subject)
    atlas_path = '/scratch/loganf/subject_brain_atlas/{}.npy'.format(subject)
    if os.path.exists(atlas_path):
        atlas = np.load(atlas_path)
        bp = np.load(bp_path)
    coords = np.concatenate([np.vstack((avg['x'], avg['y'], avg['z'])).T
                             for avg in atlas['atlases']['avg.dural']])

    # x,y,z boundries of avg brain stepped every step with a spherical radius of radius
    x_range, y_range, z_range = get_avg_surface_spheres_ranges(radius=radius, step=step)

    subject_sphere_d = {}  # Add data here

    # Use itertools to make essentially a nested for lop of them
    for sphere_center in product(x_range, y_range, z_range):

        # Find distance of each center to all coords
        dist = np.linalg.norm(coords - np.array(sphere_center), axis=1)

        # Find where it's closer than the defined radius
        matches = np.where(dist < radius)

        # If there are any matches add it to the dictionary
        if len(matches[0]) > 0:
            x, y, z = sphere_center
            # Format key as a union of x_y_x coordinates
            key = (str(x.round(2)) + '_' + str(y.round(2)) + '_' + str(z.round(2)))
            subject_sphere_d[key] = bp[matches]
    if save:
        Save_Data(subject_sphere_d, save_path.format(subject))
    return subject_sphere_d