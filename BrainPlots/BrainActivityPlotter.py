# ---------->Import relevant modules
import numpy as np
from surfer import Brain
from nibabel.freesurfer import read_label
from glob import glob
from time import time
from mayavi import mlab


def pysurf_plot(hemi, epoch_num=1, kind='pial'):
    """
    Freesurfer ships with some probabilistic labels of cytoarchitectonic
    and visual areas. Here we show how to plot labels customly created using
    PysurferSurfaceWrite code created by Jim Kregal

    Notes:
    Make sure that your subject_dir for freesurfer is set up correctly to
    reference OUR subjects not the default freesurfer subjects
    Mine is: SUBJECTS_DIR=/Volumes/rhino/data/eeg/freesurfer/subjects/

    Parameters:
        Input:
            hemi=lh, or rh for hemisphere to plot
            epoch_num= which epoch is it? This is relevant to logan code and might
                not be the best to include in a function that's supposed to be
                generalizible
            kind= what surface to play (pial etc.)
        Outputs:
            A beautiful brain plot

    TO DO: Comment better!!!
    """
    # ---------->Print header
    print(__doc__)

    """
    Note:
    Pysurf can only reliably plot one hemisphere at a time, else data may be
    plotted incorrectly. In order to render plots that are used in figures, you
    should plot each hemisphere seperately.
    """
    s = time()
    hemi = hemi.lower()
    brain = Brain("fsaverage", hemi, kind)

    """Show the morphometry with a continuous grayscale colormap."""
    brain.add_morphometry("curv", colormap="binary", min=-.8, max=.8, colorbar=False)

    """define your labels generated from get_map_stat_to_label"""
    label_path = '/Volumes/rhino/scratch/loganf/burke_sphere_epochs/theta/'
    label_files = glob(label_path + '*epoch{}_*crease-{}.label'.format(epoch_num, hemi))

    # Go through all labels, plot one color for decreases
    for f in label_files:
        # if 'increase' in f:
        # continue
        cmap = "Blues" if 'decrease' in f else "YlOrRd"
        prob_field = np.zeros_like(brain._geo.x)
        ids, probs = read_label(f, read_scalars=True)
        prob_field[ids] = np.nan_to_num(probs)
        try:
            brain.add_data(prob_field, min=0.0, max=.5, thresh=1e-5, colormap=cmap)
        except:
            print(np.unique(prob_field))
            pass

    # Go through all increasing regions, plot another color
    """change the camera to all possible views, then take a png in the folder"""
    set_camera = ['lateral', 'm', 'rostral', 'caudal', 've', 'frontal', 'par', 'dor', 'lateral']
    for views in set_camera:
        brain.show_view(views)
        brain.save_image('/Users/loganfickling/Desktop/test/' + str(views) + str('image00001') + '.png')
    print time() - s  # Time for bench mark!


# -------------->Make things fancy, animate if you want
@mlab.animate(delay=205)
def anim():
    f = mlab.gcf()
    while 1:
        # Roll the camera right to left.
        f.scene.camera.azimuth(10)
        f.scene.render()
        yield
    a = anim()  # Starts the animation.


if __name__ == '__main__':
    # -------------->Define your plot!
    brain = pysurf_plot(hemi='lh', epoch_num=3)
    mlab.show()  # interactive window



    #### In order to convert these images into a movie we can do the following:
    # ffmpeg -framerate 1/5 -i img%03d.png -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4



    # In order to put movies together to the following:
    # ffmpeg   -i input1.mp4   -i input2.mp4   -filter_complex '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]'   -map [vid]   -c:v libx264   -crf 23   -preset veryfast   output.mp4from time import time
import numpy as np
from scipy.spatial.distance import euclidean
from os import path as op
from nibabel.freesurfer.io import read_geometry
from six import b
from scipy import sparse
from scipy.sparse import coo_matrix, csr_matrix
import nibabel as nib
import os, sys

sys.path.append('/home2/loganf/SecondYear/Functions/Logans_EEGToolbox/')
from GetEEG import get_sub_tal


def Save_Data(dataframe, savedfilename):
    """
    The purpose of this code is to be able to quickly save any kind of data
    using pickle.
    """
    import pickle
    try:
        with open((savedfilename), 'wb') as handle:
            pickle.dump(dataframe, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print 'saved successfully, file save as:\n' + savedfilename
    except:
        print 'failed to save'


def get_avg_surface_spheres_ranges(radius=10, step=3):
    basedir = '/data/eeg/freesurfer/subjects/'
    rh = nib.freesurfer.read_geometry(os.path.join(basedir, 'average/surf/rh.pial'))
    lh = nib.freesurfer.read_geometry(os.path.join(basedir, 'average/surf/lh.pial'))
    surface = np.concatenate((lh[0], rh[0]))

    x_min, x_max = np.min(surface[:, 0]), np.max(surface[:, 0])
    x_range = np.arange(x_min + radius, x_max - radius, step)

    y_min, y_max = np.min(surface[:, 1]), np.max(surface[:, 1])
    y_range = np.arange(y_min + radius, y_max - radius, step)

    z_min, z_max = np.min(surface[:, 2]), np.max(surface[:, 2])
    z_range = np.arange(z_min + radius, z_max - radius, step)
    return x_range, y_range, z_range


def get_avg_surface_spheres(subject, radius=10, step=3,
                            save_path='/scratch/loganf/spheres/{}_FR1'):
    s = time()
    # Get tal and coords, use surface to make possible spheres
    m, b, t = get_sub_tal(subject, 'FR1')
    coords = np.vstack((t.avgSurf.x_snap, t.avgSurf.y_snap, t.avgSurf.z_snap)).T
    x_range, y_range, z_range = get_avg_surface_spheres_ranges(radius=radius, step=step)

    # Go through each sphere and find mataches
    subject_sphere_d = {}
    for x in x_range:
        for y in y_range:
            for z in z_range:
                sphere_center = np.array((x, y, z))
                matches = np.array([i for i, c in enumerate(coords)
                                    if euclidean(sphere_center, c) < (radius)])

                if len(matches) > 0:
                    key = (str(x.round(2)) + '_' + str(y.round(2)) + '_' + str(z.round(2)))
                    subject_sphere_d[key] = matches
    print(time() - s)

    Save_Data(subject_sphere_d, save_path.format(subject))
    # return subject_sphere_d
