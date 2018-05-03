from os import path as op
import time
import numpy as np
from nibabel.freesurfer.io import read_geometry
from six import b

from scipy import sparse
from scipy.sparse import coo_matrix, csr_matrix

import nibabel as nib
import os


class Label(object):
    """Code Written by Jim Kregal for writing annotation labels to the surface of an averaged suubject brain"""

    def __init__(self, tal, pos=None, values=None, hemi=None, comment="",
                 name=None, outdir=None, subject=None, verbose=None,
                 basedir='/data/eeg/freesurfer/subjects/', radius=3.0,
                 stat=None, smooth=0.0, x=None, y=None, z=None):
        if hemi == 'lh':
            self.surf = basedir + 'average/surf/lh.pial'

        elif hemi == 'rh':
            self.surf = basedir + 'average/surf/rh.pial'

        if x is None:
            x = tal['atlases']['avg.dural']['x']

        if y is None:
            y = tal['atlases']['avg.dural']['y']

        if z is None:
            z = tal['atlases']['avg.dural']['z']

        pos = np.vstack((x, y, z)).T

        # need to filter all the electrodes by hemisphere
        if hemi == 'lh':
            stat = stat[pos[:, 0] < 0]
            pos = pos[pos[:, 0] < 0]
        elif hemi == 'rh':
            stat = stat[pos[:, 0] > 0]
            pos = pos[pos[:, 0] > 0]

        # TODO: may want to exclude depth electrodes
        # TODO: may want to have minimum distance to surface

        self.surface = nib.freesurfer.read_geometry(os.path.join(basedir, 'average' + '/surf/' + hemi + '.pial'))
        self.cortex = np.sort(
            nib.freesurfer.read_label(os.path.join(basedir, 'average' + '/label/' + hemi + '.cortex.label')))
        self.sulc = nib.freesurfer.read_morph_data(os.path.join(basedir, 'average' + '/surf/lh.sulc'))

        self.pos = pos
        self.stat = stat

        self.radius = radius
        self.hemi = hemi
        self.comment = comment
        self.verbose = verbose
        self.subject = subject
        self.name = name
        self.outdir = outdir
        self.smooth = smooth / 2.35482

    def map_stat_to_label(self):

        self.coords, faces, vertices = self.get_vertices_from_surf()

        fnames = []
        elec_vertices = np.empty(self.pos.shape[0], dtype=int)

        filename = op.join(self.outdir, self.subject)
        path_head, name = op.split(filename)
        if name.endswith('.label'):
            name = name[:-6]
        if not (name.startswith(self.hemi) or name.endswith(self.hemi)):
            name += '-' + self.hemi
        filename = op.join(path_head, name) + '.label'

        start_time = time.time()

        self.counts = np.full(len(vertices), 0)
        self.vertices = vertices
        self.values = np.full(len(vertices), 0.)

        connectivity = mesh_dist(self.surface[1], self.surface[0])

        for i, elec_coord in enumerate(self.pos):
            elec_vertices[i] = self.get_nearest_vertex(elec_coord)[0]

        elec_dists = sparse.csgraph.dijkstra(connectivity, indices=elec_vertices)

        for i, e_dist in enumerate(elec_dists):
            self.values[np.where(((e_dist < self.radius) & (e_dist > 0.0)) | (vertices == elec_vertices[i]))] += \
                self.stat[i]
            self.counts[np.where(((e_dist < self.radius) & (e_dist > 0.0)) | (vertices == elec_vertices[i]))] += 1

        self.values = self.values / self.counts
        # TODO: may want to remove vertices, pos, values where there is no data to make a smaller label files
        label = write_label(filename, self)
        fnames.append(filename)

        return fnames

    def get_vertices_from_surf(self):
        '''

        :param hemi:
        :return:
        '''

        coords, faces = read_geometry(self.surf)
        vertices = np.arange(coords.shape[0])
        return coords, faces, vertices

    def get_nearest_vertex(self, elec_coord):
        '''

        :return:
        '''

        dist = np.linalg.norm(self.coords - elec_coord, axis=1)
        idx = np.where(dist == dist.min())[0]

        return idx


def write_label(filename, label, verbose=None):
    """Write a FreeSurfer label.
    Parameters
    ----------
    filename : string
        Path to label file to produce.
    label : Label
        The label object to save.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).
    Notes
    -----
    Note that due to file specification limitations, the Label's subject and
    color attributes are not saved to disk.
    See Also
    --------
    write_labels_to_annot
    """

    with open(filename, 'wb') as fid:
        n_vertices = len(label.vertices)
        data = np.zeros((n_vertices, 5), dtype=np.float)
        data[:, 0] = label.vertices
        data[:, 1:4] = label.coords  # self.pos #1e3 *
        data[:, 4] = label.values
        fid.write(b("#%s\n" % label.comment))
        fid.write(b("%d\n" % n_vertices))
        for d in data:
            fid.write(b("%d %f %f %f %f\n" % tuple(d)))
    return label


def average_duplicates(vertices, pos, values):
    result_vertices = np.unique(vertices)
    result_values = np.empty(result_vertices.shape)
    result_pos = np.empty((1, len(result_vertices), 3))

    for i, vertex in enumerate(result_vertices):
        result_values[i] = np.mean(values[vertices == vertex])
        result_pos[0, i, :] = np.mean(pos[0, vertices == vertex, :], axis=0)

    return result_vertices, result_pos, result_values


def gaussian(dist, sig):
    return np.exp(-np.power(dist, 2.) / (2 * np.power(sig, 2.)))


def mesh_edges(tris):
    """Return sparse matrix with edges as an adjacency matrix.
    Parameters
    ----------
    tris : array of shape [n_triangles x 3]
        The triangles.
    Returns
    -------
    edges : sparse matrix
        The adjacency matrix.
    """
    if np.max(tris) > len(np.unique(tris)):
        raise ValueError('Cannot compute connectivity on a selection of '
                         'triangles.')

    npoints = np.max(tris) + 1
    ones_ntris = np.ones(3 * len(tris))

    a, b, c = tris.T
    x = np.concatenate((a, b, c))
    y = np.concatenate((b, c, a))
    edges = coo_matrix((ones_ntris, (x, y)), shape=(npoints, npoints))
    edges = edges.tocsr()
    edges = edges + edges.T
    return edges


def mesh_dist(tris, vert):
    """Compute adjacency matrix weighted by distances.
    It generates an adjacency matrix where the entries are the distances
    between neighboring vertices.
    Parameters
    ----------
    tris : array (n_tris x 3)
        Mesh triangulation
    vert : array (n_vert x 3)
        Vertex locations
    Returns
    -------
    dist_matrix : scipy.sparse.csr_matrix
        Sparse matrix with distances between adjacent vertices
    """
    edges = mesh_edges(tris).tocoo()

    # Euclidean distances between neighboring vertices
    dist = np.linalg.norm(vert[edges.row, :] - vert[edges.col, :], axis=1)
    dist_matrix = csr_matrix((dist, (edges.row, edges.col)), shape=edges.shape)
    return dist_matrix


def Load_Data(savedfilename):
    """
    The purpose of this code is to be able to quickly load any kind of data
    using pickle.
    """
    import pickle

    try:

        with open(savedfilename, 'rb') as handle:
            loaded_data = pickle.load(handle)
            print
            'loaded successfully, fileloaded as as:\nloaded_data'
        return loaded_data
    except:
        import numpy as np
        loaded_data = np.load(savedfilename)
        return loaded_data


if __name__ == '__main__':
    import numpy as np
    from time import time

    s = time()
    import numpy as np

    tal_root = '/Volumes/rhino/home2/loganf/rotation_data/final/R1111M/R1111M'
    tal = Load_Data(tal_root + '_good_tal_structs')
    label_name = 'all_subj_500_1500'  # adds in hemi later
    coords = Load_Data('/Volumes/rhino/home2/loganf/rotation_data/final/tal_and_coords/all_subjs_l')
    tstats = np.array(Load_Data('/Volumes/rhino/home2/loganf/rotation_data/final/tal_and_coords/all_subjs_l_stats'))
    label_dir = '/Volumes/rhino/home2/loganf/labels/region_timing/'
    coords = coords[500:1500]
    tstats = tstats[500:1500]
    X = coords[:, 0]
    Y = coords[:, 1]
    Z = coords[:, 2]
    my_label = Label(tal=tal, pos=None, values=None, hemi='lh', comment="",
                     name=None, outdir=label_dir, subject=label_name, verbose=None,
                     basedir='/Volumes/rhino/data/eeg/freesurfer/subjects/', radius=12.5,
                     stat=tstats, smooth=3.0, x=X, y=Y, z=Z)
    my_label.map_stat_to_label()
    print time() - s
