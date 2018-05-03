#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 18:14:18 2016

@author: ethansolomon Created on Mon Jan  4 18:14:18 2016
-Author of original version of the script

@author: loganfickling Created on Tue Apr 11 09:42:00 2017
-Modification of the script into a class for replication of Burke et al. 2014

@author: noraherweg
- Created script for reading good/bad coordinate textfiles

Parameters:

    Inputs:
        subject: a string ID of a subject, e.g. 'R1111M'
        tal: a talirach structure generated from ptsa
        hemi: hemispheres to show. Can be 'lh', 'rh', or 'both'
        outdir: directory to save images to
        bg: background color. Can be 'white', 'black' or 'grey'
        stat: statistics to associate with a coordinate, must be a np.array
              of len(tal)
        animate: True if you want the brain to animate, else False
        trans: True if you want the brain to be transparent, recommended
               for plotting electrodes to see depth beneath surface.
    Output:
        Averaged surface plot of the brain

     Optional parameters:
        plot_coords(): plots a single subjects coordinates
        #Need to change so compatible with multiple subjects

        outline_plane(): Outlines a plane of the brain
        #Need to change so that it's rendered in a different scene that's attached

"""
import os
from mayavi.core.api import Engine
from mayavi.sources.vtk_file_reader import VTKFileReader
from mayavi.modules.surface import Surface
from mayavi import mlab
import numpy as np
from numpy import array
from ptsa.data.readers.TalReader import TalReader


class AvgBrainPlot(object):
    def __init__(self, subject=None, tal=None, hemi=None, outdir=None, bg=None,
                 stat=None, x=None, y=None, z=None, animate=False, trans=False,
                 hollow=False, coords=None):
        self.subject = subject
        self.tal = tal
        # if (self.tal is None) and (self.subject is not None):
        # self.get_good_coords()
        self.coords = coords
        self.stat = stat
        if hemi == 'lh':
            self.left_hemi = 'lh.vtk'
            self.right_hemi = None
        elif hemi == 'rh':
            self.left_hemi = None
            self.right_hemi = 'rh.vtk'
        elif hemi == 'both':
            self.left_hemi = 'lh.vtk'
            self.right_hemi = 'rh.vtk'

        # Create the Mayavi engine and start it.
        self.engine = Engine()
        self.engine.start()
        self.scene = self.engine.new_scene()

        # Read in the left hemisphere VTK file and add it
        if (hemi == 'lh') or (hemi == 'both'):
            surface_left = Surface()
            reader_left = VTKFileReader()
            reader_left.initialize(self.left_hemi)
            self.engine.add_source(reader_left)
            self.engine.add_filter(surface_left, reader_left)

            # Read in the right hemisphere VTK file and add it
        if (hemi == 'rh') or (hemi == 'both'):
            surface_right = Surface()
            reader_right = VTKFileReader()
            reader_right.initialize(self.right_hemi)
            self.engine.add_source(reader_right)
            self.engine.add_filter(surface_right, reader_right)

            # Make things look pretty.
        self.scene.scene.camera.elevation(-70)
        self.scene.scene.show_axes = True
        bg_color = {'black': (0, 0, 0),
                    'white': (1, 1, 1),
                    'grey': (.5, .5, .5)}
        self.bg = bg
        if self.bg == None:  ##### Says self has no attribute bg but it does :(
            self.bg = 'black'
        self.scene.scene.background = (bg_color[self.bg])
        self.scene.scene.x_minus_view()  # left hemi lateral view

        # Lighting parameters
        self.scene.scene.light_manager.light_mode = 'vtk'
        camera_light = self.engine.scenes[0].scene.light_manager.lights[0]
        camera_light.elevation, camera_light.azimuth = 0.0, 0.0

        # Optional parameters
        self.trans = trans
        if self.trans == True:
            self.Transparent()
        self.animate = animate
        if self.animate == True:
            a = self.anim()  # Starts the animation.
        self.hollow = hollow
        if self.hollow == True:
            self.Hollow()

    def get_good_coords(self):
        # This looks wonky so it can easily be changed to run multiple subjects
        # or a single subject as below
        talPath = []
        # Go through subjects, or here one subject and get tal paths
        if isinstance(self.subject, str):
            subjects = self.subject

        for i, j in enumerate(subjects):
            if isinstance(self.subject, str):
                subject = subjects
            tal_path = os.path.join('/Volumes/rhino/data/eeg/', subject, 'tal',
                                    subject + '_talLocs_database_bipol.mat')
            talPath.append(tal_path)

        monopolar_channels = [tal_readeri.get_monopolar_channels() for tal_readeri in
                              [TalReader(filename=talPathi) for talPathi in talPath]]
        bipolar_pairs = [tal_readeri.get_bipolar_pairs() for tal_readeri in
                         [TalReader(filename=talPathi) for talPathi in talPath]]

        # Exclude bad channels
        bpPairs = []
        mpChans = []
        for talPathi, bipolar_pairsi, monopolar_channelsi in zip(
                talPath, bipolar_pairs, monopolar_channels):
            gf = open(os.path.dirname(talPathi) + '/good_leads.txt', 'r')
            goodleads = gf.read().split('\n')
            gf.close
            if os.path.isfile(os.path.dirname(talPathi) + '/bad_leads.txt'):
                bf = open(os.path.dirname(talPathi) + '/bad_leads.txt', 'r')
                badleads = bf.read().split('\n')
                bf.close
            else:
                badleads = []
            subPairs = np.array([pairi for pairi in bipolar_pairsi
                                 if (pairi[0].lstrip('0') in goodleads)
                                 and (pairi[1].lstrip('0') in goodleads)
                                 and (pairi[0].lstrip('0') not in badleads)
                                 and (pairi[1].lstrip('0') not in badleads)])
            subPairs = np.rec.array(subPairs)
            subChans = np.array([chani for chani in monopolar_channelsi
                                 if (chani.lstrip('0') in goodleads)
                                 and (chani.lstrip('0') not in badleads)])
            bpPairs.append(subPairs)
            mpChans.append(subChans)
        if self.tal == None:
            tal_reader = TalReader(filename=tal_path)
            tal_structs = tal_reader.read()
        else:
            tal_structs = self.tal
        if isinstance(self.subject, str):
            bp = bpPairs[0]

        """Get only good pairs; int conversion necessary to convert [001,002]
        into [1,2] for comparasion match"""
        good_pairs = np.array([np.array([(int(x)) for i, x in enumerate(jj)]
                                        ) for ii, jj in enumerate(bp)])

        # Get indx of good tals
        channels_indx = []
        for i, ch in enumerate(tal_structs.channel):
            for x, pair in enumerate(good_pairs):
                if (pair[0] in ch) & (pair[1] in ch):
                    channels_indx.append(i)

        # Get good tals from all tals using index
        good_tal = np.array([tal for indx, tal in enumerate(tal_structs) if indx
                             in channels_indx]).view(np.recarray)
        # Get coords
        # Edit to only show frontal area
        tal = np.array([tal for tal in good_tal]).view(np.recarray)
        # if tal.Loc2=='Frontal Lobe' and 'L' in tal.Loc1]).view(np.recarray)

        if len(good_tal) == len(bp):
            self.tal = good_tal
            if (('avgSurf' in tal.dtype.names) == True) and (
                        'x_snap' in tal.avgSurf.dtype.names) == True:
                self.x = tal.avgSurf.x_snap
                self.y = tal.avgSurf.y_snap
                self.z = tal.avgSurf.z_snap
            else:
                if ('x' in tal.dtype.names) == True:
                    print 'no avg tals sadface'
                    self.x, self.y, self.z = tal.x, tal.y, tal.z

    def Transparent(self):
        # Left hemi parameters
        left = self.engine.scenes[0].children[0].children[0].children[0]
        left.actor.property.opacity = .2

        # Right hemi paramters
        try:  # if hemi=both
            right = self.engine.scenes[0].children[1].children[0].children[0]
            right.actor.property.opacity = .2
        except:  # if hemi=rh
            if self.hemi != 'lh':
                right = self.engine.scenes[0].children[0].children[0].children[0]
                left.actor.property.opacity = .2

    def Hollow(self):
        left = self.engine.scenes[0].children[0].children[0].children[0]
        left.actor.property.frontface_culling = True
        left.actor.property.representation = 'wireframe'
        left.actor.property.line_width = 3.0

        try:  # if hemi=both
            right = self.engine.scenes[0].children[1].children[0].children[0]
            right.actor.property.frontface_culling = True
            right.actor.property.representation = 'wireframe'
            right.actor.property.line_width = 3.0
        except:
            right = self.engine.scenes[0].children[0].children[0].children[0]
            right.actor.property.frontface_culling = True
            right.actor.property.representation = 'wireframe'
            right.actor.property.line_width = 3.0

    @mlab.animate(delay=205)
    def anim(self):
        f = mlab.gcf()
        while 1:
            f.scene.camera.azimuth(10)  # rotate right left axis
            f.scene.render()
            yield

    def plot_coords(self, coords=None, cmap=None):
        if self.coords == None:
            try:
                self.get_good_coords()
            except:
                try:
                    self.get_all_coords()
                except:
                    self.x = self.coords[:, 0]
                    self.y = self.coords[:, 1]
                    self.z = self.coords[:, 2]
            # if failed == False:
            if cmap == None:
                cmap = "spectral"
            coords = mlab.points3d(
                self.x, self.y, self.z,
                scale_factor=2, scale_mode='none',
                opacity=1, resolution=40, colormap=cmap,
                extent=[min(self.x), max(self.x),
                        min(self.y), max(self.y),
                        min(self.z), max(self.z)])
            # Set colors from data
            try:
                coords.mlab_source.dataset.point_data.scalars = self.stat
            except:
                stats = np.random.rand(len(self.x))
                coords.mlab_source.dataset.point_data.scalars = stats

        else:
            self.x = self.coords[:, 0]
            self.y = self.coords[:, 1]
            self.z = self.coords[:, 2]
            if cmap == None:
                cmap = "spectral"
            coords = mlab.points3d(
                self.x, self.y, self.z, colormap=cmap,
                scale_factor=2, scale_mode='none',
                opacity=1, resolution=40,
                extent=[min(self.x), max(self.x),
                        min(self.y), max(self.y),
                        min(self.z), max(self.z)])

            try:
                coords.mlab_source.dataset.point_data.scalars = self.stat
            except:
                stats = np.random.rand(len(self.x))
                coords.mlab_source.dataset.point_data.scalars = stats

    def Plot_coords(self, cmap=None):
        try:
            self.get_good_coords()
        except:
            try:
                self.get_all_coords()
            except:
                self.x = self.coords[:, 0]
                self.y = self.coords[:, 1]
                self.z = self.coords[:, 2]
        # if failed == False:
        if cmap == None:
            cmap = "spectral"
        coords = mlab.points3d(
            self.x, self.y, self.z,
            scale_factor=2, scale_mode='none',
            colormap=cmap, opacity=1, resolution=40,
            extent=[min(self.x), max(self.x),
                    min(self.y), max(self.y),
                    min(self.z), max(self.z)])
        # Set colors from data
        try:
            coords.mlab_source.dataset.point_data.scalars = self.stats
        except:
            stats = np.random.rand(len(self.x))
            coords.mlab_source.dataset.point_data.scalars = stats

            # Set labels to coords
            # for i, x in enumerate(self.x):
            # mlab.text3d(self.x[i], self.y[i], self.z[i], str(self.tal.tagName[i]), scale=(2, 2, 2),
            # color=(0, 0, 0))
            a = self.anim()  # Starts the animation.

    def outline_plane(self):
        # Outline lh x axis
        from mayavi.filters.decimatepro import DecimatePro
        from mayavi.modules.vector_cut_plane import VectorCutPlane
        decimate_pro_lh = DecimatePro()
        lh_file_reader = self.engine.scenes[0].children[0]
        self.engine.add_filter(decimate_pro_lh, lh_file_reader)
        cut_plane_lh = VectorCutPlane()
        self.engine.add_filter(cut_plane_lh, decimate_pro_lh)
        # Make plane
        cut_plane_lh.implicit_plane.widget.origin = array(
            [-33.8, -18.4, 15.5])
        cut_plane_lh.glyph.glyph.range = array([0., 1.])
        cut_plane_lh.implicit_plane.widget.normal = array([1., 0., 0.])
        # Set to an axis
        cut_plane_lh.implicit_plane.widget.normal_to_x_axis = True
        # Color prop
        cut_plane_lh.actor.property.specular_color = (0., 0., 0.)
        cut_plane_lh.actor.property.diffuse_color = (0., 0., 0.)
        cut_plane_lh.actor.property.ambient_color = (0., 0., 0.)
        cut_plane_lh.actor.property.color = (0., 0., 0.)


if __name__ == '__main__':
    def Load_Data(savedfilename):
        """
        The purpose of this code is to be able to quickly load any kind of data
        using pickle.
        """
        import pickle

        try:

            with open(savedfilename, 'rb') as handle:
                loaded_data = pickle.load(handle)
                print 'loaded successfully, fileloaded as as:\nloaded_data'
            return loaded_data
        except:
            import numpy as np
            loaded_data = np.load(savedfilename)
            return loaded_data


    # my_coords=np.load('/Volumes/rhino/home2/loganf/SecondYear/all_coords.npy')
    # coords = Coords[:500]
    testbrain = AvgBrainPlot(subject='R1111M', hemi='both', animate=True, bg='white',
                             trans=True).Plot_coords()
    # interactive window
    mlab.show()
from time import time
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
