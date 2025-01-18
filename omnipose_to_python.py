# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 16:25:32 2024

@author: Alexandros Papagiannakis, Christine Jacobs-Wagner lab, HHMI at Stanford University 2024
"""


import os
import scipy as sp
import numpy as np
import time
import Biviriate_medial_axis_estimation as bma
import pickle

class omnipose_to_python_timelapse(object):
    """This class can be used to incorporate the cell masks and lineages returned from Omnipose and SuperSegger to python.
    Omnipose: https://www.nature.com/articles/s41592-022-01639-4
    SuperSegger: https://pubmed.ncbi.nlm.nih.gov/27569113/
    see also: https://www.biorxiv.org/content/10.1101/2024.11.25.625259v1.full
    """
    
    def __init__(self, omni_cell_path, experiment, fluorescent_channels, min_trajectory_length, frame_interval, every_nth, save_path):
        """Initializes the class

        Args:
            omni_cell_path (_str_): the path to the "Cell" folder generated after running omnipose to timelapse images
            experiment (_str_): an ID specific for the experiment
            fluorescent_channels (_list_): A list of the fluorescent channels (e.g., ['rfp','gfp','bfp'])
            min_trajectory_length (_int_): the minimum length of the considered cell trajectories
            frame_interval (_int_): The frame interval in minutes (e.g., 2.5 min)
            every_nth (_list_): the relative frame interval of the phase contrast and fluorescent images (e.g., [1,2,2,7] means phase images every 1 frame, 
            rfp imaging every 3 frames, gfp every 3 frames, and bfp every 7 min)
            save_path (_str_): the path where the cell and lineage variables are stored
        """
        
        start = time.time()
        cropped_masks = {}
        cropped_fluor = {}
        bound_box = {}
        cell_frames = {}
        cell_centroid = {}
        divided_index = {}
        daughter_id = {}
        mother_id = {}
        cell_areas = {}
        
        cell_ids_list = os.listdir(omni_cell_path)
        
        for cl in cell_ids_list:
            
            cell_id = int(cl[4:-4])            
            cell_array = sp.io.loadmat(omni_cell_path+'/'+cl)
            birth_frame = cell_array['birth'][0][0]
            death_frame = cell_array['death'][0][0]
            divided = cell_array['divide'][0][0]
            frame_range = np.arange(birth_frame-1, death_frame)
            
            divided_index[cell_id] = divided
            if cell_array['daughterID'].shape[1]>0:
                daughter_id[cell_id] = cell_array['daughterID'][0]
            else:
                daughter_id[cell_id] = np.array([0,0])
            mother_id[cell_id] = cell_array['motherID'][0]
            
            if frame_range.shape[0] > min_trajectory_length:
                
                cell_frames[cell_id] = frame_range
                cropped_masks[cell_id] = {}
                bound_box[cell_id] = {}
                cell_centroid[cell_id] = {}
                cropped_fluor[cell_id] = {}
                cell_areas[cell_id] = []
                
                # print(cell_id, birth_frame, death_frame, divide_frame)
                # print( daughter_id[cell_id], mother_id[cell_id])
                
                for tm in range(len(cell_array['CellA'][0])):
                    # print(tm)
                    if frame_range.shape[0]>1:
                        cropped_masks[cell_id][frame_range[tm]] = sp.ndimage.binary_fill_holes(cell_array['CellA'][0][tm][0][0][3]).astype(int)
                        cell_areas[cell_id].append(np.nonzero(cell_array['CellA'][0][tm][0][0][3])[0].shape[0])
                        cropped_fluor[cell_id][frame_range[tm]] = {}
                        ch_index = 11
                        for ch in fluorescent_channels:
                            cropped_fluor[cell_id][frame_range[tm]][ch]= cell_array['CellA'][0][tm][0][0][ch_index]
                            ch_index+=3
                        cropped_fluor[cell_id][frame_range[tm]]['Phase']= cell_array['CellA'][0][tm][0][0][7]
                        bound_box[cell_id][frame_range[tm]]=cell_array['CellA'][0][tm][0][0][5]
                        cell_centroid[cell_id][frame_range[tm]]=cell_array['CellA'][0][tm][0][0][8][0][0][8][0]

        end = time.time() 
        print(round(end-start,1), 'seconds to load the cropped images and masks')
        
        self.cropped_masks = cropped_masks
        self.cropped_fluor = cropped_fluor
        self.bound_box = bound_box
        self.cell_frames = cell_frames
        self.cell_centroid = cell_centroid
        self.divided_index = divided_index
        self.daughter_id = daughter_id
        self.mother_id = mother_id
        self.cell_areas = cell_areas
        self.save_path = save_path
        self.experiment = experiment
        self.fluorescent_channels = fluorescent_channels+['Phase']
        self.interval = frame_interval
        self.channel_frequency = every_nth
                    
        
    def get_cell_out_of_boundaries(self, limits):
        """Returns the cell IDs that are located outside the specified limits

        Args:
            limits (_tuple_): tuple of lower and upper limit (e.g., (40,2008) for a 2048x2048 image)

        Returns:
            _list_: list of cell IDs outside the specified boundaries
        """
        out_of_bound = []
        
        for cl in self.cell_centroid:
            for tm in self.cell_centroid[cl]:
                if int(self.cell_centroid[cl][tm][0]) not in range(limits[0],limits[1]+1) or int(self.cell_centroid[cl][tm][1]) not in range(limits[0],limits[1]+1):
                    out_of_bound.append(cl)
    
        return list(np.unique(out_of_bound))
    
    
    def get_mothers_without_daughters(self):
        """Get a list of the mother cells that do not divide during the course of the experiment

        Returns:
            _list_: list of mother cell IDs
        """
        mothers_list = []
        
        for cl_id in self.mother_id:
            if self.mother_id[cl_id][0]==0:
                mothers_list.append(cl_id)
        
        return mothers_list
    
    
    def get_medial_axes(self, bad_cells, verb=False):
        """Get the medial axes of the segmented masks using the Biviriate_medial_axis_estimation functions.

        Args:
            bad_cells (_list_): list of bad cell IDs (these cells will be excluded from the medial axis estimation)
            verb (bool, optional): Determines if the medial axis will be plotted for each cell. Defaults to False.
        """
        medial_axis_dict = {}
        
        for cl in self.cropped_masks:
            if cl not in bad_cells:
                medial_axis_dict[cl] = {}
                for tm in self.cropped_masks[cl]:
                    if np.mean(self.cropped_fluor[cl][tm]['hu']) != 0:
                        print(cl)
                        medial_axis_dict[cl][tm] = bma.get_medial_axis(self.cropped_masks[cl][tm], radius_px=8, half_angle=22, cap_knot=13, max_degree=60, verbose=verb)
                    else:
                        print('no fluorescence image taken for cell',cl,'at time point',tm)
        with open(self.save_path+'/'+self.experiment+'_medial_axis_dict', 'wb') as handle:
            pickle.dump(medial_axis_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    
    def locate_cell_id(self, cell_position, frame, radius):
        """Returns the cell ID of a specific image location

        Args:
            cell_position (_tuple_): (x,y) coordinates 
            frame (_int_): imaging frame
            radius (_int_): radius within which the cell ID is searched for

        Returns:
            _str_: the located cell ID or 'none' if no cell ID is found
        """
        centroid_distance = radius
        cell_id_found = 'none'
        
        for cl in self.cell_centroid:
            if frame in self.cell_centroid[cl]:
                centroid_coords = self.cell_centroid[cl][frame]
                distance = np.sqrt((cell_position[0]-centroid_coords[0])**2 + (cell_position[1]-centroid_coords[1])**2)
                if distance < centroid_distance:
                    print(cl)
                    cell_id_found = cl
                    centroid_distance = distance
        
        return cell_id_found
    
    
    def get_lineage_mother(self, single_cell_id):
        """Get the mother cell ID of a lineage

        Args:
            single_cell_id (_str_): the ID of a single cell within the lineage

        Returns:
            _str_: the mother cell ID
        """
        while self.mother_id[single_cell_id][0]>0:
            single_cell_id = self.mother_id[single_cell_id][0]
            print(single_cell_id)
        
        return single_cell_id
    
    
    def get_oned_fluorescence(self, single_cell_id):
        """Project the fluorescent pixels onto the medial axis and along the cell length.

        Args:
            single_cell_id (_str_): single cell ID

        Returns:
            _dict_: a dictionary that includes the timepoints as keys and the Pandas dataframes with the relative pixel coordinates and fluorescence values as values
            _dict_: a dictionary that includes the timepoints as keys and the cell length (px) as values
        """
        oned_coords_dict = {}
        cell_length_dict = {}
        
        if os.path.isdir(self.save_path+'/'+self.experiment+'_medial_axis_dict'):
            print('Loading the medial axis dataframes...')
            with open(self.save_path+'/'+self.experiment+'_medial_axis_dict', 'rb') as handle:
                medial_axis_dict = pickle.load(handle)
        else:
            print('Drawing the medial axes...')
            medial_axis_dict = {}
            for tm in self.cropped_masks[single_cell_id]:
                if np.mean(self.cropped_fluor[single_cell_id][tm]['hu']) != 0:
                    cell_mask = self.cropped_masks[single_cell_id][tm]
                    
                    medial_axis_dict[tm] = bma.get_medial_axis(cell_mask, 
                                                               radius_px=8, half_angle=22, cap_knot=13, 
                                                               max_degree=60, verbose=True)
                    
                    medial_axis_df = medial_axis_dict[tm][0]
                    
                    cell_mask_df = bma.get_oned_coordinates(cell_mask, medial_axis_df, half_window=5)
                    print('Applying 1D fluorescence...')
                    for ch in self.fluorescent_channels:
                        crop_signal_image = self.cropped_fluor[single_cell_id][tm][ch]
                        cell_mask_df[ch+'_fluor'] = crop_signal_image[np.nonzero(cell_mask)]
                    oned_coords_dict[tm] = cell_mask_df
                    cell_length_dict[tm] = medial_axis_df.arch_length_centered.max()*2
        
        return oned_coords_dict, cell_length_dict
    
    

        



