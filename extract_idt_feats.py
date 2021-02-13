#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 16:07:04 2020

@author: arpan

@Description: Form the dictionary of IDT features using the extracted *.out files 
of strokes (extraction done using idt_strokes.sh script)
"""

import os
import numpy as np
import pandas as pd
import json

def read_pooled_partition_feats(vidsPath, labelsPath, idtFeatsPath, partition_lst, 
                                traj_len=15):
    """
    Function to iterate on all the strokes and corresponding IDT features of strokes
    and form a dictionary for the same.
    vidsPath: str
        path to the dataset containing the videos
    labelsPath: str
        path to the JSON files for the labels.
    idtFeatsPath: str
        path to the *.out files extracted using iDT binary
    partition_lst: list of video_ids
        video_ids are the filenames (without extension)
        
    Returns:
    --------
    dict of pooled stroke features (O(NFrames) x FeatSize) and list of stroke keys 
    """
    strokes_name_id = []
    all_feats = {}
    for i, v_file in enumerate(partition_lst):
        print('-'*60)
        print(str(i+1)+". v_file :: ", v_file)
        json_file = v_file + '.json'
        #print("json file :: ", json_file)
        # read labels from JSON file
        assert os.path.exists(os.path.join(labelsPath, json_file)), "{} doesn't exist!".format(json_file)
            
        with open(os.path.join(labelsPath, json_file), 'r') as fr:
            frame_dict = json.load(fr)
        frame_indx = list(frame_dict.values())[0]
        for m,n in frame_indx:
            k = v_file+"_"+str(m)+"_"+str(n)
            if n-m < traj_len:
                print("Skipping : {}".format(k+".out"))
                break
#            print("Stroke {} - {} : {}".format(m,n, n-m))
            strokes_name_id.append(k)
            # Extract the stroke features
            feat_name = k+".out"
            all_feats[k] = read_stroke_feats(os.path.join(idtFeatsPath, feat_name),  traj_len)
            # pool the features for common frame somehow and represent as a global feature
#        break
    return all_feats, strokes_name_id

def read_partition_feats(vidsPath, labelsPath, idtFeatsPath, partition_lst, traj_len=15):
    """
    Function to iterate on all the strokes and corresponding IDT features of strokes
    and form a dictionary for the same.
    vidsPath: str
        path to the dataset containing the videos
    labelsPath: str
        path to the JSON files for the labels.
    idtFeatsPath: str
        path to the *.out files extracted using iDT binary
    partition_lst: list of video_ids
        video_ids are the filenames (without extension)
        
    Returns:
    --------
    numpy array of size (nStrokes x nbins)
    """
    strokes_name_id = []
    all_feats = {}
    for i, v_file in enumerate(partition_lst):
        print('-'*60)
        print(str(i+1)+". v_file :: ", v_file)
        json_file = v_file + '.json'
        #print("json file :: ", json_file)
        # read labels from JSON file
        assert os.path.exists(os.path.join(labelsPath, json_file)), "{} doesn't exist!".format(json_file)
            
        with open(os.path.join(labelsPath, json_file), 'r') as fr:
            frame_dict = json.load(fr)
        frame_indx = list(frame_dict.values())[0]
        for m,n in frame_indx:
            k = v_file+"_"+str(m)+"_"+str(n)
            if n-m < traj_len:
                print("Skipping : {}".format(k+".out"))
                break
#            print("Stroke {} - {} : {}".format(m,n, n-m))
            strokes_name_id.append(k)
            # Extract the stroke features
            feat_name = k+".out"
            all_feats[k] = read_stroke_feats(os.path.join(idtFeatsPath, feat_name),  traj_len)
#        break
    return all_feats, strokes_name_id


def read_stroke_feats(featPath, traj_len):
    """Read the IDT features from the disk, save it in pandas dataframe and return
    the labeled dataframe with last column removed and associated column names.
    Parameters:
    -----------
    featPath : str
        full path to a strokes' feature file (*.out file) extracted using the binary.
    traj_len : int
        the trajectory length used at the time of extraction.
        
    Returns:
    --------
    pd.DataFrame with stroke features and column names.
    """
    assert os.path.isfile(featPath), "File not found {}".format(featPath)
    traj_info = ['frameNum', 'mean_x', 'mean_y', 'var_x', 'var_y', 'length', 'scale', 'x_pos',
                 'y_pos', 't_pos']
    Trajectory = ['Traj_'+str(i) for i in range(2 * traj_len)]
    HOG = ['HOG_'+str(i) for i in range(96)]
    HOF = ['HOF_'+str(i) for i in range(108)]
    MBHx = ['MBHx_'+str(i) for i in range(96)]
    MBHy = ['MBHy_'+str(i) for i in range(96)]
    
    traj_vals = Trajectory + HOG + HOF + MBHx + MBHy
    col_names = traj_info + traj_vals
    
    df = pd.read_csv(featPath, sep='\t', header=None)
    
    # Drop last column having NaN values
    df = df.iloc[:, :-1]
    assert len(col_names) == df.shape[1], "Columns mismatch."
    
    df.columns = col_names
    # drop first column which has frameNums
    df = df.iloc[:, 1:]
    return df

def read_hof_feats(featPath, traj_len):
    """Read the IDT features from the disk, save it in pandas dataframe and return
    the labeled dataframe with last column removed and associated column names.
    Parameters:
    -----------
    featPath : str
        full path to a strokes' feature file (*.out file) extracted using the binary.
    traj_len : int
        the trajectory length used at the time of extraction.
        
    Returns:
    --------
    pd.DataFrame with stroke features and column names.
    """
    assert os.path.isfile(featPath), "File not found {}".format(featPath)
    traj_info = ['frameNum', 'mean_x', 'mean_y', 'var_x', 'var_y', 'length', 'scale', 'x_pos',
                 'y_pos', 't_pos']
    Trajectory = ['Traj_'+str(i) for i in range(2 * traj_len)]
    HOG = ['HOG_'+str(i) for i in range(96)]
    HOF = ['HOF_'+str(i) for i in range(108)]
    MBHx = ['MBHx_'+str(i) for i in range(96)]
    MBHy = ['MBHy_'+str(i) for i in range(96)]
    
    traj_vals = Trajectory + HOG + HOF + MBHx + MBHy
    col_names = traj_info + traj_vals
    
    df = pd.read_csv(featPath, sep='\t', header=None)
    
    # Drop last column having NaN values
    df = df.iloc[:, :-1]
    assert len(col_names) == df.shape[1], "Columns mismatch."
    
    df.columns = col_names
    # drop first column which has frameNums
    df = df.iloc[:, 1:]
    df = df.loc[:, HOF]
    return df