#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 01:28:12 2019

@author: arpan

@Description: Read the C3D FC7 extracted features, extracted from the finetuned C3D
model on the Highlights videos.

"""

import os
import numpy as np
import json
import bovw_utils as utils


def select_trimmed_feats(c3dFC7FeatsPath, labelsPath, partition_lst, c3dWinSize=16):
    """
    Select the C3D FC7 features corresponding to trimmed videos
    
    Parameters:
    ------
    c3dFC7FeatsPath: str
        complete path to features directory
    labelsPath: str
        path to the trimmed stroke labels (JSON files)
    partition_lst: list of tuples
        video_keys for which to read the trimmed stroke features
    c3dWinSize: int
        w_{c3d} window size with which the features were extracted (>=16)
        
    Returns:
    ------
    features: dict 
        containing the C3D FC7 features.{video_key: np.array(((N-w+1), 1, 4096)),...}
    
    """
    # read all the features into one dictionary
    features = utils.readAllPartitionFeatures(c3dFC7FeatsPath, partition_lst)
    
    trimmed_feats = {}
    strokes_name_id = []
    # Iterate over the videos
    for k in sorted(list(features.keys())):
        # read the JSON trimmed localizations
        with open(os.path.join(labelsPath, k+'.json'), 'r') as fr:
            frame_dict = json.load(fr)
        frame_indx = list(frame_dict.values())[0]
        
        # Iterate over video strokes
        for m,n in frame_indx:
            # select the vectors of size Nx1x4096 and remove mid dimension of 1
            key = k+"_"+str(m)+"_"+str(n)
            trimmed_feats[key] = np.squeeze(features[k][m:(n-c3dWinSize+2),...], axis=1)
            strokes_name_id.append(key)
        
    return trimmed_feats, strokes_name_id

