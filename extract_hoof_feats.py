#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 02:57:52 2019

@author: arpan

@Description: Extract HOOF features from all files
"""

import os
import cv2
import sys
import json
import numpy as np


def extract_stroke_feats(vidsPath, labelsPath, partition_lst, nbins, mag_thresh=2, \
                         density=False, grid_size=30):
    """
    Function to iterate on all the training videos and extract the relevant features.
    vidsPath: str
        path to the dataset containing the videos
    labelsPath: str
        path to the JSON files for the labels.
    partition_lst: list of video_ids
        video_ids are the filenames (without extension)
    mag_thresh: float
        pixels with >mag_thresh will be considered significant and used for clustering
    nbins: int
        No. of bins in which the angles have to be divided.
    """
    
    strokes_name_id = []
    all_feats = {}
    bins = np.linspace(0, 2*np.pi, (nbins+1))
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
            #print("Stroke {} - {}".format(m,n))
            strokes_name_id.append(k)
            # Extract the stroke features
            all_feats[k] = extract_flow_angles(os.path.join(vidsPath, v_file+".avi"), \
                     m, n, bins, mag_thresh, density)
#            all_feats[k] = extract_flow_grid(os.path.join(vidsPath, v_file+".avi"), m, n, grid_size)
        #break
    return all_feats, strokes_name_id


def extract_flow_angles(vidFile, start, end, hist_bins, mag_thresh, density=False):
    '''
    Extract optical flow maps from video vidFile for all the frames and put the angles with >mag_threshold in different 
    bins. The bins vector is the feature representation for the stroke. 
    Use only the strokes given by list of tuples frame_indx.
    Parameters:
    ------
    vidFile: str
        complete path to a video
    start: int
        starting frame number
    end: int
        ending frame number
    hist_bins: 1d np array 
        bin divisions (boundary values). Used np.linspace(0, 2*PI, 11) for 10 bins
    mag_thresh: int
        minimum size of the magnitude vectors that are considered (no. of pixels shifted in consecutive frames of OF)
    
    '''
    cap = cv2.VideoCapture(vidFile)
    if not cap.isOpened():
        print("Capture object not opened. Aborting !!")
        sys.exit(0)
    ret = True
    stroke_features = []
    prvs, next_ = None, None
    m, n = start, end    
    #print("stroke {} ".format((m, n)))
    sum_norm_mag_ang = np.zeros((len(hist_bins)-1))  # for optical flow maxFrames - 1 size
    frameNo = m
    while ret and frameNo <= n:
        if (frameNo-m) == 0:    # first frame condition
            cap.set(cv2.CAP_PROP_POS_FRAMES, frameNo)
            ret, frame1 = cap.read()
            if not ret:
                print("Frame not read. Aborting !!")
                break
            # resize and then convert to grayscale
            #cv2.imwrite(os.path.join(flow_numpy_path, str(frameNo)+".png"), frame1)
            prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            #prvs = scale_and_crop(prvs, scale)
            frameNo +=1
            continue
            
        ret, frame2 = cap.read()
        # resize and then convert to grayscale
        next_ = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(prvs, next_, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        
        #print("Mag > 5 = {}".format(np.sum(mag>THRESH)))
        pixAboveThresh = np.sum(mag>mag_thresh)
        #use weights=mag[mag>THRESH] to be weighted with magnitudes
        #returns a tuple of (histogram, bin_boundaries)
        ang_hist = np.histogram(ang[mag>mag_thresh], bins=hist_bins, density=density)
        stroke_features.append(ang_hist[0])
        #sum_norm_mag_ang +=ang_hist[0]
#            if not pixAboveThresh==0:
#                sum_norm_mag_ang[frameNo-m-1] = np.sum(mag[mag > THRESH])/pixAboveThresh
#                sum_norm_mag_ang[(maxFrames-1)+frameNo-m-1] = np.sum(ang[mag > THRESH])/pixAboveThresh
        frameNo+=1
        prvs = next_
        #stroke_features.append(sum_norm_mag_ang/(n-m+1))
    cap.release()
    #cv2.destroyAllWindows()
    stroke_features = np.array(stroke_features)
    #Normalize row - wise
    #stroke_features = stroke_features/(1+stroke_features.sum(axis=1)[:, None])
    return stroke_features


def extract_flow_grid(vidFile, start, end, grid_size):
    '''
    Extract optical flow maps from video vidFile starting from start frame number
    to end frame no. The grid based features are flattened and appended.
    
    Parameters:
    ------
    vidFile: str
        complete path to a video
    start: int
        starting frame number
    end: int
        ending frame number
    grid_size: int
        grid size for sampling at intersection points of 2D flow.
    
    Returns:
    ------
    np.array 2D with N x (360/G * 640/G) where G is grid size
    '''
    cap = cv2.VideoCapture(vidFile)
    if not cap.isOpened():
        print("Capture object not opened. Aborting !!")
        sys.exit(0)
    ret = True
    stroke_features = []
    prvs, next_ = None, None
    m, n = start, end
    #print("stroke {} ".format((m, n)))
    #sum_norm_mag_ang = np.zeros((len(hist_bins)-1))  # for optical flow maxFrames - 1 size
    frameNo = m
    while ret and frameNo <= n:
        if (frameNo-m) == 0:    # first frame condition
            cap.set(cv2.CAP_PROP_POS_FRAMES, frameNo)
            ret, frame1 = cap.read()
            if not ret:
                print("Frame not read. Aborting !!")
                break
            # resize and then convert to grayscale
            #cv2.imwrite(os.path.join(flow_numpy_path, str(frameNo)+".png"), frame1)
            prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            #prvs = scale_and_crop(prvs, scale)
            frameNo +=1
            continue
            
        ret, frame2 = cap.read()
        # resize and then convert to grayscale
        next_ = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(prvs, next_, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        
        # stack sliced arrays along the first axis (2, 12, 16)
        sliced_flow = np.stack(( mag[::grid_size, ::grid_size], \
                                ang[::grid_size, ::grid_size]), axis=0)
                
        stroke_features.append(sliced_flow[1, ...].ravel())     # Only angles
        #feature = np.array(feature)
        #stroke_features.append(sliced_flow)
        
        frameNo+=1
        prvs = next_
        
    cap.release()
    #cv2.destroyAllWindows()
    stroke_features = np.array(stroke_features)
    #Normalize row - wise
    #stroke_features = stroke_features/(1+stroke_features.sum(axis=1)[:, None])
    return stroke_features
