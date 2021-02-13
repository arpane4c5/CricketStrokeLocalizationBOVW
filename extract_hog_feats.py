#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 02:57:52 2020

@author: arpan

@Description: Extract HOG features on resized 64x64 center crops for all the strokes
"""

import os
import cv2
import sys
import json
import numpy as np
import time
import pickle

sys.path.insert(0, '../cluster_strokes')
sys.path.insert(0, '../cluster_strokes/lib')
from utils import autoenc_utils


def extract_stroke_hog_feats(vidsPath, labelsPath, partition_lst, hogPath):
    """
    Function to iterate on all the training videos and extract the relevant HOG features.
    vidsPath: str
        path to the dataset containing the videos
    labelsPath: str
        path to the JSON files for the labels.
    partition_lst: list of video_ids
        video_ids are the filenames (without extension)
    hogPath: str
        absolute path to file hog.xml (with HOG params)
    """
    
    strokes_name_id = []
    all_feats = {}
#    bins = np.linspace(0, 2*np.pi, (nbins+1))
    for i, v_file in enumerate(partition_lst):
        print('-'*60)
        print(str(i+1)+". v_file :: ", v_file)
        if '.avi' in v_file or '.mp4' in v_file:
            v_file = v_file.rsplit('.', 1)[0]
        json_file = v_file + '.json'
        #print("json file :: ", json_file)
        
        # read labels from JSON file
        assert os.path.exists(os.path.join(labelsPath, json_file)), "{} doesn't exist!".format(json_file)
            
        with open(os.path.join(labelsPath, json_file), 'r') as fr:
            frame_dict = json.load(fr)
        frame_indx = list(frame_dict.values())[0]
        for m,n in frame_indx:
            k = v_file+"_"+str(m)+"_"+str(n)
            print("Stroke {} - {}".format(m,n))
            strokes_name_id.append(k)
            # Extract the stroke features
            all_feats[k] = extract_hog_vid(os.path.join(vidsPath, v_file+".avi"), m, n, hogPath)
            
        #break
    return all_feats, strokes_name_id


def extract_hog_vid(vidFile, start, end, hogPath):
    '''
    Extract HOG feats from video vidFile for all the frames.
    Use only the strokes given by list of tuples frame_indx.
    Parameters:
    ------
    vidFile: str
        complete path to a video
    start: int
        starting frame number
    end: int
        ending frame number
    hogPath: str
        absolute path to file hog.xml (with HOG params)
    
    '''
    cap = cv2.VideoCapture(vidFile)
    if not cap.isOpened():
        print("Capture object not opened. Aborting !!")
        sys.exit(0)
        
    W, H = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), \
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    #totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frameCount = 0
    ret = True
    stroke_features = []
    m, n = start, end    
    #print("stroke {} ".format((m, n)))
#    sum_norm_mag_ang = np.zeros((len(hist_bins)-1))  # for optical flow maxFrames - 1 size
    frameNo = m
    while ret and frameNo <= n:
        if (frameNo-m) == 0:    # first frame condition
            cap.set(cv2.CAP_PROP_POS_FRAMES, frameNo)
        ret, frame = cap.read()
        if not ret:
            print("Frame not read. Aborting !!")
            break
        # resize and then convert to grayscale
        #cv2.imwrite(os.path.join(flow_numpy_path, str(frameNo)+".png"), frame1)
        curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #curr_frame = cv2.resize(curr_frame, (W/2, H/2), cv2.INTER_AREA)
        curr_frame = cv2.resize(curr_frame, None, fx=0.25, \
                                fy=0.25, interpolation=cv2.INTER_AREA)
        (h, w) = curr_frame.shape[:2]
        # Take the centre crop of the frames (64 x 64)
        curr_frame = curr_frame[(int(h/2)-32):(int(h/2)+32), (int(w/2)-32):(int(w/2)+32)]
        # compute the HOG feature vector 
        hog = cv2.HOGDescriptor(hogPath)        # get cv2.HOGDescriptor object
        hog_feature = hog.compute(curr_frame)   # get 3600 x 1 matrix (not vec)
        # saving as a list of float matrices (dim 1 x vec_size)        
        stroke_features.append(hog_feature.flatten())
        
        frameNo+=1
        
    cap.release()
    
    stroke_features = np.array(stroke_features)
    #Normalize row - wise
    
    return stroke_features


if __name__ == '__main__':
    
    LABELS = "/home/arpan/VisionWorkspace/Cricket/scripts/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
    DATASET = "/home/arpan/VisionWorkspace/VideoData/sample_cricket/ICC WT20"
    hogPath = "/home/arpan/VisionWorkspace/Cricket/localization_rnn/hog.xml"
    feat_path = "/home/arpan/VisionWorkspace/Cricket/CricketStrokeLocalizationBOVW/logs/bow_HL_HOG"
    
    train_lst, val_lst, test_lst = autoenc_utils.split_dataset_files(DATASET)
    print("No. of training videos : {}".format(len(train_lst)))

    start = time.time()
    
    if not os.path.isfile(os.path.join(feat_path, "hog_feats.pkl")):
        if not os.path.exists(feat_path):
            os.makedirs(feat_path)
        # Extract HOG features on resized 64x64 center crops
        print("Training extraction ... ")
        features, strokes_name_id = extract_stroke_hog_feats(DATASET, LABELS, train_lst, hogPath)
        with open(os.path.join(feat_path, "hog_feats.pkl"), "wb") as fp:
            pickle.dump(features, fp)
        with open(os.path.join(feat_path, "hog_snames.pkl"), "wb") as fp:
            pickle.dump(strokes_name_id, fp)
                
    if not os.path.isfile(os.path.join(feat_path, "hog_feats_val.pkl")):
        print("Validation extraction ....")
        features_val, strokes_name_id_val = extract_stroke_hog_feats(DATASET, LABELS, val_lst, hogPath)
        with open(os.path.join(feat_path, "hog_feats_val.pkl"), "wb") as fp:
            pickle.dump(features_val, fp)
        with open(os.path.join(feat_path, "hog_snames_val.pkl"), "wb") as fp:
            pickle.dump(strokes_name_id_val, fp)

    if not os.path.isfile(os.path.join(feat_path, "hog_feats_test.pkl")):
        print("Testing extraction ....")
        features_test, strokes_name_id_test = extract_stroke_hog_feats(DATASET, LABELS, test_lst, hogPath)
        with open(os.path.join(feat_path, "hog_feats_test.pkl"), "wb") as fp:
            pickle.dump(features_test, fp)
        with open(os.path.join(feat_path, "hog_snames_test.pkl"), "wb") as fp:
            pickle.dump(strokes_name_id_test, fp)
    end = time.time()
    print("Total execution time : "+str(end-start))