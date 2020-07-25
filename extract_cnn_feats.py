#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri July 10 02:57:52 2020

@author: arpan

@Description: Extract CNN features from selected files
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, '../cluster_strokes')
sys.path.insert(0, '../cluster_strokes/lib')

from utils.extract_autoenc_feats import extract_3DCNN_feats
from utils.extract_autoenc_feats import extract_2DCNN_feats
from utils import trajectory_utils as traj_utils
#from utils import spectral_utils
from utils import plot_utils
from evaluation import eval_of_clusters

def extract_feats(DATASET, LABELS, CLASS_IDS, BATCH_SIZE, SEQ_SIZE, STEP=1, 
                  extractor='2dcnn', model_path=None, nclasses=5, part='all'):
    
    # Extract autoencoder features 
#    trajectories, stroke_names = extract_sequence_feats(model_path, DATASET, LABELS, 
#                                    CLASS_IDS, BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE, 
#                                    NUM_LAYERS, SEQ_SIZE, STEP, partition='all', 
#                                    nstrokes=3)
    
    if extractor == '2dcnn':
        # SEQ_SIZE is 1 and STEP is 1 by default
        trajectories, strokes_name_id = extract_2DCNN_feats(DATASET, LABELS, CLASS_IDS, 
                                                         BATCH_SIZE, partition=part, 
                                                         nstrokes=-1)
    elif extractor == '3dcnn':
        assert SEQ_SIZE >=16, "SEQ_SIZE should be >=16"
        trajectories, strokes_name_id = extract_3DCNN_feats(DATASET, LABELS, CLASS_IDS, 
                                                         BATCH_SIZE, SEQ_SIZE, STEP, 
                                                         model_path, nclasses, 
                                                         partition=part, nstrokes=-1)
    
    all_feats = {}
    
    trajectories = [np.stack(stroke) for vid_strokes in trajectories for stroke in vid_strokes]
    strokes_name_id = [stroke.replace('.avi', '') for vid_strokes in strokes_name_id for stroke in vid_strokes]
    strokes_name_id = [stroke.rsplit('/', 1)[-1] for stroke in strokes_name_id]
    
    for i, v in enumerate(trajectories):
        all_feats[strokes_name_id[i]] = v
        
    return all_feats, strokes_name_id

def apply_clustering(df_train, DATASET, LABELS, ANNOTATION_FILE, base_path):
    
    print("Clustering for 3 clusters... \n")
            
    km = traj_utils.kmeans(df_train, 5)
    labels = km.labels_
    
    acc_values, perm_tuples, gt_list, pred_list = \
                eval_of_clusters.get_accuracy(df_train, labels, ANNOTATION_FILE, \
                                              DATASET, LABELS)
    acc_perc = [sum(k)/df_train.shape[0] for k in acc_values]
    #best_acc[i,j] = max(acc_perc)
#    bins_vals.append(bins)
#    thresh_vals.append(thresh)
    #best_acc.append(max(acc_perc))
    best_indx = acc_perc.index(max(acc_perc))
    print("Max Acc. : ", max(acc_perc))
    print("Acc values : ", acc_perc)
    print("Acc values : ", acc_values)
    print("perm_tuples : ", perm_tuples)
    
    if df_train.shape[1] > 2:
        pca_flows = traj_utils.apply_PCA(df_train)
    else:
        pca_flows = df_train
    plot_utils.plot_clusters(pca_flows, labels, perm_tuples[best_indx], \
                             base_path, 'cluster_pca_ordered.png')
    
    pca_flows = traj_utils.apply_tsne(df_train)
    plot_utils.plot_clusters(pca_flows, labels, perm_tuples[best_indx], \
                             base_path, 'cluster_tsne_ordered.png')
    
if __name__ == '__main__':

    # Local Paths
    LABELS = "/home/arpan/VisionWorkspace/Cricket/scripts/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
    DATASET = "/home/arpan/VisionWorkspace/VideoData/sample_cricket/ICC WT20"
    CLASS_IDS = "/home/arpan/VisionWorkspace/Cricket/cluster_strokes/configs/Class Index_Strokes.txt"
    # Server Paths
    if os.path.exists("/opt/datasets/cricket/ICC_WT20"):
        LABELS = "/home/arpan/VisionWorkspace/shot_detection/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
        DATASET = "/opt/datasets/cricket/ICC_WT20"

#    model_path = "checkpoints/autoenc_gru_resnet50_ep10_w6_Adam.pt"
    SEQ_SIZE = 16
    STEP = 7
#    INPUT_SIZE = 2048
#    HIDDEN_SIZE = 32#64#1024
#    NUM_LAYERS = 2
    BATCH_SIZE = 8     
    trajectory, snames = extract_feats(DATASET, LABELS, CLASS_IDS, BATCH_SIZE, SEQ_SIZE, STEP, extractor='3dcnn')