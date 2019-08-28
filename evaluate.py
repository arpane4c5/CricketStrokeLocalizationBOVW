#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 04:47:16 2019

@author: arpan

@Description: Evaluate functions. Find maximum permutation accuracy, and write the
predicted segments into the class folders.
"""

import csv
import cv2
import os
import sys
import numpy as np
from itertools import permutations

def get_cluster_labels(cluster_labels_path):
    labs_keys = []
    labs_values = []
    with open(cluster_labels_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:    
            #print("{} :: Class : {}".format(row[0], row[1]))
            labs_keys.append(row[0])
            labs_values.append(int(row[1]))
            line_count += 1
        print("Read {} ground truth stroke labels from file.".format(line_count))
    return labs_keys, labs_values

def calculate_accuracy(gt_list, pred_list):
    """
    Get two dictionaries with labels. 
    """
    gt_clus_nos = sorted(list(set(gt_list)))
    pred_clus_nos = sorted(list(set(pred_list)))
    n_clusters_gt = len(gt_clus_nos)
    n_clusters_pred = len(pred_clus_nos)
    assert n_clusters_gt>=n_clusters_pred, "Predicted more clusters {} : GT {}".format(n_clusters_pred, n_clusters_gt)
    
    for i,x in enumerate(pred_clus_nos):
        assert x==gt_clus_nos[i], "Cluster no. mismatch {} / {}".format(gt_clus_nos[i], x)
        
    acc_values = []
    perm_list = list(permutations(pred_clus_nos))
    for perm_tuple in perm_list:
        
        pred_list_permuted = assign_clusters(perm_tuple, pred_list)
        #print(pred_list[0:11], pred_list_permuted[0:11],perm_tuple)
        acc_tuple = len(perm_tuple)*[0]
        # For each cluster in tuple, find the no. of correct predictions and save at specific tuple indx position
        for t in perm_tuple:
            acc_tuple[t] = sum([(gt_list[j]==pred and gt_list[j]==t) for j,pred in enumerate(pred_list_permuted)])
        
        acc_values.append(acc_tuple)
        
    return acc_values, perm_list, gt_list, pred_list
    
    
def assign_clusters(perm_tuple, labs):
    """
    Take input as permutation tuple and interchange labels in list labs
    Eg. if perm_tuple = (1,2,0) and labs = [0,0,0,1,1,1,2,2,2], then 
    return [1, 1, 1, 2, 2, 2, 0, 0, 0]
    """
    temp = len(labs)*[-1]
    for i,clust in enumerate(labs):
        for j,t in enumerate(perm_tuple):
            if clust==j:
                temp[i]=t
    return temp

#f = None
## save video according to their label
#def evaluate(labels, bins, thresh):
#    global f
#    flows_path = get_ordered_strokes_list()
#
#    #flows_path = sorted(os.listdir(flows_numpy_path))
#    n_clusters = max(labels) + 1
#    print("clusters, ", n_clusters)
#    for i in range(n_clusters):
#        ######
#        try:
#            os.makedirs(os.path.join(RESULTS_DIR, "bins_"+str(bins)+"_th_"+str(thresh), str(i)))
#        except Exception as e:
#            print("except", e)
#        #######
#        for count,j in enumerate(np.where(labels==i)[0]):
#            vid_data = flows_path[j].split('_')
#            m, n = map(int, vid_data[-2:])
#            vid_name = vid_data[0]
#            f = ''.join(vid_name.split(' ')[2:-1])+"_"+str(m)+"_"+str(n)
#            save_video(os.path.join(DATASET, vid_name+'.avi'), m, n, i, bins, thresh)
#            if count==9:
#                break
#
#            
#def get_frame(cap, frame_no):
#    # get total number of frames
#    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
#    # check for valid frame number
#    if frame_no >= 0 & frame_no <= totalFrames:
#        # set frame position
#        cap.set(cv2.CAP_PROP_POS_FRAMES,frame_no)
#        _, img = cap.read()
#        return img
#    print("invalid frame, ", frame_no)
#    sys.exit()
#
#def save_video(filename, m, n, label, bins, thresh):
#    global f
#    eval_path = os.path.join(RESULTS_DIR, "bins_"+str(bins)+"_th_"+str(thresh), str(label))
#    fourcc = cv2.VideoWriter_fourcc(*'XVID')
#    vid_out_path = os.path.join(eval_path, f+'.avi')
#    out = cv2.VideoWriter(vid_out_path, fourcc, 25.0, (320, 180), True)
#    cap = cv2.VideoCapture(filename)
#    if cap.isOpened():
#        pass
#    else:
#        print("closed")
#        sys.exit()
#    for i in range(m, n+1):
#        img = cv2.resize(get_frame(cap, i), (320, 180), interpolation=cv2.INTER_AREA)
#        out.write(img)
#    cap.release()
#    out.release()
