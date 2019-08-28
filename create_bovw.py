#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 04:41:12 2019

@author: arpan

@Description: Clustering and creation of BOVW dataframe
"""
import numpy as np
from scipy.cluster.vq import vq
import pandas as pd
import pickle
import os
from sklearn.cluster import KMeans

def make_codebook(vecs, nclusters):
    """
    Function to find the clusters using KMeans
    Parameters:    
        vecs: any dataframe representing the input space points
        nclusters: No. of clusters to be formed
    Returns:
        KMeans object, containing the clustering information.
    """
#    pickle.dump(train_keypoints, open(os.path.join('data',target_file), "wb"))
    print("Clustering using KMeans: Input size -> {} :: n_clusters -> {}"\
          .format(vecs.shape, nclusters))   
    
    #train_features = pickle.load(open(keypoints_path, "rb"))
    #clustering with k-means
    #kmeans = KMeans(init='k-means++', n_clusters=200, n_init=10, n_jobs=2, verbose=1)
    kmeans = KMeans(n_clusters=nclusters, n_init=10, n_jobs=2, random_state=128)
    kmeans.fit(vecs)
    print("Done Clustering!")
    return kmeans


def create_bovw_df(features, strokes_name_id, km_model, base, partition='train'):
    '''
    Form a features dataframe of C3D FC7/HOOF features kept in feats_data_dict. 
    Returns one dataframe of (nTrimmedVids, nClusters). 
    
    Parameters:
    ------
    features: dict
        {'video_id1':np.array((N, d)), ...}  , where d is vec dimension
    strokes_name_id: list of str
        key values of features, eg. video-name_34_242 
    km_model: KMeans obj
        Learned KMeans model having cluster centers of training vectors
        
    Returns:
    ------
    pd.Dataframe of size (nTrimmedVids, nClusters)
    with frequency histogram of trimmed videos
    Also return a string sequence of words with integer values representing
    cluster centers.
    
    '''
    # get the cluster centroids
    clusters_c3d = km_model.cluster_centers_
    n_strokes = len(strokes_name_id)
    
    # Create a dataframe of size n_videos X n_clusters
    print("Make bow vector for each feature vector")
    bovw_df = np.zeros((n_strokes, clusters_c3d.shape[0]))
    
    print("Shape of bovw_dataframe : {}".format(bovw_df.shape))
    
    words = []
    row_no = 0
    # Make bow vectors for all videos.
    for video_index, video in enumerate(strokes_name_id):
        # Get the starting and ending stroke frame positions        
        m, n = video.split('_')[1:]
        m, n = int(m), int(n)
        
        # select the vectors of size Nx1x4096 and remove mid dimension of 1
        stroke_feats = features[video]
        stroke_feats[np.isnan(stroke_feats)] = 0
        stroke_feats[np.isinf(stroke_feats)] = 0
        # find cluster centroid assignments for all points
        # returns a tuple, with first element having ids of the cluster centroid 
        # to which the row i belongs to. Second element is the distance between 
        # the nearest code and the ith row.
        # visual_word_ids is a 1D array
        word_ids = vq(stroke_feats, clusters_c3d)[0]  # ignoring the distances in [1]
        
        string_temp = ""
        for w_no, word_id in enumerate(word_ids):
#            if w_no<20:
#                continue
            bovw_df[row_no, word_id] += 1
            t=str(word_id)
            string_temp = string_temp+" "+t
            
        words.append(string_temp)
        row_no +=1

    if partition=='train':
        # IDF weighting
        freq = np.sum((bovw_df > 0) * 1, axis = 0)
        idf = np.log((n_strokes + 1.0) / (freq + 1.0))
        bovw_df = bovw_df * idf
        with open(os.path.join(base,"idf_ang.pkl"), "wb") as outfile:
            pickle.dump(idf, outfile)
        with open(os.path.join(base,"freq_ang.pkl"), "wb") as outfile:
            pickle.dump(freq, outfile)
        print("Saved IDF weights to disk.")
    else:   # For validation/test set
        print("Reading IDF weights to disk.")
        with open(os.path.join(base,"idf_ang.pkl"), "rb") as infile:
            idf = pickle.load(infile)
        with open(os.path.join(base,"freq_ang.pkl"), "rb") as infile:
            freq = pickle.load(infile)
        bovw_df = bovw_df * idf

#    # form the list of words (without stopwords)
#    row_no = 0
#    nstop = 2   # 5 words are considered as stopwords
#    # get list of indices as stopwords
#    stop_indices = np.argpartition(freq, -1*nstop)[(-1*nstop):]
#    for video_index, video in enumerate(strokes_name_id):
#        # Get the starting and ending stroke frame positions        
#        m, n = video.split('_')[1:]
#        m, n = int(m), int(n)
#        
#        # select the vectors of size Nx1x4096 and remove mid dimension of 1
#        stroke_feats = features[video]
#        stroke_feats[np.isnan(stroke_feats)] = 0
#        stroke_feats[np.isinf(stroke_feats)] = 0
#        # find cluster centroid assignments for all points
#        # returns a tuple, with first element having ids of the cluster centroid 
#        # to which the row i belongs to. Second element is the distance between 
#        # the nearest code and the ith row.
#        # visual_word_ids is a 1D array
#        word_ids = vq(stroke_feats, clusters_c3d)[0]  # ignoring the distances in [1]
#        
#        string_temp = ""
#        for word_id in word_ids:
#            if word_id not in stop_indices:
#                t=str(word_id)
#                string_temp = string_temp+" "+t
#            
#        words.append(string_temp)        
#        row_no +=1    
    
    bovw_df = pd.DataFrame(bovw_df, index=strokes_name_id)
    return bovw_df, words

