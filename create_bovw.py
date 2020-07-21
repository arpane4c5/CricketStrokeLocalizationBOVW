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
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.preprocessing import normalize

def make_codebook(vecs, nclusters, model_type="kmeans"):
    """
    Function to find the clusters using KMeans
    Parameters:    
        vecs: any dataframe representing the input space points
        nclusters: No. of clusters to be formed
        model_type : str
            'kmeans' or 'gmm' for selecting the clustering model.
    Returns:
        KMeans or GaussianMixture object, containing the clustering information.
    """
    assert model_type == "kmeans" or model_type == "gmm", "Invalid model_type."
    if model_type == 'kmeans':
        print("Clustering using KMeans: Input size -> {} :: n_clusters -> {}"\
              .format(vecs.shape, nclusters))   
        model = KMeans(n_clusters=nclusters, n_init=10, n_jobs=2, random_state=128)
        model.fit(vecs)
    elif model_type == 'gmm':
        print("Clustering using GMM: Input size -> {} :: n_components -> {}"\
              .format(vecs.shape, nclusters))
        model = GaussianMixture(n_components=nclusters, covariance_type='diag',
                              random_state=128).fit(vecs)        
    
    print("Done Clustering!")
    return model


def create_bovw_df(features, strokes_name_id, model, base, partition='train'):
    '''
    Form a features dataframe of C3D FC7/HOOF features kept in feats_data_dict. 
    Returns one dataframe of (nTrimmedVids, nClusters). 
    
    Parameters:
    ------
    features: dict
        {'video_id1':np.array((N, d)), ...}  , where d is vec dimension
    strokes_name_id: list of str
        key values of features, eg. video-name_34_242 
    km_model: KMeans / GaussianMixture obj
        Learned KMeans / GMM model with nClusters / nComponents
        
    Returns:
    ------
    pd.Dataframe of size (nTrimmedVids, nClusters)
    with frequency histogram of trimmed videos
    Also return a string sequence of words with integer values representing
    cluster centers.
    
    '''
    # get the cluster centroids
    if isinstance(model, KMeans):
        n_clusters = model.n_clusters
    else:
        n_clusters = model.n_components
    n_strokes = len(strokes_name_id)
    
    # Create a dataframe of size n_videos X n_clusters
    print("Make bow vector for each feature vector")
    bovw_df = np.zeros((n_strokes, n_clusters), dtype=np.float)
    
    print("Shape of bovw_dataframe : {}".format(bovw_df.shape))
    
    words = []
    row_no = 0
    # Make bow vectors for all videos.
    for video_index, video in enumerate(strokes_name_id):
        # Get the starting and ending stroke frame positions        
        m, n = video.rsplit('_', 2)[1:]
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
        if isinstance(model, KMeans):
            word_ids = vq(stroke_feats, model.cluster_centers_)[0]  # ignoring the distances in [1]
        else:
            word_ids = model.predict(stroke_feats)
        
        string_temp = ""
        for w_no, word_id in enumerate(word_ids):
#            if w_no<20:
#                continue
            bovw_df[row_no, word_id] += 1
            t=str(word_id)
            string_temp = string_temp+" "+t
            
        words.append(string_temp)
        row_no +=1

    # tf = #occurrences of word i in doc d (n_{id}) / #total word in doc d (n_{d}) 
#    tf = bovw_df.div(bovw_df.sum(axis=1), axis=0)
#    tf = bovw_df / bovw_df.sum(axis=1)[:, None]
    
    if partition=='train':
        # IDF weighting
        freq = np.sum(bovw_df, axis = 0)
        idf = np.log((n_strokes + 1.0) / (freq + 1.0))
        # log (#total docs (N) / #occurrences of term i in training set (n_{i}))
#        idf =  np.log (bovw_df.shape[0] / bovw_df.sum(axis=0))
#        bovw_df = tf * idf
        with open(os.path.join(base,"idf.pkl"), "wb") as outfile:
            pickle.dump(idf, outfile)
#        with open(os.path.join(base,"freq_ang.pkl"), "wb") as outfile:
#            pickle.dump(freq, outfile)
        print("Saved IDF weights to disk.")
    else:   # For validation/test set
        print("Reading IDF weights to disk.")
        with open(os.path.join(base,"idf.pkl"), "rb") as infile:
            idf = pickle.load(infile)
#        with open(os.path.join(base,"freq_ang.pkl"), "rb") as infile:
#            freq = pickle.load(infile)
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

def create_bovw_df_SA(features, strokes_name_id, model, base, partition='train'):
    '''
    Form a features dataframe of C3D FC7/HOOF features kept in feats_data_dict. 
    Returns one dataframe of (nTrimmedVids, nClusters). Use Soft Assignment
    
    Parameters:
    ------
    features: dict
        {'video_id1':np.array((N, d)), ...}  , where d is vec dimension
    strokes_name_id: list of str
        key values of features, eg. video-name_34_242 
    km_model: KMeans / GaussianMixture obj
        Learned KMeans / GMM model with nClusters / nComponents
        
    Returns:
    ------
    pd.Dataframe of size (nTrimmedVids, nClusters)
    with frequency histogram of trimmed videos
    Also return a string sequence of words with integer values representing
    cluster centers.
    
    '''
    # get the cluster centroids
    if isinstance(model, KMeans):
        n_clusters = model.n_clusters
    else:
        n_clusters = model.n_components
    n_strokes = len(strokes_name_id)
    
    # Create a dataframe of size n_videos X n_clusters
    print("Make bow vector for each feature vector")
    bovw_df = np.zeros((n_strokes, n_clusters), dtype=np.float)
    
    print("Shape of bovw_dataframe : {}".format(bovw_df.shape))
    
    words = []
    row_no = 0
    beta = -1.
    if features[list(features.keys())[0]].shape[1] == 2304:
        beta = -0.6     # For ofGrid10, exp operation gives large values for beta=-1
    # Make bow vectors for all videos.
    for video_index, video in enumerate(strokes_name_id):
        # Get the starting and ending stroke frame positions        
        m, n = video.rsplit('_', 2)[1:]
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
        
        if isinstance(model, KMeans):
            # calculate L2 dist of each row from all cluster centers
            cl_dists = [(np.linalg.norm(model.cluster_centers_ - stroke_feats[i,:], axis=1)) \
                        for i in range(stroke_feats.shape[0])]
            # form nFeats x nClusters (distance of a feature from all the cluster centers)
            cl_dists = np.vstack(cl_dists)      # unnormalized
#            cl_dists = normalize(cl_dists**2, axis=1, norm="l2")   # accuracy decreases 
        else:
            cl_dists = model.predict_proba(stroke_feats)
        
#        omega = np.sum(cl_dists, axis=0) / np.sum(cl_dists)
        omega = np.exp(beta * cl_dists)     # beta=1, decreasing it reduces accuracy
        omega = omega / omega.sum(axis = 1)[:, None]    # normalize
        bovw_df[row_no, :] = np.sum(omega, axis=0) / omega.shape[0]
            
#        if isinstance(model, KMeans):
#            word_ids = vq(stroke_feats, model.cluster_centers_)[0]  # ignoring the distances in [1]
#        else:
#            word_ids = model.predict(stroke_feats)
#        
#        string_temp = ""
#        for w_no, word_id in enumerate(word_ids):
##            if w_no<20:
##                continue
#            bovw_df[row_no, word_id] += 1
#            t=str(word_id)
#            string_temp = string_temp+" "+t
#            
#        words.append(string_temp)
        row_no +=1

    # tf = #occurrences of word i in doc d (n_{id}) / #total word in doc d (n_{d}) 
#    tf = bovw_df.div(bovw_df.sum(axis=1), axis=0)
#    tf = bovw_df / bovw_df.sum(axis=1)[:, None]
    
    if partition=='train':
        # IDF weighting
        freq = np.sum(bovw_df, axis = 0)
        idf = np.log((n_strokes + 1.0) / (freq + 1.0))
        # log (#total docs (N) / #occurrences of term i in training set (n_{i}))
#        idf =  np.log (bovw_df.shape[0] / bovw_df.sum(axis=0))
#        bovw_df = tf * idf
        with open(os.path.join(base,"idf.pkl"), "wb") as outfile:
            pickle.dump(idf, outfile)
#        with open(os.path.join(base,"freq_ang.pkl"), "wb") as outfile:
#            pickle.dump(freq, outfile)
        print("Saved IDF weights to disk.")
    else:   # For validation/test set
        print("Reading IDF weights to disk.")
        with open(os.path.join(base,"idf.pkl"), "rb") as infile:
            idf = pickle.load(infile)
#        with open(os.path.join(base,"freq_ang.pkl"), "rb") as infile:
#            freq = pickle.load(infile)
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


def create_bovw_df_OMP(features, strokes_name_id, model, base, partition='train'):
    '''
    Form a features dataframe of C3D FC7/HOOF features kept in feats_data_dict. 
    Returns one dataframe of (nTrimmedVids, nClusters). Use Soft Assignment
    
    Parameters:
    ------
    features: dict
        {'video_id1':np.array((N, d)), ...}  , where d is vec dimension
    strokes_name_id: list of str
        key values of features, eg. video-name_34_242 
    km_model: KMeans / GaussianMixture obj
        Learned KMeans / GMM model with nClusters / nComponents
        
    Returns:
    ------
    pd.Dataframe of size (nTrimmedVids, nClusters)
    with frequency histogram of trimmed videos
    Also return a string sequence of words with integer values representing
    cluster centers.
    
    '''
    # get the cluster centroids
    if isinstance(model, KMeans):
        n_clusters = model.n_clusters
    else:
        n_clusters = model.n_components
    n_strokes = len(strokes_name_id)
    
    # Create a dataframe of size n_videos X n_clusters
    print("Make bow vector for each feature vector")
    bovw_df = np.zeros((n_strokes, n_clusters), dtype=np.float)
    
    print("Shape of bovw_dataframe : {}".format(bovw_df.shape))
    
    words = []
    row_no = 0
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=25)
    phi = np.transpose(model.cluster_centers_)
    phi = normalize(phi, axis=0, norm='l2')
    #omp.fit(np.transpose(model.cluster_centers_), stroke_feats[0,:])

    # Make bow vectors for all videos.
    for video_index, video in enumerate(strokes_name_id):
        # Get the starting and ending stroke frame positions        
        m, n = video.rsplit('_', 2)[1:]
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
        
        stroke_feats = normalize(stroke_feats, axis=1, norm='l2')
        print("row No : {}".format(row_no))
        sparse_vecs = []
        
        for i in range(stroke_feats.shape[0]):
            omp.fit(phi, stroke_feats[i,:])
            sparse_vecs.append(omp.coef_)
            
        sparse_vecs = np.vstack(sparse_vecs)

#        omega = np.sum(cl_dists, axis=0) / np.sum(cl_dists)
        omega = np.exp(-1.0 * sparse_vecs)     # beta=1, decreasing it reduces accuracy
        omega = omega / omega.sum(axis = 1)[:, None]    # normalize
        bovw_df[row_no, :] = np.sum(omega, axis=0) / omega.shape[0]
            
#        if isinstance(model, KMeans):
#            word_ids = vq(stroke_feats, model.cluster_centers_)[0]  # ignoring the distances in [1]
#        else:
#            word_ids = model.predict(stroke_feats)
#        
#        string_temp = ""
#        for w_no, word_id in enumerate(word_ids):
##            if w_no<20:
##                continue
#            bovw_df[row_no, word_id] += 1
#            t=str(word_id)
#            string_temp = string_temp+" "+t
#            
#        words.append(string_temp)
        row_no +=1

    # tf = #occurrences of word i in doc d (n_{id}) / #total word in doc d (n_{d}) 
#    tf = bovw_df.div(bovw_df.sum(axis=1), axis=0)
#    tf = bovw_df / bovw_df.sum(axis=1)[:, None]
    
#    if partition=='train':
#        # IDF weighting
#        freq = np.sum(bovw_df, axis = 0)
#        idf = np.log((n_strokes + 1.0) / (freq + 1.0))
#        # log (#total docs (N) / #occurrences of term i in training set (n_{i}))
##        idf =  np.log (bovw_df.shape[0] / bovw_df.sum(axis=0))
##        bovw_df = tf * idf
#        with open(os.path.join(base,"idf.pkl"), "wb") as outfile:
#            pickle.dump(idf, outfile)
##        with open(os.path.join(base,"freq_ang.pkl"), "wb") as outfile:
##            pickle.dump(freq, outfile)
#        print("Saved IDF weights to disk.")
#    else:   # For validation/test set
#        print("Reading IDF weights to disk.")
#        with open(os.path.join(base,"idf.pkl"), "rb") as infile:
#            idf = pickle.load(infile)
##        with open(os.path.join(base,"freq_ang.pkl"), "rb") as infile:
##            freq = pickle.load(infile)
    bovw_df = bovw_df #* idf

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