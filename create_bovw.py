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
    kmeans = KMeans(n_clusters=nclusters, n_init=10, n_jobs=2, random_state=123)
    kmeans.fit(vecs)
    print("Done Clustering!")
    return kmeans


def create_bovw_traindf(features, strokes_name_id, km_model):
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
        # find cluster centroid assignments for all points
        # returns a tuple, with first element having ids of the cluster centroid 
        # to which the row i belongs to. Second element is the distance between 
        # the nearest code and the ith row.
        # visual_word_ids is a 1D array
        word_ids_c3d = vq(stroke_feats, clusters_c3d)[0]  # ignoring the distances in [1]
        
        string_temp = ""
        for word_id in word_ids_c3d:
            bovw_df[row_no, word_id] += 1
            t=str(word_id)
            string_temp = string_temp+" "+t
            
        words.append(string_temp)
        row_no +=1

    bovw_df = pd.DataFrame(bovw_df, index=strokes_name_id)
    
    return bovw_df, words




#        # feat is an array(2,16,12)
#        mag = [feat[0,...].ravel() for feat in frames_feats]
#        ang = [feat[1,...].ravel() for feat in frames_feats]
#        mag = np.array(mag)  # (nFlowFrames, 192)
#        ang = np.array(ang)
#        # change inf values to 0
#        mag[np.isinf(mag)] = 0
#        ang[np.isinf(ang)] = 0
#            
#        ##### Normalize
#        mag = (mag - mag_min)/(mag_max - mag_min)
#        ang = (ang - ang_min)/(ang_max - ang_min)
        # find cluster centroid assignments for all points
        # returns a tuple, with first element having ids of the cluster centroid 
        # to which the row i belongs to. Second element is the distance between 
        # the nearest code and the ith row.
        # visual_word_ids is a 1D array
        #word_ids_c3d = vq(frames_feats, clusters_c3d)[0]  # ignoring the distances in [1]
        
#        #temp=[]
#        string_temp=""
#        for word_id in word_ids_c3d:
#            bow_c3d[video_index, word_id] += 1
#            t=str(word_id)
#            string_temp = string_temp+" "+t
#            #temp.append(str(word_id))
#            #print("word id is {}".format(word_id))
#            #print(word_id)
#        
#        #print("temp is : {}".format(temp))  
#        #print("string_temp is ".format(string_temp))
#        #print("whats happening")
#        print("Done video {} : {}".format(video_index, video))
#        words.append(string_temp)
        
#    print("Applying TF-IDF weighting")
#    # This is applicable for only the training set
#    # For validation/test set, the idf will be same as for the training set
#    freq_c3d = np.sum((bow_c3d > 0) * 1, axis = 0)
#    idf_c3d = np.log((n_strokes + 1.0) / (freq_c3d + 1.0))
#    #bow_mag = bow_mag * idf_mag
#    
#    #print("idf_mag")
#    #print(idf_mag)
#    # save idf_mag to disk
#    #pickle.dump(idf_c3d, open(os.path.join(idfpath,"idf_c3d.pkl"), "wb"))
#     
