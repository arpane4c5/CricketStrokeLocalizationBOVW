#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 14:29:10 2020

@author: arpan

@Description: main file for BOW on IDT features
"""

import numpy as np
import pandas as pd
import os
import pickle

from sklearn.externals import joblib
from extract_idt_feats import read_partition_feats
from create_bovw import make_codebook
from create_bovw import create_bovw_df
from evaluate import get_cluster_labels
from evaluate import calculate_accuracy
from sklearn.svm import LinearSVC
import seaborn as sns
from matplotlib import pyplot as plt
sns.set()

import bovw_utils as utils
#import warnings

np.seterr(divide='ignore', invalid='ignore')
#warnings.filterwarnings("ignore")

# Paths and parameters
km_filename = "km_bow"
NUM_TOPICS = 5
MAX_SAMPLES = 50000
ANNOTATION_FILE = "shots_classes.txt"

# Local Paths
LABELS = "/home/arpan/VisionWorkspace/Cricket/scripts/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
DATASET = "/home/arpan/VisionWorkspace/VideoData/sample_cricket/ICC WT20"
IDT_FEATS = "/home/arpan/VisionWorkspace/Cricket/StrokeAttention/logs/idt_strokes"
base_path = "/home/arpan/VisionWorkspace/Cricket/CricketStrokeLocalizationBOVW/logs"


def main(base_name, traj_len=None, cluster_size=10):
    """
    Function to read IDT features, form BOW based model after clustering and evaluate 
    on 3/5 cluster analysis of highlights dataset.
    The videos can be visualized by writing trimmed class videos into their respective
    classes.
    
    Parameters:
    ------
    
    base_name: path to the wts, losses, predictions and log files
    
    """
    seed = 1234
    np.random.seed(seed)
    
    print(60*"#")
          
    #####################################################################
    
    # Divide the sample files into training set, validation and test sets
    train_lst, val_lst, test_lst = utils.split_dataset_files(DATASET)
    
    # form the names of the list of label files, should be at destination 
    train_lab = [f+".json" for f in train_lst]
    val_lab = [f+".json" for f in val_lst]
    test_lab = [f+".json" for f in test_lst]
    
    # get complete path lists of label files
    tr_labs = [os.path.join(LABELS, f) for f in train_lab]
    val_labs = [os.path.join(LABELS, f) for f in val_lab]
    #####################################################################
    
    sizes = [utils.getNFrames(os.path.join(DATASET, f+".avi")) for f in train_lst]
    val_sizes = [utils.getNFrames(os.path.join(DATASET, f+".avi")) for f in val_lst]
    
    ###########################################################################    
    print("No. of training videos : {}".format(len(train_lst)))
    
    print("Size : {}".format(sizes))
#    hlDataset = VideoDataset(tr_labs, sizes, seq_size=SEQ_SIZE, is_train_set = True)
    
    #####################################################################
    
    # Feature Extraction : (IDT)
    
    # Get feats for only the training videos. Get ordered histograms of freq
    print("Trajectory Length : {}, nClusters : {} ".format(traj_len, cluster_size))
    
    #####################################################################
    # read into dictionary {vidname: np array, ...}
    print("Loading features from disk...")
    # get Nx4096 numpy matrix with columns as features and rows as window placement features
    if not os.path.exists(base_name):
        os.makedirs(base_name)
        #    # Read IDT features {with trajectory length = traj_len}
        features, strokes_name_id = read_partition_feats(DATASET, LABELS, 
                                                         IDT_FEATS+"_TrajLen"+str(traj_len),
                                                         train_lst, traj_len) 

        with open(os.path.join(base_name, "idt_feats_traj"+str(traj_len)+".pkl"), "wb") as fp:
            pickle.dump(features, fp)
        with open(os.path.join(base_name, "idt_snames_traj"+str(traj_len)+".pkl"), "wb") as fp:
            pickle.dump(strokes_name_id, fp)

    with open(os.path.join(base_name, "idt_feats_traj"+str(traj_len)+".pkl"), "rb") as fp:
        features = pickle.load(fp)
    with open(os.path.join(base_name, "idt_snames_traj"+str(traj_len)+".pkl"), "rb") as fp:
        strokes_name_id = pickle.load(fp)
    
    #####################################################################
    # get small sample of the IDT features and form a matrix (N, vec_size)
    vecs = []
    for key in sorted(list(features.keys())):
        vecs.append(features[key])
    vecs = np.vstack(vecs)
    
    vecs[np.isnan(vecs)] = 0
    vecs[np.isinf(vecs)] = 0
    
    # sample points for clustering 
    if vecs.shape[0] > MAX_SAMPLES:
        vecs = vecs[np.random.choice(vecs.shape[0], MAX_SAMPLES, replace=False), :]
    
    #fc7 layer output size (4096) 
    INP_VEC_SIZE = vecs.shape[-1]
    print("INP_VEC_SIZE = ", INP_VEC_SIZE)
    
    km_filepath = os.path.join(base_name, km_filename)
#    # Uncomment only while training.
    if not os.path.isfile(km_filepath+"_C"+str(cluster_size)+".pkl"):
        km_model = make_codebook(vecs, cluster_size)  #, model_type='gmm') 
        ##    # Save to disk, if training is performed
        print("Writing the KMeans models to disk...")
        pickle.dump(km_model, open(km_filepath+"_C"+str(cluster_size)+".pkl", "wb"))
    else:
        # Load from disk, for validation and test sets.
        km_model = pickle.load(open(km_filepath+"_C"+str(cluster_size)+".pkl", 'rb'))
    
    ###########################################################################
    # Form the training dataset for supervised classification 
    # Assign the words (flow frames) to their closest cluster centres and count the 
    # frequency for each document(video). Create IDF bow dataframe by weighting
    # df_train is (nVids, 50) for magnitude, with index as videonames
    
    print("Create a dataframe for HOOF features...")
    df_train, words_train = create_bovw_df(features, strokes_name_id, km_model,\
                                                base_name, "train")

    # read the stroke annotation labels from text file.
    vids_list = list(df_train.index)
    labs_keys, labs_values = get_cluster_labels(ANNOTATION_FILE)
    if min(labs_values) == 1:
        labs_values = [l-1 for l in labs_values]
        labs_keys = [k.replace('.avi', '') for k in labs_keys]
    train_labels = np.array([labs_values[labs_keys.index(v)] for v in vids_list])
    
    ###########################################################################
                
#    apply_clustering(df_train, DATASET, LABELS, ANNOTATION_FILE, base_name)

    ###########################################################################

    ###########################################################################
    # Train SVM
    clf = LinearSVC(verbose=False, random_state=124, max_iter=3000)
    clf.fit(df_train, train_labels)
    
    print("Training Complete.")
    ###########################################################################
#    # Train a classifier on the features.

##################################################################################

    # Evaluation on validation set
    print("Validation phase ....")
    
    if not os.path.isfile(os.path.join(base_name, "idt_feats_val_traj"+str(traj_len)+".pkl")):
        features_val, strokes_name_id_val = read_partition_feats(DATASET, LABELS, \
                                                                 IDT_FEATS+"_TrajLen"+str(traj_len), 
                                                                 val_lst, traj_len) 

        with open(os.path.join(base_name, "idt_feats_val_traj"+str(traj_len)+".pkl"), "wb") as fp:
            pickle.dump(features_val, fp)
        with open(os.path.join(base_name, "idt_snames_val_traj"+str(traj_len)+".pkl"), "wb") as fp:
            pickle.dump(strokes_name_id_val, fp)
    else:
        with open(os.path.join(base_name, "idt_feats_val_traj"+str(traj_len)+".pkl"), "rb") as fp:
            features_val = pickle.load(fp)
        with open(os.path.join(base_name, "idt_snames_val_traj"+str(traj_len)+".pkl"), "rb") as fp:
            strokes_name_id_val = pickle.load(fp)

    print("Create dataframe BOVW validation set...")
    df_val_hoof, words_val = create_bovw_df(features_val, strokes_name_id_val, \
                                            km_model, base_name, "val")
    
    vids_list_val = list(df_val_hoof.index)
    val_labels = np.array([labs_values[labs_keys.index(v)] for v in vids_list_val])
    
#    labs_df = pd.DataFrame(labels, index=vids_list, columns=['label'])
    print("Evaluating on the validation set...")
#    evaluate(model_mag, df_test_mag, labs_df)
    
    # Find maximum permutation accuracy using predicted_label_val and label_val
#    acc_values, perm_tuples, gt_list, pred_list = calculate_accuracy(val_labels, \
#                                                        predicted_label_val)
#    acc_perc = [sum(k)/len(predicted_label_val) for k in acc_values]
#    
#    best_indx = acc_perc.index(max(acc_perc))
#    print("Max Acc. : ", max(acc_perc))
#    print("Acc values : ", acc_perc)
#    print("Acc values : ", acc_values)
#    print("perm_tuples : ", perm_tuples)
    
    ###########################################################################
    # Evaluate the BOW classifier (SVM)
    confusion_mat = np.zeros((NUM_TOPICS, NUM_TOPICS))
    pred = clf.predict(df_val_hoof)
    correct = 0
    for i,true_val in enumerate(val_labels):
        if pred[i] == true_val:
            correct+=1
        confusion_mat[pred[i], true_val]+=1
    print('#'*30)
    print("BOW Classification Results:")
    print("%d/%d Correct" % (correct, len(pred)))
    print("Accuracy = {} ".format( float(correct) / len(pred)))
    print("Confusion matrix")
    print(confusion_mat)
    return (float(correct) / len(pred))

    ###########################################################################
        
def plot_accuracy(x, keys, l, xlab, ylab, fname):
#    keys = ["HOG", "HOOF", "OF Grid 20", "C3D $\mathit{FC7}$: $w_{c3d}=17$"]
#    l = {keys[0]: hog_acc, keys[1]: hoof_acc, keys[2]: of30_acc, keys[3]:accuracy_17_30ep}
    cols = ['r','g','b', 'c']        
    print(l)
    fig = plt.figure(2)
    plt.title("Accuracy Vs #Words", fontsize=12)
    for i in range(len(keys)):
        acc = l[keys[i]]
        plt.plot(x[(len(x)-len(acc)):], acc, lw=1, color=cols[i], marker='.', label=keys[i])
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend(loc=4)
    plt.ylim(bottom=0, top=1)
    plt.show()
    fig.savefig(fname, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return


if __name__ == '__main__':

    nclusters = []
    grids = []
    best_acc = []
    traj = 25
#    nbins = list(range(10, 41, 10))
    cluster_sizes = list(range(1000, 1001, 10))
    keys = ["IDT Traj_Len:"+str(traj)]
#    keys = ["ofGrid:"+str(t) for t in nbins]
#    beta = [0.6, 1.0, 1.0, 1.0]
#    keys = []
#    for i, t in enumerate(nbins):
#        keys.append("ofGrid:"+str(t)+" :: $\\beta="+str(beta[i])+"$")
    l = {}
    
#    for i, grid in enumerate(nbins):
    accuracies = []
    for cluster_size in cluster_sizes:
        folder_name = "bow_HL_idt_full"+str(traj)
        nclusters.append(cluster_size)
        grids.append(traj)
        acc = main(os.path.join(base_path, folder_name), traj, cluster_size)
        accuracies.append(acc)
        best_acc.append(acc)
    l[keys[0]] = accuracies
    
#    fname = os.path.join(base_path, "ofGrid_SA_cl150_old.png")
#    plot_accuracy(cluster_sizes, keys, l, "#Words", "Accuracy", fname)
#
#    df = pd.DataFrame({"#Clusters (#Words)":nclusters, "#Grid(g)": grids, "Accuracy(percent)":best_acc})
#    df = df.pivot("#Clusters (#Words)", "#Grid(g)", "Accuracy(percent)")
#    normal_heat = sns.heatmap(df, vmin=0., vmax=1., annot=True, fmt='.4f')
#    normal_heat.figure.savefig(os.path.join(base_path, folder_name, 
#                                            "of_clusters_3classes_HA_old.png"))
    
    # "of_clusters_3classes_HA_old.png" or "..._old.png" graphs denote HA BOVW when
    # the freq of the words in IDF is computed by thresholding and not simply summing the
    # rows of the matrix.
    
    ###########################################################################
    
#    for i, grid in enumerate(nbins):
#        accuracies = []
#        for cluster_size in cluster_sizes:
#            folder_name = "bow_HL_hoof_b"+str(grid)+"_mth"+str(mth)
#            #folder_name = "bow_HL_3dres_seq16_cl20"
##            folder_name = "bow_HL_ofAng_grid"+str(grid)
#            nclusters.append(cluster_size)
#            grids.append(grid)
#            acc = main(os.path.join(base_path, folder_name), grid, None, cluster_size)
#            accuracies.append(acc)
#            best_acc.append(acc)
#        
#        l[keys[i]] = accuracies
#        
#    print(l)
#        
#    fname = os.path.join(base_path, "hoof_HA_mth2_cl150_old.png")
#    plot_accuracy(cluster_sizes, keys, l, "#Words", "Accuracy", fname)
        
#    df = pd.DataFrame({"#Clusters (#Words)":nclusters, "#nBins(b)": grids, "Accuracy(percent)":best_acc})
#    df = df.pivot("#Clusters (#Words)", "#nBins(b)", "Accuracy(percent)")
#    normal_heat = sns.heatmap(df, vmin=0., vmax=1., annot=True, fmt='.4f')
#    normal_heat.figure.savefig(os.path.join(base_path, folder_name, 
#                                            "hoof_clusters_5classes_HA.png"))
    
    ###########################################################################
