#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 14:29:10 2019

@author: arpan

@Description: main file for calling the 
"""

import numpy as np
import pandas as pd
import os
import pickle
import gensim
from gensim.corpora import Dictionary
from gensim import corpora
from sklearn.externals import joblib
from extract_hoof_feats import extract_stroke_feats
from read_c3d_feats import select_trimmed_feats
from create_bovw import make_codebook
from create_bovw import create_bovw_df
from evaluate import get_cluster_labels
from evaluate import calculate_accuracy
from sklearn.svm import LinearSVC
import seaborn as sns
sns.set()

import bovw_utils as utils

np.seterr(divide='ignore', invalid='ignore')

# Paths and parameters
#GRID_SIZE=30
km_filename = "km_bow"
mnb_modelname='lda_model'
#cluster_size=50
real_topic=3
NUM_TOPICS = 3
ANNOTATION_FILE = "stroke_labels.txt"
#c3dWinSize = 17
nbins = 40
mth = 2


# Local Paths
LABELS = "/home/arpan/VisionWorkspace/Cricket/scripts/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
DATASET = "/home/arpan/VisionWorkspace/VideoData/sample_cricket/ICC WT20"
MAIN_DATASET = "/home/arpan/VisionWorkspace/Cricket/dataset_25_fps_train_set"
MAIN_LABELS = "/home/arpan/VisionWorkspace/Cricket/dataset_25_fps_train_set_labels"
VAL_DATASET = "/home/arpan/VisionWorkspace/Cricket/dataset_25_fps_val_set"
VAL_LABELS = "/home/arpan/VisionWorkspace/Cricket/dataset_25_fps_val_set_labels"
#c3dFC7FeatsPath = "/home/arpan/VisionWorkspace/Cricket/localization_gru/c3dFinetuned_feats_"+str(c3dWinSize)
#c3dFC7MainFeatsPath = "/home/arpan/VisionWorkspace/Cricket/localization_gru/c3dFinetunedOnHLMainSeq23_mainDataset_train_feats_"+str(c3dWinSize)
#c3dFC7ValFeatsPath = "/home/arpan/VisionWorkspace/Cricket/localization_gru/c3dFinetunedOnHLMainSeq23_mainDataset_test_feats_"+str(c3dWinSize)
base_path = "/home/arpan/VisionWorkspace/Cricket/CricketStrokeLocalizationBOVW/logs"
#base_name = "/home/arpan/VisionWorkspace/Cricket/CricketStrokeLocalizationBOVW/logs/lda_HL_hoofNormed_b"+str(nbins)+"_mth"+str(mth)

# Server Paths
if os.path.exists("/opt/datasets/cricket/ICC_WT20"):
    LABELS = "/home/arpan/VisionWorkspace/shot_detection/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
    DATASET = "/opt/datasets/cricket/ICC_WT20"
    MAIN_DATASET = "/home/arpan/DATA_Drive/Cricket/dataset_25_fps_train_set"
    MAIN_LABELS = "/home/arpan/DATA_Drive/Cricket/dataset_25_fps_train_set_labels"
    VAL_DATASET = "/home/arpan/DATA_Drive/Cricket/dataset_25_fps_val_set"
    VAL_LABELS = "/home/arpan/DATA_Drive/Cricket/dataset_25_fps_val_set_labels"
#    c3dFC7FeatsPath = "/home/arpan/DATA_Drive/Cricket/extracted_feats/c3dFinetuned_feats_"+str(c3dWinSize)
#    c3dFC7MainFeatsPath = "/home/arpan/DATA_Drive/Cricket/extracted_feats/c3dFinetunedOnHLMainSeq23_mainDataset_train_feats_"+str(c3dWinSize)
#    c3dFC7ValFeatsPath = "/home/arpan/DATA_Drive/Cricket/extracted_feats/c3dFinetunedOnHLMainSeq23_mainDataset_test_feats_"+str(c3dWinSize)
    base_path = "/home/arpan/DATA_Drive/Cricket/Workspace/CricketStrokeLocalizationBOVW/logs"



f = None
# save video according to their label
#def write_strokes(labels, bins, thresh):
#    global f
#    flows_path = get_ordered_strokes_list()
#
#    #flows_path = sorted(os.listdir(flows_numpy_path))
#    n_clusters = max(labels) + 1
#    print("clusters, ", n_clusters)
#    for i in range(n_clusters):
#        ######
#        try:
#            os.makedirs(os.path.join(base_name, "bins_"+str(bins)+"_th_"+str(thresh), str(i)))
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
#    eval_path = os.path.join(base_name, "bins_"+str(bins)+"_th_"+str(thresh), str(label))
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


def main(base_name, GRID_SIZE, cluster_size): # main(base_name, c3dWinSize=16, use_gpu=False):
#if __name__ == '__main__':
    """
    Function to extract orientation features and find the directions of strokes, 
    using LDA model/clustering and evaluate on three cluster analysis on highlights.
    The videos can be visualized by writing trimmed class videos into their respective
    classes.
    
    Parameters:
    ------
    
    base_name: path to the wts, losses, predictions and log files
    use_gpu: True if training to be done on GPU, False for CPU
    
    """

    if not os.path.exists(base_name):
        os.makedirs(base_name)
    
    seed = 1234
    
    print(60*"#")
          
    #####################################################################
    
    # Form dataloaders 
#    train_lst_main_ext = get_main_dataset_files(MAIN_DATASET)   #with extensions
#    train_lst_main = [t.rsplit('.', 1)[0] for t in train_lst_main_ext]   # remove the extension
#    val_lst_main_ext = get_main_dataset_files(VAL_DATASET)
#    val_lst_main = [t.rsplit('.', 1)[0] for t in val_lst_main_ext]
    
    # Divide the samples files into training set, validation and test sets
    train_lst, val_lst, test_lst = utils.split_dataset_files(DATASET)
    #print("c3dWinSize : {}".format(c3dWinSize))
    
    # form the names of the list of label files, should be at destination 
    train_lab = [f+".json" for f in train_lst]
    val_lab = [f+".json" for f in val_lst]
    test_lab = [f+".json" for f in test_lst]
#    train_lab_main = [f+".json" for f in train_lst_main]
#    val_lab_main = [f+".json" for f in val_lst_main]
    
    # get complete path lists of label files
    tr_labs = [os.path.join(LABELS, f) for f in train_lab]
    val_labs = [os.path.join(LABELS, f) for f in val_lab]
#    tr_labs_main = [os.path.join(MAIN_LABELS, f) for f in train_lab_main]
#    val_labs_main = [os.path.join(VAL_LABELS, f) for f in val_lab_main]
    #####################################################################
    
    sizes = [utils.getNFrames(os.path.join(DATASET, f+".avi")) for f in train_lst]
    val_sizes = [utils.getNFrames(os.path.join(DATASET, f+".avi")) for f in val_lst]
#    sizes_main = [utils.getNFrames(os.path.join(MAIN_DATASET, f)) for f in train_lst_main_ext]
#    val_sizes_main = [utils.getNFrames(os.path.join(VAL_DATASET, f)) for f in val_lst_main_ext]
    
    ###########################################################################
    # Merge the training highlights and main dataset variables
#    train_lab.extend(train_lab_main)
#    tr_labs.extend(tr_labs_main)
#    sizes.extend(sizes_main)
    
    print("No. of training videos : {}".format(len(train_lst)))
    
    print("Size : {}".format(sizes))
#    hlDataset = VideoDataset(tr_labs, sizes, seq_size=SEQ_SIZE, is_train_set = True)
#    print(hlDataset.__len__())
    
    #####################################################################
    
    # Read the trimmed videos of training set and extract features
    # Extract HOOF features 
    
    # Get feats for only the training videos. Get ordered histograms of freq
    print("GRID : {}, nClusters : {} ".format(GRID_SIZE, cluster_size))
    
    #####################################################################
    # read into dictionary {vidname: np array, ...}
    print("Loading features from disk...")
    #features = utils.readAllPartitionFeatures(c3dFC7FeatsPath, train_lst)
#    mainFeatures = utils.readAllPartitionFeatures(c3dFC7MainFeatsPath, train_lst_main)
#    features.update(mainFeatures)     # Merge dicts
    # get Nx4096 numpy matrix with columns as features and rows as window placement features
#    features, strokes_name_id = select_trimmed_feats(c3dFC7FeatsPath, LABELS, \
#                                    train_lst, c3dWinSize) 
    features, strokes_name_id = extract_stroke_feats(DATASET, LABELS, train_lst, \
                                                     nbins, mth, True, GRID_SIZE) 
#    with open(os.path.join(base_name, "ang_feats.pkl"), "wb") as fp:
#        pickle.dump(features, fp)
        
    with open(os.path.join(base_name, "ang_feats.pkl"), "rb") as fp:
        features = pickle.load(fp)
#   
    #####################################################################
    # get matrix of features from dictionary (N, vec_size)
    vecs = []
    for key in sorted(list(features.keys())):
        vecs.append(features[key])
    vecs = np.vstack(vecs)
    
    vecs[np.isnan(vecs)] = 0
    vecs[np.isinf(vecs)] = 0
    
    
    #fc7 layer output size (4096) 
    INP_VEC_SIZE = vecs.shape[-1]
    print("INP_VEC_SIZE = ", INP_VEC_SIZE)
    
    km_filepath = os.path.join(base_name, km_filename+".pkl")
#    # Uncomment only while training.
    km_model = make_codebook(vecs, cluster_size) 
#    # Save to disk, if training is performed
#    print("Writing the KMeans models to disk...")
    pickle.dump(km_model, open(km_filepath+"_C"+str(cluster_size), "wb"))
    # Load from disk, for validation and test sets.
    km_model = pickle.load(open(km_filepath+"_C"+str(cluster_size), 'rb'))
    
    ###########################################################################
    # Form the training dataset for supervised classification 
    # Assign the words (flow frames) to their closest cluster centres and count the 
    # frequency for each document(video). Create IDF bow dataframe by weighting
    # df_train is (nVids, 50) for magnitude, with index as videonames
#    print("Create a dataframe for C3D FC7 features...")
#    df_train_c3d, words_train = create_bovw_c3d_traindf(features, \
#                                strokes_name_id, km_model, c3dWinSize)
    
    print("Create a dataframe for HOOF features...")
    df_train, words_train = create_bovw_df(features, strokes_name_id, km_model,\
                                                base_name, "train")

    # read the stroke annotation labels from text file.
    vids_list = list(df_train.index)
    labs_keys, labs_values = get_cluster_labels(ANNOTATION_FILE)
    train_labels = np.array([labs_values[labs_keys.index(v)] for v in vids_list])
    
    print("Training stroke labels : ")
    print(train_labels)
    print(train_labels.shape)
    
    # concat dataframe to contain features and corresponding labels
    #df_train = pd.concat([df_train_mag, labs_df], axis=1)
    
    ###########################################################################
    # Train SVM
    clf_ang = LinearSVC(verbose=True, random_state=124, max_iter=3000)
    clf_ang.fit(df_train, train_labels)
    joblib.dump(clf_ang, os.path.join(base_name, "svm_ang.pkl"))
    
    clf_ang = joblib.load(os.path.join(base_name, "svm_ang.pkl"))
    
    print("Training dataframe formed.")
    ###########################################################################
    # Train a classifier on the features.
    print("LDA execution !!! ")
    #Run LDA
    
    # Get list of lists. Each sublist contains video cluster strIDs (words). 
    # Eg. [["39","29","39","39","0", ...], ...]
    doc_clean = [doc.split() for doc in words_train]
    #print(doc_clean)
    diction=corpora.Dictionary(doc_clean)    # Form a dictionary
    print("printing dictionary after corp  {} ".format(diction))
    doc_term_matrix = [diction.doc2bow(doc) for doc in doc_clean]
    #dictionary = corpora.Dictionary(diction)
    
    # Inference using the data.
    ldamodel_obj = gensim.models.ldamodel.LdaModel(doc_term_matrix, \
                    num_topics = NUM_TOPICS, id2word=diction, passes=10, \
                    random_state=seed)
#    ldamodel_obj = gensim.models.ldaseqmodel.LdaSeqModel(doc_term_matrix, \
#                                        num_topics=3, time_slice=[351])
#    ldamodel_obj = gensim.models.LsiModel(doc_term_matrix, num_topics=3, \
#                                          id2word = diction)

    print("training complete saving to disk ")
    #save model to disk 
    joblib.dump(ldamodel_obj, os.path.join(base_name, mnb_modelname+".pkl"))

    # Load trained model from disk
    ldamodel_obj = joblib.load(os.path.join(base_name, mnb_modelname+".pkl"))
    
    # Print all the topics
    for i,topic in enumerate(ldamodel_obj.print_topics(num_topics=3, num_words=10)):
        #print("topic is {}".format(topic))
        word = topic[1].split("+")
        print("{} : {} ".format(topic[0], word))
        
    # actions are rows and discovered topics are columns
    topic_action_map = np.zeros((real_topic, NUM_TOPICS))
    
    predicted_labels = []
    #vids_list = list(df_train_mag.index)
    for j,vname in enumerate(vids_list):
        label_vid = train_labels[j]
        # sort the tuples with descending topic probabilities
        for index, score in sorted(ldamodel_obj[doc_term_matrix[j]], key=lambda tup: -1*tup[1]):
#        for index in [ldamodel_obj[doc_term_matrix[j]].argmax(axis=0)]:
         #   print("Score is : {} of Topic: {}".format(score,index))
            #if score>0.5:
            #    topic_action_map[label_vid][index]+=1
#            score = ldamodel_obj[doc_term_matrix[j]][index]
            topic_action_map[label_vid][index]+=score
            predicted_labels.append(index)  
            break
    print("Training Time : topic action mapping is : ")
    print("topic0  topic1  topic2")
    #coloumn are topics and rows are labels
    print(topic_action_map)
    acc_values_tr, perm_tuples_tr, gt_list, pred_list = calculate_accuracy(train_labels,\
                                                            predicted_labels)
    acc_perc = [sum(k)/len(predicted_labels) for k in acc_values_tr]
    
    best_indx = acc_perc.index(max(acc_perc))
    print("Max Acc. : ", max(acc_perc))
    print("Acc values : ", acc_perc)
    print("Acc values : ", acc_values_tr)
    print("perm_tuples : ", perm_tuples_tr)
    
    #model_ang = joblib.load(os.path.join(destpath, mnb_modelname+"_ang.pkl"))
##################################################################################

    # Evaluation on validation set

#    features_val, strokes_name_id_val = select_trimmed_feats(c3dFC7FeatsPath, val_lst, \
#                                                     c3dWinSize) 
    features_val, strokes_name_id_val = extract_stroke_feats(DATASET, LABELS, val_lst, \
                                                     nbins, mth, True, GRID_SIZE) 
#    with open(os.path.join(base_name, "ang_feats_val.pkl"), "wb") as fp:
#        pickle.dump(features_val, fp)

    with open(os.path.join(base_name, "ang_feats_val.pkl"), "rb") as fp:
        features_val = pickle.load(fp)

    print("Create dataframe BOVW validation set...")
    df_val_hoof, words_val = create_bovw_df(features_val, strokes_name_id_val, \
                                            km_model, base_name, "val")
    
    vids_list_val = list(df_val_hoof.index)
    val_labels = np.array([labs_values[labs_keys.index(v)] for v in vids_list_val])
    
    topic_action_map_val = np.zeros((real_topic, NUM_TOPICS))
    doc_clean_val = [doc.split() for doc in words_val]
    # Creating Dictionary for val set words
    diction_val=corpora.Dictionary(doc_clean_val)
    
    doc_term_matrix_val = [diction_val.doc2bow(doc) for doc in doc_clean_val]
    predicted_label_val = []
    for j,vname in enumerate(vids_list_val):
        label_vid = val_labels[j]
        for index, score in sorted(ldamodel_obj[doc_term_matrix_val[j]], key=lambda tup: -1*tup[1]):
#        for index in [ldamodel_obj[doc_term_matrix[j]].argmax(axis=0)]:
#            score = ldamodel_obj[doc_term_matrix[j]][index]
         #   print("Score is : {} of Topic: {}".format(score,index))
            #if score>0.5:
            #    topic_action_map_val[label_vid][index]+=1
            topic_action_map_val[label_vid][index]+=score
            predicted_label_val.append(index)  
            break
            
    print(topic_action_map_val)
    
#    labs_df = pd.DataFrame(labels, index=vids_list, columns=['label'])
#    
    print("Evaluating on the validation set...")
#    evaluate(model_mag, df_test_mag, labs_df)
    
    # Find maximum permutation accuracy using predicted_label_val and label_val
    acc_values, perm_tuples, gt_list, pred_list = calculate_accuracy(val_labels, \
                                                        predicted_label_val)
    acc_perc = [sum(k)/len(predicted_label_val) for k in acc_values]
    
    best_indx = acc_perc.index(max(acc_perc))
    print("Max Acc. : ", max(acc_perc))
    print("Acc values : ", acc_perc)
    print("Acc values : ", acc_values)
    print("perm_tuples : ", perm_tuples)
    
    ###########################################################################
    # Evaluate the BOW classifier (SVM)
    confusion_mat = np.zeros((NUM_TOPICS, NUM_TOPICS))
    pred = clf_ang.predict(df_val_hoof)
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
    
#def visualize_topics(ldamodel_obj):
    

if __name__ == '__main__':

    nclusters = []
    grids = []
    best_acc = []
    
    best_acc = [0.8, 0.780952380952381, 0.780952380952381, 0.819047619047619, \
                0.7428571428571429, 0.7904761904761904, \
                0.8, 0.780952380952381, 0.780952380952381, 0.819047619047619, \
                0.780952380952381, 0.8285714285714286, 0.7714285714285715, \
                0.8, 0.819047619047619, 0.7714285714285715, 0.7428571428571429, \
                0.7904761904761904, 0.7904761904761904, 0.7714285714285715]
    
    #cluster_size = 50
    for grid in list(range(10, 41, 10)):
        for cluster_size in list(range(10, 51, 10)):
            #folder_name = "lda_HL_hoofNormed_b"+str(nbins)+"_mth"+str(mth)
            folder_name = "bow_HL_ofAng_grid"+str(grid)
            nclusters.append(cluster_size)
            grids.append(grid)
#            best_acc.append(main(os.path.join(base_path, folder_name), grid, cluster_size))
        
    
    
    df = pd.DataFrame({"#Clusters (#Words)":nclusters, "Grid(g)": grids, "Accuracy(percent)":best_acc})
    df = df.pivot("#Clusters (#Words)", "Grid(g)", "Accuracy(percent)")
    normal_heat = sns.heatmap(df, vmin=0., vmax=1., annot=True, fmt='.2f')
    normal_heat.figure.savefig(os.path.join(base_path, "normal_heat_hoof_clust.png"))

