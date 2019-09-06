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
from create_bovw import create_bovw_traindf
from evaluate import get_cluster_labels
from evaluate import calculate_accuracy

import bovw_utils as utils
import cv2
import sys

# Paths and parameters
GRID_SIZE=30
km_filename = "km_hoof"
mnb_modelname='lda_mod_hoof'
cluster_size=50
real_topic=3
NUM_TOPICS = 3
ANNOTATION_FILE = "stroke_labels.txt"
#c3dWinSize = 17
nbins = 30
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
#base_name = "/home/arpan/VisionWorkspace/Cricket/CricketStrokeLocalizationBOVW/logs/lda_HL_ofGridUnnormed_grid"+str(GRID_SIZE)
base_name = "/home/arpan/VisionWorkspace/Cricket/CricketStrokeLocalizationBOVW/logs/lda_HL_hoofNormed_b"+str(nbins)+"_mth"+str(mth)

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
    base_name = "/home/arpan/VisionWorkspace/localization_gru/logs/lda_HL_hoof"



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

            
def get_frame(cap, frame_no):
    # get total number of frames
    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # check for valid frame number
    if frame_no >= 0 & frame_no <= totalFrames:
        # set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame_no)
        _, img = cap.read()
        return img
    print("invalid frame, ", frame_no)
    sys.exit()

def save_video(filename, m, n, label, bins, thresh):
    global f
    eval_path = os.path.join(base_name, "bins_"+str(bins)+"_th_"+str(thresh), str(label))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vid_out_path = os.path.join(eval_path, f+'.avi')
    out = cv2.VideoWriter(vid_out_path, fourcc, 25.0, (320, 180), True)
    cap = cv2.VideoCapture(filename)
    if cap.isOpened():
        pass
    else:
        print("closed")
        sys.exit()
    for i in range(m, n+1):
        img = cv2.resize(get_frame(cap, i), (320, 180), interpolation=cv2.INTER_AREA)
        out.write(img)
    cap.release()
    out.release()


if __name__ == "__main__": # main(base_name, c3dWinSize=16, use_gpu=False):
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
    
    seed = 1236
    
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
#    all_feats[np.isinf(all_feats)] = 0
    print("bin:{}, thresh:{} ".format(nbins, mth))    
    
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
                                                     nbins, mth, GRID_SIZE) 
#    with open(os.path.join(base_name, "feats.pkl"), "wb") as fp:
#        pickle.dump(features, fp)
        
    with open(os.path.join(base_name, "feats.pkl"), "rb") as fp:
        features = pickle.load(fp)
#   
    #####################################################################
    # get matrix of features from dictionary (N, vec_size)
    vecs = []
    for key in sorted(list(features.keys())):
        vecs.append(features[key])
    vecs = np.vstack(vecs)
    #fc7 layer output size (4096) 
    INP_VEC_SIZE = vecs.shape[-1]
    print("INP_VEC_SIZE = ", INP_VEC_SIZE)
    
    km_filepath = os.path.join(base_name, km_filename+".pkl")
#    # Uncomment only while training.
    km_model = make_codebook(vecs, cluster_size) 
#    # Save to disk, if training is performed
#    print("Writing the KMeans models to disk...")
    pickle.dump(km_model, open(km_filepath, "wb"))
    # Load from disk, for validation and test sets.
    km_model = pickle.load(open(km_filepath, 'rb'))
    
    ###########################################################################
    # Form the training dataset for supervised classification 
    # Assign the words (flow frames) to their closest cluster centres and count the 
    # frequency for each document(video). Create IDF bow dataframe by weighting
    # df_train is (nVids, 50) for magnitude, with index as videonames
#    print("Create a dataframe for C3D FC7 features...")
#    df_train_c3d, words_train = create_bovw_c3d_traindf(features, \
#                                strokes_name_id, km_model, c3dWinSize)
    
    print("Create a dataframe for HOOF features...")
    df_train, words_train = create_bovw_traindf(features, strokes_name_id, km_model)

    # read the stroke annotation labels from text file.
    vids_list = list(df_train.index)
    labs_keys, labs_values = get_cluster_labels(ANNOTATION_FILE)
    train_labels = np.array([labs_values[labs_keys.index(v)] for v in vids_list])
    
    print("Training stroke labels : ")
    print(train_labels)
    print(train_labels.shape)
    
    # concat dataframe to contain features and corresponding labels
    #df_train = pd.concat([df_train_mag, labs_df], axis=1)
    
    print("Training dataframe formed.")
    ###########################################################################
    labs_df = pd.DataFrame(train_labels, index=vids_list, columns=['label'])
    # Train a classifier on the features.
    print("Training with SVM (ang)")
    #clf_ang = RandomForestClassifier(max_depth=5, n_estimators=1000, random_state=124)
    #clf_ang = SVC(kernel="linear",verbose=True)
    clf_ang = LinearSVC(verbose=True, random_state=124, max_iter=2000)
    clf_ang.fit(df_train, train_labels)
    
    #print("Training complete. Saving to disk.")
    # Save model to disk
    joblib.dump(clf_ang, os.path.join(base_name, "clf_ang.pkl"))
    # Load trained model from disk
    clf_ang = joblib.load(os.path.join(base_name, "clf_ang.pkl"))

    # Train a classifier on both the features.
    #print("Training with SVM")
    #df_train = pd.concat([df_train_mag, df_train_ang], axis=1)
    #clf_both = SVC(kernel="linear",verbose=True)
    #clf_both = LinearSVC(verbose=True, random_state=123, max_iter=2000)
    #clf_both.fit(df_train, labels)
    #print("Training with SVM (ang)")
    #clf_ang = SVC(kernel="linear",verbose=True)
    #clf_ang.fit(df_train_ang, labels)
    
    print("Eval on train set mag, ang and both")
    evaluate(clf_ang, df_train, labs_df)
    
    ###########################################################################
    # Evaluation on validation set
    # extract the optical flow information from the validation set videos and form dictionary
    #bgthresh = 70000 # should be <=90k, to prevent error vq(mag, clusters_mag[0])
    # 
#    target_file = os.path.join(destpath, flow_filename+"_test_BG"+str(bgthresh)+".pkl")
#    features_val = extract_flow_val(DATASET, bgthresh, grid_size=gdsize, partition="testing")
#    with open(target_file, "wb") as outfile:
#        pickle.dump(features_val, outfile)

    # Load feaures from disk
    with open(target_file, "rb") as infile:
        features_val = pickle.load(infile)
    
    features_val, strokes_name_id_val = extract_stroke_feats(DATASET, LABELS, val_lst, \
                                                     nbins, mth, GRID_SIZE) 
    
    print("Create dataframe BOVW validation set (mag)")
    df_test_mag, df_test_ang = create_bovw_testdf(features_val, km_mag, km_ang, \
                                                  mag_min, mag_max, ang_min, ang_max,\
                                                  destpath)
    vids_list = list(df_test_mag.index)
    labels = np.array([get_video_label(v) for v in vids_list])
    labs_df = pd.DataFrame(labels, index=vids_list, columns=['label'])
    
    print("Evaluating on the validation set (mag)")
    evaluate(clf_mag, df_test_mag, labs_df)
    
    print("Evaluating on the validation set (ang)")
    evaluate(clf_ang, df_test_ang, labs_df)

    print("Evaluating on the validation set (both features)")
    df_test = pd.concat([df_test_mag, df_test_ang], axis=1)
    evaluate(clf_both, df_test, labs_df)
    
    
    
    
    
    
    
    print("Training dataframe formed.")
    ###########################################################################
    # Train a classifier on the features.
    print("LDA execution !!! ")
#Run LDA

    #clusters_mag = km_mag.cluster_centers_
    #print(type(clusters_mag))
    #cluster_dict={}
    #cluster_str=[]
    #for i,j in enumerate(clusters_mag):
    #    cluster_dict[str(i)]=j
    #    cluster_str.append(str(i))
           
    #clusters_mag = clusters_mag.tostring()
    ##print(cluster_str)
    #cl=[cluster_str]
    
    # Get list of lists. Each sublist contains video cluster strIDs (words). Eg. [["39","29","39","39","0", ...], ...]
    doc_clean = [doc.split() for doc in words_train]
    #print(doc_clean)
    diction=corpora.Dictionary(doc_clean)    # Form a dictionary
    print("printing dictionary after corp  {} ".format(diction))
    doc_term_matrix = [diction.doc2bow(doc) for doc in doc_clean]
    #dictionary = corpora.Dictionary(diction)
    #print(df_train_mag)
    
    # Inference using the data.
    ldamodel_obj = gensim.models.ldamodel.LdaModel(doc_term_matrix, \
                    num_topics = NUM_TOPICS, id2word=diction, passes=10, random_state=seed)

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
         #   print("Score is : {} of Topic: {}".format(score,index))
            #if score>0.5:
            #    topic_action_map[label_vid][index]+=1
            topic_action_map[label_vid][index]+=score
            predicted_labels.append(index)  
            break
    print("topic action mapping is : ")
    print("topic0  topic1  topic2")
    #coloumn are topics and rows are labels
    print(topic_action_map)
    acc_values_tr, perm_tuples_tr, gt_list, pred_list = calculate_accuracy(train_labels, predicted_labels)
    acc_perc = [sum(k)/len(predicted_labels) for k in acc_values_tr]
    
    best_indx = acc_perc.index(max(acc_perc))
    print("Max Acc. : ", max(acc_perc))
    print("Acc values : ", acc_perc)
    print("Acc values : ", acc_values_tr)
    print("perm_tuples : ", perm_tuples_tr)
    
    #model_ang = joblib.load(os.path.join(destpath, mnb_modelname+"_ang.pkl"))
#####################################################################################################################################1

    # Evaluation on validation set
    
    #bgthresh = 0
    #target_file = os.path.join(base_name, flow_filename+"_val_BG"+str(bgthresh)+".pkl")

    #pickle.dump(features_val, open(target_file, "wb"))
#    features_val, strokes_name_id_val = select_trimmed_feats(c3dFC7FeatsPath, val_lst, \
#                                                     c3dWinSize) 
    features_val, strokes_name_id_val = extract_stroke_feats(DATASET, LABELS, val_lst, \
                                                     nbins, mth, GRID_SIZE) 
#    with open(os.path.join(base_name, "feats_val.pkl"), "wb") as fp:
#        pickle.dump(features_val, fp)

    with open(os.path.join(base_name, "feats_val.pkl"), "rb") as fp:
        features_val = pickle.load(fp)

    vecs = []
    for key in list(features_val.keys()):
        vecs.append(features_val[key])
    vecs = np.vstack(vecs)

    print("Create dataframe BOVW validation set")
    df_val_hoof, words_val = create_bovw_traindf(features_val, strokes_name_id_val, km_model)
    
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
         #   print("Score is : {} of Topic: {}".format(score,index))
            #if score>0.5:
            #    topic_action_map_val[label_vid][index]+=1
            topic_action_map_val[label_vid][index]+=score
            predicted_label_val.append(index)  
            break
            
    print(topic_action_map_val)
    
#    labs_df = pd.DataFrame(labels, index=vids_list, columns=['label'])
#    
#    print("Evaluating on the validation set (mag)")
#    evaluate(model_mag, df_test_mag, labs_df)
    
    # Find maximum permutation accuracy using predicted_label_val and label_val
    acc_values, perm_tuples, gt_list, pred_list = calculate_accuracy(val_labels, predicted_label_val)
    acc_perc = [sum(k)/len(predicted_label_val) for k in acc_values]
    
    best_indx = acc_perc.index(max(acc_perc))
    print("Max Acc. : ", max(acc_perc))
    print("Acc values : ", acc_perc)
    print("Acc values : ", acc_values)
    print("perm_tuples : ", perm_tuples)
    
    ###########################################################################
    
#    evaluate

    ###########################################################################


#if __name__=='__main__':
#    
##    SEQ_SIZE = 16   # has to >=c3dWinSize (ie. the number of frames used for c3d input)
##    BATCH_SIZE = 256
##    # Parameters and DataLoaders
##    HIDDEN_SIZE = 1000
##    N_EPOCHS = 90
##    N_LAYERS = 1        # no of hidden layers
##    threshold = 0.5
##    seq_threshold = 0.5
##    use_gpu = torch.cuda.is_available()
#    #use_gpu = True
#    
#    description = "Script for training RNNs on C3D FC7 features"
#    p = argparse.ArgumentParser(description=description)
#    
##    p.add_argument('-ds', '--DATASET', type=str, default=DATASET,
##                   help=('input directory containing input videos'))
##    p.add_argument('-labs', '--LABELS', type=str, default=LABELS,
##                   help=('output directory for c3d feats'))
##    p.add_argument('-feats', '--featuresPath', type=str, default=c3dFC7FeatsPath,
##                   help=('extracted c3d FC7 features path'))
#    p.add_argument('-dest', '--base_name', type=str, default=base_name, 
#                   help=('wts, losses and log file path'))
#    
#    p.add_argument('-c3dwin', '--c3dWinSize', type=int, default=c3dWinSize)
##    p.add_argument('-w', '--SEQ_SIZE', type=int, default=SEQ_SIZE)
##    p.add_argument('-b', '--BATCH_SIZE', type=int, default=BATCH_SIZE)
##    p.add_argument('-hs', '--HIDDEN_SIZE', type=int, default=HIDDEN_SIZE)
##    p.add_argument('-n', '--N_EPOCHS', type=int, default=N_EPOCHS)
##    p.add_argument('-l', '--N_LAYERS', type=int, default=N_LAYERS)    
##    p.add_argument('-t', '--threshold', type=int, default=threshold)
##    p.add_argument('-s', '--seq_threshold', type=int, default=seq_threshold)
##    p.add_argument('-g', '--use_gpu', type=bool, default=use_gpu)
#    
#    # create dictionary of tiou values and save to destination 
#    tiou_dict = {}
#    
#    for seq in range(33, 34):
#        #p.set_defaults(SEQ_SIZE = seq)
#        main(**vars(p.parse_args()))
#        #tiou_dict[seq] = tiou
#    
#    dest = vars(p.parse_args())['base_name']
#    print("Topics found !!")
    
#    #####################################################################
    ###########################################################################
    # Step 1: Extract optical flow from training videos and save to disk
    # calculate optical flow vectors of training dataset
#    print("Extract optical flow data for training set...")
#    
#    #features = extract_flow_seq_train(DATASET, hist_bins=np.linspace(0, 7060, (nbins+1)), \
#    #                                  mag_thresh=2)
#    #pickle.dump(features, open(flow_filepath, "wb"))
#    print("Written training features to disk...")
#    features = pickle.load(open(flow_filepath, 'rb'))
#    
#    # extract keypoints (optical flow vectors of consecutive frames for
#    # all videos in training dataset)
#    #mag, ang = extract_vec_points(flow_filepath)
#    #mag = extract_vec_points(flow_filepath)
#    mag = extract_hoof_points(flow_filepath)
#    # change -inf value to 0,  
#    mag[np.isinf(mag)] = 0

