#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 00:29:27 2018

@author: Arpan
@Description: Use RNN/LSTM on sequence of features extracted from frames for action 
localization of cricket strokes on Highlight videos dataset.
"""

import torch
import os

from torch.utils.data import DataLoader
#import matplotlib.pyplot as plt
import pickle
import time
import utils
import torch.nn as nn
import numpy as np
import pandas as pd

from torch.autograd import Variable
from Video_Dataset import VideoDataset
from model_gru import RNNClassifier
from sklearn.cluster import KMeans
from scipy.cluster.vq import vq

torch.manual_seed(777)  # reproducibility

# Local Paths
LABELS = "/home/arpan/VisionWorkspace/Cricket/scripts/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
DATASET = "/home/arpan/VisionWorkspace/VideoData/sample_cricket/ICC WT20"

# Server Paths
if os.path.exists("/opt/datasets/cricket/ICC_WT20"):
    LABELS = "/home/arpan/VisionWorkspace/shot_detection/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
    DATASET = "/opt/datasets/cricket/ICC_WT20"

THRESHOLD = 0.5
# Parameters and DataLoaders
HIDDEN_SIZE = 1000
N_LAYERS = 1
BATCH_SIZE = 256
N_EPOCHS = 10
INP_VEC_SIZE = 1152      # taking grid_size = 20 get this feature vector size 
#INP_VEC_SIZE = 576     # only magnitude features
SEQ_SIZE = 5
threshold = 0.5
seq_threshold = 0.5



# Form a magnitude and angle dataframes using extracted features kept in 
# feats_data_dict. BOW rows are calculated using counts of 
# Returns two dataframes of (nVids, nClusters). 
# assignment to one of the centroids.
# feats_data_dict: {'filename':[array(2,16,12), array(2,16,12), ....]}
def create_bovw_traindf(feats_df, kmeans_mod): # mag_min, mag_max, idfpath):
    """
    Form a sequence of one hot vectors corresponding to input features.
    feats_df: pd.DataFrame (rows as keys(all vids), 71797 x 528)
    kmeans_mod: model 
    """
    clusters = kmeans_mod.cluster_centers_
    
    # Create a dataframe of size n_videos X n_clusters
    #print("Make bow vector for each frame")
    words_mat = np.zeros((feats_df.shape[0], clusters.shape[0]))
    word_ids = vq(feats_df, clusters)[0]  # ignoring the distances in [1]
        
    for idx, word_id in enumerate(word_ids):
        words_mat[idx, word_id] = 1
        

    return pd.DataFrame(words_mat, index=feats_df.index)

def plot_cluster_hist(words):
    import matplotlib.pyplot as plt
    plt.hist(words, bins=(np.max(words)+1))
    plt.xlabel("Cluster no.")
    plt.show()

def create_feats_df(features):
    """
    Create a pandas dataframe of features with key values as the row names.
    """
    # make copies of keys for the row names and convert the matrices from 
    # N x 1 x vec_size to N x vec_size
    indices = []
    vals = 0
    for k,v in features.items():
        indices.extend(v.shape[0]*[k])
        if type(vals)==int:
            vals = np.squeeze(v, axis=1)
        else:
            vals = np.vstack((vals, np.squeeze(v, axis=1)))
        
    vals[np.isinf(vals)] = 0
    # form the dataframe and set index 
    df = pd.DataFrame(vals, index = indices)
    return df
        

# function to find the clusters using KMeans
# vecs: any dataframe representing the input space points
# nclusters: No. of clusters to be formed
# returns the KMeans object, containing the cluster centroids
def make_codebook(vecs, nclusters, seed=153):
    print("Clustering using KMeans: Input size -> {} :: n_clusters -> {}"\
          .format(vecs.shape, nclusters))   
    
    #clustering with k-means
    #kmeans = KMeans(init='k-means++', n_clusters=200, n_init=10, n_jobs=2, verbose=1)
    kmeans = KMeans(n_clusters=nclusters, n_init=10, n_jobs=2, random_state=seed)
    kmeans.fit(vecs)
    print("Done Clustering!")
    return kmeans

def getBatchFeatures(features_df, videoFiles, sequences, motion=True):
    """Select only the batch features from the dictionary of features (corresponding
    to the given sequences) and return them as a list of lists. 
    OFfeatures: a dictionary of features {vidname: numpy matrix, ...}
    videoFiles: the list of filenames for a batch
    sequences: the start and end frame numbers in the batch videos to be sampled.
    SeqSize should be >= 2 for atleast one vector in sequence.
    """
    #grid_size = 20
    batch_feats = []
    # Iterate over the videoFiles in the batch and extract the corresponding feature
    for i, videoFile in enumerate(videoFiles):
        # get key value for the video. Use this to read features from dictionary
        videoFile = videoFile.split('/')[1].rsplit('.', 1)[0]
            
        start_frame = sequences[0][i].item()   # starting point of sequences in video
        end_frame = sequences[1][i].item()     # end point
        # Load features
        # (N-1) sized list of vectors of 1152 dim
        vidFeats = features_df.loc[videoFile, :]
        if motion:
            vid_feat_seq = vidFeats.iloc[start_frame:end_frame, :]
        else:
            vid_feat_seq = vidFeats.iloc[start_frame:(end_frame+1), :]
        
        #vid_feat_seq = np.squeeze(vid_feat_seq, axis = 1)
        batch_feats.append(vid_feat_seq.values)
        
    return batch_feats


if __name__=="__main__":
    # Divide the samples files into training set, validation and test sets
    train_lst, val_lst, test_lst = utils.split_dataset_files(DATASET)
    print(train_lst, len(train_lst))
    print(60*"-")
    
    gridSize = 30
    
    train_lab = [f+".json" for f in train_lst]
    val_lab = [f+".json" for f in val_lst]
    test_lab = [f+".json" for f in test_lst]
    
    #####################################################################
    
    tr_labs = [os.path.join(LABELS, f) for f in train_lab]
    sizes = [utils.getNFrames(os.path.join(DATASET, f+".avi")) for f in train_lst]
    print("Size : {}".format(sizes))
    hlDataset = VideoDataset(tr_labs, sizes, seq_size=SEQ_SIZE, is_train_set = True)
    print(hlDataset.__len__())
    
    #####################################################################
    # Run extract_denseOF_par.py before executing this file, using same grid_size
    # Features already extracted and dumped to disk 
    # Read those features using the given path and grid size
    #featuresPath = os.path.join(os.getcwd(),"OF_npy_grid"+str(gridSize))
    featuresPath = os.path.join(os.getcwd(),"hog_feats_64x64")
    #HOGfeaturesPath = os.path.join(os.getcwd(),"hog_feats_new")
    # change the name to "OF_npy_grid"+str(gridSize) or "c3d_feats_"+str(depth)
    #featuresPath = os.path.join(os.getcwd(), "hog_feats_64x64")
    
    # Uncomment the lines below to extract features for a different gridSize
#    from extract_denseOF_par import extract_dense_OF_vids
#    start = time.time()
#    extract_dense_OF_vids(DATASET, OFfeaturesPath, grid_size=gridSize, stop='all')
#    end = time.time()
#    print "Total execution time : "+str(end-start)
    
    #####################################################################
    
    # Create a DataLoader object and sample batches of examples. 
    # These batch samples are used to extract the features from videos parallely
    train_loader = DataLoader(dataset=hlDataset, batch_size=BATCH_SIZE, shuffle=True)

    train_losses = []
    # read into dictionary {vidname: np array, ...}
    print("Loading features from disk...")
    #OFfeatures = utils.readAllOFfeatures(OFfeaturesPath, train_lst)
    #HOGfeatures = utils.readAllHOGfeatures(HOGfeaturesPath, train_lst)
    features = utils.readAllPartitionFeatures(featuresPath, train_lst)
    print(len(train_loader.dataset))
    #####################################################################
    
    # form a matrix(dataframe) of NPOINTS x VECSIZE
    features_df = create_feats_df(features)
    # Cluster the features
#    km_feats = make_codebook(features_df, 200, seed=123)
#    # Save to disk, if training is performed
#    print("Writing the KMeans models to disk...")
#    with open("kmeans_model_hog", "wb") as outfile:
#        pickle.dump(km_feats, outfile)

    # Load from disk, for validation and test sets.
    with open("kmeans_model_hog", 'rb') as infile:
        km_feats = pickle.load(infile)
    
    words_df = create_bovw_traindf(features_df, km_feats)
    
    
    #####################################################################
    INP_VEC_SIZE = words_df.shape[-1]
    #INP_VEC_SIZE = features[list(features.keys())[0]].shape[-1] 
    print("INP_VEC_SIZE = ", INP_VEC_SIZE)
    
    # Creating the RNN and training
    classifier = RNNClassifier(INP_VEC_SIZE, HIDDEN_SIZE, 1, N_LAYERS, use_gpu=True)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [33, xxx] -> [11, ...], [11, ...], [11, ...] on 3 GPUs
        classifier = nn.DataParallel(classifier)

    if torch.cuda.is_available():
        classifier.cuda()

    #optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
    optimizer = torch.optim.SGD(classifier.parameters(), lr = 0.01, \
                                momentum=0.9)
    #criterion = nn.CrossEntropyLoss()
    sigm = nn.Sigmoid()
    criterion = nn.BCELoss()

    start = time.time()
    
    print("Training for %d epochs..." % N_EPOCHS)
    for epoch in range(N_EPOCHS):
        total_loss = 0
        for i, (keys, seqs, labels) in enumerate(train_loader):
            # Run your training process
            #print(epoch, i) #, "keys", keys, "Sequences", seqs, "Labels", labels)
            #batchFeats = utils.getBatchFeatures(features, keys, seqs, motion=True)
            batchFeats = getBatchFeatures(words_df, keys, seqs, motion=False)
            #break

            # Training starts here
            inputs, target = utils.make_variables_new(batchFeats, labels, motion=False, use_gpu=True)
            #inputs, target = utils.make_variables(batchFeats, labels, motion=False)
            output = classifier(inputs)

            loss = criterion(sigm(output.view(output.size(0))), target)
            #total_loss += loss.data[0]
            total_loss += loss.item()

            classifier.zero_grad()
            loss.backward()
            optimizer.step()

            #if i % 2 == 0:
            #    print('Train Epoch: {} :: Loss: {:.2f}'.format(epoch, total_loss))
            #if (i+1) % 10 == 0:
            #    break
        train_losses.append(total_loss)
        print('Train Epoch: {} :: Loss: {:.2f}'.format(epoch+1, total_loss))
    
    
    # Save only the model params
    torch.save(classifier.state_dict(), "gru_100_epoch10_BCE.pt")
    print("Model saved to disk...")
    
    # Save losses to a txt file
    with open("losses.pkl", "wb") as fp:
        pickle.dump(train_losses, fp)
    
    # To load the params into model
    ##the_model = RNNClassifier(INP_VEC_SIZE, HIDDEN_SIZE, 1, N_LAYERS)
    ##the_model.load_state_dict(torch.load("gru_100_epoch10_BCE.pt"))    
    #classifier.load_state_dict(torch.load("gru_100_epoch10_BCE.pt"))
    
#    save_checkpoint({
#            'epoch': epoch + 1,
#            'arch': args.arch,
#            'state_dict': classifier.state_dict(),
#            'best_prec1': best_prec1,
#            'optimizer' : optimizer.state_dict(),
#        }, is_best)
#    #####################################################################
#    
#    # Loading and resuming from dictionary
#    # Refer : https://github.com/pytorch/examples/blob/master/imagenet/main.py#L139
#    if args.resume:
#        if os.path.isfile(args.resume):
#            print("=> loading checkpoint '{}'".format(args.resume))
#            checkpoint = torch.load(args.resume)
#            args.start_epoch = checkpoint['epoch']
#            best_prec1 = checkpoint['best_prec1']
#            model.load_state_dict(checkpoint['state_dict'])
#            optimizer.load_state_dict(checkpoint['optimizer'])
#            print("=> loaded checkpoint '{}' (epoch {})"
#                  .format(args.resume, checkpoint['epoch']))
#        else:
#            print("=> no checkpoint found at '{}'".format(args.resume))
            
    #####################################################################
    
    # Test a video or calculate the accuracy using the learned model
    print("Prediction video meta info.")
    val_labs = [os.path.join(LABELS, f) for f in val_lab]
    val_sizes = [utils.getNFrames(os.path.join(DATASET, f+".avi")) for f in val_lst]
    print("Size : {}".format(val_sizes))
    hlvalDataset = VideoDataset(val_labs, val_sizes, is_train_set = False)
    print(hlvalDataset.__len__())
    
    # Create a DataLoader object and sample batches of examples. 
    # These batch samples are used to extract the features from videos parallely
    val_loader = DataLoader(dataset=hlvalDataset, batch_size=BATCH_SIZE, shuffle=False)
    print(len(val_loader.dataset))
    correct = 0
    val_keys = []
    predictions = []
    print("Loading validation/test features from disk...")
    #OFValFeatures = utils.readAllOFfeatures(OFfeaturesPath, val_lst)
    #HOGValFeatures = utils.readAllHOGfeatures(HOGfeaturesPath, val_lst)   
    valFeatures = utils.readAllPartitionFeatures(featuresPath, val_lst)
    
    # form a matrix(dataframe) of NPOINTS x VECSIZE
    valFeatures_df = create_feats_df(valFeatures)
    
    val_words_df = create_bovw_traindf(valFeatures_df, km_feats)
    
    print("Predicting on the validation/test videos...")
    for i, (keys, seqs, labels) in enumerate(val_loader):
        
        # Testing on the sample
        #feats = getFeatureVectors(DATASET, keys, seqs)      # Parallelize this
        #batchFeats = utils.getBatchFeatures(valFeatures, keys, seqs, motion=True)
        batchFeats = getBatchFeatures(val_words_df, keys, seqs, motion=False)
        #batchFeats = utils.getFeatureVectorsFromDump(HOGValFeatures, keys, seqs, motion=False)
        #break
        # Validation stage
        inputs, target = utils.make_variables_new(batchFeats, labels, motion=False, use_gpu=True)
        #inputs, target = utils.make_variables(batchFeats, labels, motion=False)
        output = classifier(inputs) # of size (BATCHESxSeqLen) X 1

        #pred = output.data.max(1, keepdim=True)[1]  # get max value in each row
        pred_probs = sigm(output.view(output.size(0))).data  # get the normalized values (0-1)
        #preds = pred_probs > THRESHOLD  # ByteTensor
        #correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        val_keys.append(keys)
        predictions.append(pred_probs)  # append the 
        
        #loss = criterion(m(output.view(output.size(0))), target)
        #total_loss += loss.data[0]

        #if i % 2 == 0:
        #    print('i: {} :: Val keys: {} : seqs : {}'.format(i, keys, seqs)) #keys, pred_probs))
        #if (i+1) % 10 == 0:
        #    break
    print("Predictions done on validation/test set...")
    #####################################################################
    
    with open("predictions.pkl", "wb") as fp:
        pickle.dump(predictions, fp)
    
    with open("val_keys.pkl", "wb") as fp:
        pickle.dump(val_keys, fp)
    
#    with open("predictions.pkl", "rb") as fp:
#        predictions = pickle.load(fp)
#    
#    with open("val_keys.pkl", "rb") as fp:
#        val_keys = pickle.load(fp)
    
    from get_localizations import getLocalizations

    # [4949, 4369, 4455, 4317, 4452]
    #predictions = [p.cpu() for p in predictions]  # convert to CPU tensor values
    localization_dict = getLocalizations(val_keys, predictions, BATCH_SIZE, \
                                         threshold, seq_threshold)

    print(localization_dict)
    
    import json        
#    for i in range(0,101,10):
#        filtered_shots = filter_action_segments(localization_dict, epsilon=i)
#        filt_shots_filename = "predicted_localizations_th0_5_filt"+str(i)+".json"
#        with open(filt_shots_filename, 'w') as fp:
#            json.dump(filtered_shots, fp)

    # Apply filtering    
    i = 60  # optimum
    filtered_shots = utils.filter_action_segments(localization_dict, epsilon=i)
    #i = 7  # optimum
    #filtered_shots = filter_non_action_segments(filtered_shots, epsilon=i)
    filt_shots_filename = "predicted_localizations_th0_5_filt"+str(i)+".json"
    with open(filt_shots_filename, 'w') as fp:
        json.dump(filtered_shots, fp)
    print("Prediction file written to disk !!")
    #####################################################################

    print("#Parameters : {} ".format(utils.count_parameters(classifier)))