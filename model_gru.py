#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 00:50:31 2018

@author: Arpan
Description: RNN Model defined in this file. 

"""

import torch
import torch.nn as nn
import utils

class RNNClassifier(nn.Module):
    # Our model
    def __init__(self, input_size, hidden_size, output_size, n_layers=1,\
                 bidirectional=False, use_gpu=False):
        super(RNNClassifier, self).__init__()
        self.use_gpu = use_gpu
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = int(bidirectional) + 1

        #self.embedding = nn.Embedding(input_size, hidden_size)
        #self.rnn = nn.RNN(input_size=input_size,
        #                  hidden_size=hidden_size, batch_first=True)
        self.gru = nn.GRU(input_size, hidden_size, n_layers, batch_first=True,
                          bidirectional=bidirectional)
        #self.gru1 = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True,
        #                  bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size, output_size)
        #self.soft = nn.Softmax()

    def forward(self, input):
        # Note: we run this all at once (over the whole input sequence)
        # input shape: B x S (input size)
        # transpose to make S(sequence) x B (batch)
        #input = input.t()
        #batch_size = input.size(1)
        batch_size = input.size(0)
        seq_len = input.size(1)        
        #print "Seq Len : {} :: Batch size : {}".format(seq_len, batch_size)

        # Make a hidden
        hidden = self._init_hidden(batch_size)

        # Embedding S x B -> S x B x I (embedding size)
        #embedded = self.embedding(input.view(seq_len, batch_size, -1))
        # B X S X 1152
        embedded = input

        # Pack them up nicely
        #gru_input = pack_padded_sequence(
        #    embedded, seq_lengths.data.cpu().numpy())

        # To compact weights again call flatten_parameters().
        self.gru.flatten_parameters()
        #output, hidden = self.rnn(embedded, hidden)
        output, hidden = self.gru(embedded, hidden)
        #self.gru1.flatten_parameters()
        #output, hidden = self.gru1(output, hidden)

        # Use the last layer output as FC's input
        # No need to unpack, since we are going to use hidden
        return self.fc(output.contiguous().view(-1, self.hidden_size))
        #fc_output = self.fc(hidden[-1])
        #return fc_output

    def _init_hidden(self, batch_size):
         #* self.n_directions
        hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, \
                             self.hidden_size)
        return utils.create_variable(hidden, self.use_gpu)


class LSTMModel(nn.Module):
    # Our model
    def __init__(self, input_size, hidden_size, output_size, n_layers=1,\
                 bidirectional=False, use_gpu=False):
        super(LSTMModel, self).__init__()
        self.use_gpu = use_gpu
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = int(bidirectional) + 1

        #self.embedding = nn.Embedding(input_size, hidden_size)
        #self.rnn = nn.RNN(input_size=input_size,
        #                  hidden_size=hidden_size, batch_first=True)
        self.lstm1 = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)
        #self.soft = nn.Softmax()

    def forward(self, input):
        # Note: we run this all at once (over the whole input sequence)
        # input shape: B x S (input size)
        # transpose to make S(sequence) x B (batch)
        #input = input.t()
        #batch_size = input.size(1)
        batch_size = input.size(0)
        seq_len = input.size(1)
        #print "Seq Len : {} :: Batch size : {}".format(seq_len, batch_size)

        # Make a hidden
        h0 = self._init_hidden(batch_size)
        c0 = self._init_hidden(batch_size)

        # Embedding S x B -> S x B x I (embedding size)
        #embedded = self.embedding(input.view(seq_len, batch_size, -1))
        # B X S X 1152
        embedded = input

        # Pack them up nicely
        #gru_input = pack_padded_sequence(
        #    embedded, seq_lengths.data.cpu().numpy())

        # To compact weights again call flatten_parameters().
        #self.lstm1.flatten_parameters()
        #output, hidden = self.rnn(embedded, hidden)
        output, (hn, cn) = self.lstm1(embedded, (h0, c0))
        #self.gru1.flatten_parameters()
        #output, hidden = self.gru1(output, hidden)

        # Use the last layer output as FC's input
        # No need to unpack, since we are going to use hidden
        return self.fc(output.contiguous().view(-1, self.hidden_size))
        #fc_output = self.fc(hidden[-1])
        #return fc_output

    def _init_hidden(self, batch_size):
         #* self.n_directions
        hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, \
                             self.hidden_size)
        return utils.create_variable(hidden, self.use_gpu)


#class RNN(nn.Module):
#
#    def __init__(self, num_classes, input_size, hidden_size, num_layers):
#        super(RNN, self).__init__()
#
#        self.num_classes = num_classes
#        self.num_layers = num_layers
#        self.input_size = input_size
#        self.hidden_size = hidden_size
#        self.sequence_length = sequence_length
#
#        self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, \
#                          batch_first=True)
#
#    def forward(self, x):
#        # Initialize hidden and cell states
#        # (num_layers * num_directions, batch, hidden_size) for batch_first=True
#        h_0 = Variable(torch.zeros(
#            self.num_layers, x.size(0), self.hidden_size))
#
#        # Reshape input
#        x.view(x.size(0), self.sequence_length, self.input_size)
#
#        # Propagate input through RNN
#        # Input: (batch, seq_len, input_size)
#        # h_0: (num_layers * num_directions, batch, hidden_size)
#
#        out, _ = self.rnn(x, h_0)
#        return out.view(-1, num_classes-1)
