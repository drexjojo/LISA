import math
import torch
import sys
import numpy as np
import torch.nn as nn
from torch.autograd import Function
from constants import *

class ReverseGrad(Function):

    @staticmethod
    def forward(ctx, x, LAMBDA):
        ctx.save_for_backward(LAMBDA)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        LAMBDA, = ctx.saved_tensors
        return grad_output.neg() * LAMBDA, None

class LISA(nn.Module):
    def __init__(self,embeddings,config):
        super().__init__()
        self.LAMBDA = torch.tensor(config.LAMBDA)
        self.MUSE_embeddings = nn.Embedding(len(embeddings), len(embeddings[0]),padding_idx=0)
        self.MUSE_embeddings.load_state_dict({'weight': torch.from_numpy(embeddings)})
        self.MUSE_embeddings.weight.requires_grad = False
        self.lstm = nn.LSTM(EMBEDDING_DIM, config.HIDDEN_SIZE_LSTM,batch_first = True)
        self.dropout = nn.Dropout(config.LSTM_DROPOUT)
        self.linear_language1 =  nn.Linear(config.HIDDEN_SIZE_LSTM, config.HIDDEN_SIZE_LANG)
        self.linear_language2 =  nn.Linear(config.HIDDEN_SIZE_LANG, NUM_LANG)
        self.linear_sentiment1 =  nn.Linear(config.HIDDEN_SIZE_LSTM, config.HIDDEN_SIZE_SENT)
        self.linear_sentiment2 =  nn.Linear(config.HIDDEN_SIZE_SENT, OUTPUT_SIZE)
        self.activation_function = nn.LeakyReLU(True)

    def forward(self, x, X_lengths,device):
        batch_size, seq_len = x.size()
        embedded = self.MUSE_embeddings(x)

        X = nn.utils.rnn.pack_padded_sequence(embedded, X_lengths, batch_first=True)
        output, (hidden, cell) = self.lstm(X)
        fc1 = self.dropout(self.activation_function(self.linear_sentiment1(hidden)))
        sentiment_predictions = self.linear_sentiment2(fc1).squeeze(0)
        reverse_features = ReverseGrad.apply(hidden,self.LAMBDA)
        lang_predictions_temp = self.linear_language1(reverse_features)
        lang_predictions = self.linear_language2(lang_predictions_temp).squeeze(0)
        return sentiment_predictions, lang_predictions

class LSTM_NOLD(nn.Module):
    def __init__(self,embeddings):
        super().__init__()
        self.MUSE_embeddings = nn.Embedding(len(embeddings), len(embeddings[0]),padding_idx=0)
        self.MUSE_embeddings.load_state_dict({'weight': torch.from_numpy(embeddings)})
        self.MUSE_embeddings.weight.requires_grad = False
        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_SIZE,batch_first = True)
        self.dropout = nn.Dropout(LSTM_DROPOUT)
        self.linear_sentiment1 =  nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear_sentiment2 =  nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        self.activation_function = nn.LeakyReLU(True)

    def forward(self, x, X_lengths,device):
        batch_size, seq_len = x.size()
        embedded = self.MUSE_embeddings(x)

        X = nn.utils.rnn.pack_padded_sequence(embedded, X_lengths, batch_first=True)
        output, (hidden, cell) = self.lstm(X)

        # X, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        
        sentiment_predictions_temp = self.dropout(self.activation_function(self.linear_sentiment1(hidden)))
        sentiment_predictions = self.linear_sentiment2(sentiment_predictions_temp).squeeze(0)
        return sentiment_predictions

class LSTM_word_disc(nn.Module):
    def __init__(self,embeddings):
        super().__init__()
        self.MUSE_embeddings = nn.Embedding(len(embeddings), len(embeddings[0]),padding_idx=0)
        self.MUSE_embeddings.load_state_dict({'weight': torch.from_numpy(embeddings)})
        self.MUSE_embeddings.weight.requires_grad = False
        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_SIZE,batch_first = True)
        self.dropout = nn.Dropout(LSTM_DROPOUT)
        self.linear_language1 =  nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear_language2 =  nn.Linear(HIDDEN_SIZE, NUM_LANG)
        self.linear_sentiment1 =  nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear_sentiment2 =  nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        self.activation_function = nn.LeakyReLU(True)

    def forward(self, x, X_lengths,device):
        batch_size, seq_len = x.size()
        embedded = self.MUSE_embeddings(x)

        X = nn.utils.rnn.pack_padded_sequence(embedded, X_lengths, batch_first=True)
        output, (hidden, cell) = self.lstm(X)

        # X, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        
        # averaged_word_vectors = torch.zeros(X.shape[0],X.shape[2]).to(device)
        # for i,batch in enumerate(X) :
        #     averaged_word_vectors[i,:] = X[i,:X_lengths[i],:].mean(0)

        last_layer = hidden
        sentiment_predictions_temp = self.dropout(self.activation_function(self.linear_sentiment1(last_layer)))
        sentiment_predictions = self.linear_sentiment2(sentiment_predictions_temp).squeeze(0)

        reverse_features = grad_reverse(last_layer)
        lang_predictions_temp = self.activation_function(self.linear_language1(last_layer))
        lang_predictions = self.linear_language2(lang_predictions_temp).squeeze(0)
        return sentiment_predictions, lang_predictions