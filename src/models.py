
"""
Author: Erica Wei
UNI: cw3137
models.py contains models LSTM and BiLSTM for the experiment in our project.
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
cd = os.getcwd()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ExperimentalRNN(nn.Module):
    """
    BiLSTM model.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, drop_prob=0.2):
        super(ExperimentalRNN, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, bidirectional=True)
        self.linear = nn.Linear(hidden_dim * 4, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_o, _ = self.lstm(x)
        avg_pool = torch.mean(lstm_o, 0)
        max_pool, _ = torch.max(lstm_o, 0)
        out = torch.cat((avg_pool, max_pool), -1)
        out = self.relu(self.linear(out))
        out = self.dropout(out)
        out = self.out(out)
        scores = F.log_softmax(out, dim=1) # might delete if result is bad
        return scores


class LSTMNet(nn.Module):
    """
    Simple LSTM model.
    """
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=10, num_layers=2, drop_prob=0.2):
        super(LSTMNet, self).__init__()
        self.input_size = input_dim
        self.output_size = output_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.batch_size= batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_dim, self.num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, hidden = self.lstm(x)
        out = self.fc(lstm_out[-1])
        scores = F.log_softmax(out, dim=1)

        return scores

    def init_hidden(self):
        return (
            torch.zeros(self.n_layers, self.batch_size, self.hidden_dim),
            torch.zeros(self.n_layers, self.batch_size, self.hidden_dim),
        )

