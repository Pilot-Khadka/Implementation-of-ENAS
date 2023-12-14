import torch
import torch.nn as nn
import torch.optim as optim
from vocab import *
import numpy as np
from parameters import *

# Two cells (2 nodes - say: A, B) are treated as inputs
# these cells must be the output of the previous cell
#
# Remaining B-2 nodes are used by the controller.
#     -> between the previous two nodes (A,B), which one to use as input
#
# Decisions made by the controller:
#     -> which to use as input between two nodes
#     -> which operation to apply to the two sampled nodes(on the previous nodes)
#         -> operation includes:
#             -> identity
#             -> seperable convolution with 3x3
#             -> seperable convolution with 5x5
#             -> average pooling with 3x3
#             -> max pooling with 3x3

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=number_of_cells,embedding_dim=hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        x =self.embedding_layer(x)
        lstm_out, _ = self.lstm(x)
        x = self.fc(lstm_out)
        x = self.softmax(x)
        return x


