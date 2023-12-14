import torch
import torch.nn as nn
import torch.nn.functional as F
from parameters2 import *


class Controller(nn.Module):
    def __init__(self):
        super(Controller, self).__init__()
        self.num_cells = num_cells
        self.lstm_size = lstm_size
        self.out_features = 6
        self.num_layers = num_layers
        self.seq = []

        self.lstm = nn.LSTM(self.lstm_size, self.lstm_size)
        self.linear = nn.Linear(self.lstm_size, self.num_cells)

        self.embedding = nn.Embedding(1, self.lstm_size)
        self.word_embedding = nn.Embedding(self.num_cells, self.lstm_size)

        self.attn_1 = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
        self.attn_2 = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
        self.attn_v = nn.Linear(self.lstm_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=0)

    def forward(self):
        seq = []
        all_h, all_weighted_h = [], []
        input = torch.zeros(1).long()
        embed = self.embedding(input)
        # prev_c, prev_h = [], []

        prev_c = torch.zeros(1, lstm_size)
        prev_h = torch.zeros(1, lstm_size)

        for layer in range(2):
            output, (next_h,next_c) = self.lstm(embed, (prev_h, prev_c))
            a,(b,c) = self.lstm(embed)
            prev_c, prev_h = next_c, next_h
            all_h.append(next_h)
            all_weighted_h.append(self.attn_1(next_h))

        for layer_id in range(2,num_layers):
            for i in range(2):
                output, (next_h, next_c) = self.lstm(embed,(prev_h,prev_c))
                prev_c, prev_h = next_c, next_h

                # query = self.attn_2(next_h[-1])
                # query = torch.cat(all_weighted_h[:-1], dim=0)
                # query2 = torch.cat(all_weighted_h[:-1], dim=1)

                # for lstm cell
                query3 = torch.cat(all_weighted_h[:layer_id] ,dim=0)

                # for stacked lstm
                # query3 = torch.cat(all_weighted_h,dim=0)

                # query = query + self.attn_2(next_h[-1])
                # query2 = query2 + self.attn_2(next_h[-1])
                query3 = query3 + self.attn_2(next_h)

                # alignment_score = self.attn_v(torch.tanh(query))
                # alignment_score2 = self.attn_v(torch.tanh(query2))
                alignment_score3 = self.attn_v(torch.tanh(query3))
                score = torch.reshape(alignment_score3,[1,layer_id])
                # print(alignment_score)
                # print("Alignment 2 score: ")
                # print(alignment_score2)
                # print('Alignment score 3')
                # print(alignment_score3)
                # print('\n')
                all_h.append(next_h)
                all_weighted_h.append(self.attn_1(next_h))

                # original implementation based on squared difference

                # trying softmax
                skip_connections = self.softmax(score)
                skip_index = torch.multinomial(skip_connections,1)
                skip_index = torch.reshape(skip_index,[1])
                seq.append(skip_index)

                # select input
                concat_tensor = torch.cat(all_h,dim=0)
                embed = torch.index_select(concat_tensor,dim=0,index=skip_index)

            for i in range(2):
                output, (next_h, next_c) = self.lstm(embed,(prev_h,prev_c))
                prev_c, prev_h = next_c, next_h
                logit = self.linear(next_h)

                # requires non negative value
                logit = self.softmax(logit)
                op = torch.multinomial(logit, 1)
                op = torch.reshape(op, [1])
                seq.append(op)

                # select input
                concat_tensor = torch.cat(all_h,dim=0)
                embed = torch.index_select(concat_tensor,dim=0,index=skip_index)

            self.seq = seq
            # all_h.append(next_h)
            # all_weighted_h.append(self.attn_1(next_h))
        return seq


if __name__ == "__main__":
    c = Controller()
    op = c()
    print(op)
