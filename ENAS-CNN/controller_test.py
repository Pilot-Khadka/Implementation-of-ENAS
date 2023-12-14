import unittest
from controller import *

import torch

# class MyTestCase(unittest.TestCase):
#
#     def test_model(self):
#         input_data = torch.rand(1,10)
#
#         model = controller()
#         print(model)


if __name__ == "__main__":
    # unittest.main()
    lstm = torch.nn.LSTM(input_size=10,hidden_size=100)
    embed = torch.nn.Embedding(6,10)

    inputs = torch.zeros(1).long()

    inputs = embed(inputs)
    prev_c = torch.zeros(1,100)
    prev_h = torch.zeros(1,100)

    # next_h, next_c = lstm(inputs, (prev_h,prev_c))
    # print(next_h.size())
    # print(next_c)
    w_attn1 = torch.nn.Linear(in_features=100,out_features=100)
    w_attn2 = torch.nn.Linear(in_features=100,out_features=100)
    w_v = torch.nn.Linear(in_features=100,out_features=100)

    all_h, all_h_w = [],[]

    for i in range(6):
        output, (next_h, next_c) = lstm(inputs ,(prev_h,prev_c))
        prev_h, prev_c = next_h, next_c

        all_h.append(next_h[-1])
        all_h_w.append(w_attn1(next_h[-1]))

        query = w_attn1(next_h[-1])
        query = torch.tanh(query)

        logits = w_v(query)
        