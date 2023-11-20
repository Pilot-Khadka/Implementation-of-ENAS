import torch.nn as nn
import torch.optim as optim
from vocab import *

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = self.fc(lstm_out)
        x = self.softmax(x)
        return x


class controller(LSTM):
    def __init__(self):

        self.input_size = max_len -1
        self.hidden_size = 100

        self.vocab = vocab()
        self.vocab_idx = list(self.vocab.vocab.keys())
        self.output_size = len(self.vocab_idx)
        super().__init__(self.input_size, self.hidden_size, self.output_size)
        self.model = self.init_controller()

    def init_controller(self):
        model = LSTM(input_size = self.input_size,
                     hidden_size = self.hidden_size,
                     output_size = self.output_size)
        loss_function = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        return model

    def generate_sequence(self):
        seed = []

        while len(seed) < max_len - 1:
            sequence = torch.zeros(max_len - 1)
            sequence = sequence.reshape(1, 1, max_len - 1)

            with torch.no_grad():
                probab = self.model(sequence)

            probab = np.array(probab)
            normalized_probab = probab / probab.sum()
            next = np.random.choice(self.vocab_idx, p=normalized_probab[0][0])
            if next == len(self.vocab_idx):
                break
            seed.append(next)
        seed.append(len(self.vocab_idx))
        return seed
