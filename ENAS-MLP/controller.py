import torch
import torch.nn as nn
import torch.optim as optim
from vocab import *
import numpy as np

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
        self.lr = LSTM_learning_rate
        self.input_size = max_len -1
        self.hidden_size = 100
        self.epoch = LSTM_training_epochs

        self.vocab = vocab()
        self.vocab_idx = list(self.vocab.vocab.keys())
        self.output_size = len(self.vocab_idx)
        super().__init__(self.input_size, self.hidden_size, self.output_size)
        self.model = self.init_controller()

    def init_controller(self):
        model = LSTM(input_size = self.input_size,
                     hidden_size = self.hidden_size,
                     output_size = self.output_size)

        return model

    def generate_sequence(self, num_of_samples):
        samples =[]

        while len(samples) < num_of_samples:
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
            samples.append(seed)
        return samples

    def train_and_update_lstm(self,sequence):
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        sequence_tensor = torch.tensor(sequence)
        for epoch in range(self.epoch):
            output = self.model(sequence_tensor)
            loss = loss_function(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

