import torch

from controller import *
import pandas as pd
from architecture import *


class NAS(controller):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        super().__init__()

    # Try generating a single sequence
    def search(self):
        print("Generating architecture")
        sequence = self.generate_sequence()
        model = CustomModel(mlp_input_shape=self.x.size(),
                            mlp_dropout=0.3,
                            sequence=sequence)

        print(model)
        history = model.train_model(model, self.x, self.y)
        print(history)


if __name__ == '__main__':
    data = pd.read_csv('DATASETS/wine-quality.csv')
    x = data.drop('quality_label', axis=1, inplace=False).values
    x = torch.tensor(x)
    x = x.float()
    # print
    y = pd.get_dummies(data['quality_label']).values
    y = torch.tensor(y,dtype=torch.float32)
    nas = NAS(x, y)

    nas.search()
