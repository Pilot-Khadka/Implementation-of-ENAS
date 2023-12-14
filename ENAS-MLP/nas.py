import torch
from controller import *
import pandas as pd
from architecture import *
from parameters import *


class NAS(controller):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.num_samples_generated = samples_generated_per_controller_epoch
        super().__init__()

    # Try generating a single sequence
    def search(self):

        print("Generating architecture")
        sequences = self.generate_sequence(self.num_samples_generated)

        for i, sequence in enumerate(sequences):
            model = Generate_Architecture(mlp_input_shape=self.x.size(),
                                          sequence=sequence)
            # print(model)
            history = model.train_model( self.x, self.y)

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

