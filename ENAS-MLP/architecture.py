import torch
import torch.nn as nn
from collections import OrderedDict
import os
from vocab import *
import pandas as pd
from utils import data_loader


class CustomModel(nn.Module):
    def __init__(self,mlp_input_shape, layer_configs, mlp_dropout=0.3):
        super(CustomModel, self).__init__()
        # self.mlp_input_shape = mlp_input_shape
        # self.layer_configs = layer_configs
        self.layers = self.initialize_layers(mlp_input_shape,layer_configs,mlp_dropout)

    def initialize_layers(self, mlp_input_shape, layer_configs, mlp_dropout):
        temp_layers = []
        if isinstance(mlp_input_shape, int):
            prev_nodes = mlp_input_shape
        else:
            prev_nodes = mlp_input_shape[1]

        if len(mlp_input_shape) > 1:
            temp_layers.append(('{}'.format(0), nn.Flatten()))

        layer_index = 1  # Initialize a separate variable for the layer index

        for i, layer_conf in enumerate(layer_configs):
            if layer_conf['type'] == 'dropout':
                temp_layers.append(('{}'.format(layer_index), nn.Dropout(mlp_dropout)))
                layer_index += 1

            elif layer_conf['type'] == 'output':
                linear_layer = nn.Linear(prev_nodes, layer_conf['nodes'])
                prev_nodes = layer_conf['nodes']
                temp_layers.append(('{}'.format(layer_index), linear_layer))
                layer_index += 1
                temp_layers.append(('{}'.format(layer_index), nn.Softmax(dim=1)))
                layer_index += 1

            else:
                activation_name = layer_conf['activation']
                activation = getattr(nn, activation_name)() if activation_name else None
                linear_layer = nn.Linear(prev_nodes, layer_conf['nodes'])
                prev_nodes = layer_conf['nodes']
                temp_layers.append(('{}'.format(layer_index), linear_layer))
                layer_index += 1
                if activation:
                    temp_layers.append(('{}'.format(layer_index), activation))
                    layer_index += 1

        print(temp_layers)
        return nn.Sequential(OrderedDict(temp_layers))

    def forward(self, x):
        return self.layers(x)

class Generate_Architecture():
    def __init__(self,mlp_input_shape, sequence):
        self.mlp_input_shape = mlp_input_shape
        self.sequence = sequence
        self.vocab_init = vocab()
        self.layer_configs = self.vocab_init.decode_architecture(self.sequence)
        self.config_ids = self.generate_node_pair()
        # super().__init__(mlp_input_shape, layer_configs)
        self.model = CustomModel(self.mlp_input_shape, self.layer_configs)
        print(self.model)

        self.weights_file = 'shared_weights.pkl'
        self.shared_weights = pd.DataFrame({'id': [], 'weights': []})
        if not os.path.exists(self.weights_file):
            print("Initializing shared weights dictionary...")
            self.shared_weights.to_pickle(self.weights_file)

    def generate_node_pair(self):
        # print(sequence)
        v = vocab()
        layer_config = v.decode_architecture(self.sequence)
        # print(layer_config)
        config = ['input','flatten']
        for layer in layer_config:
            print(layer)
            if layer['type'] != 'dropout':
                config.append((layer['nodes'], layer['activation']))

        # print(config)
        config_ids = []
        for i in range(1,len(config)):
            config_ids.append((config[i - 1], config[i]))

        return config_ids

    def set_model_weight(self):
        # layer_configs = ['input']
        # for layer in model.children():
        #     if isinstance(model.layer, nn.Flatten):
        #         layer_configs.append('flatten')
        #     elif not isinstance(layer, nn.Dropout):
        #         # access layer in_feature size
        #         # print("from set_model_weight:",model.layer)
        #         # print("layer_node size :",model.layer.weight)
        #         layer_configs.append((layer.size(0), layer.activation.__name__))
        #
        #
        # config_ids = []
        # for i in range(1, len(layer_configs)):
        #     config_ids.append((layer_configs[i - 1], layer_configs[i]))
        j = 1
        for layer in self.model.children():
            if not isinstance(layer, nn.Dropout):
                current_config = self.config_ids[j - 1]

                search_index = []
                for idx, row in self.shared_weights.iterrows():
                    if current_config == (row['id'], row['weights']):
                        search_index.append(idx)

                if len(search_index) > 0:
                    print("Transferring weights for layer:", current_config)

                    self.model.layers[j].weight = torch.tensor(self.shared_weights['weights'].iloc[search_index[0]])
                j += 1

    def update_model_weight(self, config_ids):
        j = 1
        for layer in self.model.children():
            if not isinstance(layer, nn.Dropout):
                current_config = config_ids[j - 1]

                search_index = []
                for idx, row in self.shared_weights.iterrows():
                    if current_config == (row['id'], row['weights']):
                        search_index.append(idx)

                if not search_index:
                    with torch.no_grad():
                        data_to_be_appended = pd.DataFrame({
                            'id': [config_ids[j]],
                            'weights': [self.model.layers[j].weight]})

                    if self.shared_weights is None:
                        self.shared_weights = data_to_be_appended
                    else:
                        self.shared_weights = pd.concat([self.shared_weights, data_to_be_appended], ignore_index=True)

                else:
                    self.shared_weights.at[search_index[0], 'weights'] = self.model.layer[j].weight
                j+=1
            self.shared_weights.to_pickle(self.weights_file)

    def train_model(self, x, y, epochs=8, batch_size=32):
        # model = model.to(device)
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        train_loader, val_loader = data_loader(x, y, batch_size)

        print('model params:', self.model.parameters())
        # self.set_model_weight( config_ids)
        print('after model params:', self.model.parameters())

        criteria = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criteria(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            average_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(average_train_loss)

            self.model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = self.model(inputs)
                    loss = criteria(outputs, targets)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    _, targets = torch.max(targets.data, 1)
                    total_val += targets.size(0)
                    correct_val += (predicted == targets).sum().item()

            average_val_loss = val_loss / len(val_loader)
            history['val_loss'].append(average_val_loss)
            val_accuracy = correct_val / total_val
            history['val_accuracy'].append(val_accuracy)

        self.update_model_weight(self.config_ids)
        return history

# class
