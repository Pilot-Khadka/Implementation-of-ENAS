from parameters import *


class vocab:
    def __init__(self):
        self.vocab = self.vocab_dict()

    def vocab_dict(self):
        arch_vocab = {}
        layer_id = 1

        for node in nodes:
            for activation in activations:
                arch_vocab[layer_id] = {'type': 'hidden', 'nodes': node, 'activation': activation}
                layer_id += 1

        arch_vocab[layer_id] = {'type': 'dropout'}
        layer_id += 1

        output_activation = 'Sigmoid' if target_classes == 2 else 'Softmax'
        arch_vocab[layer_id] = {'type': 'output', 'nodes': target_classes, 'activation': output_activation}

        return arch_vocab



    def decode_architecture(self, sequence):
        seed = []
        for _ in sequence:
            original_param = self.vocab[_]
            seed.append(original_param)

        # print(original_param)
        return seed
