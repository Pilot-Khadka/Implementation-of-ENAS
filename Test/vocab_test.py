import unittest
import sys

sys.path.append('/home/pilot/Code/Github/Neural-Architecture-Search-NAS-')
from vocab import *
from parameters import *

class MyTestCase(unittest.TestCase):

    def test_vocabdict(self):
        vocab_instance = vocab()
        vocab_dict = vocab_instance.vocab

        # All combination of nodes and activation
        # along with dropout and softmax layer
        expected_layers = len(nodes) * len(activations) + 2
        self.assertEqual(len(vocab_dict),expected_layers)

        # Test the properties of the last layer
        output_layer = vocab_dict[expected_layers]
        self.assertEqual(output_layer['type'], 'output')
        self.assertEqual(output_layer['nodes'], target_classes)
        self.assertIn(output_layer['activation'], ['Sigmoid', 'Softmax'])

if __name__ == '__main__':
    unittest.main()

