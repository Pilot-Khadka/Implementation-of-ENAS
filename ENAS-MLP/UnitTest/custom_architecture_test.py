import unittest
from architecture import *
from controller import *

control = controller()
sequence = control.generate_sequence()
print(sequence)


class MyTestCase(unittest.TestCase):

    def test_forward_pass(self):
        input_data= torch.rand(1,10)

        model = CustomModel(mlp_input_shape=input_data,
                            mlp_dropout=0.3,
                            sequence = sequence)
        print(model)
        # output = self.model.train_model(input_data)



if __name__ == '__main__':
    unittest.main()
