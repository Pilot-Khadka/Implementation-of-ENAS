import unittest
from controller import *

max_len = 10

def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(MyTestCase))
    return test_suite


class MyTestCase(unittest.TestCase):
    def test_generate_sequence(self):
        self.control = controller()
        result = self.control.generate_sequence()
        print('Generated Sequence:',result)

        # Assertions
        self.assertIsInstance(result, list)
        # self.assertTrue(all(isinstance(x, int) for x in result))

        # Check if all the elements generated in the sequence belong to the vocabulary
        self.assertTrue(all(x in self.control.vocab_idx for x in result))

        # Check if the last layer is Final output layer
        self.assertTrue(len(self.control.vocab_idx) in result)

        # Check to ensure there are no elements after final output layer
        # final output layer is encoded: len(vocab_idx)
        if len(self.control.vocab_idx) in result:
            # Find the index of len(vocab_idx) in the sequence
            index_of_len_vocab_idx = result.index(len(self.control.vocab_idx))

            # Assert that len(vocab_idx) is the last element in the sequence
            self.assertEqual(index_of_len_vocab_idx, len(result) - 1, "len(vocab_idx) is not the last element in the sequence.")

if __name__ == '__main__':
    for _ in range(20):
        # unittest.main(argv=[''],exit=False)
        runner = unittest.TextTestRunner()
        runner.run(suite())