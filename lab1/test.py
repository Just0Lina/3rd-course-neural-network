import unittest
import numpy as np
import pandas as pd

from main import entropy, gain_ratio

class TestEntropy(unittest.TestCase):
    def test_entropy(self):
        labels = np.array([0, 0, 1, 1, 1])
        expected_entropy = 0.971
        calculated_entropy = entropy(labels)
        self.assertAlmostEqual(expected_entropy, calculated_entropy, places=3)

    def test_entropy_single_class(self):
        labels = np.array([0, 0, 0, 0, 0])
        expected_entropy = 0.0
        calculated_entropy = entropy(labels)
        self.assertAlmostEqual(expected_entropy, calculated_entropy)

    def test_entropy_multiple_classes(self):
        labels = np.array([0, 1, 0, 1, 2, 0, 1])
        expected_entropy = 1.44132
        calculated_entropy = entropy(labels)
        self.assertAlmostEqual(expected_entropy, calculated_entropy)


class TestGainRatio(unittest.TestCase):
    def test_gain_ratio(self):
        data = pd.DataFrame({'feature': [1, 1, 2, 2, 3], 'target': [0, 0, 1, 1, 1]})
        target = 'target'
        feature = 'feature'
        expected_gain_ratio = 0.63797
        calculated_gain_ratio = gain_ratio(data, target, feature)
        self.assertAlmostEqual(expected_gain_ratio, calculated_gain_ratio, places=4)

    def test_gain_ratio_equal_values(self):
        data = pd.DataFrame({'feature': [1, 1, 1, 1, 1], 'target': [0, 0, 1, 1, 1]})
        target = 'target'
        feature = 'feature'
        expected_gain_ratio = 0.0
        calculated_gain_ratio = gain_ratio(data, target, feature)
        self.assertAlmostEqual(expected_gain_ratio, calculated_gain_ratio)

    def test_gain_ratio_different_values(self):
        data = pd.DataFrame({'feature': [1, 1, 2, 2, 3], 'target': [0, 0, 1, 1, 1]})
        target = 'target'
        feature = 'feature'
        expected_gain_ratio = 0.637974026
        calculated_gain_ratio = gain_ratio(data, target, feature)
        self.assertAlmostEqual(expected_gain_ratio, calculated_gain_ratio)

if __name__ == '__main__':
    unittest.main()