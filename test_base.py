import sys
sys.path.append('src/utils')
from base import mean_square_elements, canonical_angle, canonical_angle_matrix
import unittest 
import numpy as np

class TestBase(unittest.TestCase):
    def setUp(self):
       self.matrix1 = np.array([[1, 2], [3, 4]])
    # Define basis matrices for testing canonical_angle and canonical_angle_matrix
       self.X = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
       self.Y = [np.array([[2, 3], [5, 6]]), np.array([[8, 9], [11, 12]])]

    def test_mean_square_elements(self):
        self.assertEqual(mean_square_elements(self.matrix1), 15)
    
    def test_canonical_angle(self):
        expected_output = 2574.0
        self.assertEqual(canonical_angle(self.X[0], self.X[1]), expected_output)

    def test_canonical_angle_matrix(self):
        data = [[1099., 6031.], [6419., 35639.]]
        expected_output = np.array(data)
        np.testing.assert_allclose(canonical_angle_matrix(self.X, self.Y), expected_output)

if __name__ == '__main__':
    unittest.main()
