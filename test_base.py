import sys
sys.path.append('src/utils')
from base import mean_square_elements, canonical_angle, canonical_angle_matrix, _eigh, _get_eigvals, _eigen_basis, subspace_bases, dual_vectors
import unittest 
import numpy as np

class TestBase(unittest.TestCase):
    def setUp(self):
       self.matrix1 = np.array([[1, 2], [3, 4]])
    # Define basis matrices for testing canonical_angle and canonical_angle_matrix
       self.X = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
       self.Y = [np.array([[2, 3], [5, 6]]), np.array([[8, 9], [11, 12]])]
        # This is a symmetric matrix
       self.matrix2 = np.array([[4, 1], [1, 4]])
       self.K = np.array([[4, 4], [3, 3]])

        # These are the expected eigenvalues and eigenvectors of matrix2
       self.expected_eigenvalues = np.array([5, 3])
       self.expected_eigenvectors = np.array([[0.70710678, -0.70710678], [0.70710678, 0.70710678]])

    def test_mean_square_elements(self):
        self.assertEqual(mean_square_elements(self.matrix1), 15)
    
    def test_canonical_angle(self):
        expected_output = 2574.0
        self.assertEqual(canonical_angle(self.X[0], self.X[1]), expected_output)

    def test_canonical_angle_matrix(self):
        data = [[1099., 6031.], [6419., 35639.]]
        expected_output = np.array(data)
        np.testing.assert_allclose(canonical_angle_matrix(self.X, self.Y), expected_output)
    
    def test_eigh(self):

        eigenvalues, eigenvectors = _eigh(self.matrix2)

        # Check that the computed eigenvalues match the expected eigenvalues
        np.testing.assert_array_almost_equal(eigenvalues, self.expected_eigenvalues, decimal=8)

        # Check that the computed eigenvectors match the expected eigenvectors
        np.testing.assert_array_almost_equal(eigenvectors, self.expected_eigenvectors, decimal=8)

    def test_get_eigvals(self):
        # Here, we assume n is 10 and n_subdims is 3
        n = 10
        n_subdims = 3

        # When higher is True
        expected_eigvals_higher = (7, 9) # Expected to return 7, 8, 9
        self.assertEqual(_get_eigvals(n, n_subdims, True), expected_eigvals_higher)

        # When higher is False
        expected_eigvals_lower = (0, 2) # Expected to return 0, 1, 2
        self.assertEqual(_get_eigvals(n, n_subdims, False), expected_eigvals_lower)
    
    def test_eigen_basis(self):
        # Without specifying eigvals
        eigenvalues, eigenvectors = _eigen_basis(self.matrix2)

        # Check that the computed eigenvalues match the expected eigenvalues
        np.testing.assert_array_almost_equal(eigenvalues, self.expected_eigenvalues, decimal=8)

        # Check that the computed eigenvectors match the expected eigenvectors
        np.testing.assert_array_almost_equal(eigenvectors, self.expected_eigenvectors, decimal=8)

        # When specifying eigvals
        eigenvalues, eigenvectors = _eigen_basis(self.matrix2, eigvals=(0, 0))  # here it only return the first eigenvalue/vector

        # # Check that the computed eigenvalue matches the first expected eigenvalue
        np.testing.assert_almost_equal(eigenvalues[0], self.expected_eigenvalues[1], decimal=8)


    def test_subspace_bases(self):
        expected_bases = np.array([[-0.8, 0.6], [-0.6, -0.8]])
        np.testing.assert_array_almost_equal(subspace_bases(self.K), expected_bases, decimal=8)


if __name__ == '__main__':
    unittest.main()
