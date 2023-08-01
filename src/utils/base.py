import numpy as np

def mean_square_elements(matrix):
    """
    Calculate the mean square of the elements in the input matrix.

    Parameters:
    -----------
    matrix : array-like, shape: (n, m)
        The input matrix with n rows and m columns.

    Returns:
    --------
    mean_square : float
        The mean square of the elements, computed as the sum of all squared elements divided by the smaller dimension of the matrix (either number of rows or columns).
    """
    # Compute the sum of all squared elements in the matrix
    sum_of_squares = (matrix * matrix).sum()
    
    # Determine the smaller dimension of the matrix (either number of rows or columns)
    smaller_dimension = min(matrix.shape)
    
    # Compute the mean square of the elements
    mean_square = sum_of_squares / smaller_dimension
    
    return mean_square


##TODO - This function is not correct. It should be the mean square of the elements in the matrix resulting from Y @ X^T.
def canonical_angle(X, Y):
    """
    Calculate similarity between subspaces represented by basis matrices X and Y.
    Note: This function does not actually calculate canonical angles. It calculates 
    the mean square of the elements in the matrix resulting from Y @ X.T.

    Parameters:
    -----------
    X : array-like, shape: (n_subdim_X, n_dim)
        Basis matrix representing a subspace.
    Y : array-like, shape: (n_subdim_Y, n_dim)
        Basis matrix representing another subspace.

    Returns:
    --------
    similarity: float
        Similarity of X, Y.
    """
    return mean_square_elements(Y @ X.T)

def canonical_angle_matrix(X, Y):
    """
    Calculate similarity matrix between sets of subspaces.

    Parameters:
    -----------
    X : list of array-like, each with shape: (n_subdim, n_dim)
        Set of basis matrices, each representing a subspace.
    Y : list of array-like, each with shape: (n_subdim, n_dim)
        Set of basis matrices, each representing a subspace.

    Returns:
    --------
    C : array-like, shape: (n_set_X, n_set_Y)
        Similarity matrix, with each element representing the similarity between a pair of subspaces.
    """
    n_set_X, n_set_Y = len(X), len(Y)
    C = np.empty((n_set_X, n_set_Y))

    for x, sub_X in enumerate(X):
        for y, sub_Y in enumerate(Y):
            C[x, y] = canonical_angle(sub_X, sub_Y)

    return C

# Define basis matrices for testing canonical_angle and canonical_angle_matrix

X = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
Y = [np.array([[ 2, 3], [ 5, 6]]), np.array([[8, 9], [11, 12]])]
print(canonical_angle(X[0], X[1]))
print(canonical_angle_matrix(X, Y))