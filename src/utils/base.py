import numpy as np
from scipy.linalg import eigh

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

def _eigh(X, eigvals=None):
    """
    A wrapper function of numpy.linalg.eigh and scipy.linalg.eigh.

    Parameters
    ----------
    X: array-like, shape (a, a)
        A symmetric matrix for which to compute eigenvalues and eigenvectors.

    eigvals: tuple, (lo, hi)
        Indices of the smallest and largest (in ascending order) eigenvalues and corresponding eigenvectors
        to be returned: 0 <= lo <= hi <= M-1. If omitted, all eigenvalues and eigenvectors are returned.

    Returns
    -------
    e: array-like, shape (a) or (n_dims)
        Eigenvalues in descending order.

    V: array-like, shape (a, a) or (a, n_dims)
        Corresponding eigenvectors, where each column is an eigenvector.
    """
    if eigvals != None:
        e, V = eigh(X, eigvals=eigvals)
    else:
        # numpy's eigh is faster than scipy's when all calculating eigenvalues and eigenvectors
        e, V = np.linalg.eigh(X)

    e, V = e[::-1], V[:, ::-1]

    return e, V
    
    
def _eigen_basis(X, eigvals=None):
    """
    Compute the eigenbasis of the input matrix.

    Parameters
    ----------
    X: array-like, shape (a, a)
        Input matrix for which to compute the eigenbasis.

    eigvals: tuple (low, high), default=None
        Indices of the smallest and largest (in ascending order) eigenvalues and corresponding eigenvectors
        to be returned: 0 <= lo <= hi <= M-1. If omitted, all eigenvalues and eigenvectors are returned.

    Returns
    -------
    e: array-like, shape (a) or (n_dims)
        Eigenvalues in descending order.

    V: array-like, shape (a, a) or (a, n_dims)
        Corresponding eigenvectors, where each column is an eigenvector.
    """

    try:
        e, V = _eigh(X, eigvals=eigvals)
    except np.linalg.LinAlgError:
        # if it not converges, try with tiny salt
        salt = 1e-8 * np.eye(X.shape[0])
        e, V = eigh(X + salt, eigvals=eigvals)

    return e, V


def _get_eigvals(n, n_subdims, higher):
    """
        Calculate the range of eigenvalues to compute in terms of indices.

    Parameters
    ----------
    n: int
        Total number of dimensions in the input data.

    n_subdims: int, default=None
        Desired number of dimensions in the subspace. If None, all dimensions are used.

    higher: boolean, default=True
        If True, compute the largest `n_subdim` eigenvalues. If False, compute the smallest.

    Returns
    -------
    eigvals: tuple of 2 integers (low, high)
        Indices of the smallest and largest (in ascending order) eigenvalues to be computed.
    """

    if n_subdims is None:
        return None

    if higher:
        low = max(0, n - n_subdims)
        high = n - 1
    else:
        low = 0
        high = min(n - 1, n_subdims - 1)

    return low, high 



def subspace_bases(X, n_subdims=None, higher=True, return_eigvals=False):
    """
    Return subspace basis using PCA

    Parameters
    ----------
    X: array-like, shape (n_dimensions, n_vectors)
        data matrix
    n_subdims: integer
        number of subspace dimension
    higher: bool
        if True, this function returns eigenvectors collesponding
        top-`n_subdims` eigenvalues. default is True.
    return_eigvals: bool
        if True, this function also returns eigenvalues.
    Returns
    -------
    V: array-like, shape (n_dimensions, n_subdims)
        bases matrix
    w: array-like shape (n_subdims)
        eigenvalues
    """

    if X.shape[0] <= X.shape[1]:
        eigvals = _get_eigvals(X.shape[0], n_subdims, higher)
        # get eigenvectors of autocorrelation matrix X @ X.T
        w, V = _eigen_basis(X @ X.T, eigvals=eigvals)
    else:
        # use linear kernel to get eigenvectors
        A, w = dual_vectors(X.T @ X, n_subdims=n_subdims, higher=higher)
        V = X @ A

    if return_eigvals:
        return V, w
    else:
        return V


def dual_vectors(K, n_subdims=None, higher=True, eps=1e-6):
    """
    Calc dual representation of vectors in kernel space

    Parameters:
    -----------
    K :  array-like, shape: (n_samples, n_samples)
        Grammian Matrix of X: K(X, X)
    n_subdims: int, default=None
        Number of vectors of dual vectors to return
    higher: boolean, default=None
        If True, this function returns eigenbasis corresponding to 
            higher `n_subdims` eigenvalues in descending order.
        If False, this function returns eigenbasis corresponding to 
            lower `n_subdims` eigenvalues in descending order.
    eps: float, default=1e-20
        lower limit of eigenvalues

    Returns:
    --------
    A : array-like, shape: (n_samples, n_samples)
        Dual replesentation vectors.
        it satisfies lambda[i] * A[i] @ A[i] == 1, where lambda[i] is i-th biggest eigenvalue
    e:  array-like, shape: (n_samples, )
        Eigen values descending sorted
    """

    eigvals = _get_eigvals(K.shape[0], n_subdims, higher)
    e, A = _eigen_basis(K, eigvals=eigvals)

    # replace if there are too small eigenvalues
    e[(e < eps)] = eps

    A = A / np.sqrt(e)

    return A, e


def cross_similarities(refs, inputs):
    """
    Calc similarities between each reference spaces and each input subspaces

    Parameters:
    -----------
    refs: list of array-like (n_dims, n_subdims_i)
    inputs: list of array-like (n_dims, n_subdims_j)

    Returns:
    --------
    similarities: array-like, shape (n_refs, n_inputs)
    """

    similarities = []
    for _input in inputs:
        sim = []
        for ref in refs:
            sim.append(mean_square_elements(ref.T @ _input))
        similarities.append(sim)

    similarities = np.array(similarities)

    return similarities

