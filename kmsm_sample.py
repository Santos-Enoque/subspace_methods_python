import h5py
import numpy as np
from scipy.linalg import block_diag
from sklearn.metrics.pairwise import rbf_kernel as _rbf_kernel
from scipy.linalg import eigh
from tqdm import tqdm as track
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import normalize as _normalize, LabelEncoder
from scipy.linalg import block_diag
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

file_path = "data/Face_data.mat"

def mean_square_singular_values(matrix):
    # Singular Value Decomposition
    U, s, Vh = np.linalg.svd(matrix, full_matrices=False)
    
    # Mean square of the singular values
    mean_square = np.mean(s ** 2)
    
    return mean_square


##TODO - This function is not correct. It should be the mean square of the elements in the matrix resulting from Y @ X^T.
def canonical_angle(X, Y):
    return mean_square_singular_values(Y @ X.T)

def canonical_angle_matrix(X, Y):
    n_set_X, n_set_Y = len(X), len(Y)
    C = np.empty((n_set_X, n_set_Y))

    for x, sub_X in enumerate(X):
        for y, sub_Y in enumerate(Y):
            C[x, y] = canonical_angle(sub_X, sub_Y)

    return C

def _eigh(X, eigvals=None):
    if eigvals != None:
        e, V = eigh(X, eigvals=eigvals)
    else:
        e, V = np.linalg.eigh(X)

    e, V = e[::-1], V[:, ::-1]

    return e, V
    
    
def _eigen_basis(X, eigvals=None):
    try:
        e, V = _eigh(X, eigvals=eigvals)
    except np.linalg.LinAlgError:
        # if it not converges, try with tiny salt
        salt = 1e-8 * np.eye(X.shape[0])
        e, V = eigh(X + salt, eigvals=eigvals)

    return e, V


def _get_eigvals(n, n_subdims, higher):
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
    eigvals = _get_eigvals(K.shape[0], n_subdims, higher)
    e, A = _eigen_basis(K, eigvals=eigvals)

    e[(e < eps)] = eps

    A = A / np.sqrt(e)

    return A, e


def cross_similarities(refs, inputs):
    similarities = []
    for _input in inputs:
        sim = []
        for ref in refs:
            sim.append(mean_square_singular_values(ref.T @ _input))
        similarities.append(sim)

    similarities = np.array(similarities)

    return similarities


#ANCHOR - Evaluation
def calc_er(X, y, labels=None, data_type='S'):
    _, _pred_class, mappings = _calc_preparations(X, labels, data_type)

    if mappings is not None:
        _y = np.array([mappings[v] for v in y])
    else:
        _y = y

    er = (_pred_class == _y).mean()

    return er


def calc_eer(X, labels=None, data_type='S'):
    _labels, _pred_class, _ = _calc_preparations(X, labels, data_type)
    n_classes = len(np.unique(_labels))
    print(_labels)
    B = np.eye(n_classes)[_labels][:, _pred_class]

    # make them 1D
    _X = X.reshape(-1)
    _B = B.reshape(-1)

    # get sorted index regarding to data_type
    if data_type == 'S':
        sort_idx = np.argsort(_X)
    else:
        sort_idx = np.argsort(_X)[:, :, -1]

    # number of positive/negative prediction
    n_pos = _B[_B == 1].size
    n_neg = _B.size - n_pos

    # make _B sorted
    _B = _B[sort_idx]

    # False Acceptance Rate
    far = 1 - np.cumsum(_B == 0) / n_neg
    # False Rejection Rate
    frr = np.cumsum(_B == 1) / n_pos

    thresh_idx = np.abs(far - frr).argmin()
    eer = (far[thresh_idx] + frr[thresh_idx]) / 2
    thresh = _X[sort_idx][thresh_idx]

    return eer, thresh


def _calc_preparations(X, labels, data_type):
    # check data_type
    if data_type not in {'S', 'D'}:
        raise ValueError('`data_type` must be \'S\' or \'D\'')

    # check shape
    if len(X.shape) != 2:
        raise ValueError('`X` must be 2 dimensional matrix')
    if labels is None:
        _labels = np.arange(X.shape[1])
        mappings = None
    else:
        classes = np.unique(labels)
        mappings = np.stack((classes, np.arange(len(classes))), axis=1)
        mappings = dict(mappings)
        _labels = np.array([mappings[v] for v in labels])

    if data_type == 'S':
        idx = X.argmax(axis=1)
    else:
        idx = X.argmin(axis=1)

    _pred_class = _labels[idx]

    return _labels, _pred_class, mappings


#ANCHOR - Kernel Methods
def l2_kernel(X, Y):
    XX = (X**2).sum(axis=0)[:, None]
    YY = (Y**2).sum(axis=0)[None, :]
    x = XX - 2 * (X.T @ Y) + YY
    
    return x


def rbf_kernel(X, Y, sigma=None):
    n_dims = X.shape[0]
    if sigma is None:
        sigma = np.sqrt(n_dims / 2)

    x = l2_kernel(X, Y)

    # gausiann kernel, (n_samples_X, n_samples_Y)
    x = np.exp(-0.5 * x / (sigma**2))
    return x


def linear_kernel(X, Y):
    return X.T @ Y

#ANCHOR - Base Methods
class SMBase(BaseEstimator, ClassifierMixin):
    param_names = {'normalize', 'n_subdims'}

    def __init__(self, n_subdims, normalize=False, faster_mode=False):
        self.n_subdims = n_subdims
        self.normalize = normalize
        self.faster_mode = faster_mode
        self.le = LabelEncoder()
        self.dic = None
        self.labels = None
        self.n_classes = None
        self._test_n_subdims = None
        self.params = ()

    def get_params(self, deep=True):
        return {name: getattr(self, name) for name in self.param_names}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def _prepare_X(self, X):
        # normalize each vectors
        if self.normalize:
            X = [_normalize(_X) for _X in X]

        X = [_X.T for _X in X]

        return X

    def _prepare_y(self, y):
        # converted labels
        y = self.le.fit_transform(y)
        self.labels = y

        # number of classes
        self.n_classes = self.le.classes_.size

        # number of data
        self.n_data = len(y)

        return y

    def fit(self, X, y):
        # ! X[i] will transposed for conventional
        X = self._prepare_X(X)
        y = self._prepare_y(y)

        self._fit(X, y)

    def _fit(self, X, y):
        breakpoint()
        dic = [subspace_bases(_X, self.n_subdims) for _X in X]
        # dic,  (n_classes, n_dims, n_subdims)
        dic = np.array(dic)
        self.dic = dic

    def predict(self, X):
        if self.faster_mode and hasattr(self, 'fast_predict_proba'):
            proba = self.fast_predict_proba(X)
        else:
            proba = self.predict_proba(X)

        salt = 1e-3
        assert proba.min() > 0.0 - salt, 'some probabilities are smaller than 0! min value is {}'.format(proba.min())
        assert proba.max() < 1.0 + salt, 'some probabilities are bigger than 1! max value is {}'.format(proba.max())
        proba = np.clip(proba, 0, 1)
        return self.proba2class(proba)

    def proba2class(self, proba):
        pred = self.labels[np.argmax(proba, axis=1)]
        return self.le.inverse_transform(pred)

    def predict_proba(self, X):
        X = self._prepare_X([X])[0]
        pred = self._predict_proba(X)
        return pred

    def _predict_proba(self, X):
        raise NotImplementedError('_predict is not implemented')


class KernelSMBase(SMBase):
    param_names = {'normalize', 'n_subdims', 'sigma'}

    def __init__(self, n_subdims, normalize=False, sigma=None, faster_mode=False):
        super(KernelSMBase, self).__init__(n_subdims, normalize, faster_mode)
        self.sigma = sigma

    def _fit(self, X, y):
        coeff = []
        for _X in track(X):
            K = rbf_kernel(_X, _X, self.sigma)
            _coeff, _ = dual_vectors(K, self.n_subdims)
            coeff.append(_coeff)

        self.dic = list(zip(X, coeff))


class MSMInterface(object):
    def predict_proba(self, X):
        X = self._prepare_X(X)

        pred = []
        for _X in X:
            gramians = self._get_gramians(_X)
            c = [mean_square_singular_values(g) for g in gramians]
            pred.append(c)
        return np.array(pred)

    def _get_gramians(self, X):
        raise NotImplementedError('_get_gramians is not implemented')
        
    @property
    def test_n_subdims(self):
        if self._test_n_subdims is None:
            return self.n_subdims
        return self._test_n_subdims
    
    @test_n_subdims.setter
    def test_n_subdims(self, v):
        assert isinstance(v, int)
        self._test_n_subdims = v
        
    @test_n_subdims.deleter
    def test_n_subdims(self):
        self._test_n_subdims = None

#ANCHOR - KMSM IMPLEMENTATION



class KernelMSM(MSMInterface, KernelSMBase):
    def _get_gramians(self, X):
        K = rbf_kernel(X, X, self.sigma)
        in_coeff, _ = dual_vectors(K, self.test_n_subdims)
        
        gramians = []
        for i in range(self.n_data):
            ref_X, ref_coeff = self.dic[i]
            _K = rbf_kernel(ref_X, X, self.sigma)
            S = ref_coeff.T.dot(_K.dot(in_coeff))
            gramians.append(S)
        return np.array(gramians)


    def fast_predict_proba(self, X):
        n_input = len(X)
        n_ref =  len(self.dic)
        X = self._prepare_X(X)
        
        ref_Xs, ref_coeffs = [], []
        for ref_X, ref_coeff in self.dic:
            ref_Xs.append(ref_X)
            ref_coeffs.append(ref_coeff)
    
        ref_mappings = np.array([i for i in range(len(ref_Xs)) for _ in range(ref_coeffs[i].shape[1])])
        ref_Xs = np.hstack(ref_Xs)
        ref_coeffs = block_diag(*ref_coeffs)
        
        in_coeffs = []
        for _X in X:
            K = rbf_kernel(_X, _X, self.sigma)
            in_coeff, _ = dual_vectors(K, self.test_n_subdims)
            in_coeffs.append(in_coeff)
        in_mappings = np.array([i for i in range(n_input) for _ in range(in_coeffs[i].shape[1])])
        in_Xs = np.hstack(X)
        in_coeffs = block_diag(*in_coeffs)
        
        K = rbf_kernel(in_Xs, ref_Xs, self.sigma)
        del ref_Xs, in_Xs
        
        S = in_coeffs.T.dot(K).dot(ref_coeffs)
        del in_coeffs, ref_coeffs, K
        
        # Split matrix into (n_input x n_ref) blocks
        in_split = np.where(np.diff(np.pad(in_mappings, (1, 0), 'constant')))[0]
        ref_split = np.where(np.diff(np.pad(ref_mappings, (1, 0), 'constant')))[0]
        S = [np.hsplit(_S, ref_split) for _S in np.vsplit(S, in_split)]
        
        vmssv = np.vectorize(lambda i, j: mean_square_singular_values(S[i][j]))
        pred = vmssv(*np.meshgrid(np.arange(n_input), np.arange(n_ref))).T
        
        del S, X
        return np.array(pred)
   
    

def load_data(file_name):
    with h5py.File(file_name, "r") as f:
        # Extract the data from the file
        train_X = np.array(f["X1"])
        train_y = np.arange(len(train_X))
        test_X = np.array(f["X2"])
        test_X = test_X.reshape(-1, test_X.shape[-2], test_X.shape[-1])
        test_y = np.array([[i] * 36 for i in range(10)]).flatten()

    return train_X, train_y, test_X, test_y

train_X, train_y, test_X, test_y = load_data(file_path)

model = KernelMSM(n_subdims=5, sigma=100, faster_mode=True)
model.fit(train_X, train_y)

pred = model.predict(test_X)
print(f"pred: {pred}\ntrue: {test_y}\naccuracy: {(pred == test_y).mean()}")