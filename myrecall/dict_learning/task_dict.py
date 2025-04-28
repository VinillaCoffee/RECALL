import os
import pickle
from copy import deepcopy

import numpy as np
from scipy import spatial
from sklearn.decomposition import DictionaryLearning, sparse_encode
from sklearn.linear_model import Lasso
from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import randomized_svd
from sklearn.metrics.pairwise import cosine_similarity


def overlap_1(arr1, arr2):
    return np.mean(arr1 == arr2)

def overlap_2(arr1, arr2):
    return np.mean(np.logical_and(arr1, arr2))

def overlap_3(arr1, arr2):
    n_union = np.sum(np.logical_or(arr1, arr2))
    n_intersected = np.sum(np.logical_and(arr1, arr2))
    return n_intersected / n_union


def clip_by_norm(x, c):
    clip_coef = c / (np.linalg.norm(x) + 1e-6)
    clip_coef_clipped = min(1.0, clip_coef)
    return x * clip_coef_clipped


def _update_dict(
    dictionary,
    Y,
    code,
    A=None,
    B=None,
    c=1e-1,
    verbose=False,
    random_state=None,
    positive=False,
):
    """Update the dense dictionary factor in place.

    Parameters
    ----------
    dictionary : ndarray of shape (n_components, n_features)
        Value of the dictionary at the previous iteration.

    Y : ndarray of shape (n_samples, n_features)
        Data matrix.

    code : ndarray of shape (n_samples, n_components)
        Sparse coding of the data against which to optimize the dictionary.

    A : ndarray of shape (n_components, n_components), default=None
        Together with `B`, sufficient stats of the online model to update the
        dictionary.

    B : ndarray of shape (n_features, n_components), default=None
        Together with `A`, sufficient stats of the online model to update the
        dictionary.

    verbose: bool, default=False
        Degree of output the procedure will print.

    random_state : int, RandomState instance or None, default=None
        Used for randomly initializing the dictionary. Pass an int for
        reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    positive : bool, default=False
        Whether to enforce positivity when finding the dictionary.
    """
    n_samples, n_components = code.shape
    random_state = check_random_state(random_state)

    if A is None:
        A = code.T @ code
    if B is None:
        B = Y.T @ code

    n_unused = 0
    for k in range(n_components):
        if A[k, k] > 1e-6:
            # 1e-6 is arbitrary but consistent with the spams implementation
            # -np.inf means that never resample atoms.
            dictionary[k] += (B[:, k] - A[k] @ dictionary) / A[k, k]
        else:
            # kth atom is almost never used -> sample a new one from the data
            newd = Y[random_state.choice(n_samples)]

            # add small noise to avoid making the sparse coding ill conditioned
            noise_level = 1.0 * (newd.std() or 1)  # avoid 0 std
            noise = random_state.normal(0, noise_level, size=len(newd))
            dictionary[k] = newd + noise
            code[:, k] = 0
            n_unused += 1

        if positive:
            np.clip(dictionary[k], 0, None, out=dictionary[k])

        # Projection on the constraint set ||V_k||_2 <= c
        dictionary[k] = clip_by_norm(dictionary[k], c)

    if verbose and n_unused > 0:
        print(f"{n_unused} unused atoms resampled.")

    return dictionary


class OnlineDictLearnerV2(object):
    def __init__(self, 
                 n_features: int, 
                 n_components: int,
                 seed: int=0,
                 init_sample: np.ndarray=None,
                 c: float=1e-2,
                 scale: float=1.0,
                 alpha: float=1e-3,
                 method: str='lasso_lars', # ['lasso_cd', 'lasso_lars', 'threshold']
                 positive_code: bool=False,
                 scale_code: bool=False,
                 verbose=True):

        self.N = 0
        self.rng = np.random.RandomState(seed=seed)
        self.A = np.zeros((n_components, n_components))
        self.B = np.zeros((n_features, n_components))

        if init_sample is None:
            dictionary = self.rng.normal(loc=0.0, scale=scale, size=(n_components, n_features))
            # Projection on the constraint set ||V_k||_2 <= c
            for j in range(n_components):
                dictionary[j] = clip_by_norm(dictionary[j], c)
                
        else:
            _, S, dictionary = randomized_svd(init_sample, n_components, random_state=self.rng)
            dictionary = S[:, np.newaxis] * dictionary
            r = len(dictionary)
            if n_components <= r:
                dictionary = dictionary[:n_components, :]
            else:
                dictionary = np.r_[
                    dictionary,
                    np.zeros((n_components - r, dictionary.shape[1]), dtype=dictionary.dtype),
                ]

        dictionary = check_array(dictionary, order="F", copy=False)
        self.D = np.require(dictionary, requirements="W")
        
        self.C = None
        self.c = c
        self.alpha = alpha
        self.method = method
        self.archives = None
        self._verbose = verbose
        self.arch_code = None
        self._positive_code = positive_code
        self._scale_code = scale_code
        self.change_of_dict = []
        
    def get_alpha(self, sample: np.ndarray):
        code = sparse_encode(
            sample,
            self.D,
            algorithm=self.method, 
            alpha=self.alpha,
            check_input=False,
            positive=self._positive_code,
            max_iter=10000)

        # recording
        if self.arch_code is None:
            self.arch_code = code
        else:
            self.arch_code = np.vstack([self.arch_code, code])

        if self._scale_code:
            scaled_code = self._scale_coeffs(code)
            assert np.max(scaled_code) == 1.0
        else:
            scaled_code = code
        
        if self._verbose:
            recon = np.dot(code, self.D)
            print('Sparse Coding of Task Embedding')
            print(f'Rate of deactivate: {1-np.mean(np.heaviside(scaled_code, 0)):.4f}')
            print(f'Rate of activate: {np.mean(np.heaviside(scaled_code, 0)):.4f}')
            print(f'Reconstruction loss: {np.mean((sample - recon)**2):.4e}')
            print('----------------------------------')

        return scaled_code

    def update_dict(self, codes: np.ndarray, sample: np.ndarray):
        self.N += 1
        if self._scale_code:
            codes = self._rescale_coeffs(codes)
        # recording
        if self.C is None:
            self.C = codes
            self.archives = sample
        else:
            self.C = np.vstack([self.C, codes])
            self.archives = np.vstack([self.archives, sample])
        assert self.C.shape[0] == self.N
            
        # Update the auxiliary variables
        self.A += np.dot(codes.T, codes)
        self.B += np.dot(sample.T, codes)

        # pre-verbose
        if self._verbose:
            recons = np.dot(self.C, self.D)
            print('Dictionary Learning')
            print(f'Pre-MSE loss: {np.mean((self.archives - recons)**2):.4e}')

        old_D = deepcopy(self.D)

        # Update dictionary
        self.D = _update_dict(
            self.D,
            sample,
            codes,
            self.A,
            self.B,
            self.c,
            verbose=self._verbose,
            random_state=self.rng,
            positive=self._positive_code
        )
        
        self.change_of_dict.append(np.linalg.norm(old_D - self.D)**2/self.D.size)

        # post-verbose
        if self._verbose:
            recons = np.dot(self.C, self.D)
            print(f'Post-MSE loss: {np.mean((self.archives - recons)**2):.4e}')
            print('----------------------------------')

    def _scale_coeffs(self, alpha: np.ndarray):
        # constrain the alpha to [0, 1]
        assert self._positive_code
        self.factor = np.max(alpha)
        return alpha / self.factor

    def _rescale_coeffs(self, alpha_scaled: np.ndarray):
        assert self._positive_code
        return alpha_scaled * self.factor

    def _compute_overlapping(self):
        binary_masks = np.heaviside(self.arch_code, 0)
        overlap_mat = np.empty((binary_masks.shape[0], binary_masks.shape[0]))

        for i in range(binary_masks.shape[0]):
            for j in range(binary_masks.shape[0]):
                overlap_mat[i, j] = overlap_3(binary_masks[i], binary_masks[j])

        return overlap_mat

    def save(self, save_path: str):
        
        saved_dict = {
            "A": self.A,
            "B": self.B,
            "D": self.D,
        }

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(saved_dict, f)

    def load(self, load_path: str):
        with open(load_path, 'rb') as f:
            loaded_dict = pickle.load(f)

        self.A = loaded_dict["A"]
        self.B = loaded_dict["B"]
        self.D = loaded_dict["D"] 