from modules import *
from functions.utils import *


def mmd_from_kernel(kernel_matrix, n, m):
    k_xx = kernel_matrix [:n, :n]; np.fill_diagonal(k_xx, 0)
    k_yy = kernel_matrix [m:, m:]; np.fill_diagonal(k_yy, 0)
    k_xy = kernel_matrix [:n, m:]    
    obs_value = k_xx.sum()/(n*(n-1)) +  k_yy.sum()/(m*(m-1)) - 2* k_xy.sum()/(n*m)
    return obs_value

# ------------------------ # ------------------------ # ------------------------ # ------------------------ 

class MMD_two_sample_test:
    """
    Two-sample MMD test to determine if two samples come from the same distribution.
    Based on the method described in [Gretton et al., '12].

    Arguments:
    X (ndarray): First sample (n x d).
    Y (ndarray): Second sample (m x d).
    kernel (function): Kernel function. Default is Laplacian kernel, with bandiwth set to median heuristic

    Attributes:
    obs_value (float): Observed MMD statistic value.

    Methods:
    __call__: Perform the permutation test and return the p-value.
    """

    def __init__(self, X, Y, kernel = None):
        """
        Initialize the test class.
        """
        self.X = X; self.Y = Y
        n = len(X)
        m = len(Y)
        fullsample = np.vstack([X, Y])
        if kernel is None:
            pairwise_dists = cdist(fullsample, fullsample, 'euclidean')
            median_dist = np.median(pairwise_dists[pairwise_dists > 0]) 
            bandwidth = 2 * median_dist
            self.kernel_matrix  = np.exp( - pairwise_dists / bandwidth)
        else:
            self.kernel_matrix  = kernel(fullsample, fullsample)  
        self.obs_value = mmd_from_kernel(self.kernel_matrix, n, m)
        
    def __call__(self, num_permutations=1000, return_stats=False):
        """
        Perform the permutation test and return the p-value.

        Parameters:
        num_permutations (int): Number of permutations for calibrating the test.
        return_stats (bool): If True, return the permuted statistics as well.

        Returns:
        float: p-value of the test.
        tuple: (permuted_stats, p_value) if return_stats is True.
        """
        n, m = len(self.X), len(self.Y)
        permuted_stats = []
        for _ in range(num_permutations):
            permuted_indices = np.random.permutation(n + m)
            reordered_kernel = self.kernel_matrix[permuted_indices][:, permuted_indices]
            _value = mmd_from_kernel(reordered_kernel, n, m)
            permuted_stats.append(_value)
        p_value = float(np.mean(np.array(permuted_stats) >= self.obs_value))
        if not return_stats:
            return p_value
        else:
            return permuted_stats, p_value


class spectral_reg_MMD:
    """
    Modification to the MMD test based on spectral regularization by taking into account the covariance information 
    Based on the method described in [Hagrass et al., '24].

    Arguments:
    X (ndarray): First sample (n x d).
    Y (ndarray): Second sample (m x d).
    kernel (function): Kernel function.  ('laplacian', ' guassian', ...).
    reg (float): Regularization parameter for the covariance matrices.

    Attributes:
    obs_value (float): Observed MMD statistic value.

    Methods:
    __call__: Perform the permutation test and return the p-value.
    """