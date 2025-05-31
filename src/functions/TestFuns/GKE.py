from modules import *
from functions.utils import *

# ------------------------ # ------------------------ # ------------------------ # ------------------------ 
def HS_test_stat(SX, SY,  SX_ED,  SY_ED = None):
    N = SX.shape[0]
    return np.linalg.norm(inv_sqrtm_ED(SX_ED)@ (SY - SX) @ inv_sqrtm_ED(SX_ED),'fro')
# ------------------------ # ------------------------ # ------------------------ # ------------------------ 
def KL_test_stat(mX, mY, SX, SY, SX_ED, SY_ED):
    N = SX.shape[0]
    return np.linalg.norm(inv_sqrtm_ED(SX_ED)@(mY-mX).T) - logdet2(inv_sqrtm_ED(SX_ED)@ (SY - SX) @ inv_sqrtm_ED(SX_ED),'fro')
# ------------------------ # ------------------------ # ------------------------ # ------------------------ 

class GKE_HS_two_sample_test:
    """
    Two-sample test to determine if two samples come from the same distribution.
    Based on Hilbert-Schmidt discrepancy in [Santoro, Waghmare and Panaretos '24]

    Arguments:
    X (ndarray): First sample (n x d).
    Y (ndarray): Second sample (m x d).
    kernel (function): Kernel function. Default is Laplacian kernel, with bandiwth set to median heuristic

    Attributes:
    obs_value (float): Observed MMD statistic value.

    Methods:
    __call__: Perform the permutation test and return the p-value.
    """

    def __init__(self, X, Y, kernel=None):
        """
        Initialize the test class.
        """
        self.X = X; self.Y = Y;
        n = len(X); m = len(Y)
        fullsample = np.concatenate([X, Y])

        if kernel is None:
            pairwise_dists = cdist(fullsample, fullsample, 'euclidean')
            median_dist = np.median(pairwise_dists[pairwise_dists > 0]) 
            bandwidth = 2 * median_dist
            self.kernel_matrix  = np.exp( - pairwise_dists / bandwidth)
        else:
            self.kernel_matrix  = kernel(fullsample, fullsample)  

        kxx = self.kernel_matrix[:n, :n]; kxy = self.kernel_matrix[:n, n:]; kyy = self.kernel_matrix[n:, n:]
        SX, SX_ED, SY, SY_ED = get_Smats_X_Y(kxx, kxy, kyy)
        self.obs_value = HS_test_stat(SX, SY, SX_ED, SY_ED)

        
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
        permuted_stats =[]
        for _ in np.arange(num_permutations):
            permuted_indices = np.random.permutation(n + n)
            reordered_kernel = self.kernel_matrix[permuted_indices][:, permuted_indices]
            _kxx = reordered_kernel[:n, :n] ; _kyy = reordered_kernel[n:, n:]; _kxy = reordered_kernel[:n, n:]
            _SX, _SX_ED, _SY, _SY_ED = get_Smats_X_Y(_kxx, _kxy, _kyy)
            permuted_stats.append(HS_test_stat(_SX, _SY, _SX_ED, _SY_ED))

        p_value = float(np.mean(permuted_stats > self.obs_value))

        if not return_stats:
            return p_value
        else:
            return permuted_stats, p_value

# ------------------------ # ------------------------ # ------------------------ # ------------------------ 

class GKE_KL_two_sample_test:
    """
    Two-sample test to determine if two samples come from the same distribution.
    Based on Regularised Kernel-KL-divergence in [Santoro, Waghmare and Panaretos '24]

    Arguments:
    X (ndarray): First sample (n x d).
    Y (ndarray): Second sample (m x d).
    kernel (function): Kernel function. Default is Laplacian kernel, with bandiwth set to median heuristic

    Attributes:
    obs_value (float): Observed MMD statistic value.

    Methods:
    __call__: Perform the permutation test and return the p-value.
    """

    def __init__(self, X, Y, kernel=None):
        """
        Initialize the test class.
        """
        self.X = X; self.Y = Y;
        n = len(X); m = len(Y)
        fullsample = np.concatenate([X, Y])

        if kernel is None:
            pairwise_dists = cdist(fullsample, fullsample, 'euclidean')
            median_dist = np.median(pairwise_dists[pairwise_dists > 0]) 
            bandwidth = 2 * median_dist
            self.kernel_matrix  = np.exp( - pairwise_dists / bandwidth)
        else:
            self.kernel_matrix  = kernel(fullsample, fullsample)  

        kxx = self.kernel_matrix[:n, :n]; kxy = self.kernel_matrix[:n, n:]; kyy = self.kernel_matrix[n:, n:]
        mX, mY = get_mvecs_X_Y(kxx, kxy, kyy)
        SX, SX_ED, SY, SY_ED = get_Smats_X_Y(kxx, kxy, kyy)
        
        self.obs_value = KL_test_stat(mX, mY, SX, SY, SX_ED, SY_ED)

        
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
        permuted_stats =[]
        for _ in np.arange(num_permutations):
            permuted_indices = np.random.permutation(n + n)
            reordered_kernel = self.kernel_matrix[permuted_indices][:, permuted_indices]
            _kxx = reordered_kernel[:n, :n] ; _kyy = reordered_kernel[n:, n:]; _kxy = reordered_kernel[:n, n:]
            _mX, _mY = get_mvecs_X_Y(_kxx, _kxy, _kyy)
            _SX, _SX_ED, _SY, _SY_ED = get_Smats_X_Y(_kxx, _kxy, _kyy)
            permuted_stats.append(KL_test_stat(_mX, _mY, _SX, _SY, _SX_ED, _SY_ED))
        p_value = float(np.mean(permuted_stats > self.obs_value))

        if not return_stats:
            return p_value
        else:
            return permuted_stats, p_value

# ------------------------ # ------------------------ # ------------------------ # ------------------------ 
