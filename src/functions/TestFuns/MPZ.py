from modules import *
from functions.utils import *


def mpz_test_stat(X, Y, center = True):
    """
    Perform the Masarotto-Panaretos-Zemel for equality of (centered) distributions.
    
    Parameters:
    X (np.ndarray): Sample from the first distribution, shape (n, d).
    Y (np.ndarray): Sample from the second distribution, shape (m, d).
    center (bool): If True, center the samples before computing the test statistic.
    
    Returns:
    float: Test statistic
    """
    if center:
        X = X - X.mean(0)
        Y = Y - Y.mean(0)
    d = X.shape[1]

    SX, UX = efficient_cov_eigdec(X - X.mean(0))
    SY, UY = efficient_cov_eigdec(Y - Y.mean(0))

    sqrtm_CX = sqrtm_ED((SX, UX))
    sqrtm_inv_CX = inv_sqrtm_ED((SX, UX))
    
    S_MT, U_MT  = EIG_DEC(sqrtm_CX @ (np.cov(Y.T)) @ sqrtm_CX)
    otmap_XY = sqrtm_inv_CX@sqrtm_ED((S_MT, U_MT))@sqrtm_inv_CX

    otmap_YX = sqrtm_CX@inv_sqrtm_ED((S_MT, U_MT))@sqrtm_CX

    return np.linalg.norm( otmap_XY - np.eye(d),'fro') + np.linalg.norm(otmap_YX - np.eye(d),'fro')

class MPZ_two_sample_test:
    """
    Two-sample OT-based test to determine if two samples come from the same distribution.
    Based on [ Masarotto, Panaretos & Zemel, '24]

    Arguments:
    X (ndarray): First sample (n x d).
    Y (ndarray): Second sample (m x d)
    reg (float): Regularization parameter for the covariance matrices, default 1e-5

    Attributes:
    obs_value (float): Observed statistic value.

    Methods:
    __call__: Perform the permutation test and return the p-value.
    """

    def __init__(self, X, Y, reg = 1e-5):
        """
        Initialize the test class.
        """
        self.X = X; self.Y = Y
        self.Z = np.vstack([X, Y]) - np.vstack([X, Y]).mean(0)
        self.obs_value = mpz_test_stat(X,Y)

    def __call__(self, num_permutations=100, return_stats=False):
        """
        Perform the permutation test and return the p-value.

        Parameters:
        num_permutations (int): Number of permutations for calibrating the test.
        return_stats (bool): If True, return the permuted statistics as well.

        Returns:
        float: p-value of the test.
        tuple: (permuted_stats, p_value) if return_stats is True.
        """
        n = len(self.X); m = len(self.Y)
        d = self.X.shape[1]
        permuted_stats = []
        for _ in range(num_permutations):
            permuted_indices = np.random.permutation(n + m)
            _X = self.Z[permuted_indices[:n]]
            _Y = self.Z[permuted_indices[n:]]
            permuted_stats.append( mpz_test_stat(_X,_Y))

        p_value = float(np.mean(np.array(permuted_stats) >= self.obs_value))
        if not return_stats:
            return p_value
        else:
            return permuted_stats, p_value
