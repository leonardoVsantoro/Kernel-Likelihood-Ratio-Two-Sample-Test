from modules import *

from functions.tools import *



# ------------------------ # ------------------------ # ------------------------ # ------------------------ 
@profile
def FH_test_stat(KX, KY,  KX_ED,  KY_ED):
    N = KX.shape[0]
    return np.linalg.norm(inv_sqrtm_ED(KX_ED)@ KY @ inv_sqrtm_ED(KX_ED) - np.eye(N),'fro')/N**.5
# ------------------------ # ------------------------ # ------------------------ # ------------------------ 

class CKE_two_sample_test:
    """
    Two-sample test to determine if two samples come from the same distribution.
    Based on the "Covariance Kernel Embedding" (CKE) method proposed by [Santoro, Waghmare and Panaretos '24]

    Arguments:
    X (ndarray): First sample (n x d).
    Y (ndarray): Second sample (m x d).
    kernel (function): Kernel function; default is Laplacian kernel with \sigma = 10
    kappa_K (float): Implicit regularization parameter, sets the maximum conditioning number of the embedded covariances. (default is 1e6)

    Attributes:
    obs_value (float): Observed MMD statistic value.

    Methods:
    __call__: Perform the permutation test and return the p-value.
    """

    def __init__(self, X, Y, kappa_K=1e4, kernel=None):
        """
        Initialize the test class.
        """
        self.X = X; self.Y = Y; self.kappa_K = kappa_K
        n = len(X); m = len(Y)
        fullsample = np.concatenate([X, Y])

        if kernel is None:
            pairwise_dists = cdist(fullsample, fullsample, 'euclidean')
            median_dist = np.median(pairwise_dists[pairwise_dists > 0])  # Avoid zero distances
            bandwidth = 2 * median_dist
            self.kernel_matrix  = np.exp( - pairwise_dists / bandwidth)
        else:
            self.kernel_matrix  = kernel(fullsample, fullsample)  

        kxx = self.kernel_matrix[:n, :n]; kxy = self.kernel_matrix[:n, n:]; kyy = self.kernel_matrix[n:, n:]
        KX, KX_ED, KY, KY_ED = get_Kmats_X_Y(kxx, kxy, kyy, kappa_K)
        self.obs_value = FH_test_stat(KX, KY, KX_ED, KY_ED)

        
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
            _KX, _KX_ED, _KY, _KY_ED = get_Kmats_X_Y(_kxx, _kxy, _kyy, self.kappa_K)
            permuted_stats.append(FH_test_stat(_KX, _KY, _KX_ED, _KY_ED))

        p_value = float(np.mean(permuted_stats > self.obs_value))

        if not return_stats:
            return p_value
        else:
            return permuted_stats, p_value

# ------------------------ # ------------------------ # ------------------------ # ------------------------ 


def mmd_from_kernel(kernel_matrix, n, m):

    k_xx = kernel_matrix [:n, :n]
    np.fill_diagonal(k_xx, 0)
    k_yy = kernel_matrix [m:, m:]
    np.fill_diagonal(k_yy, 0)
    k_xy = kernel_matrix [:n, m:]    

    obs_value = k_xx.sum()/(n*(n-1)) +  k_yy.sum()/(m*(m-1)) - 2* k_xy.sum()/(n*m)
    return obs_value

class MMD_two_sample_test:
    """
    Two-sample MMD test to determine if two samples come from the same distribution.
    Based on the method described in [Gretton et al., '12].

    Arguments:
    X (ndarray): First sample (n x d).
    Y (ndarray): Second sample (m x d).
    kernel (function): Kernel function.
    reg (float): Regularization parameter for the covariance matrices.

    Attributes:
    obs_value (float): Observed MMD statistic value.

    Methods:
    __call__: Perform the permutation test and return the p-value.
    """


    def __init__(self, X, Y, kernel_name = 'gaussian',  kernel_bandwith = None):
        """
        Initialize the test class.
        """
        self.X = X; self.Y = Y
        n = len(X)
        m = len(Y)
        combined = np.vstack([X, Y])
        if kernel_name == 'laplacian':
            pairwise_dists = cdist(combined, combined, 'euclidean')
        else:
            pairwise_dists = cdist(combined, combined, 'sqeuclidean')
        if kernel_bandwith is None:
            median_dist = np.median(pairwise_dists[pairwise_dists > 0])
            kernel_bandwith = (2 * median_dist)

        self.kernel_matrix  = np.exp( - pairwise_dists /  kernel_bandwith)

        k_xx = self.kernel_matrix [:n, :n]    
        np.fill_diagonal(k_xx, 0)
        k_yy = self.kernel_matrix [m:, m:]    
        np.fill_diagonal(k_yy, 0)
        k_xy = self.kernel_matrix [:n, m:]    
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

# ------------------------ # ------------------------ # ------------------------ # ------------------------ 

class KNN_two_sample_test:
    """
    Two-sample KNN test to determine if two samples come from the same distribution.

    Arguments:
    X (ndarray): First sample (n x d).
    Y (ndarray): Second sample (m x d).
    k (int): Number of nearest neighbors to consider.

    Attributes:
    obs_value (float): Observed MMD statistic value.

    Methods:
    __call__: Perform the permutation test and return the p-value.
    """
    def __init__(self,X, Y, k=1):
        """
        Initialize the test class.
        """
        self.X = X; self.Y = Y
        Z = np.vstack([X, Y])
        n, m = len(X), len(Y)
        self.k = k 
        labels = np.array([1] * n + [2] * m)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
        distances, indices = nn.kneighbors(Z)
        self.indices  = indices[:, 1:]
        same_sample_count = 0
        for i, neighbors in enumerate(self.indices):
            same_sample_count += sum(labels[i] == labels[neighbors])
        obs_value = same_sample_count / (k * (n + m))
        self.labels = labels
        self.obs_value = obs_value

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
            permuted_labels = np.random.permutation(self.labels)
            permuted_count = 0
            for i, neighbors in enumerate(self.indices):
                permuted_count += sum(permuted_labels[i] == permuted_labels[neighbors])
            permuted_stats.append(permuted_count / (self.k * (n + m)))
        permuted_stats = np.array(permuted_stats)
        p_value = float(np.mean(np.array(permuted_stats) >= self.obs_value))
        if not return_stats:
            return p_value
        else:
            return permuted_stats, p_value

# ------------------------ # ------------------------ # ------------------------ # ------------------------ 

class FR_two_sample_test:
    """
    Two-sample FR Smirnov test to determine if two samples come from the same distribution.
    Generalises the Kolmogorov-Smirnov test to the multivariate setting.
    Based on [Friedman and Rafsky, '79]

    Arguments:
    X (ndarray): First sample (n x d).
    Y (ndarray): Second sample (m x d).

    Attributes:
    obs_value (float): Observed MMD statistic value.

    Methods:
    __call__: Perform the permutation test and return the p-value.
    """
    def __init__(self, X, Y):
        """
        Initialize the test class.
        """
        self.X = X; self.Y = Y
        Z = np.vstack([X, Y])
        n, m = len(X), len(Y)
        self.labels = np.array([1] * n + [2] * m)
        dist_matrix = distance_matrix(Z, Z)
        mst = minimum_spanning_tree(dist_matrix).toarray()
        self.edges  = np.array(np.nonzero(mst)).T
        runs = 0
        for edge in self.edges :
            if self.labels[edge[0]] != self.labels[edge[1]]:
                runs += 1
        self.obs_value = runs


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
        n,m = len(self.X), len(self.Y)
        permuted_stats = []
        for _ in range(num_permutations):
            permuted_labels = np.random.permutation(self.labels)
            permuted_runs_count = 0
            for edge in self.edges :
                if permuted_labels[edge[0]] != permuted_labels[edge[1]]:
                    permuted_runs_count += 1
            permuted_stats.append(permuted_runs_count)
        permuted_stats = np.array(permuted_stats)
        p_value = float(np.mean(permuted_stats <= self.obs_value)) # reversed here!
        if not return_stats:
            return p_value
        else:
            return permuted_stats, p_value

# ------------------------ # ------------------------ # ------------------------ # ------------------------ 
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

# ------------------------ # ------------------------ # ------------------------ # ------------------------ 


def hall_tajvidi_test_stat(X, Y, k=10):
    """
    Perform the Hall and Tajvidi nearest-neighbor test for equality of distributions.
    Base od [ Hall and Tajvidi, '02]
    
    Parameters:
    X (np.ndarray): Sample from the first distribution, shape (n, d).
    Y (np.ndarray): Sample from the second distribution, shape (m, d).
    k (int): Number of nearest neighbors to consider (default is 1).

    Returns:
    float: Test statistic (fraction of nearest neighbors from the opposite sample).
    """
    # Combine both samples and label them
    Z = np.vstack([X, Y])  # Combined dataset
    n, d = X.shape
    m = Y.shape[0]
    labels = np.array([0] * n + [1] * m)  # 0 for X, 1 for Y
    
    # Compute pairwise distances
    distances = distance.cdist(Z, Z)  # Pairwise distances between all points
    
    # Initialize count for nearest neighbors from the opposite sample
    count = 0
    
    for i in range(len(Z)):
        # Get sorted indices of distances for point i, excluding itself
        sorted_indices = np.argsort(distances[i])
        sorted_indices = sorted_indices[sorted_indices != i]  # Exclude itself
        
        # Find the k nearest neighbors
        nearest_labels = labels[sorted_indices[:k]]
        
        # Count how many of these neighbors belong to the opposite class
        if labels[i] == 0:  # Point is from X
            count += np.sum(nearest_labels == 1)
        else:  # Point is from Y
            count += np.sum(nearest_labels == 0)
    
    # Normalize the count to get the test statistic
    test_stat = count / (k * len(Z))
    
    return test_stat

class HT_two_sample_test:
    """
    Two-sample Hall and Tajvidi nearest-neighbor test for equality of distributions.
    Based on [ Hall and Tajvidi, '02]

    Arguments:
    X (ndarray): First sample (n x d).
    Y (ndarray): Second sample (m x d)
    k (int): Number of nearest neighbors to consider (default is 10).

    Attributes:
    obs_value (float): Observed statistic value.

    Methods:
    __call__: Perform the permutation test and return the p-value.
    """

    def __init__(self, X, Y, k = 10):
        """
        Initialize the test class.
        """
        self.X = X; self.Y = Y
        self.k = k
        # self.Z = np.vstack([X, Y]) - np.vstack([X, Y]).mean(0)        
        self.Z = np.vstack([X, Y])
        self.obs_value = hall_tajvidi_test_stat(X,Y,k)

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
            permuted_stats.append( hall_tajvidi_test_stat(_X,_Y,self.k))

        p_value = float(np.mean(np.array(permuted_stats) <= self.obs_value)) #reversed here!
        if not return_stats:
            return p_value
        else:
            return permuted_stats, p_value




class GKE_two_sample_test:
    """
    Two-sample test to determine if two samples come from the same distribution.
    Based on the "Gaussian Kernel Embedding" (GKE) method proposed by [Santoro, Waghmare and Panaretos '24 B]

    Arguments:
    X (ndarray): First sample (n x d).
    Y (ndarray): Second sample (m x d).
    kernel (function): Kernel function; default is Laplacian kernel with \sigma = 10
    kappa_K (float): Implicit regularization parameter, sets the maximum conditioning number of the embedded covariances. (default is 1e6)

    Attributes:
    obs_value (float): Observed MMD statistic value.

    Methods:
    __call__: Perform the permutation test and return the p-value.
    """

    def __init__(self, X, Y, kappa_K=1e4, kernel=None):
        """
        Initialize the test class.
        """
        self.X = X; self.Y = Y; self.kappa_K = kappa_K
        n = len(X); m = len(Y)
        fullsample = np.concatenate([X, Y])

        if kernel is None:
            pairwise_dists = cdist(fullsample, fullsample, 'euclidean')
            median_dist = np.median(pairwise_dists[pairwise_dists > 0])  # Avoid zero distances
            bandwidth = 2 * median_dist
            self.kernel_matrix  = np.exp( - pairwise_dists / bandwidth)
        else:
            self.kernel_matrix  = kernel(fullsample, fullsample)  

        kxx = self.kernel_matrix[:n, :n]; kxy = self.kernel_matrix[:n, n:]; kyy = self.kernel_matrix[n:, n:]
        KX, KX_ED, KY, KY_ED = get_Kmats_X_Y(kxx, kxy, kyy, kappa_K)
        muX = self.kernel_matrix[:n, :].sum(1)/n
        muY = self.kernel_matrix[n:, :].sum(1)/m
        self.obs_value = FH_test_stat(KX, KY, KX_ED, KY_ED) + np.linalg.norm(np.linalg.inv(KX + KY)@(muX - muY), 2)

        
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
            _KX, _KX_ED, _KY, _KY_ED = get_Kmats_X_Y(_kxx, _kxy, _kyy, self.kappa_K)
            _muX = reordered_kernel[:n, :].sum(1)/n
            _muY = reordered_kernel[n:, :].sum(1)/m
            permuted_stats.append(FH_test_stat(_KX, _KY, _KX_ED, _KY_ED) + np.linalg.norm(np.linalg.inv(_KX + _KY)@(_muX - _muY), 2))
            
        p_value = float(np.mean(permuted_stats > self.obs_value))

        if not return_stats:
            return p_value
        else:
            return permuted_stats, p_value

# ------------------------ # ------------------------ # ------------------------ # ------------------------ 