from src.modules import *
from src.utils import *

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
        decision = 1 if p_value < 0.05 else 0
        return decision, p_value
        
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
        decision = 1 if p_value < 0.05 else 0
        return decision, p_value


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
        decision = 1 if p_value < 0.05 else 0
        return decision, p_value

# ------------------------ # ------------------------ # ------------------------ # ------------------------ 

def optmap(F,G, eigdec_F = None):
    if eigdec_F is None:
        UF, SF = EIG_DEC(F)
    else:
        UF, SF = eigdec_F
    sqrtm_F = sqrtm_ED((UF, SF))
    sqrtm_inv_F =  inv_sqrtm_ED((UF, SF))
    return sqrtm_inv_F@sqrtm_ED(EIG_DEC(sqrtm_F@G@sqrtm_F))@sqrtm_inv_F
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
        decision = 1 if p_value < 0.05 else 0
        return decision, p_value

