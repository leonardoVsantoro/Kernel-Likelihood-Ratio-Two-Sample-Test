from modules import *
from functions.utils import *

def split_kernel(kernel_matrix,n,m):
    kxx = kernel_matrix [:n, :n]
    kyy = kernel_matrix [m:, m:]
    kxy = kernel_matrix [:n, m:]  
    return kxx, kyy, kxy  

# ------------------------ # ------------------------ # ------------------------ # ------------------------ 

def KLR0_test_stat(kernel_matrix, n, m, ridge):
    kxx, kyy, kxy = split_kernel(kernel_matrix, n, m)
    SX, SY = get_Smats_X_Y(kxx, kxy, kyy)
    eigvals_SX ,eigvecs_SX = SX_ED_from_kernel_via_svd(kernel_matrix[:,:n]); 
    vals = []
    for r in ridge:
        eigvals_SX+= r; SX_ED = (eigvals_SX, eigvecs_SX)
        vals.append( np.linalg.norm(inv_sqrtm_ED(SX_ED)@ (SY - SX) @ inv_sqrtm_ED(SX_ED),'fro'))
    return np.array(vals)

def KLR_test_stat(kernel_matrix, n, m, ridge):
    kxx, kyy, kxy = split_kernel(kernel_matrix, n, m)  
    SX, SY = get_Smats_X_Y(kxx, kxy, kyy)
    mX, mY = get_mvecs_X_Y(kxx, kxy, kyy)
    eigvals_SX ,eigvecs_SX = SX_ED_from_kernel_via_svd(kernel_matrix[:,:n])
    vals = []
    for r in ridge:
        eigvals_SX+= r; SX_ED = (eigvals_SX, eigvecs_SX)
        vals.append( np.linalg.norm(inv_sqrtm_ED(SX_ED)@(mY-mX)) + np.abs(logdet2(inv_sqrtm_ED(SX_ED)@ (SY - SX) @ inv_sqrtm_ED(SX_ED))))
    return np.array(vals)

def CM_test_stat(kernel_matrix, n, m, ridge):
    kxx, kyy, kxy = split_kernel(kernel_matrix, n, m)
    SX, SY = get_Smats_X_Y(kxx, kxy, kyy)
    mX, mY = get_mvecs_X_Y(kxx, kxy, kyy)
    eigvals_SX ,eigvecs_SX = SX_ED_from_kernel_via_svd(kernel_matrix[:,:n])
    vals = []
    for r in ridge:
        eigvals_SX+= r; SX_ED = (eigvals_SX, eigvecs_SX)
        vals.append(np.linalg.norm(inv_sqrtm_ED(SX_ED)@(mY-mX)))
    return np.array(vals)

def mmd_test_stat(kernel_matrix, n, m, ridge = None):
    kxx, kyy, kxy = split_kernel(kernel_matrix, n, m)  
    np.fill_diagonal(kxx, 0); np.fill_diagonal(kyy, 0)
    obs_value = kxx.sum()/(n*(n-1)) +  kyy.sum()/(m*(m-1)) - 2* kxy.sum()/(n*m)
    return obs_value

def spec_reg_mmd(kernel_matrix, n, m, ridge):
    kxx, kyy, kxy = split_kernel(kernel_matrix, n, m)
    mX, mY = get_mvecs_X_Y(kxx, kxy, kyy)
    C = get_Cmat(kernel_matrix)
    vals = []
    for r in ridge:
        C_ED = EIG_DEC(C + np.eye(C.shape[0]) * r)   
        vals.append(np.linalg.norm(inv_sqrtm_ED(C_ED)@(mY-mX)))
    return np.array(vals)
    
# ------------------------ # ------------------------ # ------------------------ # ------------------------ 

def KernelTwoSampleTest(name):
    """
    Factory function to create a two-sample kernel test class.

    Arguments:
    name (str): Name of the test. Options include 
        'Maximum Mean Discrepancy (MMD)',
        'Spectral Regularised MMD (SpecReg-MMD)',
        'Kernel Likelihood Ratio (KLR)',
        'Centered Kernel Likelihood Ratio (KLR-0)',
        'Cameron-Martin (CM)'.
    """

    test_function = {
        'MMD': mmd_test_stat,
        'SpecReg-MMD': spec_reg_mmd,
        'KLR': KLR_test_stat,
        'KLR-0': KLR0_test_stat,
        'CM': CM_test_stat
    }
    if name not in test_function:
        raise ValueError(f"Test name '{name}' not recognized. Available tests: {list(test_function.keys())}")
    test_stat = test_function[name]
 
    

    class TestClass:
        f"""
        Two-sample {name} test to determine if two samples come from the same distribution.

        Arguments:
        X (ndarray): First sample (n x d).
        Y (ndarray): Second sample (m x d).
        kernel: 'sqeuclidean' (default) for Gaussian, 'euclidean' for laplacian; see scipy.cdist documentation for alternatives. Bandiwith set to median heuristic

        Attributes:
        obs_value (float): Observed statistic value.

        Methods:
        __call__: Perform the permutation test and return the p-value."
        """

        def __init__(self, X, Y, kernel_name = 'sqeuclidean', band_factor = 1, ridge = 1):
            """
            Initialize the test class.
            """
            self.ridge = ridge if len(ridge)>0 else [ridge]
            self.band_factor = band_factor if len(band_factor)>0 else [band_factor]
            self.n = len(X); self.m = len(Y)
            n = len(X); m = len(Y)
            fullsample = np.concatenate([X, Y])
            pairwise_dists = cdist(fullsample, fullsample, kernel_name)
            bandwidth = 2 * np.median(pairwise_dists[pairwise_dists > 0]) 
            self.kernel_matrix  = { BF : np.exp( - pairwise_dists / (bandwidth*BF)) for BF in band_factor}
            self.obs_value =  { BF : test_stat(self.kernel_matrix[BF], n, m, ridge).reshape(-1,1) for BF in band_factor } 
            
        def __call__(self, num_permutations = 500, level = 0.05):
            """
            Perform the permutation test and return the p-value.

            Parameters:
            num_permutations (int): Number of permutations for calibrating the test.

            Returns:
            float: p-value of the test.
            """
            p_values = [] 
            for BF in self.band_factor:
                if np.abs(self.obs_value[BF]) < 1e-15:
                    p_values += [1.0] * len(self.ridge)
                else:
                    permuted_stats = np.array([
                        test_stat(
                            self.kernel_matrix[BF][np.ix_(perm := np.random.permutation(self.n + self.m), perm)],
                            self.n, self.m,
                            self.ridge,
                        )
                        for _ in range(num_permutations)
                    ]).reshape(-1, num_permutations)
                    
                    p_values += list(np.mean(permuted_stats > self.obs_value[BF] , axis = 1).astype(float)) 

            decision = 1 if min(p_values) < level/len(p_values) else 0
            p_value = min(p_values)*len(self.ridge)
            return decision, p_value
    return TestClass
    
# ------------------------ # ------------------------ # ------------------------ # ------------------------ 
