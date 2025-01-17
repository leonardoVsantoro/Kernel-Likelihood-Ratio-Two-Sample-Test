from modules import *
# ------------------------ # ------------------------ # ------------------------ # ------------------------ 
# MISCELLANEOUS



@profile
def get_Kmats_X_Y(kxx, kxy, kyy, regularisation_param):
    kyx = kxy.T
    n = kxx.shape[0]
    KmatX = np.concatenate([kxx, kyx])
    # KmatY = np.concatenate([kyx, kyy])
    KmatY = np.concatenate([kxy, kyy])
    KX = KmatX @ (KmatX.T) / n
    KY = KmatY @ (KmatY.T) / n
    
    # regularize_with_condition_number_byRidge, regularize_with_condition_number_byTruncation, regularize_with_ridge
    KX, KX_ED = regularize_with_condition_number_byRidge(KX, regularisation_param)
    KY, KY_ED = regularize_with_condition_number_byRidge(KY, regularisation_param)
    
    return KX, KX_ED, KY, KY_ED

# ------------------------ # ------------------------ # ------------------------ # ------------------------ 


min_val_to_clip = 1e-10
def project_to_psd(A):
    eigvals, eigvecs = LA.eigh(A)
    U = eigvecs
    S = np.clip(eigvals, min_val_to_clip, None)
    return U @ np.diag(S) @ U.T

@profile
def EIG_DEC(A):
    #assert np.allclose(A, A.T), "Matrix is not symmetric"
    eigvals, eigvecs = LA.eigh(A)
    S = np.clip(eigvals, min_val_to_clip, None)
    U = eigvecs
    # if not np.allclose(U @ np.diag(S) @ U.T, A):
    #     raise ValueError("SVD failed")
    return S, U

@profile
def sqrtm_ED(eigdec):
    S, U = eigdec
    return U @ np.diag( np.clip(S, min_val_to_clip, None)**.5)  @ U.T

@profile
def inv_sqrtm_ED(eigdec):
    S, U = eigdec
    return U @ np.diag(np.clip(S, min_val_to_clip, None)**-.5)  @ U.T



def efficient_cov_eigdec(X):
    """
    Compute the eigendecomposition of X.T @ X using SVD in an efficient way,
    depending on whether n > d or d > n.

    Parameters:
    X (ndarray): Input matrix of shape (n, d).

    Returns:
    eigenvalues (ndarray): Eigenvalues of X.T @ X.
    eigenvectors (ndarray): Eigenvectors of X.T @ X.
    """
    n, d = X.shape

    if n >= d:
        S, U, Vt = np.linalg.svd(X, full_matrices=False)
        eigenvalues = S**2/n
        eigenvectors = Vt.T
    else:
        S, U, Vt = np.linalg.svd(X.T, full_matrices=False)
        eigenvalues = S**2/n
        eigenvectors = U

    return eigenvectors, eigenvalues

@profile
def reorder_kernel(kernel_matrix, permuted_indices_1, permuted_indices_2):
    return kernel_matrix[permuted_indices_1, :][:, permuted_indices_2]
    #return kernel_matrix[np.ix_(permuted_indices_1, permuted_indices_2)]


# @profile
# def reorder_kernel(kernel_matrix, permuted_indices):
#     n = len(permuted_indices)//2
#     kxx = kernel_matrix[permuted_indices[:n], :][:, permuted_indices[:n]]
#     kxy = kernel_matrix[permuted_indices[:n], :][:, permuted_indices[n:]]
#     kyy = kernel_matrix[permuted_indices[n:], :][:, permuted_indices[n:]]
#     return kxx, kxy, kyy



# ------------------------ # ------------------------ # ------------------------ # ------------------------ 
# REGULARIZATION

@profile
def regularize_with_condition_number_byRidge(A, max_condition_number):
    S, U  = EIG_DEC(A)
    # cond_num = S[-1]/S[0]
    cond_num = S.max()/S.min()
    if cond_num < max_condition_number:
        return (A, (S, U))
    else:
        S[S<0] = 0
        S = S + np.ones(S.size)*(max_condition_number*S.min() - S.max())/(1- max_condition_number)
        return (A, (S, U))
        # return (U @ np.diag(S) @ (U.T), (S, U))

def regularize_with_ridge(A, ridge):
    assert np.allclose(A, A.T), "Matrix is not symmetric"
    regularised_A = A + ridge*np.eye(A.shape[0])
    return regularised_A, EIG_DEC(regularised_A)


# ------------------------ # ------------------------ # ------------------------ # ------------------------ 
# BURES WASSERSTEIN
@profile
def optmap(F,G, eigdec_F = None):
    if eigdec_F is None:
        UF, SF = EIG_DEC(F)
    else:
        UF, SF = eigdec_F
    sqrtm_F = sqrtm_ED((UF, SF))
    sqrtm_inv_F =  inv_sqrtm_ED((UF, SF))
    return sqrtm_inv_F@sqrtm_ED(EIG_DEC(sqrtm_F@G@sqrtm_F))@sqrtm_inv_F


# ------------------------ # ------------------------ # ------------------------ # ------------------------ 
# Kernels

def gaussian_kernel(X, Y, sigma = 1):
    pairwise_dists = cdist(X, Y, 'sqeuclidean')
    return np.exp(-pairwise_dists / (2*sigma**2))

def laplacian_kernel(X, Y, sigma = 1):
    pairwise_dists = cdist(X, Y, 'euclidean')
    return np.exp(-pairwise_dists / (2*sigma**2))
# ------------------------ # ------------------------ # ------------------------ # ------------------------ 

# SOME KNOWN COVARIANCES
def spiked_covariance_off_diag(d, num_spikes=1, spike_value=10, template_cov=None):
    if template_cov is None:
        cov_matrix = np.eye(d)
    else:
        cov_matrix = template_cov(d)
    indices = np.triu_indices(d)
    selected_indices = np.random.choice(len(indices[0]), num_spikes, replace=False)
    for idx in selected_indices:
        i, j = indices[0][idx], indices[1][idx]
        cov_matrix[i, j] = spike_value
        cov_matrix[j, i] = spike_value
    return project_to_psd(cov_matrix)

def spiked_covariance(d, num_spikes=1, spike_value=10, template_cov=None):
    if template_cov is None:
        cov_matrix = np.eye(d)
    else:
        cov_matrix = template_cov(d)

    selected_indices = np.random.choice(d, num_spikes, replace=False)
    for i in selected_indices:
        cov_matrix[i, i] = spike_value
    return cov_matrix

from sklearn.gaussian_process.kernels import Matern
def matern_cov(d,nu): 
    matern_grid = np.array([ (
            np.linspace(0,1,d).ravel()[:, np.newaxis], 
            np.linspace(0,1,d).ravel()[:, np.newaxis]) ])[0].T.reshape(-1,2)
    return Matern(nu=nu)(matern_grid)

def BM_cov(d): 
    grid = np.linspace(0,1,d)
    return np.array([[min(s,t) for s in grid] for t in grid])

def BB_cov(d):
    grid = np.linspace(0,1,d)
    return np.array([[min(s,t) - s*t for s in grid] for t in grid])

def toeplitz_covariance(d, rho):
    return np.array([[rho**abs(i-j) for j in range(d)] for i in range(d)])

def block_diagonal_covariance(blocks):
    return np.block([[block if i == j else np.zeros_like(block) for j, block in enumerate(blocks)] for i, block in enumerate(blocks)])

def band_covariance(d, k, value=1):
    cov_matrix = np.eye(d)
    for i in range(d):
        for j in range(max(0, i-k), min(d, i+k+1)):
            if i != j:
                cov_matrix[i, j] = value
    return cov_matrix

def compound_symmetry_covariance(d, sigma2, tau2):
    return sigma2 * np.ones((d, d)) + (tau2 - sigma2) * np.eye(d)

def ar1_covariance(d, rho):
    return toeplitz_covariance(d, rho)

    # if bootstrap_type == '2centred':
    #     cov_X = np.cov(X.T) + np.eye(X.shape[1])*1e-5
    #     cov_Y = np.cov(Y.T) + np.eye(X.shape[1])*1e-5
    #     cov_null = geodesic_midpoint(cov_X, cov_Y)
    #     mu_X = X.mean(0)
    #     mu_Y = Y.mean(0)
    #     mu_null = .5*(mu_X + mu_Y)
    #     invsqrtCOV_X = np.linalg.inv(sqrtm(cov_X))
    #     invsqrtCOV_Y = np.linalg.inv(sqrtm(cov_Y))
    #     return np.concatenate([(sqrtm(cov_null)@invsqrtCOV_X@(X-mu_X).T).T, (sqrtm(cov_null)@invsqrtCOV_Y@(Y-mu_Y).T).T], axis = 0)+ mu_null
# ------------------------ # ------------------------ # ------------------------ # ------------------------ 

def get_kernel(kernel_name,sigma=1):
    if kernel_name == 'gaussian':
        return lambda X,Y :  gaussian_kernel(X,Y, sigma)
    if kernel_name == 'laplacian':
        return lambda X,Y : laplacian_kernel(X,Y, sigma)
    else:
        return None

def get_fullsample(X,Y, bootstrap_type = 'naive'):
    if bootstrap_type == 'naive':    
        return np.concatenate([X,Y], axis = 0)
    
    if bootstrap_type == '1centred':
        return np.concatenate([X - X.mean(0), Y - Y.mean(0)], axis = 0) + .5*(X.mean(0) + Y.mean(0))