from src.modules import *
from src.modules import *

def regularize_to_condition_number(A, max_condition_number=1e9):
    """
    Regularizes symmetric PSD matrix A by adding alpha * I
    so that its condition number is at most max_condition_number.
    """
    assert A.shape[0] == A.shape[1], "Matrix must be square"
    
    eigvals = np.clip( np.linalg.eigvalsh(A), 1e-15, None)  # more stable for symmetric matrices
    lambda_min = np.min(eigvals)
    lambda_max = np.max(eigvals)
    
    cond = lambda_max / lambda_min
    if cond <= max_condition_number:
        return A  # no regularization needed
    
    alpha = max(0, (lambda_max - max_condition_number * lambda_min) / (max_condition_number - 1))
    
    return A + alpha * np.eye(A.shape[0])

def get_mvecs_X_Y(kxx, kxy, kyy):
    kyx = kxy.T
    n = kxx.shape[0]
    m = kyy.shape[0]
    mY = np.sum( np.concatenate([kxx, kyx]), axis=1, keepdims=True)/n
    mX = np.sum( np.concatenate([kxy, kyy]), axis=1, keepdims=True)/m
    return mX, mY

def get_Smats_X_Y(kxx, kxy, kyy, max_cond_number = 1e9):
    kyx = kxy.T
    n = kxx.shape[0]
    m = kyy.shape[0]
    KmatX = np.concatenate([kxx, kyx])
    KmatY = np.concatenate([kxy, kyy])
    SX = KmatX @ (KmatX.T)/n 
    SY = KmatY @ (KmatY.T)/m 
    SX = regularize_to_condition_number(SX, max_condition_number=max_cond_number)
    SY = regularize_to_condition_number(SY, max_condition_number=max_cond_number)
    return SX, SY

def get_Cmat(kernel_matrix):
    N = kernel_matrix.shape[0]
    m_emb = np.sum(kernel_matrix, axis=1, keepdims=True)/N
    return kernel_matrix@(kernel_matrix.T)/ N  - m_emb@(m_emb.T) 


def SX_ED_from_kernel_via_svd(kernel_matrix,):
    U, S, _ = np.linalg.svd(kernel_matrix, full_matrices=True)
    eigvegs = U
    eigvals = np.zeros(kernel_matrix.shape[0]); eigvals[:len(S)] = S**2/kernel_matrix.shape[1]**2
    return (eigvals, eigvegs)

min_val_to_clip = 1e-10  # Minimum value to clip eigenvalues to avoid numerical issues
def sqrtm_ED(eigdec):
    S, U = eigdec
    return U @ np.diag(np.array([s**0.5 if s > 0 else 0 for s in S])) @ U.T

def inv_sqrtm_ED(eigdec):
    S, U = eigdec
    return U @ np.diag(np.array([s**-0.5 if s > 0 else 0 for s in S])) @ U.T

def reorder_kernel(kernel_matrix, permuted_indices_1, permuted_indices_2):
    return kernel_matrix[permuted_indices_1, :][:, permuted_indices_2]
   


def det2(A):
    """Compute the Carleman–Fredholm determinant det_2(I + A)"""
    I = np.eye(A.shape[0])
    eigvals =  LA.eigh(A, eigvals_only=True); eigvals[eigvals<-1]=1e-10
    return np.prod(1 + eigvals)* np.exp(-np.sum(eigvals))

def logdet2(A):
    """Compute the log Carleman–Fredholm determinant log det_2(I + A)"""
    eigvals =  np.clip( LA.eigh(A, eigvals_only=True), 1e-5 - 1 , None); 
    return - np.sum(eigvals) + np.sum(np.log(1 + eigvals))