import numpy as np
from collections import defaultdict
import scipy.linalg
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed
from scipy.stats import norm
from utils.other_tests import *
import pandas as pd
import math
import scipy



def wilson_score_interval(successes, n, confidence=0.95):
    z = norm.ppf(1 - (1 - confidence) / 2)  # z-score for desired confidence level
    phat = successes / n
    denominator = 1 + z**2 / n
    center = phat + z**2 / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n)
    lower = (center - margin) / denominator
    upper = (center + margin) / denominator
    return lower.round(3), upper.round(3)

def split_kernel(kernel_matrix,n,m):
    kxx = kernel_matrix [:n, :n]
    kyy = kernel_matrix [m:, m:]
    kxy = kernel_matrix [:n, m:]  
    return kxx, kyy, kxy  

# def inv_sqrtm_ED(eigdec, r ):
#     S, U = eigdec
#     S = np.clip(S, 1e-15, None)  # Avoid sqrt(0)
#     return U @ np.diag(np.array([1/np.sqrt(s+r) if s > 0 else 0 for s in S])) @ U.T
def inv_sqrtm_ED(eigdec):
    S, U = eigdec
    S = np.clip(S, 1e-15, None)  # Avoid sqrt(0)
    return U @ np.diag(np.array([s**-0.5 if s > 0 else 0 for s in S])) @ U.T
def inv_ED(eigdec):
    S, U = eigdec
    S = np.clip(S, 1e-15, None)  # Avoid sqrt(0)
    return U @ np.diag(np.array([1/s if s > 0 else 0 for s in S])) @ U.T


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

# ------------------------------------------------------------------------------------------------------------------------------------------


def update_dict(original, kernel_matrix, ridge_ls, BF):
    max_cond_number = 1e9
    islist = original.default_factory == list
    n =  kernel_matrix.shape[0]//2; m = n
    # project_dim = n 
    kxx, kyy, kxy = split_kernel(kernel_matrix, n, m); kyx = kxy.T
    KmatX = np.concatenate([kxx, kyx])
    KmatY = np.concatenate([kxy, kyy])
    mX = np.mean(KmatX, axis=1, keepdims=True)
    mY = np.mean(KmatY, axis=1, keepdims=True)
    SY = KmatY @ (KmatY.T)/m
    SX = KmatX @ (KmatX.T)/n 
    
    # S_avg = regularize_to_condition_number( (SX+SY)/2, 
    smartreg = 1e-2
    m_avg = (1-smartreg)*mX+smartreg*mY
    S_avg = regularize_to_condition_number( (1-smartreg)*SX+smartreg*SY,
                                            max_condition_number=max_cond_number)
    C  = regularize_to_condition_number( np.cov(kernel_matrix.T),
                                            max_condition_number=max_cond_number)
    
    eigvals_C, eigvecs_C = np.linalg.eigh(C)
    eigvals_S_avg, eigvecs_S_avg = np.linalg.eigh(S_avg)
    # eigvals_SY, eigvecs_SY = np.linalg.eigh(SY)
       
    projection_matrix = eigvecs_S_avg[:,:] @ eigvecs_S_avg[:, :].T
    SY = projection_matrix @ SY @ projection_matrix.T

    for r in ridge_ls:

        inv_sqrtm_S_avg = inv_sqrtm_ED((eigvals_S_avg +r ,eigvecs_S_avg))
        # inv_S_avg = inv_ED((eigvals_S_avg + r, eigvecs_S_avg))

        CM_term = 0.5*np.linalg.norm(inv_sqrtm_S_avg @ ( m_avg - mY ))**2
        H = inv_sqrtm_S_avg @ ( S_avg - SY ) @ inv_sqrtm_S_avg
        eigvals_H =  np.clip(np.linalg.eigvalsh(H), 1e-15 - 1, None)  # Avoid log(0)]
        logdet2_term = .5*np.abs(- np.sum(eigvals_H) + np.sum(np.log(1 + eigvals_H)))
        HS_term = np.sum(eigvals_H**2)#np.linalg.norm(H,'fro')
        # inv_SY = inv_ED((eigvals_SY + r, eigvecs_SY))
        # logdet2_term = .25*np.abs(np.trace((S_avg - SY)@(inv_S_avg - inv_SY)))
        inv_sqrtm_C = inv_sqrtm_ED((eigvals_C+r, eigvecs_C) )
        specreg_term = np.linalg.norm(inv_sqrtm_C @ (mY-mX))

        if islist:
            original['KLR', r, BF].append(CM_term + logdet2_term)
            original['KLR0', r, BF].append(HS_term)
            original['*KLR*', r, BF].append(CM_term + HS_term)
            original['logdet2', r, BF].append(logdet2_term)
            original['CM', r, BF].append(CM_term)
            original['SpecReg-MMD', r, BF].append(specreg_term)
        else:
            original['KLR', r, BF] = CM_term + logdet2_term
            original['KLR0', r, BF] = HS_term
            original['*KLR*', r, BF] = CM_term + HS_term
            original['logdet2', r, BF] = logdet2_term
            original['CM', r, BF] = CM_term
            original['SpecReg-MMD', r, BF] = specreg_term

    if islist:
        original['AggMMD', BF].append(np.linalg.norm(mX - mY))
    else:
        original['AggMMD', BF] = np.linalg.norm(mX - mY)
    
    return original

def run_fast(X,Y, num_permutations, kernel_name, ridge_ls, band_factor_ls, light = False, return_all = False):
    n = len(X); m = len(Y)
    fullsample = np.concatenate([X, Y])
    pairwise_dists = cdist(fullsample, fullsample, kernel_name)
    bandwidth = 2 * np.median(pairwise_dists[pairwise_dists>0])
    obs_value = defaultdict()
    p_values = []
    obs_value, permuted_values = defaultdict(), defaultdict(list)
    for BF in band_factor_ls:
        kernel_matrix  =  np.exp( - pairwise_dists / (bandwidth*BF)) 
        obs_value = update_dict(obs_value, kernel_matrix, ridge_ls, BF)
        for _ in range(num_permutations):
            perm = np.random.permutation(n + m)
            _kernel_matrix = kernel_matrix[np.ix_(perm, perm)]
            permuted_values = update_dict(permuted_values, _kernel_matrix, ridge_ls, BF)
    
    p_values = defaultdict(list)
    for key, values in permuted_values.items():
        name = key[0]
        p_values[name].append( np.mean(np.array(values) > obs_value[key] - 1e-8) ) 

    if not light:
        p_values['HT'] = [ HT_two_sample_test(X,Y)(num_permutations)[1] ]
        p_values['FR'] = [ FR_two_sample_test(X,Y)(num_permutations)[1] ]
        # p_values['KNN'] = [ KNN_two_sample_test(X,Y)(num_permutations)[1] ]
    
    if return_all:
        return p_values, obs_value, permuted_values
    return {name: min(ps) < 0.05/len(ps) for name, ps in p_values.items()}

def run_fast_parallel(n, d,  _model_, model_params, num_permutations, N_iters, NUM_CORES, kernel_name, ridge, band_factor , null=False, light = False, print_results = False):
    XY = []
    # for _ in range(N_iters):
    #     X = _model_(**model_params)(d).sample_X(n)
    #     Y = _model_(**model_params)(d).sample_Y(n)
    #     if null:
    #         Z = np.concatenate([X, Y])
    #         np.random.shuffle(Z)
    #         X = Z[:n]
    #         Y = Z[n:]
    #     XY.append((X, Y))
    if null:
        light = True
    for _ in range(N_iters):
        if null:
            X = _model_(**model_params)(d).sample_X(n)
            Y = _model_(**model_params)(d).sample_X(n)
        else:
            X = _model_(**model_params)(d).sample_X(n)
            Y = _model_(**model_params)(d).sample_Y(n)
        XY.append((X, Y))
    iter_args = [(X, Y, num_permutations, kernel_name, ridge, band_factor, light) for X,Y in XY]
    out = Parallel(n_jobs=NUM_CORES)(delayed(run_fast)(*args) for args in iter_args)
    if print_results:
        try:
            results = [[n, d, test_name, np.mean( [_[test_name] for _ in out]), wilson_score_interval(np.sum( [_[test_name] for _ in out]), 8)] for test_name in out[0].keys()]
            data = pd.DataFrame(results, columns=['sample_size', 'dimension', 'test_name', 'rejection_rate', 'CI'])
        except:
            results = [[n, d, test_name, np.mean( [_[test_name] for _ in out])] for test_name in out[0].keys()]
            data = pd.DataFrame(results, columns=['sample_size', 'dimension', 'test_name', 'rejection_rate'])
        print(data.sort_values(by=['rejection_rate'], ascending=False).reset_index(drop=True))
    return out



def run_fast_parallel_sampling(Xpd,Ypd, sample_size, num_permutations, N_iters, NUM_CORES, kernel_name, ridge, band_factor , light = False, print_results = False, null = False):
    if null:
        Xpd, Ypd = Xpd, Xpd
        # Xpd, Ypd = pd.concat([Xpd, Ypd], ignore_index=True), pd.concat([Xpd, Ypd], ignore_index=True)
        # Xpd = Xpd.sample(frac=1).reset_index(drop=True)
        # Ypd = Xpd.sample(frac=1).reset_index(drop=True)
    iter_args = [(Xpd.sample(sample_size), Ypd.sample(sample_size),
                   num_permutations, kernel_name, ridge, band_factor, light) for _ in range(N_iters)]
    out = Parallel(n_jobs=NUM_CORES)(delayed(run_fast)(*args) for args in iter_args)
    if print_results:
        try:
            results = [[test_name, np.mean( [_[test_name] for _ in out]), wilson_score_interval(np.sum( [_[test_name] for _ in out]), 8)] for test_name in out[0].keys()]
            data = pd.DataFrame(results, columns=['test_name', 'rejection_rate', 'CI'])
        except:
            results = [[test_name, np.mean( [_[test_name] for _ in out])] for test_name in out[0].keys()]
            data = pd.DataFrame(results, columns=['test_name', 'rejection_rate'])
        print(data.sort_values(by=['rejection_rate'], ascending=False).reset_index(drop=True))
    return out


