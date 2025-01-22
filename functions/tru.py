
from modules import *
from functions.tools import *
from functions.TestFuns import *
from models import models_classes as models
from functions.run import *
from functions.tru import *
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore




title_dict = {'isotropic_different_means' : 'Isotropic Gaussian data with different means',
              'isotropic_vs_DiagSpiked' : 'Isotropic and diagonal spiked Gaussian data',}

def sample_test_vals(sample_X, sample_Y, n, test_names, kappa_K = 1e6, kernel = None ):
    X = sample_X(n)
    Y = sample_Y(n)  
    pooled = np.vstack([X, Y])
    values  = []
    for test_name in test_names:
        if test_name in all_CovKerEmb_tests:
            if kernel is None:
                pairwise_dists = cdist(pooled, pooled, 'euclidean')
                median_dist = np.median(pairwise_dists[pairwise_dists > 0])  # Avoid zero distances
                bandwidth = 2 * median_dist
                kernel_matrix  = np.exp( - pairwise_dists / bandwidth)
            else:
                kernel_matrix  = kernel(pooled, pooled)  
            kxx = kernel_matrix[:n, :n]
            kxy = kernel_matrix[:n, n:]
            kyy = kernel_matrix[n:, n:]
            KX, KX_ED, KY, KY_ED = get_Kmats_X_Y(kxx, kxy, kyy, kappa_K)
            values.append(CovKerEmb_test_stat(test_name, KX, KY, KX_ED, KY_ED))
        if test_name == 'MMD':
            values.append(MMD_two_sample_test(X,Y).obs_value)
        if test_name == 'KNN':
            values.append(KNN_two_sample_test(X,Y).obs_value)
        if test_name == 'FR':
            values.append(FR_two_sample_test(X,Y).obs_value)
        if test_name == 'HT':
            values.append(HT_two_sample_test(X, Y, k=10).obs_value)
    return values


def H0_H1(n,d, _model_,test_names, kappa_K, num_reps, kernel = None, NUM_CORES=8):
    # sample from null 
    iter_args = [(_model_(d).sample_X, _model_(d).sample_X, n, test_names, kappa_K, kernel) for _ in range(num_reps)]
    null_vals = Parallel(n_jobs=NUM_CORES)(delayed(sample_test_vals)(*args) for args in iter_args)

    # sample from alternative 
    iter_args = [(_model_(d).sample_X, _model_(d).sample_Y, n, test_names, kappa_K, kernel) for _ in range(num_reps)]
    alternative_vals = Parallel(n_jobs=NUM_CORES)(delayed(sample_test_vals)(*args) for args in iter_args)

    return null_vals, alternative_vals
