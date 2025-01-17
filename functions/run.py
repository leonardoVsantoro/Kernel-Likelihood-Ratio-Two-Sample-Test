from modules import *

from functions.tools import *
from functions.TestFuns import *
from models.models_classes import *


# ------------------------ # ------------------------ # ------------------------ # ------------------------ 
@profile
def CovKerEmb_tests(kernel_matrix, kappa_K, test_names, permute = False):
    n = kernel_matrix.shape[0]//2
    if permute:
        permuted_indices = np.random.permutation(n + n)
        kernel_matrix = kernel_matrix[permuted_indices][:, permuted_indices]
    _kxx = kernel_matrix[:n, :n] ; _kyy = kernel_matrix[n:, n:]; _kxy = kernel_matrix[:n, n:]
    _KX, _KX_ED, _KY, _KY_ED = get_Kmats_X_Y(_kxx, _kxy, _kyy, kappa_K)
    test_values =  [ CovKerEmb_test_stat(test_name, _KX, _KY, _KX_ED, _KY_ED) for test_name in test_names] 
    return test_values
# ------------------------ # ------------------------ # ------------------------ # ------------------------ 
@profile
def run_CKEpart(X,Y, kernel_name, kernel_bandwith, kappa_K, test_names, num_permutations, parallelise_permutations = False, NUM_CORES = 8):
    EmbKer_tests = list(filter(lambda x: x in test_names, all_CovKerEmb_tests))
    n = len(X); m = len(Y)
    fullsample = np.concatenate([X, Y])

    if kernel_name == 'gaussian':
        pairwise_dists = cdist(fullsample, fullsample, 'sqeuclidean')
    elif kernel_name == 'laplacian':
        pairwise_dists = cdist(fullsample, fullsample, 'euclidean')
    else:
        pairwise_dists = cdist(fullsample, fullsample, 'euclidean')

    if kernel_bandwith is None:
            median_dist = np.median(pairwise_dists[pairwise_dists > 0])  
            kernel_bandwith = 2 * median_dist
    
    kernel_matrix  = np.exp( - pairwise_dists / kernel_bandwith)

    _ = CovKerEmb_tests(kernel_matrix, kappa_K, test_names)
    obs_stat = {test_name: val for test_name, val in zip(test_names, _)}
    if parallelise_permutations:
        _ = Parallel(n_jobs=NUM_CORES)(delayed(CovKerEmb_tests)(kernel_matrix, kappa_K, test_names, permute = True) for _ in range(num_permutations))
    else:
        _ = [CovKerEmb_tests(kernel_matrix, kappa_K, test_names, permute = True) for _ in range(num_permutations)]
    permuted_stats = {test_name: np.array(_)[:,i] for i, test_name in enumerate(test_names)}
    pvals = {test_name: np.mean(permuted_stats[test_name] > obs_stat[test_name]) for test_name in EmbKer_tests}
    return pvals
# ------------------------ # ------------------------ # ------------------------ # ------------------------ 
@profile
def run_iteration(n, d, _model_, model_params, test_names, kernel_name, kernel_bandwith, kappa_K = 1e6, num_permutations = 100, parallelise_permutations = False, NUM_CORES = 8):

    model = _model_(**model_params)(d)
    X = model.sample_X(n)
    Y = model.sample_Y(n)

    pvals =  run_CKEpart(X,Y, kernel_name, kernel_bandwith, kappa_K, test_names, num_permutations, parallelise_permutations, NUM_CORES)

    if 'KNN' in test_names:
        pvals.update({'KNN' : KNN_two_sample_test(X, Y, k=1)(num_permutations)})
    if 'FR' in test_names:
        pvals.update({'FR' : FR_two_sample_test(X, Y)(num_permutations)})
    if 'HT' in test_names:
        pvals.update({'HT' : HT_two_sample_test(X, Y, k=10)(num_permutations)})
    if 'MMD' in test_names:
        pvals.update({'MMD' : MMD_two_sample_test(X, Y, kernel_name, kernel_bandwith)(num_permutations)})
    if 'MPZ' in test_names:
        pvals.update({'MPZ' : MPZ_two_sample_test(X, Y)(num_permutations)})                  
    return { test_name : 1 if pval<0.05 else 0 for test_name, pval in pvals.items()}

# ------------------------ # ------------------------ # ------------------------ # ------------------------