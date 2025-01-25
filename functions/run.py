from modules import *

from functions.tools import *
from functions.TestFuns import *
from models.models_classes import *


def run_iteration(n, d, _model_, model_params, test_names, kernel_name, kernel_bandwith, kappa_K = 1e6, num_permutations = 100):
    model = _model_(**model_params)(d)
    X = model.sample_X(n)
    Y = model.sample_Y(n)
    pvals = {}
    if 'FH' in test_names:
        pvals.update({'FH' : CKE_two_sample_test(X, Y, kappa_K = kappa_K)(num_permutations)})
    if 'FullFH' in test_names:
        pvals.update({'FullFH' : GKE_two_sample_test(X, Y, kappa_K = kappa_K)(num_permutations)})
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

def run_parallel(n, d, _model_, model_params, test_names, kernel_name, kernel_bandwith,  kappa_K, num_permutations, N_iters, NUM_CORES):
    iter_args = [(n, d, _model_,model_params, test_names, kernel_name, kernel_bandwith, kappa_K, num_permutations) for _ in range(N_iters)]
    try:
        results = Parallel(n_jobs=NUM_CORES)(delayed(run_iteration)(*args) for args in tqdm(iter_args))
    except:
        results = Parallel(n_jobs=NUM_CORES)(delayed(run_iteration)(*args) for args in iter_args)
    return results
