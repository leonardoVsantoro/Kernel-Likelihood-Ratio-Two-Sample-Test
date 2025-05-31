from modules import *
from functions.TestFuns import *
from models import *


def run_iteration(n, d, _model_, model_params, test_names, num_permutations = 100):
    model = _model_(**model_params)(d)
    X = model.sample_X(n)
    Y = model.sample_Y(n)
    pvals = {}
    if '*HS*' in test_names:
        pvals.update({'*HS*' : GKE_HS_two_sample_test(X, Y)(num_permutations)})
    if '*KL*' in test_names:
        pvals.update({'*KL*' : GKE_KL_two_sample_test(X, Y)(num_permutations)})
    if 'KNN' in test_names:
        pvals.update({'KNN' : KNN_two_sample_test(X, Y, k=1)(num_permutations)})
    if 'FR' in test_names:
        pvals.update({'FR' : FR_two_sample_test(X, Y)(num_permutations)})
    if 'HT' in test_names:
        pvals.update({'HT' : HT_two_sample_test(X, Y, k=10)(num_permutations)})
    if 'MMD' in test_names:
        pvals.update({'MMD' : MMD_two_sample_test(X, Y)(num_permutations)})
    if 'MPZ' in test_names:
        pvals.update({'MPZ' : MPZ_two_sample_test(X, Y)(num_permutations)})                  
    return { test_name : 1 if pval<0.05 else 0 for test_name, pval in pvals.items()}

def run_parallel(n, d, _model_, model_params, test_names, num_permutations, N_iters, NUM_CORES):
    iter_args = [(n, d, _model_,model_params, test_names, num_permutations) 
                 for _ in tqdm( range(N_iters), total = N_iters, desc = 'Running n,d = {},{}'.format(n,d), ncols = 100)]
    results = Parallel(n_jobs=NUM_CORES)(delayed(run_iteration)(*args) for args in iter_args)
    return results

