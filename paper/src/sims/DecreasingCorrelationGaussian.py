null = True #set if run under H0 or H1
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
os.chdir(parent_dir)
sys.path.insert(0, parent_dir)
from utils import *
output_dir = f'../out/sims'
os.makedirs(output_dir, exist_ok=True)
# --------- set parameters ---------------------------------------------------------------------------------------------------------------------
NUM_CORES = 72
num_permutations = 200
N_iters = 200
ridge_ls = np.logspace(-7, 0, 8)
band_factor_ls = [0.05, 0.1, 1, 5,10]
kernel_name = 'euclidean'
# --------- set model parameters ---------------------------------------------------------------------------------------------------------------------
n,d = 100, 500
ls_eps = [0.05, 0.1, 0.15, 0.20, 0.25]
_model_,model_params = (models.DecreasingCorrelationGaussian, {'alpha': 0.5})
# --------- run -------------------------------------------------------------------------------------------------------------------------------
model_name = _model_.__name__
if null: # type: ignore
    model_name += '-null'
alpha = model_params['alpha']
results = []
for eps in ls_eps:
    model_params.update({'eps': eps})
    out = run_fast_parallel(*(n, d, _model_, model_params, num_permutations, N_iters, NUM_CORES, kernel_name, ridge_ls, band_factor_ls,null))# type: ignore
    for test_name in out[0].keys():
        decisions = [_[test_name] for _ in out]
        results += [[eps, test_name, np.mean(decisions)]]
pd.DataFrame( results, columns = [ 'eps', 'test', 'rejection rate']).to_csv(f'{output_dir}/{model_name}.csv', index=False)