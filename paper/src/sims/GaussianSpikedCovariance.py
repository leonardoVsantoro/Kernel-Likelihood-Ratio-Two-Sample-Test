null = False #set if run under H0 or H1
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
num_permutations = 300
N_iters = 200
ridge_ls = np.logspace(-6, 0, 7)
band_factor_ls = [0.05, 0.1, 1, 5,10]
kernel_name = 'sqeuclidean'
NUM_CORES = 72
# # --------- set model parameters ---------------------------------------------------------------------------------------------------------------------
n = 100
n_d_values = [(n, 50),(n, 250),(n, 500),(n, 1000),(n, 1500)]
_model_, model_params = (models.GaussianSpikedCovariance, {'spike_value' : 3, 'num_spikes' : 5})
# --------- run -------------------------------------------------------------------------------------------------------------------------------
model_name = _model_.__name__
if null: # type: ignore
    model_name += '-null'
results = []
for n,d in n_d_values:
    args = (n, d, _model_, model_params, num_permutations, N_iters, NUM_CORES, kernel_name, ridge_ls, band_factor_ls,null)# type: ignore
    out = run_fast_parallel(*args)
    for test_name in out[0].keys():
        decisions = [_[test_name] for _ in out]
        results += [[d, test_name, np.mean(decisions)]]
pd.DataFrame(   results, 
                columns = ['dimension', 'test', 'rejection rate']
                ).to_csv(f'{output_dir}/{model_name}.csv', index=False)

# ls_spike_values = [1.25,1.5,2, 3,4]
# # --------- run -------------------------------------------------------------------------------------------------------------------------------
# model_name = _model_.__name__
# results = []
# for spike in ls_spike_values:
#     model_params.update({'spike_value': spike})
#     args = (n, d, _model_, model_params, num_permutations, N_iters, NUM_CORES, kernel_name, ridge_ls, band_factor_ls)
#     out = run_fast_parallel(*args)
#     for test_name in out[0].keys():
#         decisions = [_[test_name] for _ in out]
#         results += [[spike, test_name, np.mean(decisions)]]
# pd.DataFrame(   results, 
#                 columns = ['spike intensity', 'test', 'rejection rate']
#                 ).to_csv(f'{output_dir}/{model_name}.csv', index=False)
