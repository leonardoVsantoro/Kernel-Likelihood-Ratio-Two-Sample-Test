import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
os.chdir(parent_dir)
sys.path.insert(0, parent_dir)
from modules import *
from functions import *
import models.models_classes as models
ts = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
# --------- set parameters ---------------------------------------------------------------------------------------------------------------------
NUM_CORES = 72
num_permutations = 250
N_iters = 100
band_factor= [0.05, 0.1, 0.5, 1, 5]
ridge = np.logspace(-8, 1, 10)
kernel_name = 'sqeuclidean'
n,d = 100, 500
eps_values = [np.linspace(0, 0.1, 6)[1:], [0.01, 0.02, 0.03, 0.05, 0.06, 0.08]]
lsmodels = [ (models.MODEL_7, {'alpha': 0.75}), (models.MODEL_8, {'alpha': 0.5})] 
# --------- run -------------------------------------------------------------------------------------------------------------------------------
output_dir = f'../out/sims/{ts}'
os.makedirs(output_dir, exist_ok=True)
summary = 'sims_perm_eps.py\n\n'
for (_model_,model_params), ls_eps in zip(lsmodels, eps_values) :
    model_name = _model_.__name__
    tqdm.write(f"Running model: {model_name}")
    alpha = model_params['alpha']
    summary += 'Model {} with alpha: {} and eps : {}\n'.format(model_name[-1], alpha, ls_eps)
    results = []
    for eps in ls_eps:
        model_params.update({'eps': eps})
        tqdm.write(f"   eps : {eps}")
        out = run_fast_parallel(*(n, d, _model_, model_params, num_permutations, N_iters, NUM_CORES, kernel_name, ridge, band_factor))
        for test_name in out[0].keys():
            decisions = [_[test_name] for _ in out]
            results += [[eps, test_name, np.mean(decisions)]]
    pd.DataFrame( results, columns = [ 'eps', 'test', 'rejection rate']).to_csv(f'{output_dir}/{model_name}.csv', index=False)
summary+= f'\n\nsample_size : {n}\ndimension : {d}'        
summary+= f'\n\nbandwith_factor : {band_factor}\n\ridge : {ridge}\nkernel_name : {kernel_name}\nnum_permutations : {num_permutations}\nN_iters : {N_iters}'        
with open(f'{output_dir}/summary.txt', 'w') as f:
    f.write(summary)
# --------- end -------------------------------------------------------------------------------------------------------------------------------