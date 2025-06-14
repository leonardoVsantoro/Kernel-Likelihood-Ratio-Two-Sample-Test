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
d = 2
n_d_values = [(50, d),(75, d),(100, d),(125, d),(150, d)]
num_permutations = 250
N_iters = 100
band_factor= [0.05, 0.1, 0.5, 1, 5]
ridge = np.logspace(-8, 1, 10)
kernel_name = 'sqeuclidean'
lsmodels = [ (models.MODEL_6 , {'eps' : 0.1}), (models.MODEL_9 , {'kappa' : .75})] 
# --------- run -------------------------------------------------------------------------------------------------------------------------------
output_dir = f'../out/sims/{ts}'
os.makedirs(output_dir, exist_ok=True)
summary = 'sims_perm_n.py\n\n'
for _model_,model_params in lsmodels:
    model_name = _model_.__name__
    tqdm.write(f"Running model: {_model_.__name__} with parameters: {model_params}")
    results = []
    for n,d in n_d_values:
        tqdm.write(f"   Running n={n}, d={d}")
        args = (n, d, _model_, model_params, num_permutations, N_iters, NUM_CORES, kernel_name, ridge, band_factor)
        out = run_fast_parallel(*args)
        for test_name in out[0].keys():
            decisions = [_[test_name] for _ in out]
            results += [[n, test_name, np.mean(decisions)]]
    pd.DataFrame( results, 
                 columns = ['sample size', 'test', 'rejection rate']
                 ).to_csv(f'{output_dir}/{model_name}.csv', index=False)
    summary += 'Model {} {}\n'.format(_model_.__name__[-1], tuple(model_params.values()))
summary+= f'\n\ndimension : {d}\nbandwith_factor : {band_factor}\nridge : {ridge}\nkernel_name : {kernel_name}\nnum_permutations : {num_permutations}\nN_iters : {N_iters}'        
summarytxt_path = f'{output_dir}/summary.txt'
with open(summarytxt_path, 'w') as f:
    f.write(summary)
# --------- end -------------------------------------------------------------------------------------------------------------------------------
