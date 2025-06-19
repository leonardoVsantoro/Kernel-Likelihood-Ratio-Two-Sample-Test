import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
os.chdir(parent_dir)
sys.path.insert(0, parent_dir)

from src import *
from tests import models_classes as models
from tests.plot import *
from tests.run import *


# --------- set parameters ---------------------------------------------------------------------------------------------------------------------
NUM_CORES = 72
num_permutations = 250
N_iters = 100
band_factor= [0.05, 0.1, 0.5, 1, 5]
ridge = np.logspace(-8, 1, 10)
kernel_name = 'sqeuclidean'
# --------- run - PART 1 -----------------------------------------------------------------------------------------------------------------------
n = 100
n_d_values = [(n, 25),(n, 50),(n, 250),(n, 500),(n, 1000),(n, 1500),(n, 2000)]
lsmodels = [ (models.MODEL_2, {'spike_value' : 3, 'num_spikes' : 8}), (models.MODEL_4 , {'epsilon' : 0.1, 'P' : 30})]
for _model_, model_params in lsmodels:
    model_name = _model_.__name__
    results = []
    for n,d in n_d_values:
        args = (n, d, _model_, model_params, num_permutations, N_iters, NUM_CORES, kernel_name, ridge, band_factor)
        out = run_parallel(*args)
        for test_name in out[0].keys():
            decisions = [_[test_name] for _ in out]
            results += [[d, test_name, np.mean(decisions)]]
    pd.DataFrame(   results, 
                    columns = ['dimension', 'test', 'rejection rate']
                 ).to_csv(f'out/data/D/{model_name}.csv', index=False)
    NUM_CORES = 72
# --------- run - PART 2 -----------------------------------------------------------------------------------------------------------------------
d = 2
n_d_values = [(50, d),(75, d),(100, d),(125, d),(150, d)]
lsmodels = [ (models.MODEL_6 , {'eps' : 0.1}), (models.MODEL_9 , {'kappa' : .75})] 
for _model_,model_params in lsmodels:
    model_name = _model_.__name__
    results = []
    for n,d in n_d_values:
        args = (n, d, _model_, model_params, num_permutations, N_iters, NUM_CORES, kernel_name, ridge, band_factor)
        out = run_parallel(*args)
        for test_name in out[0].keys():
            decisions = [_[test_name] for _ in out]
            results += [[n, test_name, np.mean(decisions)]]
    pd.DataFrame( results, 
                 columns = ['sample size', 'test', 'rejection rate']
                 ).to_csv(f'out/data/N/{model_name}.csv', index=False)
# --------- run - PART 2 -----------------------------------------------------------------------------------------------------------------------
n,d = 100, 500
eps_values = [np.linspace(0, 0.1, 6)[1:], [0.01, 0.02, 0.03, 0.05, 0.06, 0.08]]
lsmodels = [ (models.MODEL_7, {'alpha': 0.75}), (models.MODEL_8, {'alpha': 0.5})] 
# --------- run -------------------------------------------------------------------------------------------------------------------------------
for (_model_,model_params), ls_eps in zip(lsmodels, eps_values) :
    model_name = _model_.__name__
    tqdm.write(f"Running model: {model_name}")
    alpha = model_params['alpha']
    results = []
    for eps in ls_eps:
        model_params.update({'eps': eps})
        tqdm.write(f"   eps : {eps}")
        out = run_parallel(*(n, d, _model_, model_params, num_permutations, N_iters, NUM_CORES, kernel_name, ridge, band_factor))
        for test_name in out[0].keys():
            decisions = [_[test_name] for _ in out]
            results += [[eps, test_name, np.mean(decisions)]]
    pd.DataFrame( results, columns = [ 'eps', 'test', 'rejection rate']).to_csv(f'out/data/EPS/{model_name}.csv', index=False)
# --------- PLOT -------------------------------------------------------------------------------------------------------------------------------
folders = [f for f in os.listdir('out/data') if os.path.isdir(os.path.join('../out/sims', f))]
for folder in folders:
    print(folder); plot(folder)
    plt.savefig(f'out/figures/{folder}.png', dpi=300, bbox_inches='tight')


