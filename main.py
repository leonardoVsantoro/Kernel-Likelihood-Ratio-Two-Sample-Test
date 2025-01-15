from modules import *
from functions.tools import *
from functions.TestFuns import *
import models.models_classes as models
from functions.run import *


# --------- set parameters

NUM_CORES = 8

n = 75

n_d_values = [(n, 25),(n, 50),(n, 250),(n, 500),(n, 1000)]
num_permutations = 100
N_iters = 100

kappa_K = 1e5

test_names = ['FH', 'MMD', 'KNN', 'FR']

# --------- select model

# _model_ = models.isotropic_different_means
# model_params = {'mu' : 1.15, 'numDiffLocs' :  1000}

# _model_ = models.uni_vs_bimodal
# model_params = {'width' : 1}

# _model_ = models.isotropic_vs_DiagSpiked 
# model_params = {'num_spikes' : 5, 'spike_value' : 3}

# _model_ = models.isotropic_vs_scaledIsotropic
# model_params = {'sigma' : 1.2}


_model_ = models.isotropic_different_means
model_params = {'mu' : 1.25, 'numDiffLocs' :  50}

kernel_name = 'default'
kernel_bandwith = None

# --------- do modify below this line

def run_parallel(n, d, _model_, model_params, test_names, kernel_name, kernel_bandwith,  kappa_K, num_permutations, N_iters):
    iter_args = [(n, d, _model_,model_params, test_names, kernel_name, kernel_bandwith, kappa_K, num_permutations) for _ in range(N_iters)]
    results = Parallel(n_jobs=NUM_CORES)(delayed(run_iteration)(*args) for args in tqdm(iter_args, position=1, leave=False))
    return results

start_time = datetime.now()
results = []
for n, d in tqdm(n_d_values,position=0, leave=True):
    decisions_list = run_parallel(n, d, _model_, model_params, test_names, kernel_name, kernel_bandwith, kappa_K, num_permutations, N_iters)
    results.append(decisions_list)
end_time = datetime.now()
print("Run time:", end_time - start_time)

      
out = []
for name in test_names:
    for _, (n,d) in zip(results, n_d_values):
        out.append( [name, n,d, N_iters, kernel_name, kernel_bandwith, kappa_K, num_permutations,  pd.DataFrame(_).mean(0)[name] ])

model_name = _model_(**model_params)(0).name


if len( model_params.items()) > 1:
    folder_name = ''.join('{} : {}, '.format(key, value) for key, value in model_params.items())
else:
    folder_name = ''.join('{} : {}'.format(key, value) for key, value in model_params.items())

os.makedirs('out/' + model_name + '/' + folder_name, exist_ok=True)

ts = datetime.now().strftime("%Y-%m-%d_%H:%M")
data_out = pd.DataFrame(out, 
             columns= ['test', 'sample size', 'dimension', 'N_iters', 'kernel', 'kernel_bandwith', 'max conditioning number', 'num_permutations', 'percent of rejections']
             )
data_out.to_csv(f'out/{model_name}/{folder_name}/{ts}.csv')

print("Results saved successfully.")
