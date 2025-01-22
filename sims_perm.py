from modules import *
from functions.tools import *
from functions.TestFuns import *
import models.models_classes as models
from functions.run import *


# --------- set parameters ---------------------------------------------------------------------------------------------------------------------
NUM_CORES = 72
n = 50
n_d_values = [(n, 25),(n, 50),(n, 250),(n, 500),(n, 1000),(n, 1500)]
n = 100
n_d_values += [(n, 25),(n, 50),(n, 250),(n, 500),(n, 1000),(n, 1500)]

num_permutations = 500
N_iters = 100
kappa_K = 1e5
kernel_name = 'default'
kernel_bandwith = None
test_names = ['FH', 'MMD', 'KNN', 'FR', 'HT']


# --------- select model ---------------------------------------------------------------------------------------------------------------------
lsmodels = []

_model_ = models.isotropic_different_means
model_params = {'mu' : 1.3, 'numDiffLocs' :  20}
lsmodels.append((_model_ , model_params))

_model_ = models.isotropic_vs_DiagSpiked
model_params = {'num_spikes' : 10,  'spike_value' : 5}
lsmodels.append((_model_ , model_params))

_model_ = models.isotropic_vs_DiagSpiked_different_means
model_params = {'num_spikes' : 10,  'spike_value' : 5, 'mu' : 1.3, 'numDiffLocs' :  20}
lsmodels.append((_model_ , model_params))


# --------- run -------------------------------------------------------------------------------------------------------------------------------
def run_parallel(n, d, _model_, model_params, test_names, kernel_name, kernel_bandwith,  kappa_K, num_permutations, N_iters):
    iter_args = [(n, d, _model_,model_params, test_names, kernel_name, kernel_bandwith, kappa_K, num_permutations) for _ in range(N_iters)]
    results = Parallel(n_jobs=NUM_CORES)(delayed(run_iteration)(*args) for args in iter_args)
    return results


lsout = [] 
for _model_, model_params in lsmodels:
    results = []
    for n, d in n_d_values:
        decisions_list = run_parallel(n, d, _model_, model_params, test_names, kernel_name, kernel_bandwith, kappa_K, num_permutations, N_iters)
        results.append(decisions_list)
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
    lsout.append(f'out/{model_name}/{folder_name}/{ts}.csv')

# --------- plot ------------------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


title_dict = {'isotropic_different_means' : 'Isotropic Gaussian data with different means',
              'isotropic_vs_DiagSpiked' : 'Isotropic and diagonal spiked Gaussian data',}
for file in lsout:
    df = pd.read_csv(file)
    ns = df['sample size'].unique()
    
    fig, axs = plt.subplots(figsize=(5.5*len(ns) , 6), ncols = len(ns))

    fig.suptitle('{}'.format(title_dict[file.split('/')[1]]))
    for n,ax in zip(ns,axs):
        sns.lineplot(data=df[df['sample size'] == n], x="dimension",  y="percent of rejections", hue="test",style="test", markers=True, dashes=False, ax=ax)
        ax.set_xlabel("Dimension (log)")
        ax.set_ylabel("Rejection Percentage")
        ax.set_title('sample size: {}'.format( n), fontsize = 10)
        ax.legend(title="Test")
        ax.set_ylim(0, 1.025)
        ax.set_xscale('log') 
        plt.tight_layout()


	
    output_dir = f'figures/{ts}'
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f'{output_dir}/simulations_{file.split("/")[1]}.png')
    
# --------- end -------------------------------------------------------------------------------------------------------------------------------