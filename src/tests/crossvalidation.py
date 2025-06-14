# Set working directory to the parent folder of this script
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
os.chdir(parent_dir)
sys.path.insert(0, parent_dir)
from modules import *
from functions import *

import models.models_classes as models
ts = datetime.now().strftime("%Y-%m-%d_%H:%M")

# ------------------------------ set parameters ---------------------------------------------------------------------------------------------------------------------
NUM_CORES = 72
n, d = 50, 250
N_iters = 100
num_permutations = 250
kernel_name = 'sqeuclidean'
band_factor_ls = [0.01, 0.05, 0.1, 0.5, 1, 5, 10]
ridge_ls = np.logspace(-9, -1, 9)

# NUM_CORES = 4; n,d = 5,100; N_iters = 2; num_permutations = 2; band_factor_ls = [1]; ridge_ls = [1]
# ------------------------------ run -------------------------------------------------------------------------------------------------------------------------------
summary = ''
for _model_,model_params in models.lsmodels:
    tqdm.write(f"Running model: {_model_.__name__} with parameters: {model_params}")
    results = []
    for band_factor in band_factor_ls:
        for ridge in ridge_ls:
            tqdm.write(f"   Running band_factor = {band_factor}, ridge = {ridge}" )
            args = (n, d, _model_, model_params, num_permutations, N_iters, NUM_CORES, kernel_name, [ridge], [band_factor], True )
            out = run_fast_parallel(*args)
            test_names = out[0].keys()
            for test_name in test_names:
                decisions = [_[test_name] for _ in out]
                results += [[band_factor, ridge, test_name, np.mean(decisions), np.std(decisions)]]

    data = pd.DataFrame( results, columns = ['band_factor','ridge', 'test_name', 'rejection_rate', 'std'])
    fig, axs = plt.subplots(1, len(test_names), figsize=(15, 5), sharey=True)
    fig.suptitle(f'{_model_.__name__}', fontsize=16)
    for ax, name in zip(axs, test_names):
        subset = data[data.test_name == name].pivot(index='ridge', columns='band_factor', values='rejection_rate')
        sns.heatmap(subset, ax=ax, annot=True, fmt=".1f", cmap="RdBu", cbar_kws={'label': 'Rejection Rate'}, vmin=0, vmax=1, cbar=False)
        ax.set_title(name)
        ax.set_xlabel('band_factor')
        ax.set_ylabel('ridge')
        ax.set_aspect('equal', adjustable='box')

# ------------------------------ save ------------------------------------------------------------------------------------------------------------------------------
    output_dir = f'../out/crossvalidation/{ts}/'
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f'{output_dir}/{_model_.__name__}.png', bbox_inches='tight')
    summary += 'Model {} {}\n'.format(_model_.__name__[-1], tuple(model_params.values()))
    data.to_csv(f'{output_dir}/{_model_.__name__}.csv', index=False)
summary += f'\n\nbandwith_factor : {band_factor_ls}\nridge : {ridge_ls}\nkernel_name : {kernel_name}\nnum_permutations : {num_permutations}\nN_iters : {N_iters}'
summarytxt_path = f'{output_dir}/summary.txt'
with open(summarytxt_path, 'w') as f:
    f.write(summary)    