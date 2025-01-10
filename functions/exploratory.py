

from modules import *

from functions.tools import *
from functions.TestFuns import *
from models import models_classes as models
from functions.run import *


from matplotlib import pyplot as plt # type: ignore
from matplotlib.ticker import ScalarFormatter # type: ignore
import seaborn as sns # type: ignore

# import torch# type: ignore
# import subprocess# type: ignore
# def run_terminal_command(command):
#     process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     stdout, stderr = process.communicate()
#     if process.returncode != 0:
#         print(f"Error: {stderr.decode('utf-8')}")
#     else:
#         print(stdout.decode('utf-8'))

# run_terminal_command('pip install git+https://github.com/josipd/torch-two-sample.git')

# from torch_two_sample.statistics_nondiff import FRStatistic, KNNStatistic  # type: ignore
# from torch_two_sample.statistics_diff import MMDStatistic # type: ignore


def sample_test_vals(sample_X, sample_Y, n, test_names, kappa_K = 1e6, kernel = None ):
    X = sample_X(n)
    Y = sample_Y(n)  
    pooled = np.vstack([X, Y])

    values  = []
        
    for test_name in test_names:
        if test_name in all_CovKerEmb_tests:
            if kernel is None:
                pairwise_dists = cdist(pooled, pooled, 'euclidean')
                median_dist = np.median(pairwise_dists[pairwise_dists > 0])  # Avoid zero distances
                bandwidth = 2 * median_dist
                kernel_matrix  = np.exp( - pairwise_dists / bandwidth)
            else:
                kernel_matrix  = kernel(pooled, pooled)  

            kxx = kernel_matrix[:n, :n]
            kxy = kernel_matrix[:n, n:]
            kyy = kernel_matrix[n:, n:]
            KX, KX_ED, KY, KY_ED = get_Kmats_X_Y(kxx, kxy, kyy, kappa_K)
            values.append(CovKerEmb_test_stat(test_name, KX, KY, KX_ED, KY_ED))
        if test_name == 'MMD':
            values.append(MMD_two_sample_test(X,Y).obs_value)
        # if test_name == 'torchMMD':
        #     _stat = MMDStatistic(n,n) # type: ignore
        #     X_torch = torch.autograd.Variable(torch.tensor(X, dtype=torch.float32)) # type: ignore
        #     Y_torch = torch.autograd.Variable(torch.tensor(Y, dtype=torch.float32)) # type: ignore
        #     values.append(_stat(X_torch, Y_torch, [1e-2]).item())
        if test_name == 'KNN':
            values.append(KNN_two_sample_test(X,Y).obs_value)
        if test_name == 'FR':
            values.append(FR_two_sample_test(X,Y).obs_value)
        if test_name == 'MPZ':
            values.append(MPZ_two_sample_test(X,Y).obs_value)
    return values


def H0_H1(n,d, _model_,test_names, kappa_K, num_reps, kernel = None, NUM_CORES=4):

    # sample from null 
    iter_args = [(_model_(d).sample_X, _model_(d).sample_X, n, test_names, kappa_K, kernel) for _ in range(num_reps)]
    null_vals = Parallel(n_jobs=NUM_CORES)(delayed(sample_test_vals)(*args) for args in iter_args)
    # null_vals = {test_name : np.array(null_vals)[:,i] for i, test_name in enumerate(test_names)}

    # sample from alternative 
    iter_args = [(_model_(d).sample_X, _model_(d).sample_Y, n, test_names, kappa_K, kernel) for _ in range(num_reps)]
    alternative_vals = Parallel(n_jobs=NUM_CORES)(delayed(sample_test_vals)(*args) for args in iter_args)
    # alternative_vals = {test_name : np.array(alternative_vals)[:,i] for i, test_name in enumerate(test_names)}

    return null_vals, alternative_vals

def plot_H0_H1(null_vals, alternative_vals, test_names, n, d, kappa_K= 'not given'):
    # plot results
    fig, axs = plt.subplots(1,len(test_names), figsize=(15, 2)); axs = axs.ravel()
        
    fig.suptitle(f'n={n}, d={d}, $\kappa$={kappa_K}', y=1.3)
    for ax, test in zip(axs, test_names):
        fromNULL = np.array(null_vals[test])
        fromALTERNATIVE = np.array(alternative_vals[test])
        sns.histplot(x = fromNULL, ax=ax, color='b', label = 'null')
        sns.histplot(x = fromALTERNATIVE, ax=ax, color='r', label = 'alternative')

        rthresh = np.quantile(fromNULL, .95)
        power = np.mean(fromALTERNATIVE > rthresh); level = np.mean(fromNULL > rthresh)
        ax.axvline(x=rthresh, color='g', linestyle='--', label = 'threshold', alpha =.9)

        ax.set_title(f'{test}\npower: {power:.2f} | level: {level:.2f}', y=1.1)
        ax.legend();
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False)); ax.ticklabel_format(style='plain', axis='x')
        for label in ax.get_xticklabels():
            label.set_rotation(90)
    plt.show()
    return None