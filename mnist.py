import os
from modules import *
from functions.tools import *
from functions.TestFuns import *

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict
from scipy.ndimage import gaussian_filter
from datetime import datetime

ts = datetime.now().strftime("%Y-%m-%d_%H:%M")

output_directories = [
    'figures',
    f'out/{ts}', 
    f'out/mnist/{ts}'
]
for directory in output_directories:
    os.makedirs(directory, exist_ok=True)

# -------------------------------- import data ---------------------------------------------------------------------------------

mnist_data_train = pd.read_csv('datasets/MNIST/mnist_train.csv').set_index('label')
mnist_data_train /= 255

# -------------------------------- plot sample data: original and perturbed -----------------------------------------------------

for i in range(3):
    fig, axs = plt.subplots(figsize=(8, 3), nrows=2, ncols=5)
    for num, ax in enumerate(axs.ravel()):
        X = mnist_data_train.loc[num].values[0]

        im = X.reshape(28, 28)
        im_white = (X + np.random.normal(0, .4, X.shape)).reshape(28, 28)
        im_blur = gaussian_filter(X.reshape(28, 28), 2)

        sns.heatmap([im, im_white, im_blur][i], square=True, cbar=False, ax=ax, vmin=-.2, vmax=1.2)
        ax.axis('off')
    fig.savefig(f'figures/mnist_{["original", "white_noise", "blur"][i]}.png')

# -------------------------------- comparing 2 sets of digits -----------------------------------------------------------------
group_1 = [9, 6, 8]
group_2 = [4, 8]
NUM_CORES = 10
num_replications = 200
num_samples = 75
num_permutations = 500
test_names = ['FH-G', 'FH-C', 'MMD', 'KNN', 'FR', 'HT']
kappa_K = 1e5

def run_perturbed_minst_test(X, Y, num_permutations, test_names):
    out = []
    for test in test_names:
        if test == 'MMD':
            out.append((test,  1 if MMD_two_sample_test(X, Y)(num_permutations) < 0.05 else 0))
        elif test == 'FH-G':
            out.append(( test, 1 if GKE_two_sample_test(X,Y, kappa_K = kappa_K)(num_permutations) < 0.05 else 0 ) )
        elif test == 'FH-C':
            out.append(( test, 1 if CKE_two_sample_test(X,Y, kappa_K = kappa_K)(num_permutations) < 0.05 else 0 ) )
        elif test == 'KNN':
            out.append((test,  1 if KNN_two_sample_test(X, Y, k=1)(num_permutations) < 0.05 else 0))
        elif test == 'FR':
            out.append((test,  1 if FR_two_sample_test(X, Y)(num_permutations) < 0.05 else 0))
        elif test == 'HT':
            out.append((test,  1 if HT_two_sample_test(X, Y, k=10)(num_permutations) < 0.05 else 0))
    return out

XYpairs = [(mnist_data_train.loc[group_1].sample(num_samples).values,
            mnist_data_train.loc[group_2].sample(num_samples).values,
            ) for _ in range(num_replications)]

# --------------------------------additive Gaussian white noise -----------------------------------------------------------------
sigmas = np.linspace(0, 1.5, 7)
results = {}
for sigma in sigmas:
    iter_args = [(X + np.random.normal(0, sigma, X.shape),
                  Y + np.random.normal(0, sigma, Y.shape),
                  num_permutations, test_names)
                 for (X, Y) in XYpairs]

    results[sigma] = Parallel(n_jobs=NUM_CORES)(
        delayed(run_perturbed_minst_test)(*args) for args in iter_args
    )
data = []
for sigma in sigmas:
    for line in results[sigma]:
        for el in line:
            test_name, value = el
            data.append({"sigma": sigma, "test_name": test_name, "value": value})
df = pd.DataFrame(data)
rej_perc_df = df.groupby(["sigma", "test_name"])["value"].mean().reset_index()

rej_perc_df.to_csv(f'out/mnist/{ts}/rp_additive.csv', index=False)

# -------------------------------- Blurring - Gaussian convolution --------------------------------------------------------------
sigmas = np.linspace(0, 5, 7)
results = {}
for sigma in sigmas:
    iter_args = [(np.array([gaussian_filter((_ + np.random.normal(0, .25, _.shape)).reshape(28, 28), sigma).flatten() for _ in X]),
                  np.array([gaussian_filter((_ + np.random.normal(0, .25, _.shape)).reshape(28, 28), sigma).flatten() for _ in Y]),
                  num_permutations, test_names)
                 for (X, Y) in XYpairs]

    results[sigma] = Parallel(n_jobs=NUM_CORES)(
        delayed(run_perturbed_minst_test)(*args) for args in iter_args
    )
data = []
for sigma in sigmas:
    for line in results[sigma]:
        for el in line:
            test_name, value = el
            data.append({"sigma": sigma, "test_name": test_name, "value": value})
df = pd.DataFrame(data)
rej_perc_df = df.groupby(["sigma", "test_name"])["value"].mean().reset_index()
rej_perc_df.to_csv(f'out/mnist/{ts}/rp_blurred.csv', index=False)

# -------------------------------- Plot percentage of rejections vs Noise -----------------------------------------------------
fig, [axl, axr] = plt.subplots(figsize=(12, 6), ncols=2)
df_additive = pd.read_csv(f'out/mnist/{ts}/rp_additive.csv')
df_blurred = pd.read_csv(f'out/mnist/{ts}/rp_blurred.csv')
for ax, df, title in zip([axl, axr], [df_additive, df_blurred], ['Additive Noise', 'Blurred Image']):
    sns.lineplot(
        data=df,
        x="sigma",
        y="value",
        hue="test_name",
        style="test_name",
        markers=True,
        dashes=False,
        ax=ax
    )
    ax.set_xlabel("Sigma")
    ax.set_ylabel("Rejection Percentage")
    ax.set_title(title)
    ax.legend(title="Test Name")
    ax.set_aspect('equal', adjustable='datalim')
plt.tight_layout()
fig.savefig(f'figures/mnist_rp_additive_blurred.png')


# -------------------------------- End ---------------------------------------------------------------------------------------
