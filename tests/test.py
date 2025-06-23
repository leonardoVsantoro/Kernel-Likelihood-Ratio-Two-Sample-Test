import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
os.chdir(parent_dir)
sys.path.insert(0, parent_dir)
from src import *
np.random.seed(42)
# --------- set parameters -----------------------------------------------------------------------------------------
num_permutations = 500
band_factor_ls = [0.05, 0.1, 0.5, 1, 5]
ridge_ls = np.logspace(-8, 1, 10)
kernel_name = 'sqeuclidean'
# --------- define sampling model ---------------------------------------------------------------------------------
n = 50
d = 1000
X = np.random.multivariate_normal(np.zeros(d), np.eye(d), n)
Y = np.random.multivariate_normal(np.zeros(d), np.eye(d), n)
# --------- run different tests and store results -----------------------------------------------------------------
keriterargs_dict = {'X' : X, 
                    'Y': Y,
                    'kernel_name': kernel_name,
                    'band_factor_ls': band_factor_ls,
                    'ridge_ls': ridge_ls,
                    'symmetrise': True,
                    'project': True
                    }
out = {
    'KLR': KLR(**keriterargs_dict)(num_permutations=num_permutations, NUM_CORES=10),
    'KLR0': KLR0(**keriterargs_dict)(num_permutations=num_permutations, NUM_CORES=10),
    'Agg_MMD': Agg_MMD(**keriterargs_dict)(num_permutations=num_permutations, NUM_CORES=10),
    'SpecReg_MMD': SpecReg_MMD(**keriterargs_dict)(num_permutations=num_permutations, NUM_CORES=10),
    'CM': CM(**keriterargs_dict)(num_permutations=num_permutations, NUM_CORES=10),
    'HT': HT(X, Y)(num_permutations=num_permutations),
    'KNN': KNN(X, Y)(num_permutations=num_permutations),
    'FR': FR(X, Y)(num_permutations=num_permutations)
}
# --------- display results ---------------------------------------------------------------------------------------
print(f"{'Test Name':<15} {'p-value':<10}")
print("-" * 25)
for name, test in sorted(out.items(), key=lambda item: item[1].p_value):
    print(f"{name:<15} {test.p_value:<10.4f}")