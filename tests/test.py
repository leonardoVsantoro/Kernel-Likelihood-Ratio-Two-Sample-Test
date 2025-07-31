import os
import sys
import time
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
os.chdir(parent_dir)
sys.path.insert(0, parent_dir)
from src import *
np.random.seed(42)
# --------- set parameters -----------------------------------------------------------------------------------------
num_permutations = 250
band_factor_ls = [0.05, 0.1, 0.5, 1, 5]
ridge_ls = np.logspace(-8, 1, 10)
kernel_name = 'sqeuclidean'
# --------- define sampling model ---------------------------------------------------------------------------------
n = 50
d = 500
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
test_funcs = {
    'KLR': lambda: KLR(**keriterargs_dict)(num_permutations=num_permutations, NUM_CORES=10),
    'KLR0': lambda: KLR0(**keriterargs_dict)(num_permutations=num_permutations, NUM_CORES=10),
    'Agg_MMD': lambda: Agg_MMD(**keriterargs_dict)(num_permutations=num_permutations, NUM_CORES=10),
    'SpecReg_MMD': lambda: SpecReg_MMD(**keriterargs_dict)(num_permutations=num_permutations, NUM_CORES=10),
    'CM': lambda: CM(**keriterargs_dict)(num_permutations=num_permutations, NUM_CORES=10),
    'HT': lambda: HT(X, Y)(num_permutations=num_permutations),
    'KNN': lambda: KNN(X, Y)(num_permutations=num_permutations),
    'FR': lambda: FR(X, Y)(num_permutations=num_permutations)
}

out = {}
runtimes = {}
_ = Agg_MMD(**keriterargs_dict)(num_permutations=1, NUM_CORES=10) # "dummy" parallel job before the loop to warm up the pool:
for name, func in test_funcs.items():
    start = time.time()
    result = func()
    end = time.time()
    out[name] = result
    runtimes[name] = end - start

# --------- display results ---------------------------------------------------------------------------------------
print(f"{'Test Name':<15} {'p-value':<10} {'runtime (s)':<12}")
print("-" * 40)
for name, test in sorted(out.items(), key=lambda item: item[1].p_value):
    print(f"{name:<15} {test.p_value:<10.4f} {runtimes[name]:<12.2f}")