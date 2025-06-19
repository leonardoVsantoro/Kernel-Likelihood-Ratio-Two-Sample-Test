import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
os.chdir(parent_dir)
sys.path.insert(0, parent_dir)

from src import *
from tests import models_classes as models


def run_fast(X,Y, num_permutations, kernel_name, ridge, band_factor_ls, symmetrise = True, project = True):
    
    pvals = {
                name : KernelTwoSampleTest(name)(
                    X = X, 
                    Y= Y, 
                    kernel_name = kernel_name,
                    band_factor_ls = band_factor_ls, 
                    ridge_ls = ridge,
                    symmetrise = True, 
                    project = True 
                    )(num_permutations=num_permutations)
                    for name in ['KLR-0', 'KLR', 'CM', 'Agg-MMD', 'SpecReg-MMD']
             }

def run_parallel(n, d,  _model_, model_params, num_permutations, N_iters, NUM_CORES, kernel_name, ridge, band_factor , light = False):
    model = _model_(**model_params)(d)
    X = model.sample_X(n)
    Y = model.sample_Y(n)
    return run_fast(X, Y, num_permutations, kernel_name, ridge, band_factor)
    
