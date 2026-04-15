null = True #set if run under H0 or H1

import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
os.chdir(parent_dir)
sys.path.insert(0, parent_dir)
from utils import *
import torchvision.transforms as transforms
import torchvision
transform = transforms.Compose([transforms.ToTensor(),])

cifar10_1 = np.load("data/cifar/cifar10_1/cifar10.1_v6_data.npy")
cifar10 = torchvision.datasets.CIFAR10(root='./data/cifar/cifar10', train=True, download=False, transform=transform)
cifar10 = np.array([np.array(image.permute(1, 2, 0)) for image, _ in cifar10])*255  # convert CHW to HWC
cifar10_1_flat = np.array([_.flatten() for _ in cifar10_1])
cifar10_flat = np.array([_.flatten() for _ in cifar10])
pd_cifar10_1 = pd.DataFrame(cifar10_1_flat)
pd_cifar10 = pd.DataFrame(cifar10_flat)


N_iters = 72
NUM_CORES = 72
num_permutations = 100
kernel_name = 'euclidean'
ridge_ls = np.logspace(-10, 0, 7)
band_factor_ls = [0.1, 1, 10]
sample_sizes = [250,500,750,1000]
results = []
for sample_size in sample_sizes:
    out = run_fast_parallel_sampling(
        pd_cifar10, pd_cifar10_1, sample_size,
        num_permutations, N_iters, NUM_CORES,
        kernel_name, ridge_ls, band_factor_ls, light=True, null = null
    )
    df = pd.DataFrame(out).mean(0).rename('rejection_rate').to_frame()
    df['sample_size'] = sample_size
    df = df.rename_axis('test').reset_index()
    results.append(df)
final_df = pd.concat(results, ignore_index=True)

if null: # type: ignore
    _name = 'cifar-null'
else:
    _name = 'cifar'
final_df.to_csv(f'../out/real/{_name}.csv', index=False)