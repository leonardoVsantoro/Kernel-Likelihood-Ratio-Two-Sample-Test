null = True #set if run under H0 or H1
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
os.chdir(parent_dir)
sys.path.insert(0, parent_dir)
from utils import *
import torch 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import pickle
import pandas as pd
mnistdata = torch.utils.data.DataLoader(
    datasets.MNIST(
        "./data/mnist",
        train=False,
        download=False,
        transform=transforms.Compose(
            [transforms.Resize(32), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=10000,
    shuffle=True,
)

with open('./data/mnist/fake_mnist.pckl', 'rb') as file:
    fake_mnist = pd.DataFrame( pickle.load(file)[0].reshape(10000, -1))


real_mnist = pd.DataFrame(next(iter(mnistdata))[0].reshape(10000, -1).detach().cpu().numpy())

real_mnist = real_mnist.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
fake_mnist = fake_mnist.apply(lambda x: (x - x.mean()) / x.std(), axis=1)


N_iters = 72
NUM_CORES = 72
num_permutations = 300
kernel_name = 'euclidean'
ridge_ls = np.logspace(-7, 0, 8)
band_factor_ls = [0.1, 1, 10]

sample_sizes = [100, 200, 300]
results = []

for sample_size in sample_sizes:
    out = run_fast_parallel_sampling(
        real_mnist, fake_mnist, sample_size,
        num_permutations, N_iters, NUM_CORES,
        kernel_name, ridge_ls, band_factor_ls, light=False, null = null
    )
    df = pd.DataFrame(out).mean(0).rename('rejection_rate').to_frame()
    df['sample_size'] = sample_size
    df = df.rename_axis('test').reset_index()
    results.append(df)

final_df = pd.concat(results, ignore_index=True)


if null: # type: ignore
    _name = 'mnist-null'
else:
    _name = 'mnist'
final_df.to_csv(f'../out/real/{_name}.csv', index=False)