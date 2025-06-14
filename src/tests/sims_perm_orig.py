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


# --------- set parameters ---------------------------------------------------------------------------------------------------------------------
NUM_CORES = 72
n = 50
n_d_values = [(n, 25),(n, 50),(n, 250),(n, 500),(n, 1000),(n, 1500), (n, 2000)]
n = 100
n_d_values += [(n, 25),(n, 50),(n, 250),(n, 500),(n, 1000),(n, 1500),(n, 2000)]
num_permutations = 250
N_iters = 100
band_factor= [0.05, 0.1, 0.5, 1, 5]
ridge = np.logspace(-9, -1, 9)
kernel_name = 'sqeuclidean'
# --------- run -------------------------------------------------------------------------------------------------------------------------------
# testvals
# n_d_values = [(50,20),(50,50), (75,20), (75,50)]; num_permutations = 100; N_iters = 10; NUM_CORES = 4; ridge = [0.5,5]; band_factor = [0.1, 0.5, 1]

lsout = [] 
output_dir = f'../out/sims/{ts}'
os.makedirs(output_dir, exist_ok=True)
summary = ''

for _model_,model_params in [models.lsmodels[-1]]:
    model_name = _model_.__name__
    lsout.append(f'{output_dir}/{model_name}.csv')
    tqdm.write(f"Running model: {_model_.__name__} with parameters: {model_params}")
    results = []
    for n,d in n_d_values:
        tqdm.write(f"   Running n={n}, d={d}")
        args = (n, d, _model_, model_params, num_permutations, N_iters, NUM_CORES, kernel_name, ridge, band_factor)
        out = run_fast_parallel(*args)
        for test_name in out[0].keys():
            decisions = [_[test_name] for _ in out]
            results += [[n, d, test_name, np.mean(decisions)]]

    pd.DataFrame( results, 
                 columns = ['sample size', 'dimension', 'test', 'rejection rate']
                 ).to_csv(f'{output_dir}/{model_name}.csv', index=False)
    

    summary += 'Model {} {}\n'.format(_model_.__name__[-1], tuple(model_params.values()))

summary+= f'\n\nbandwith_factor : {band_factor}\nridge : {ridge}\nkernel_name : {kernel_name}\nnum_permutations : {num_permutations}\nN_iters : {N_iters}'        
summarytxt_path = f'{output_dir}/summary.txt'
with open(summarytxt_path, 'w') as f:
    f.write(summary)
    

# --------- plot ------------------------------------------------------------------------------------------------------------------------------
titles = [line for line in open(summarytxt_path).read().splitlines()]
for file,title in zip(lsout, titles):
    model_name = file.split('/')[-1].split('.csv')[0]
    df = pd.read_csv(file)

    ns = df['sample size'].unique()
    ds = df['dimension'].unique()

    sns.set_style("whitegrid")
    sns.set_palette("bright")
    sns.set(font="DejaVu Sans")

    fig, axs = plt.subplots(figsize=(5.5 * len(ns), 5.75), ncols=len(ns))
    fig.suptitle(title, fontweight="bold")
    if len(ns) == 1:
        axs = [axs]
    tests = df['test'].unique()
    for n, ax in zip(ns, axs):
        subset = df[df['sample size'] == n]
        sns.lineplot(data=subset, x="dimension", y="rejection rate", hue="test", style="test", marker='o', dashes=True, ax=ax, alpha = .75, lw=1.5)
        for test in ['KLR', 'KLR-0']:
            if test in subset['test'].values:
                klr_data = subset[subset['test'] == test]
                ax.plot( klr_data['dimension'], klr_data['rejection rate'],  linewidth=10, alpha=0.15,color=ax.get_lines()[list(tests).index(test)].get_color(), zorder=1)
        ax.set_xlabel("Dimension", fontsize=12)
        ax.set_ylabel("Rejection rate", fontsize=12)
        ax.set_title(f"Sample size: {n}", fontsize=13)
        ax.set_ylim(0, 1.025)
        ax.set_xscale('log')
        ax.tick_params(labelsize=11)
        ax.get_legend().remove()
        ax.set_xticks(ds)
        ax.set_xticklabels(ds, fontsize=11, rotation=90)
        ax.set_ylim(-0.025, 1.025)
    fig.legend(axs[0].get_legend_handles_labels()[0], axs[0].get_legend_handles_labels()[1], 
            loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=len(tests), fontsize=11)
    sns.despine()
    plt.tight_layout()

    output_dir = f'../out/sims/{ts}/'
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f'{output_dir}/{model_name}.png', bbox_inches='tight')

    
# --------- end -------------------------------------------------------------------------------------------------------------------------------