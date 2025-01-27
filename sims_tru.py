from modules import *
from functions.tools import *
from functions.TestFuns import *
from models import models_classes as models
from functions.run import *
from functions.tru import *

test_names = ['FH', 'MMD', 'KNN', 'FR', 'HT']

for _model_ in [ models.MODEL_1(.25, 20),
                 models.MODEL_2(10, 3),
                 models.MODEL_3(.25,3, 20,10)]:
    
#----------------------- KDE PART ----------------------------------------------------------------------    
    n, d = 50, 250
    num_reps = 200
    kappa_K = 1e6

    null_vals, alternative_vals= H0_H1(n, d, _model_, test_names, kappa_K, num_reps, kernel = None, NUM_CORES=8)
    null_vals = {name : np.array(null_vals)[:,i] for i,name in enumerate(test_names)}
    alternative_vals = {name : np.array(alternative_vals)[:,i] for i,name in enumerate(test_names)}

    # ----- PLOT 
    fig = plt.figure(figsize=(8, 6))
    fig.suptitle('{}\nsample size : {}, dimension : {}'.format(title_dict[_model_(d).name],n,d), y=1)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[0, 1])
    gs.update(hspace=0.3)
    gs_bottom = gs[1, :].subgridspec(1, 3)  
    ax3 = fig.add_subplot(gs_bottom[0]); ax4 = fig.add_subplot(gs_bottom[1]); ax5 = fig.add_subplot(gs_bottom[2])
    for ax, test in zip([ax1,ax2,ax3,ax4,ax5], test_names):
        fromNULL = np.array(null_vals[test])
        fromALTERNATIVE = np.array(alternative_vals[test])
        sns.kdeplot(x = fromNULL, ax=ax, color='b', label = 'null', fill=True, common_norm = True)
        sns.kdeplot(x = fromALTERNATIVE, ax=ax, color='r', label = 'alternative', fill=True, common_norm = True)
        power = max( [np.mean(fromALTERNATIVE > np.quantile(fromNULL, .95)), np.mean(fromALTERNATIVE < np.quantile(fromNULL, .05))]); 
        level = 0.05
        ax.set_title(f'{test}\npower: {power:.2f} | level: {level:.2f}', y=1)
        ax.set_xticks([]); ax.set_yticks([])
        for label in ax.get_xticklabels():
            label.set_rotation(90)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2)
    fig.savefig('figures/truth_kde_{}.png'.format(_model_(d).name))


#----------------------- POWER PART ----------------------------------------------------------------------    
    n_d__values =  [(50, 25), (50, 50),(50, 150), (50, 300), (50, 500), (50, 1000)]
    powers = {test : [] for test in test_names}
    for n, d in (n_d__values):
        null_vals, alternative_vals= H0_H1(n, d, _model_, test_names, kappa_K, num_reps, kernel = None, NUM_CORES=8)
        null_vals = {name : np.array(null_vals)[:,i] for i,name in enumerate(test_names)}
        alternative_vals = {name : np.array(alternative_vals)[:,i] for i,name in enumerate(test_names)}
        for test in test_names:
            fromNULL = np.array(null_vals[test]); fromALTERNATIVE = np.array(alternative_vals[test])
            powers[test].append(  max( [np.mean(fromALTERNATIVE > np.quantile(fromNULL, .95)), np.mean(fromALTERNATIVE < np.quantile(fromNULL, .05))]) )

    # ----- PLOT 
    fig, ax = plt.subplots(figsize=(6.5, 6))
    sns.lineplot(pd.DataFrame(powers).assign(d=[d for n,d in n_d__values]).set_index('d'), markers=True, dashes=False, ax=ax)
    ax.set_xscale('log')
    ax.set_ylabel('power')
    fig.savefig('figures/truth_power_{}.png'.format(_model_(d).name))

