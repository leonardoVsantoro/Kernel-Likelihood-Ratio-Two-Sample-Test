from src import *
def plot(folder):
    path = f'../out/sims/{folder}'
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    summary = pd.read_csv(path + '/summary.txt', sep='\t')
    print(summary)
    fig, axs = plt.subplots(figsize=(5.5 * len(csv_files), 4.75), ncols=len(csv_files))
    for i, (file,ax) in enumerate(zip(csv_files, axs)):
        model_name = file.split('/')[-1].split('.csv')[0]
        df = pd.read_csv(path + '/' + file)
        if summary.columns[0] == 'sims_perm_n.py':
            iterable = df['sample size'].unique()
            iterable_label = 'sample size'
        elif summary.columns[0] == 'sims_perm_d.py':
            iterable = df['dimension'].unique()
            iterable_label = 'dimension'
        elif summary.columns[0] == 'sims_perm_eps.py':
            iterable = df['eps'].unique()
            iterable_label = 'eps'

        sns.set_style("whitegrid")
        sns.set_palette("bright")
        sns.set(font="DejaVu Sans")

        title = summary.values[i][0]
        if title[-2] == ',':
            title = title[:-2] + ')'
        ax.set_title(title, fontweight="bold")

        tests = df['test'].unique()
        sns.lineplot(data=df, x=iterable_label, y="rejection rate", hue="test", style="test", marker='o', dashes=True, ax=ax, alpha = .75, lw=1.5)
        for test in ['KLR', 'KLR-0']:
            if test in df['test'].values:
                klr_data = df[df['test'] == test]
                ax.plot( klr_data[iterable_label], klr_data['rejection rate'],  linewidth=10, alpha=0.15,color=ax.get_lines()[list(tests).index(test)].get_color(), zorder=1)
        ax.set_xlabel(iterable_label, fontsize=12)
        ax.set_ylabel("Rejection rate", fontsize=12)
        ax.set_ylim(0, 1.025)
        # if iterable_label in ['eps', 'dimension']:
        #     ax.set_xscale('log')
        ax.tick_params(labelsize=11)
        ax.get_legend().remove()
        # ax.set_xticks(iterable)
        # ax.set_xticklabels([str(_) for _ in iterable], fontsize=11, rotation=90)
        ax.set_ylim(-0.025, 1.025)
    fig.legend(axs[0].get_legend_handles_labels()[0], axs[0].get_legend_handles_labels()[1], 
                loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=len(tests), fontsize=11)
    sns.despine()
    plt.tight_layout()
    plt.show()
        # fig.savefig(f'{path}/{model_name}.png', bbox_inches='tight')