import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import subprocess
import os
import glob  
from PIL import Image, ImageChops

modelmapper = {'GaussianSparseMeanShift': 'Model 1',
               'LaplaceSparseMeanShift': 'Model 2',
               'GaussianMixture': 'Model 3',
               'GaussianSpikedCovariance': 'Model 4',
               'EquiCorrelationGaussian': 'Model 5',
               'DecreasingCorrelationGaussian': 'Model 6',
               'UniformThinHypercube': 'Model 7',
               'ConcentricSpheres': 'Model 8'}


def wilson_score_interval(successes, n, confidence=0.95):
    """
    Compute the Wilson score confidence interval for a Bernoulli proportion.

    Parameters:
    - successes: number of observed successes (1s)
    - n: total number of trials
    - confidence: confidence level (default 0.95)

    Returns:
    - (lower_bound, upper_bound): tuple of confidence interval bounds
    """
    if n == 0:
        raise ValueError("Number of trials n must be > 0")

    z = norm.ppf(1 - (1 - confidence) / 2)  # z-score for desired confidence level
    phat = successes / n
    denominator = 1 + z**2 / n
    center = phat + z**2 / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n)
    lower = (center - margin) / denominator
    upper = (center + margin) / denominator
    return lower.round(3), upper.round(3)

def wilson_score_precision(successes, n, confidence=0.95):
    z = norm.ppf(1 - (1 - confidence) / 2)  # z-score for desired confidence level
    phat = successes / n
    denominator = 1 + z**2 / n
    center = phat + z**2 / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n)
    return ((center - margin) / denominator).round(3)

def plot(file, figax=None):
    """
    Plot the results of the simulations from the specified folder.
    Args:
    folder (str): Name of the folder containing the simulation results.
    """
    if figax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig, ax = figax
    
    model_name = file.split('.csv')[0]
    df = pd.read_csv(f'../out/sims/{model_name}.csv')
    df = df[df['test'].isin(['KLR', 'KLR0', 'SpecReg-MMD', 'AggMMD', 'FR','HT', 'KNN' ])]
    # df = df[df['test'].isin(['KLR0', 'KLR','*KLR*', 'SpecReg-MMD', 'AggMMD', 'FR','HT', 'KNN' ])]
    iterable_label = df.columns[0]  
    
    sns.set_style("whitegrid")
    sns.set_palette("bright")
    sns.set_theme(font="DejaVu Sans")

    title = model_name
    if title[-2] == ',':
        title = title[:-2] + ')'
    title = title.split(' with')[0]
    if figax != None: 
        ax.set_title(title, fontweight="bold")

    tests = df['test'].unique()
    sns.lineplot(data=df, x=iterable_label, y="rejection rate", hue="test", style="test", marker='o', dashes=True, ax=ax, alpha = .75, lw=1.5)
    for test in ['KLR', 'KLR0']:
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
    # fig.legend(ax.get_legend_handles_labels()[0], ax.get_legend_handles_labels()[1], 
    #             loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=len(tests), fontsize=11)
    ax.legend()
    sns.despine()
    plt.tight_layout()
    if figax is None:
        path = '../out/plots'
        os.makedirs(path, exist_ok=True)
        fig.savefig(f'{path}/{model_name}.png', bbox_inches='tight')

        # fig.savefig(f'{path}/{model_name}.png', bbox_inches='tight')


def get_table(file):
    df = pd.read_csv(f'../out/sims/{file}.csv')
    folder = '../out/tables/'
    tex_filename = os.path.join(folder, f'{file}.tex')
    pdf_filename = os.path.join(folder, f'{file}.pdf')
    png_filename = os.path.join(folder, f'{file}.png')
    # Write LaTeX source
    latex_lines = []
    latex_lines.append(r'\documentclass{article}')
    latex_lines.append(r'\usepackage[margin=1in]{geometry}')
    latex_lines.append(r'\usepackage{amsmath}')
    latex_lines.append(r'\setlength{\arrayrulewidth}{0.5mm}')
    latex_lines.append(r'\setlength{\tabcolsep}{7pt}')
    latex_lines.append(r'\renewcommand{\arraystretch}{1.5}')
    latex_lines.append(r'\begin{document}')
    latex_lines.append(r'\noindent\begin{tabular}{ |p{1cm}|p{2.1cm}|p{2.1cm}|p{2.1cm}|p{2.1cm}|p{2.1cm}|p{2.1cm}| }')
    latex_lines.append(r'\hline')
    iterable_name = df.columns[0]
    latex_lines.append(f'{iterable_name[:3]} & KLR & KLR-0 & SpecRegMMD & AggMMD & HT & FR \\\\')
    latex_lines.append(r'\hline')
    iterables = df[iterable_name].unique()
    tests = ['KLR', 'KLR0', 'SpecReg-MMD', 'AggMMD', 'HT', 'FR']
    for ix in iterables:
        ixdf = df.loc[df[iterable_name] == ix].set_index('test')['rejection rate']
        rates = [ixdf.loc[test] for test in tests]
        max_rate = max(rates)
        values = []
        for test, rate in zip(tests, rates):
            err = (np.sqrt(rate * (1 - rate) / 200) * 1.65)
            value = f'{rate:.2f} $\\pm$ {err:.2f}'
            if np.isclose(rate, max_rate):
                value = r'\textbf{' + value + '}'
            values.append(value)
        latex_lines.append(f'{ix} & ' + ' & '.join(values) + r' \\')
    latex_lines.append(r'\hline')
    latex_lines.append(r'\end{tabular}')
    latex_lines.append(r'\end{document}')
    with open(tex_filename, 'w') as f:
        f.write('\n'.join(latex_lines))
    subprocess.run(['pdflatex', '-output-directory', folder, tex_filename], check=True)
    subprocess.run(['convert', '-density', '300', pdf_filename, '-quality', '100', png_filename], check=True)
    for f in glob.glob(os.path.join(folder, '*')):
        if not f.endswith('.png'):
            os.remove(f)
    im = Image.open(png_filename)
    if iterable_name == 'dimension':
        crop_box = (280, 280, 2320, 800)  # crop from top-left corner down to max_height
    else:
        crop_box = (280, 280, 2320, 880)  # crop from top-left corner down to max_height
    im_cropped = im.crop(crop_box)
    im_cropped.save(png_filename)
    return None

def get_tableH0(df_out, outname, iterable):
    folder = '../out/tables/'
    os.makedirs(folder, exist_ok=True)
    tex_filename = os.path.join(folder, f'{outname}.tex')
    pdf_filename = os.path.join(folder, f'{outname}.pdf')
    png_filename = os.path.join(folder, f'{outname}.png')
    # Start LaTeX code
    latex_lines = []
    latex_lines.append(r'\documentclass{article}')
    latex_lines.append(r'\usepackage[margin=1in]{geometry}')
    latex_lines.append(r'\usepackage{amsmath}')
    latex_lines.append(r'\setlength{\arrayrulewidth}{0.5mm}')
    latex_lines.append(r'\setlength{\tabcolsep}{7pt}')
    latex_lines.append(r'\renewcommand{\arraystretch}{1.5}')
    latex_lines.append(r'\begin{document}')
    n_cols = df_out.shape[1]
    col_format = '|p{2.1cm}' + '|p{2.1cm}' * n_cols + '|'
    latex_lines.append(r'\noindent\begin{tabular}{ ' + col_format + ' }')
    latex_lines.append(r'\hline')
    col_header = f'{iterable} & ' + ' & '.join(str(x) for x in df_out.columns) + r' \\'
    latex_lines.append(col_header)
    latex_lines.append(r'\hline')
    N = df_out.shape[0]
    for model, row in df_out.iterrows():
        values = []
        max_val = row.max()
        for val in row:
            err = (np.sqrt(val * (1 - val) / 200) * 1.65)
            entry = f'{val:.2f} $\\pm$ {err:.2f}'
            # if np.isclose(val, max_val):
            #     entry = r'\textbf{' + entry + '}'
            values.append(entry)
        latex_lines.append(f'{model} & ' + ' & '.join(values) + r' \\')
    latex_lines.append(r'\hline')
    latex_lines.append(r'\end{tabular}')
    latex_lines.append(r'\end{document}')
    with open(tex_filename, 'w') as f:
        f.write('\n'.join(latex_lines))
    subprocess.run(['pdflatex', '-output-directory', folder, tex_filename], check=True)
    subprocess.run(['convert', '-density', '300', pdf_filename, '-quality', '100', png_filename], check=True)
    for f in glob.glob(os.path.join(folder, '*')):
        if not f.endswith('.png'):
            os.remove(f)
    im = Image.open(png_filename)
    if iterable =='dimension':
        crop_box = (285, 285, 2150, 850)  
    else:
        crop_box = (285, 285, 2150, 480) 
    im_cropped = im.crop(crop_box)
    im_cropped.save(png_filename)
    return None




def get_table_real(file, null =False):
    df = pd.read_csv(f'../out/real/{file}.csv').rename(columns={'rejection_rate': 'rejection rate', 'sample_size' : 'sample size'})
    folder = '../out/tables/'
    tex_filename = os.path.join(folder, f'{file}.tex')
    pdf_filename = os.path.join(folder, f'{file}.pdf')
    png_filename = os.path.join(folder, f'{file}.png')
    latex_lines = []
    latex_lines.append(r'\documentclass{article}')
    latex_lines.append(r'\usepackage[margin=1in]{geometry}')
    latex_lines.append(r'\usepackage{amsmath}')
    latex_lines.append(r'\setlength{\arrayrulewidth}{0.5mm}')
    latex_lines.append(r'\setlength{\tabcolsep}{7pt}')
    latex_lines.append(r'\renewcommand{\arraystretch}{1.5}')
    latex_lines.append(r'\begin{document}')
    latex_lines.append(r'\noindent\begin{tabular}{ |p{2.1cm}|p{2.1cm}|p{2.1cm}|p{2.1cm}|p{2.1cm}| }')
    latex_lines.append(r'\hline')
    iterable_name = 'sample size'
    latex_lines.append(f'{iterable_name} & KLR & KLR-0 & SpecRegMMD & AggMMD \\\\')
    latex_lines.append(r'\hline')
    iterables = df[iterable_name].unique()
    for ix in iterables:
        ixdf = df.loc[df[iterable_name] == ix].set_index('test')['rejection rate']
        methods = ['KLR', 'KLR0', 'SpecReg-MMD', 'AggMMD']
        rates = [ixdf.loc[test] for test in methods]
        max_rate = max(rates)
        values = []
        for test, rate in zip(methods, rates):
            err = (np.sqrt(rate * (1-rate) / 72)*1.65).round(3)
            value = f'{rate:.2f} $\\pm$ {err:.2f}'
            if not null:
                if np.isclose(rate, max_rate):
                    value = r'\textbf{' + value + '}'
            values.append(value)
        latex_lines.append(f'{ix} & ' + ' & '.join(values) + r' \\')
    latex_lines.append(r'\hline')
    latex_lines.append(r'\end{tabular}')
    latex_lines.append(r'\end{document}')
    with open(tex_filename, 'w') as f:
        f.write('\n'.join(latex_lines))
    subprocess.run(['pdflatex', '-output-directory', folder, tex_filename], check=True)
    subprocess.run(['convert', '-density', '300', pdf_filename, '-quality', '100', png_filename], check=True)
    for f in glob.glob(os.path.join(folder, '*')):
        if not f.endswith('.png'):
            os.remove(f)
    im = Image.open(png_filename)
    crop_box = (280, 280, 1880, 640) 
    if file == 'cifar' or file == 'cifar-null':
        crop_box = (280, 280, 1880, 700)
    im_cropped = im.crop(crop_box)
    im_cropped.save(png_filename)
    return None




