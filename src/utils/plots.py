import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

params = {
    'legend.fontsize': 48,
    'figure.figsize': (54, 32),
    'axes.labelsize': 60,
    'axes.titlesize': 60,
    'xtick.labelsize': 60,
    'ytick.labelsize': 60,
    'lines.linewidth': 20,
    'scatter.marker': 'o',  
}
plt.rcParams.update(params)


def plot_histogram(data, title, xlabel, ylabel, filename):
    plt.figure()
    plt.hist(data, bins=50, color='#86bf91', edgecolor='black', alpha=0.7)
    plt.title(title, fontweight='bold')
    plt.xlabel(xlabel,)
    plt.ylabel(ylabel)
    plt.xticks()
    plt.yticks()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_scatter(data1, data2, title, xlabel, ylabel, filename):
    plt.figure()
    plt.scatter(data1, data2, s=350, alpha=0.5)
    plt.title(title,  fontweight='bold')
    plt.xlabel(xlabel, )
    plt.ylabel(ylabel, )
    plt.xticks()
    plt.yticks()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_bar(data, title, xlabel, ylabel, filename):
    plt.figure()
    plt.bar(data.keys(), data.values(), color='#86bf91')
    plt.title(title,  fontweight='bold')
    plt.xlabel(xlabel, )
    plt.ylabel(ylabel, )
    plt.xticks()
    plt.yticks()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def plot_box(data, title, ylabel, filename):
    plt.figure()
    sns.boxplot(y=data)
    plt.title(title,  fontweight='bold')
    plt.ylabel(ylabel, )
    plt.xticks()
    plt.yticks()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_violin(data, title, ylabel, filename):
    plt.figure()
    sns.violinplot(data)
    plt.title(title,  fontweight='bold')
    plt.ylabel(ylabel, )
    plt.xticks()
    plt.yticks()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_log_histogram(data, title, xlabel, ylabel, filename):
    plt.figure()
    plt.hist(data, bins=50, color='#86bf91', edgecolor='black', alpha=0.7)
    plt.yscale('log')
    plt.title(title,  fontweight='bold')
    plt.xlabel(xlabel, )
    plt.ylabel(ylabel, )
    plt.xticks()
    plt.yticks()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_cdf(data, title, xlabel, ylabel, filename):
    plt.figure()
    sns.ecdfplot(data)
    plt.title(title,  fontweight='bold')
    plt.xlabel(xlabel, )
    plt.ylabel(ylabel, )
    plt.xticks()
    plt.yticks()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_pairplot(data, filename):
    sns.pairplot(data)
    plt.savefig(filename)
    plt.close()


def plot_combined_histogram(data1, data2, title, xlabel, ylabel, filename):
    plt.figure()
    plt.hist([data1, data2], bins=50, color=['#86bf91', '#FF6F61'], edgecolor='black', alpha=0.7, label=['Length', 'Count'])
    plt.title(title)
    plt.xlabel(xlabel, )
    plt.ylabel(ylabel, )
    plt.legend()
    plt.xticks()
    plt.yticks()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_combined_log_histogram(data1, data2, title, xlabel, ylabel, filename):
    plt.figure()
    plt.hist([data1, data2], bins=50, color=['#86bf91', '#FF6F61'], edgecolor='black', alpha=0.7, label=['Length', 'Count'])
    plt.yscale('log')
    plt.title(title)
    plt.xlabel(xlabel, )
    plt.ylabel(ylabel, )
    plt.legend()
    plt.xticks()
    plt.yticks()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_token_distribution(tokens, natural_values, generated_values, filename):
    x = np.arange(len(tokens))
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - width / 2, natural_values, width, label='Natural RNA')
    ax.bar(x + width / 2, generated_values, width, label='Generated RNA')
    ax.set_xlabel('Nucleotide')
    ax.set_ylabel('Frequency')
    ax.set_title('Token Distribution in RNA Sequences')
    ax.set_xticks(x)
    ax.set_xticklabels(tokens)
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_token_distribution_single(tokens, generated_values, filename):
    x = np.arange(len(tokens))
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x + width / 2, generated_values, width, label='Generated RNA')
    ax.set_xlabel('Nucleotide')
    ax.set_ylabel('Frequency')
    ax.set_title('Token Distribution in RNA Sequences - Generated')
    ax.set_xticks(x)
    ax.set_xticklabels(tokens)
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_mfe_distribution(natural_values, generated_values, filename):
    x = np.concatenate((natural_values, generated_values))
    y_natural = ['Natural' for i in range(len(natural_values))]
    y_generated = ['Generated' for i in range(len(generated_values))]
    y = np.concatenate((y_natural,y_generated))
    data = pd.DataFrame({'x':y,'y':x})
    plot_violin(data,'MFE Distribution Comparison',ylabel='',filename=filename)

def plot_mfe_distribution_single(generated_values, filename):

    y_generated = ['Generated' for i in range(len(generated_values))]
    data = pd.DataFrame({'x':generated_values,'y':y_generated})
    plot_violin(data,'MFE Distribution - Generated Sequences',ylabel='',filename=filename)
    

def plot_multi_histogram(data, cols, title, xlabel, ylabel, filename):
    for col in cols:
        sns.histplot(data[col].dropna(), kde=True, label=col)

    plt.title(f'Distribution of Lengths Greater Than {1024}')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.legend()
    plt.savefig(filename)
    plt.close()

def plot_lost_data_bar(length_data_filtered, length_columns, data, title="Lost Data Comparison", xlabel='Length Columns', ylabel='Sum of Interactions', filename="./"):
    interactions = {}
    for col in length_columns:
        interactions[col] = data.loc[length_data_filtered[col].dropna().index, 'Count'].sum()

    # Plot the interactions
    plt.figure()
    plt.bar(interactions.keys(), interactions.values())
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.savefig(filename)
    plt.close()

