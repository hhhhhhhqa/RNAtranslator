import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd

params = {'legend.fontsize': 48,
        'figure.figsize': (54, 32),
        'axes.labelsize': 60,
        'axes.titlesize':60,
        'xtick.labelsize':60,
        'ytick.labelsize':60,
        'lines.linewidth': 10}

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
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker

# Violin Plot
def plot_violin_compare(data_arrays, labels, ylabel, filename, limit=True):
    # Create the violin plot
    plt.figure()
    sns.violinplot(
        data=data_arrays,
        inner='box', 
        palette=["#65879F", "#8B8C89", "#425062", "#8F5C5C", "#CFACAC"],
        width=0.6,  
        linewidth=1.5,  
        fliersize=4,  
        whis=1.5  
    )
    # Customize the plot
    plt.xticks(range(len(data_arrays)), labels)
    plt.ylabel(ylabel)
    # if limit:
    #     plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Ridge Plot
def plot_ridge_compare(data_arrays, labels, xlabel, filename):
    # Convert input data to a DataFrame
    data_dict = {label: data for label, data in zip(labels, data_arrays)}
    data = pd.DataFrame(data_dict)

    # Melt the DataFrame for seaborn compatibility
    df_melted = data.melt(var_name='Category', value_name='Value')

    # Dynamic plot based on number of categories
    g = sns.FacetGrid(df_melted, row="Category", hue="Category", aspect=3, height=1.5, 
                      palette=["#65879F", "#8B8C89", "#425062", "#8F5C5C", "#CFACAC"],)
    g.map(sns.kdeplot, "Value", fill=True, alpha=0.6)

    # Customize plot
    g.set_axis_labels(xlabel, "Density")
    g.set_titles(size=10)
    g.set_xlabels(xlabel, fontsize=10)
    g.set_ylabels("Density", fontsize=10)
    g.set_xticklabels(fontsize=10)
    
    g.set(yticks=[])
    g.set(xlim=(0, 1))
    
    for ax in g.axes.flat:
        ax.grid(True, which='major', linestyle='--', linewidth=0.6)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# Density Plot
def plot_density_compare(data_arrays, labels, xlabel, filename, limit=True):
    plt.figure()
    
    for data, label in zip(data_arrays, labels):
        sns.kdeplot(data, fill=True, label=label, alpha=0.3)

    plt.xlabel(xlabel, labelpad=20)
    plt.legend()

    if limit:
        plt.xlim(0, 1)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Box Plot
def plot_box_compare(data_arrays, labels, ylabel, filename):
    plt.figure()
    sns.boxplot(
        data=data_arrays,
        palette=["#65879F", "#8B8C89", "#425062", "#8F5C5C", "#CFACAC"],  # Dynamic color palette based on input size
        width=0.6,  
        linewidth=1.5,  
        fliersize=4,  
        whis=1.5  
    )
    plt.xticks(range(len(data_arrays)), labels)
    plt.ylabel(ylabel)
    plt.grid(True, axis='y', linestyle='--', linewidth=0.6, alpha=0.7)
    
    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=False))
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
