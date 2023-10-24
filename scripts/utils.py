import os
import pandas as pd
from pyomo.environ import value, AbstractModel
import matplotlib.pyplot as plt

# --------------- OUTPUT ---------------

def generate_output_dfs(
    instance:AbstractModel,
    dfs:dict,
    S:int,
    T:int
):
    """ Fill output DataFrames with optimization result data.
    
    Parameters:
    -----------
        dfs: Dictionary of pandas DataFrames to fill with results.
        instance: Pyomo AbstractModel instance containing optimization results.
        S: Number of scenarios.
        T: Number of time points.

    Returns:
    --------
        dfs: Dictionary of filled pandas DataFrames.
    """
    for s in range(1, S+1):
        dfs['stocks'].loc[0, s] = value(instance.X0[1, s])
        dfs['bonds'].loc[0, s] = value(instance.X0[2, s])
        dfs['liability'].loc[0, s] = value(instance.L0)
        dfs['solvency'].loc[0, s] = value(instance.sr0)
        
        for t in range(1, T+1):
            dfs['stocks'].loc[t, s] = value(instance.X[1, t, s])
            dfs['bonds'].loc[t, s] = value(instance.X[2, t, s])
            dfs['liability'].loc[t, s] = value(instance.L[t, s])
            dfs['solvency'].loc[t, s] = value(instance.sr[t, s])
            dfs['p'].loc[t, s] = value(instance.p[t, s])
            dfs['b16'].loc[t, s] = value(instance.b16[t, s])
            dfs['C'].loc[t, s] = value(instance.C[t, s])

    dfs['value'] = dfs['stocks'] + dfs['bonds']
    dfs['allocation'] = dfs['stocks'] / dfs['value']

    return dfs

def create_output_folder(
    T:int,
    S:int,
    SR:float,
    sr0:float
) -> str:
    """Create the output folder and return its path.

    Parameters:
    -----------
        T: Number of time points.
        S: Number of scenarios.
        SR: Solvency ratio threshold.
        sr0: Initial solvency ratio.
    Returns:
    --------
        dir_path: The path of the created directory.
        simulation_name: The name of the simulation, based on the provided parameters.
    """
    simulation_name = f"T{T}_S{S}_SR{SR}_sr0{sr0}".replace('.', '')
    dir_path = f'results/{simulation_name}'
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return dir_path, simulation_name

def save_results_to_excel(
    dfs: dict[str, pd.DataFrame],
    dir_path: str,
    simulation_name: str
) -> None:
    """
    Save the results stored in pandas DataFrames to Excel files.
    
    Parameters:
    -----------
        dfs: A dictionary containing the DataFrames to be saved.
        dir_path: The directory where the Excel files will be saved.
        simulation_name: The name of the simulation, used to generate Excel file names.
    """
    for df_name, df in dfs.items():
        df.to_excel(f'{dir_path}/{df_name}_res_{simulation_name}.xlsx')

# -------------- PLOTTING --------------

def plot_dual_subplots(
    df1:pd.DataFrame,
    df2:pd.DataFrame,
    titles:[str, str],
    xlabels:[str, str],
    filename:str,
    legend_threshold:int=10
):
    """ 
    Create a subplot for two different DataFrames.
    
    Parameters:
    -----------
        df1, df2: DataFrames to plot.
        titles: List of titles for the subplots.
        xlabels: List of x-axis labels for the subplots.
        filename: Filename to save the plot.
        legend_threshold: Maximum number of columns to include in the legend.
    """
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    
    for s in df1.columns:
        axs[0].plot(df1.loc[:, s], label=f'Scenario {s}')
    for s in df2.columns:
        axs[1].plot(df2.loc[:, s], label=f'Scenario {s}')

    axs[0].set_title(titles[0])
    axs[0].set_xlabel(xlabels[0])
    axs[1].set_title(titles[1])
    axs[1].set_xlabel(xlabels[1])

    if len(df1.columns) <= legend_threshold:
        axs[0].legend()
    if len(df2.columns) <= legend_threshold:
        axs[1].legend()

    plt.savefig(filename)
    plt.show()

def plot_solvency(
    df:pd.DataFrame,
    title:str,
    xlabel:str,
    filename:str,
    SR:float,
    T:int,
    legend_threshold:int=10
):
    """Plot the solvency data.
    
    Parameters:
    -----------
        df: DataFrame containing the solvency data.
        title: Title of the plot.
        xlabel: x-axis label of the plot.
        filename: Filename to save the plot.
        SR: Solvency ratio threshold.
        T: Number of time points.
        legend_threshold: Maximum number of columns to include in the legend.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    for s in df.columns:
        ax.plot(df.loc[:, s], label=f'Scenario {s}')
    
    ax.hlines(SR, 0, T, colors=['red'], linestyle='dashed', label='Solvency limit')
    ax.set_title(title)
    ax.set_xlabel(xlabel)

    if len(df.columns) <= legend_threshold:
        ax.legend()

    plt.savefig(filename)
    plt.show()