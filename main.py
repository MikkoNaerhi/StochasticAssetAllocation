from typing import Tuple
import pandas as pd
import json

from scripts.optimization import *
from scripts import simulation, utils

from pyomo.environ import *

def load_config(file_name:str) -> dict:
    """Load configuration from a JSON file.
    
    Parameters:
    -----------
        file_name: Name of the JSON file to load.
            
    Returns:
    --------
        config: Dictionary containing the loaded configuration.
    """
    with open(file_name, 'r') as f:
        return json.load(f)

def prepare_optimization_data(
    params: dict,
    equity_return_simulations: pd.DataFrame,
    bond_return_simulations: pd.DataFrame
) -> Tuple[int,int,float,float,dict]:
    """Prepare the data needed for optimization.
    
    Parameters:
    -----------
        opt_params: Dictionary of parameters.
        equity_return_simulations: DataFrame containing equity return simulations.
        bond_return_simulations: DataFrame containing bond return simulations.
            
    Returns:
    --------
        Tuple containing T, S, SR, sr0, and data.
    """
    N = params['optimization_params']['N']
    T = params['optimization_params']['T']
    S = params['optimization_params']['S']
    SR = params['optimization_params']['SR']
    sr0 = params['optimization_params']['sr0']
    pr = params['optimization_params']['pr']
    i0 = params['optimization_params']['i0']
    l = params['optimization_params']['l']
    L0 = params['optimization_params']['L0']
    alpha = params['optimization_params']['alpha']
    p_mean = params['optimization_params']['p_mean']
    w = params['optimization_params']['w']

    # Define max values for time horizon (T) and number of simulations (S)
    max_time_horizon = (params['simulation_params']['time_horizon']*4)-1
    max_num_of_sims = (params['simulation_params']['num_of_simulations'])-1

    assert T < max_time_horizon, f"Parameter value {T} is invalid, should be at most {max_time_horizon}"
    assert S < max_num_of_sims, f"Parameter value {S} is invalid, should be at most {max_num_of_sims}"
    assert N == 2, f"Parameter value {N} is invalid, should be equal to 2"

    data = {None: {
        'N': {None: N},
        'T': {None: T},
        'S': {None: S},
        'i0': {None: i0},
        'l': {None: l},
        'pr': {s: pr for s in range(1,S+1)},
        'R': {},
        'cR': {},
        'L0': {None: L0},
        'sr0': {None: sr0},
        'SR': {None: SR},
        'alpha': {None: alpha},
        'p_mean': {None: p_mean},
        'w': {None: w}
    }}

    for t in range(T):
        for s in range(S):
            data[None]['R'][(1,t+1,s+1)] = equity_return_simulations.iloc[t+1, s+1]
            data[None]['R'][(2,t+1,s+1)] = bond_return_simulations.iloc[t+1, s+1]
    
    for s in range(S):
        data[None]['cR'][(1,1,s+1)] = equity_return_simulations.iloc[1, s+1]
        data[None]['cR'][(2,1,s+1)] = bond_return_simulations.iloc[1, s+1]

    for t in range(1,T):
        for s in range(S):
            data[None]['cR'][(1,t+1,s+1)] = (1 + data[None]['cR'][(1,t,s+1)]) * (1 + data[None]['R'][(1,t+1,s+1)]) - 1
            data[None]['cR'][(2,t+1,s+1)] = (1 + data[None]['cR'][(2,t,s+1)]) * (1 + data[None]['R'][(2,t+1,s+1)]) - 1

    return (T, S, SR, sr0, data)
    
def main():
    """ Main function to run the program.
    """
    config = load_config('config.json')

    # ------------------------------------------------
    # ------------------ SIMULATION ------------------
    # ------------------------------------------------

    equity_return_simulations, bond_return_simulations = simulation.returns_simulation_main(
        simulation_params = config['simulation_params']
    )

    # ------------------------------------------------
    # ----------------- OPTIMIZATION -----------------
    # ------------------------------------------------

    T, S, SR, sr0, data = prepare_optimization_data(
        params=config,
        equity_return_simulations=equity_return_simulations,
        bond_return_simulations=bond_return_simulations
    )

    instance = model.create_instance(data)
    solver = SolverFactory('gurobi')
    solver.options['NonConvex'] = 2
    results = solver.solve(instance, tee=True)

    print(results)
    print(f'Objective / Portfolio value: {value(instance.OBJ)}')

    dfs = {
        'summary': pd.DataFrame(),
        'stocks': pd.DataFrame(),
        'bonds': pd.DataFrame(),
        'value': pd.DataFrame(),
        'allocation': pd.DataFrame(),
        'liability': pd.DataFrame(),
        'solvency': pd.DataFrame(),
        'p': pd.DataFrame(),
        'b16': pd.DataFrame(),
        'C': pd.DataFrame(),
    }

    dfs['summary'].loc[0, ['T', 'S', 'SR', 'sr0', 'Objective', 'a0_stocks', 'a0_bonds']] = [
        T, S, SR, sr0, value(instance.OBJ), value(instance.a0[1]), value(instance.a0[2])
    ]

    dfs = utils.generate_output_dfs(
        dfs=dfs,
        instance=instance,
        S=S,
        T=T
    )

    dir_path, simulation_name = utils.create_output_folder(
        T=T,
        S=S,
        SR=SR,
        sr0=sr0
    )

    # Excel Output
    if config['output_params']['output_excel']:
        utils.save_results_to_excel(
            dfs=dfs,
            dir_path=dir_path,
            simulation_name=simulation_name)

    # Plot
    if config['output_params']['plot']:
        utils.plot_dual_subplots(
            df1=dfs['allocation'],
            df2=1 - dfs['allocation'],
            titles=['Equity allocation', 'Bond allocation'],
            xlabels=['Quarter', 'Quarter'],
            filename=f'{dir_path}/allocations_{simulation_name}.png'
        )

        utils.plot_dual_subplots(
            df1=dfs['value'],
            df2=dfs['liability'],
            titles=['Assets', 'Liabilities'],
            xlabels=['Quarter', 'Quarter'],
            filename=f'{dir_path}/assetsliabilities_{simulation_name}.png'
        )

        utils.plot_solvency(
            df=dfs['solvency'],
            title='Solvency ratio',
            xlabel='Quarter',
            filename=f'{dir_path}/solvency_{simulation_name}.png',
            SR=SR,
            T=T
        )

if __name__ == '__main__':
    main()