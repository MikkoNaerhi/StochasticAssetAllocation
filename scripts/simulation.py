from typing import Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize

import utils

def read_data(filename:str) -> Tuple[pd.Series, pd.Series]:
    """ Read raw Returns data for equities and bonds.

    Parameters:
    -----------
        filename: The name of the file containing raw data.

    Returns:
    --------
        Tuple[pd.Series, pd.Series]
            A tuple containing series of returns for equities and bonds.
    """
    data_raw = pd.read_excel(f'data/{filename}', skiprows=1)
    data_raw = data_raw.set_index('date')
    return (data_raw['Equity.1'], data_raw['Bond.1'])

def quarterly_drift_and_vol(returns: pd.DataFrame) -> Tuple[float, float]:
    """ Calculate the quarterly drift and volatility based on daily returns.
    
    Parameters:
    -----------
        returns: Series of daily returns.

    Returns:
    --------
        Tuple[float, float]:
            Quarterly drift and volatility.
    """
    # Assuming 252 trading days in a year
    quarterly_drift_equity = (np.mean(returns) * 252) / 4
    quarterly_vol_equity = (np.std(returns) * np.sqrt(252)) / np.sqrt(4)

    return (quarterly_drift_equity, quarterly_vol_equity)


# --------------------------
# ---------- BOND ----------
# --------------------------

def cir_function(
    r: float,
    a: float,
    b: float 
) -> float:
    """ Definition of the CIR-model function.
    
    Parameters:
    -----------
        r: The interest rate at time t.
        a: The speed of reversion to the mean.
        b: The long-term mean interest rate.

    Returns:
    --------
        float: The result of the CIR model function.
    """
    return a * (b - r)

def log_likelihood_cir(
    params: [float, float],
    r: pd.Series
) -> float:
    """ Log-likelihood function for CIR-model.
    
    Parameters:
    -----------
        params: Parameters a and b for the CIR model.
        r: Interest rates.

    Returns:
    --------
        float: Negative log-likelihood.
    """
    a, b = params
    n = len(r)
    dt = 1.0
    sum1 = sum((cir_function(r[i-1], a, b) - r[i]) ** 2 for i in range(1, n))
    sum2 = sum(cir_function(r[i-1], a, b) ** 2 for i in range(1, n))
    likelihood = (-(n-1)/2)*np.log(sum1/n) - (n-1)/2*np.log(2*np.pi*dt) - (1/2*dt)*sum2/n
    return -likelihood

def cir_MLE(
    returns:pd.Series,
    a_init:float,
    b_init:float,
) -> Tuple[float,float]:
    """ Function to execute MLE for CIR-model parameters.

    Parameters:
    -----------
        returns: Series of returns data.
        a_init: Initial parameter estimate for a.
        b_init: Initial parameter estimate for b.

    Returns:
    --------
        Tuple[float, float]
            A tuple containing new estimates for a and b
    """
    result = optimize.minimize(log_likelihood_cir, [a_init, b_init], args=returns)

    a=result.x[0]
    b=result.x[1]
    
    return (a,b)

def CIR(
    simulated_prices:np.ndarray,
    volatility:float,
    random_var:np.ndarray,
    num_of_simulations:int,
    time_horizon:int,
    a:float,
    b:float
) -> np.ndarray:
    """ Function to simulate prices for bonds using CIR model.

    Parameters:
        simulated_prices: Empty NumPy array used for storing simulated prices.
        volatility: Volatility for bonds.
        random_var: A correlated random variable drawn from multivariate normal distribution.
        num_of_simulations: Number of simulations.
        time_horizon: The number of time periods to simulate.
        a: Estimated value for a.
        b: Estimated value for b.
    Returns:
        simulated_prices: A NumPy array filled with simulated price data.
    """
    dt=1
    for i in range(num_of_simulations):
        for j in range(1,time_horizon):
            dr = a * (b - simulated_prices[i,j-1]) * dt + volatility * np.sqrt(simulated_prices[i,j-1]) * random_var[i,j,1]
            simulated_prices[i,j] = simulated_prices[i,j-1] + dr

    return simulated_prices

# --------------------------
# --------- EQUITY ---------
# --------------------------

def brownian_motion(
    simulated_prices:np.ndarray,
    drift:float,
    volatility:float,
    random_var:np.ndarray,
    num_of_simulations:int,
    time_horizon:int
) -> np.ndarray:
    """ Function to simulate prices for equities using Brownian Motion.

    Parameters:
        simulated_prices: Empty NumPy array used for storing simulated prices.
        drift: Drift for equity.
        volatility: Volatility for equity.
        random_var: A correlated random variable drawn from multivariate normal distribution.
        num_of_simulations: Number of simulations.
        time_horizon: The number of time periods to simulate.
    Returns:
        simulated_prices: A NumPy array filled with simulated price data.
    """
    dt=1
    for i in range(num_of_simulations):
        for j in range(1,time_horizon):
            drift_temp = drift - 0.5 * volatility**2
            diffusion = volatility * random_var[i,j,0]
            simulated_prices[i,j] = simulated_prices[i,j-1] * np.exp(drift_temp*dt + diffusion*np.sqrt(dt))

    return simulated_prices


def simulate_prices(
    volatility:float,
    random_var:np.ndarray,
    init_price:int=100,
    num_of_simulations:int=100,
    time_horizon:int=8,
    method:str='brownian_motion',
    drift:float=None,
    a:float=None,
    b:float=None
) -> np.ndarray:
    """ A wrapper for simulating price data for both equities and bonds. 

    Parameters:
        volatility: Volatility for the asset.
        random_var: A correlated random variable drawn from multivariate normal distribution.
        init_price: Initial price for simulations.
        num_of_simulations: Number of simulations.
        time_horizon: The number of time periods to simulate.
        method: Whether to use Brownian motion or CIR model for simulating price data.
        drift: Drift for the asset. 
        a: Estimated value for a.
        b: Estimated value for b.
    Returns:
        simulated_prices: A NumPy array filled with simulated price data.
    """
    simulated_prices = np.zeros((num_of_simulations, time_horizon))
    simulated_prices[:, 0] = init_price

    if method=='brownian_motion': # For equity
        simulated_prices = brownian_motion(
            simulated_prices=simulated_prices,
            drift=drift,
            volatility=volatility,
            random_var=random_var,
            num_of_simulations=num_of_simulations,
            time_horizon=time_horizon
        )
    elif method=='CIR': # For bond
        simulated_prices = CIR(
            simulated_prices=simulated_prices,
            volatility=volatility,
            random_var=random_var,
            num_of_simulations=num_of_simulations,
            time_horizon=time_horizon,
            a=a,
            b=b
        )

    return simulated_prices

def convert_np_price_to_df_returns(price_data:np.ndarray) -> pd.DataFrame:
    """ Convert multidimensional price data NumPy array to a DataFrame of returns.

    Parameters:
    -----------
        price_data: Price data as numpy array.
    
    Returns:
    --------
        df_simulations_returns: Returns as DataFrame. 
    """
    scenario_column_names = [f'Scenario {num}' for num in range(price_data.shape[0])]
    
    simulations_df = pd.DataFrame(price_data.T, columns=scenario_column_names)
    df_simulations_returns = simulations_df.pct_change()
    
    return df_simulations_returns
    
def returns_simulation_main(
    simulation_params:dict,
    plot:bool=False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Main function to simulate returns for equities and bonds.

    Parameters:
    -----------
        simulation_params: Dictionary containing various parameters needed for the simulation.
        plot: Whether to plot the simulated prices or not. Defaults to False.
    Returns:
    --------
        A tuple containing DataFrames for simulated returns of equities and bonds. 
    """
    a_init = simulation_params['a_init']
    b_init = simulation_params['b_init']
    init_price = simulation_params['init_price']
    num_of_simulations = simulation_params['num_of_simulations']
    time_horizon = simulation_params['time_horizon']
    quarterly_time_horizon = time_horizon * 4

    daily_returns_equity, daily_returns_bond = read_data(filename=simulation_params['data_filename'])

    # Generate correlated random variables using multivariate normal distribution
    corr_matrix = np.corrcoef(daily_returns_equity[1:], daily_returns_bond[1:])
    
    np.random.seed(seed=42)
    random_variables_flat = np.random.multivariate_normal(mean=np.zeros(2), cov=corr_matrix, size=num_of_simulations * quarterly_time_horizon)
    random_variables = random_variables_flat.reshape(num_of_simulations, quarterly_time_horizon, 2)

    # ----------------- EQUITY -----------------
    quarterly_drift_equity, quarterly_vol_equity = quarterly_drift_and_vol(daily_returns_equity)

    equity_simulated_prices = simulate_prices(
        drift=quarterly_drift_equity,
        volatility=quarterly_vol_equity,
        random_var=random_variables,
        init_price=init_price,
        num_of_simulations=num_of_simulations,
        time_horizon=quarterly_time_horizon,
        method='brownian_motion'
    )

    # ----------------- BOND -----------------
    quarterly_returns_bond = daily_returns_bond.resample('Q').apply(lambda x: np.log(1 + x).sum())
    quarterly_vol_bond = (np.std(daily_returns_bond) * np.sqrt(252)) / np.sqrt(4)

    a,b = cir_MLE(
        returns=quarterly_returns_bond,
        a_init=a_init,
        b_init=b_init
    )

    bond_simulated_prices = simulate_prices(
        volatility=quarterly_vol_bond,
        random_var=random_variables,
        init_price=init_price,
        num_of_simulations=num_of_simulations,
        time_horizon=quarterly_time_horizon,
        method='CIR',
        a=a,
        b=b
    )

    equity_simulated_returns = convert_np_price_to_df_returns(price_data=equity_simulated_prices)
    bond_simulated_returns = convert_np_price_to_df_returns(price_data=bond_simulated_prices)

    if plot:

        equity_prices_df = pd.DataFrame(equity_simulated_prices.T)
        ax = equity_prices_df.plot(figsize=(10, 5))
        plt.ylabel("Price")
        plt.xlabel("Months $(t)$")
        plt.show()

        bond_prices_df = pd.DataFrame(bond_simulated_prices.T)
        ax = bond_prices_df.plot(figsize=(10, 5))
        plt.ylabel("Price")
        plt.xlabel("Months $(t)$")
        plt.show()

    return (equity_simulated_returns, bond_simulated_returns)