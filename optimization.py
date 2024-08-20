import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as spo
from util import get_data, plot_data


# This is the function that will be tested by the autograder  		  	   		   	 		  		  		    	 		 		   		 		  
# The student must update this code to properly implement the functionality  		  	   		   	 		  		  		    	 		 		   		 		  
def optimize_portfolio(
        sd=dt.datetime(2008, 6, 1),
        ed=dt.datetime(2009, 6, 1),
        syms=["IBM", "X", "GLD", "JPM"],
        gen_plot=True,
):
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe  		  	   		   	 		  		  		    	 		 		   		 		  
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of  		  	   		   	 		  		  		    	 		 		   		 		  
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take  		  	   		   	 		  		  		    	 		 		   		 		  
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and  		  	   		   	 		  		  		    	 		 		   		 		  
    statistics.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		   	 		  		  		    	 		 		   		 		  
    :type sd: datetime  		  	   		   	 		  		  		    	 		 		   		 		  
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		   	 		  		  		    	 		 		   		 		  
    :type ed: datetime  		  	   		   	 		  		  		    	 		 		   		 		  
    :param syms: A list of symbols that make up the portfolio (note that your code should support any  		  	   		   	 		  		  		    	 		 		   		 		  
        symbol in the data directory)  		  	   		   	 		  		  		    	 		 		   		 		  
    :type syms: list  		  	   		   	 		  		  		    	 		 		   		 		  
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your  		  	   		   	 		  		  		    	 		 		   		 		  
        code with gen_plot = False.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type gen_plot: bool  		  	   		   	 		  		  		    	 		 		   		 		  
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,  		  	   		   	 		  		  		    	 		 		   		 		  
        standard deviation of daily returns, and Sharpe ratio  		  	   		   	 		  		  		    	 		 		   		 		  
    :rtype: tuple  		  	   		   	 		  		  		    	 		 		   		 		  
    """

    # Read in adjusted closing prices for given symbols, date range  		  	   		   	 		  		  		    	 		 		   		 		  
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY  		  	   		   	 		  		  		    	 		 		   		 		  
    prices = prices_all[syms]  # only portfolio symbols  		  	   		   	 		  		  		    	 		 		   		 		  
    prices_SPY = prices_all["SPY"]  # only SPY, for comparison later  		  	   		   	 		  		  		    	 		 		   		 		  

    # find the allocations for the optimal portfolio
    # note that the values here ARE NOT meant to be correct for a test case

    # ----------- Normalize prices dataframe for each column/stock  ----------- #
    for col in list(prices):
        prices[col] = prices[col] / prices[col][0]

    # Minimize get_sharpe_ratio function with scipy
    allocs = np.asarray(call_optimizer(prices, get_sharpe_ratio))

    # Get stats of optimal portfolio
    cr, adr, sddr, sr, port_val = get_port_stats(allocs, prices)

    # Get daily portfolio value
    # Compare daily portfolio value with SPY using a normalized plot  		  	   		   	 		  		  		    	 		 		   		 		  
    if gen_plot:
        # add code to plot here
        # normalize SPY values, port_val is already normalized
        prices_SPY = prices_SPY / prices_SPY[0]

        # Temp dataframe to plot
        df_temp = pd.concat(
            [port_val, prices_SPY], keys=["Portfolio", "SPY"], axis=1
        )

        # Plot stock prices with a custom title and meaningful axis labels.
        ax = df_temp.plot(title="Daily Portfolio Value and SPY", fontsize=12)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        plt.legend(df_temp.keys())
        plt.savefig("Figure1.png")

        pass

    return allocs, cr, adr, sddr, sr




def get_sharpe_ratio(allocations, port_data_df):
    """
    Parameters
    ----------
    allocations: 1d array of allocation %'s
    port_data_df: normalized numpy arrays with prices/stock in column, dates in row

    Assumptions
    -----------
    risk free rate = 0
    daily return factor (k) = 252
    portfolio starting value = 1
    """

    # Get port values and returns by applying allocations
    position_values = allocations * port_data_df * 1
    port_value = position_values.sum(axis=1)
    daily_rets = (port_value / port_value.shift(1) - 1)

    # Return Sharpe ratio
    return -1 * np.sqrt(252) * daily_rets.mean() / daily_rets.std()


def get_port_stats(allocations, port_data_df):
    """
    Parameters
    ----------
    allocations: 1d array of allocation %'s
    port_data_df: normalized pandas dataframe with prices/stock in column, dates in row

    Assumptions
    -----------
    risk free rate = 0
    daily return factor (k) = 252
    portfolio starting value = 1
    """

    # Get port values and returns by applying allocations
    position_values = allocations * port_data_df * 1
    port_value = position_values.sum(axis=1)

    # Cumulative Return
    cr = port_value[-1] / port_value[0]

    # Daily return calcs
    daily_rets = (port_value / port_value.shift(1) - 1)
    adr = daily_rets.mean()
    sddr = daily_rets.std()

    # Sharpe ratio
    sr = np.sqrt(252) * adr / sddr

    return cr, adr, sddr, sr, port_value


def call_optimizer(price_data, sharpe_function):
    # Initial Guess
    n = len(list(price_data))
    guess_allocations = np.asarray([1 / n for i in range(n)])  # List of 1/n for each

    # Test initial sharpe ratio
    # print("\n\n Initial sharpe ratio is: ", sharpe_function(guess_allocations, price_data), "\n\n")

    # Bounds, each allocation must be in [0,1]
    b = (0, 1)
    bounds = list()
    for i in range(n):
        bounds.append(b)
    bnds = tuple(bounds)

    # Constraints, all allocations must sum to 1
    con1 = {'type': 'eq', 'fun': constraint_1}
    cons = [con1]

    result = spo.minimize(sharpe_function, guess_allocations, method='SLSQP', bounds=bnds, constraints=cons,
                          args=(price_data,),
                          options={'disp': False})
    return result.x


def constraint_1(allocations):
    sum_allocations = 1
    for i in range(len(list(allocations))):
        sum_allocations -= allocations[i]
    return sum_allocations


def test_code():
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    This function WILL NOT be called by the auto grader.  		  	   		   	 		  		  		    	 		 		   		 		  
    """

    start_date = dt.datetime(2009, 1, 1)
    end_date = dt.datetime(2010, 1, 1)
    symbols = ["GOOG", "AAPL", "GLD", "XOM", "IBM"]

    # Assess the portfolio  		  	   		   	 		  		  		    	 		 		   		 		  
    allocations, cr, adr, sddr, sr = optimize_portfolio(
        sd=start_date, ed=end_date, syms=symbols, gen_plot=False
    )

    # Print statistics  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    print(f"Symbols: {symbols}")
    print(f"Allocations:{allocations}")
    print(f"Sharpe Ratio: {sr}")
    print(f"Volatility (stdev of daily returns): {sddr}")
    print(f"Average Daily Return: {adr}")
    print(f"Cumulative Return: {cr}")


if __name__ == "__main__":  		  	   		  		 		  		  		    	 		 		   		 		  
    # This code WILL NOT be called by the auto grader  		  	   		  		 		  		  		    	 		 		   		 		  
    # Do not assume that it will be called  		  	   		  		 		  		  		    	 		 		   		 		  
    test_code()  
Ungraded
(TESTING) Project 2: optimize_portfolio(gen_plot=True)
Student
Cherry Tin Latt
Total Points
- / 0 pts

Autograder Score
0.0 / 0.0
Passed Tests
1) optimize_portfolio(gen_plot) execution (1/1)
2) Check for missing files (1/1)
3) Check for files produced (1/1)
