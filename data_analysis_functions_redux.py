import csv
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import warnings
import copy
import helperfunctions as hfx
import numba

from pandas.plotting import register_matplotlib_converters
from numba import jit, double
from numba.typed import List
register_matplotlib_converters()
plt.style.use('dark_background')

defaults = {
    'alphas': [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, .55, .6, .65, .7, .75],
    'betas': [0.25, .3, .35, .4, .45, .5, .55, .6, .65, .7, .75],
    'date_time_index': 'Ship Date',
    'gammas': [.5, .55, .6, .65, .7, .75, .8, .85],
    'input_path': 'condensed_product_sales.csv',
    'indices': ['Branch', 'Product No.', 'Ship Date'],
    'lambda_sigmas': [0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
}

def start_database(input_path, indices=[], date_time_index = ''):
    #Type verification
    if not isinstance(input_path, str):
        raise TypeError("Parameter 'input_path' should be a string")
    elif not isinstance(indices, list):
        raise TypeError("Parameter 'indices' should be a list")
    elif not isinstance(date_time_index, str):
        raise TypeError("Parameter 'date_time_index' should be a string")

    #Read from CSV file
    df = pd.read_csv(input_path)
    #Convert date_time_index if it is given to date_time
    if date_time_index is not '':
        df[date_time_index] = pd.to_datetime(df[date_time_index])
    
    #If indices were listed, set the appropriate columns.
    if len(indices) > 0:
        df = df.set_index(indices)
    
    return df

def split_dataframe(df, percent_train = 0.8, seasonality_split = None):
    
    if len(df) == 0:
        raise ValueError('Invalid DataFrame')

    if percent_train > 1 and percent_train <= 100:
        percent_train /= 100
    elif percent_train > 100 or percent_train <= 0:
        raise ValueError('Invalid Percentage')

    inv_percent = 1 / percent_train
    dataframe_length = len(df)
    train_cut = int(dataframe_length // inv_percent)

    if seasonality_split is not None:
        if not hfx.is_number(seasonality_split):
            raise ValueError("Seasonality Split must be either a number or 'False'")
        else:
            train_cut = seasonality_split * round(train_cut / seasonality_split)
    
    if train_cut == 0:
        train_cut = 1
    elif train_cut >= len(df) - 1:
        train_cut = len(df) - 2
    
    return df[:train_cut + 1], df[train_cut + 1:]
    
def exponential_smoothing(df_input, alpha, beta, startup_period = 0.15, colname='', m=1, length_forecasted = 3, initial_level = None, initial_trend = None, phi = None, verbose=False):

    if not isinstance(df_input.index, pd.DatetimeIndex):
        raise ValueError('Input DataFrame is not time-indexed')
    
    #Check if there's one or zero rows
    if len(df_input) <= 1:
        raise ValueError('Too few rows in input DataFrame')
    
    if isinstance(df_input, pd.DataFrame):
        if df_input.shape[1] >= 1:
            #If the input dataframe has multiple columns and hasn't specified a column name, give a warning and grab the column indexed at 0
            if colname is '':
                warnings.warn("Input DataFrame has more than one column, analysis will be run on column indexed at 0. Please enter a colname argument or input a dataframe with just one column if this is not the desired behavior.")
                df = df_input.iloc[:,0]
            #If the column name is given, grab that column
            else:
                df = df_input[colname]
    
        #If for some reason there's zero columns, raise a ValueError
        else:
            raise ValueError('Invalid DataFrame')
    elif isinstance(df_input, pd.Series):
        df = df_input
        
    #Give a warning if there's only 3 rows or less
    if len(df) <= 3:
        warnings.warn("DataFrame is too small to make a meaningful prediction")


    if initial_level is not None:
        prev_level = initial_level
    else:
        train, test = split_dataframe(df)
        prev_level = np.mean(train)

    if beta > 0:
        if initial_trend is None:
            k = int(len(df) // 2)
            initial_trend_array = np.empty((k,1))
            for i in range(k):
                initial_trend_array[i] = (df.iat[i + k] - df.iat[i])/k
            initial_trend_array = np.squeeze(initial_trend_array)
            
            prev_trend = np.mean(initial_trend_array)
        else:
            prev_trend = initial_trend
    else:
        prev_trend = 0

    if phi is None:
        phi = 1.0
    else:
        phi_hat = phi if phi != 0 else 1
        prev_trend = initial_trend / phi_hat

    levels = [initial_level]
    trends = [initial_trend]
    forecasts = [prev_level + prev_trend]
    
    for i in range(len(df)):
        level = alpha * df.iloc[i] + (1 - alpha) * (prev_level + phi * prev_trend)
        trend = beta * (level - prev_level) + (1 - beta) * phi * prev_trend
        forecast = level + phi * trend

        prev_trend, prev_level = trend, level
        levels.append(level)
        trends.append(trend)
        forecasts.append(forecast)

    if verbose:
        for i in range(length_forecasted - 1):
            forecasts.append(forecasts[-1] + phi * trend)
        forecasts.insert(0, np.nan)
        actuals = df
        actuals = actuals.resample('M').sum()
        actuals.loc[actuals.index.max()+pd.Timedelta(days=(29 * length_forecasted))] = np.nan
        actuals.loc[actuals.index.min() - pd.Timedelta(days=32)] = np.nan
        actuals = actuals.resample('M').sum()
        for i in range((len(actuals) - len(df)) + 1):
            actuals.iloc[-i] = np.nan
        return_df = pd.DataFrame([actuals.to_list(), forecasts, levels, trends]).T
        return_df.columns = ['Actuals', 'Forecasts', 'Levels', 'Trends']
        return_df['Month'] = actuals.index
        return_df = return_df.set_index('Month')
        return_df['Squared Errors'] = (return_df['Forecasts'] - return_df['Actuals']) ** 2
        return return_df.round(2)
    else:
        forecasts = [forecast]
        for i in range(length_forecasted - 1):
            forecasts.append(forecasts[-1] + phi * trend)
        
        return forecasts

def triple_exponential_smoothing(df_input, alpha, beta, gamma, initial_level, initial_trend, s_list):
    df = df_input

    prev_level, prev_trend = initial_level, initial_trend
    s = len(s_list)

    levels = []
    trends = []
    forecasts = [prev_level + prev_trend + s_list[0]]

    for i in range(len(df)):
        level = alpha * (df.iat[i] - s_list[-s]) + (1 - alpha) * (prev_level + prev_trend)
        trend = beta * (level - prev_level) + (1 - beta) * prev_trend
        s_list.append(gamma * (df.iat[i] - prev_level - prev_trend) + (1 - gamma) * s_list[-s])
        forecast = level + trend + s_list[-s]

        prev_trend, prev_level = trend, level
        levels.append(level)
        trends.append(trend)
        forecasts.append(forecast)
    
    return forecasts, trends, levels, s_list

@jit(double(double[:], double, double), nopython=True)
def _fast_ets_aia(li, alpha, initial_level):
    """Fast, precompiled version of exponential smoothing. Requires alpha and initial_level to be floats
    
    Arguments:
        li {float[]} -- List of floats describing the actual data
        alpha {float} -- Smoothing parameter alpha
        initial_level {float} -- Initial Level
    
    Returns:
        float -- Sum of Squared Errors of the initial parameters
    """
    forecast = initial_level
    sse = 0

    for i in li:
        sse -= (i - forecast) ** 2
        forecast = alpha * i + (1 - alpha) * forecast
    
    return sse

@jit(double(double[:], double, double, double, double), nopython=True)
def _fast_ets_abiaib(li, alpha, beta, initial_level, initial_trend):
    """Fast, precompiled version of exponential smoothing. Requires alpha, beta, initial level, and initial trend
    
    Arguments:
        li {float[]} -- List of floats describing the actual data
        alpha {float} -- Smoothing parameter alpha
        beta {float} -- Smoothing parameter beta
        initial_level {float} -- initial level
        initial_trend {float} -- initial trend
    
    Returns:
        float -- Sum of Squared Errors of the initial parameters
    """
    
    prev_level = initial_level
    prev_trend = initial_trend

    sse = 0

    for i in li:
        level = alpha * i + (1 - alpha) * (prev_level + prev_trend)
        trend = beta * (level - prev_level) + (1 - beta) * prev_trend

        sse -= (i - prev_level - prev_trend) ** 2
        prev_level, prev_trend = level, trend
    
    return sse

@jit(double(double[:], double, double, double, double), nopython=True)
def _fast_ets_abiaib_m(li, alpha, beta, initial_level, initial_trend):
    """Fast precompiled version of exponential smoothing with multiplicative trend. Requires alpha, beta, initial level, and initial trend to be floats
    
    Arguments:
        li {float[]} -- List of floats describing the actual data
        alpha {float} -- Smoothing parameter alpha
        beta {float} -- Smoothing parameter beta
        initial_level {float} -- initial level
        initial_trend {float} -- initial trend
    
    Returns:
        float -- Sum of Squared Errors of the initial parameters
    """

    prev_level = initial_level
    prev_trend = initial_trend

    sse = 0

    for i in li:
        level = alpha * i + (1 - alpha) * (prev_level * prev_trend)
        trend = beta * (level / prev_level) + (1 - beta) * prev_trend

        sse -= (i / (prev_level * prev_trend)) ** 2
        prev_level, prev_trend = level, trend

    return sse

@jit(double(double[:], double, double, double, double, double), nopython=True)
def _fast_ets_abiaibd(li, alpha, beta, initial_level, initial_trend, phi):
    """Fast precompiled version of damped exponential smoothing. Requires alpha, beta, initial_level, initial_trend, and phi to be floats
    
    Arguments:
        li {float[]} -- List of floats describing the actual data
        alpha {float} -- Smoothing parameter alpha
        beta {float} -- Smoothing parameter beta
        initial_level {float} -- Initial Level
        initial_trend {float} -- Initial Trend
        phi {float} -- Dampening parameter phi
    
    Returns:
        float -- Sum of Squared Errors of the initial parameters
    """

    prev_level = initial_level
    prev_trend = initial_trend / phi

    sse = 0

    for i in li:
        level = alpha * i + (1 - alpha) * (prev_level + phi * prev_trend)
        trend = beta * (level - prev_level) + (1 - beta) * phi * prev_trend

        sse -= (i - prev_level - phi * prev_trend) ** 2
        prev_level, prev_trend = level, trend
    
    return sse

@jit(double(double[:], double, double, double, double, double, double[:]), nopython=True)
def _fast_ets_abiaibgs(li, alpha, beta, initial_level, initial_trend, gamma, s_list):
    """Fast precompiled version of seasonal exponential smoothing
    
    Arguments:
        li {float[]} -- List of floats describing the actual data
        alpha {float} -- Smoothing parameter alpha
        beta {float} -- Smoothing parameter beta
        initial_level {float} -- Initial Level
        initial_trend {float} -- Initial Trend
        gamma {float} -- Smoothing parameter gamma
        s_list {float[]} -- List of floats describing the seasonal data
    
    Returns:
        float -- Sum of Squared Errors of the initial parameters
    """
    prev_level = initial_level
    prev_trend = initial_trend
    counter = 0
    s = len(s_list)

    sse = 0

    for i in li:
        level = alpha * (i - s_list[counter]) + (1 - alpha) * (prev_level + prev_trend)
        trend = beta * (level - prev_level) + (1 - beta) * prev_trend
        sse -= (i - prev_level - prev_trend - s_list[counter]) ** 2

        s_list[counter] = gamma * (i - prev_level - prev_trend) + (1 - gamma) * s_list[counter]
        prev_level, prev_trend = level, trend
        counter = (counter + 1) % s
    
    return sse
