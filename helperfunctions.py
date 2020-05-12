import csv
import pandas as pd
import math
import numpy as np
import warnings
import copy

def is_number(n):
    try:
        float(n)
        return True
    except ValueError:
        return False


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
        if not is_number(seasonality_split):
            raise ValueError("Seasonality Split must be either a number or 'False'")
        else:
            train_cut = seasonality_split * round(train_cut / seasonality_split)
    
    if train_cut == 0:
        train_cut = 1
    elif train_cut >= dataframe_length - 1:
        train_cut = dataframe_length - 2
    
    return df[:train_cut + 1], df[train_cut + 1:]


def repeated_median(df, colname = ''):
    #Implementation of repeated median for a dataframe or series

    if not (isinstance(df, pd.DataFrame) or isinstance(df, pd.Series)):
        raise ValueError("Input not valid")
    
    if isinstance(df, pd.DataFrame):
        if colname is '':
            warnings.warn("Input DataFrame has more than one column, analysis will be run on column indexed at 0. Please enter a colname argument or input a dataframe with just one column if this is not the desired behavior.")
            df = df.iloc[:,0]
        else:
            df = df[colname]
    
    #initialize a numpy array for the outer median
    ilist = np.empty([len(df),1])

    #Establish an epsilon tolerance. Values too small will just be equated to zero
    epsilon_tolerance = 1e-16
    
    for i in range(len(df)):
        
        #initialize a numpy array for inner median
        jlist = np.empty([len(df) - 1,1])

        #Compensate for the fact that the median excludes the i=j case
        passed_j_equals_i = False
        for j in range(len(df)):
           
            #Avoid dividing by zero
            if i == j:
                passed_j_equals_i = True
                continue

            
            numerator = df.iat[i] - df.iat[j]
            denominator = i - j

            #Assign the numpy array value based off of whether the i=j case has been passed
            if not passed_j_equals_i:
                jlist[j] = (numerator / denominator)
            else:
                jlist[j - 1] = (numerator / denominator)
        
        #Make all items smaller than the epsilon tolerance equal to zero
        jlist[abs(jlist) < epsilon_tolerance] = 0
        
        np.nan_to_num(jlist)

        #calculate the inner median
        ilist[i] = np.median(jlist)
    
    np.nan_to_num(ilist)
    #Make all items smaller than the epsilon tolerance equal to zero
    ilist[abs(ilist) < epsilon_tolerance] = 0

    #calculate the outer median and return it
    return np.median(ilist)

def huber_psi(x, k = 2):
    if abs(x) < k:
        return x
    else:
        return (x / abs(x)) * k

def MAD(df, nonzero = False):
    #An alternative formulation of Standard Deviation that is more resistant to outliers
    
    #If the dataframe has more than one column, grab the column at position 0
    if isinstance(df, pd.DataFrame):
        df = df.iloc[:,0]
    
    df = np.trim_zeros(np.asarray(df))
    rs = np.median(df)
    z =  np.median(np.abs(df - rs))
    if z == 0 and not nonzero:
        return np.mean(np.abs(df - rs))
    else:
        return z

def rho(x, k = 2, ck = 2.52):
    if abs(x) <= k:
        return ck * (1 - (1 - (x / k) ** 2) ** 3)
    else:
        return ck

def rho_accelerated(x, acceleration, power_input, k = 2, ck = 2.52):
    if power_input == 0:
        power = 0
    else:
        power = power_input - 1
    if abs(x) <= k:
        return [ck * (1 - (1 - (x / k) ** 2) ** 3), 0]
    else:
        return [ck * acceleration ** power, power_input + 1]

def MSFE(test_df, pred_df):
    #Finds the Mean Squared Forecast Error

    if len(test_df) != len(pred_df):
        raise ValueError("Lengths of both inputs must be the same!")

    test_df = np.asarray(test_df)
    pred_df = np.asarray(pred_df)

    return np.sum((test_df - pred_df) ** 2)

def robust_MSFE(test_df, pred_df, acceleration):
    #Finds robust Mean Squared Forecast Error

    if len(test_df) != len(pred_df):
        raise ValueError("Lengths of both inputs must be the same!")

    test_df = np.asarray(test_df)
    pred_df = np.asarray(pred_df)

    #calculate error
    r_t = test_df - pred_df
    #calculate s_t. 1.48 is currently a magic number to me
    s_t = 1.48 * np.median(np.abs(r_t))

    inner_rho = r_t / s_t
    k = 0
    power = 0
    for i in inner_rho:
        rho_value, power = rho(i, acceleration, power)
        k += rho_value

    return s_t ** 2 / len(test_df) * k

def clean_dataset(predicted, actual, sigma):
    return huber_psi((actual - predicted) / sigma) * sigma + predicted

def update_sigma(lambda_sigma, sigma, residual, acceleration, power):
    rho_value, power = rho(residual / sigma, acceleration, power)
    next_sigma_squared = lambda_sigma * rho_value * sigma ** 2 + (1 - lambda_sigma) * sigma ** 2
    return [math.sqrt(next_sigma_squared), power]

