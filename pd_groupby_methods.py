import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import numpy as np


def SecondDerivative_pdgroupby(obj = pd.core.groupby.generic.SeriesGroupBy) -> pd.Series:
    '''
    Computes numerical second derivative, specifically  meant for data in a multi-index pandas dataframe.
    Derivative is of the form y_{i+1} - 2*y{i} + y_{i-1}, from the finite difference approach.

    Args: 
    obj: Pass in the groupby object of the pandas dataframe.
    '''
    return obj.shift(1)+obj.shift(-1) -2*obj.shift(0)

def GaussKernelSmoother_pandas(series:pd.Series, sigma=3, order = 0)->pd.Series:
    '''
    1D gaussian kernel smoother meant for smoothing probability distribution functions.
    Compatible with pandas groupby.

    Args:
    series: The data to smooth.
    sigma: Similar to the averaging window.
    order: Typically kept at 0 for gaussian kernel smoothing.  
    '''

    smoothed = gaussian_filter1d(series.dropna(), sigma, order)
    smoothed_series = pd.Series(index = series.index)
    smoothed_series[series.isna()] = np.nan
    smoothed_series[~series.isna()] = smoothed
    return smoothed_series

def savgol_pandas(series:pd.Series, window_length, polyorder)->pd.Series:
    '''
    Savitzky-Golay filter for a pandas groupby operation.  
    For polyorder=1 this filter roughly acts like a moving average.

    Args:
    Series: Noise data to smooth
    window_length: Size of the sliding window 
    polyorder:  interpolation polynomial order, usually 1 is just fine.  Higher orders can cause boundary issues.
    '''
    data = savgol_filter(series.values, window_length, polyorder)
    return pd.Series(data = data, index = series.index)

def gradient_pandas(series:pd.Series, index_level = -1)->pd.Series:
    '''
    Custom numpy gradient tool for groupby operations on multi-index pandas dataframes.  
    X data is taken to be in the index of the series, Y data is in series.values.
    Gradient is used to return a centered derivative (i.e. the return is the same length as the input.)
    
    Args:
    series: series.values should have the y data, series.index should have the X data in one of the index levels.
    index_level: the tier of the index that continas the X data
    '''

    price = series.values
    index = series.index.get_level_values(index_level)
    data = np.gradient(price)/np.gradient(index)
    return pd.Series(data=data, index = series.index)

