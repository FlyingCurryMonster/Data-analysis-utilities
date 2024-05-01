import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import numpy as np


def pandas_2nd_derivative(obj = pd.core.groupby.generic.SeriesGroupBy) -> pd.Series:
    return obj.shift(1)+obj.shift(-1) -2*obj.shift(0)

def pandas_savgol(series:pd.Series, window_length, polyorder)->pd.Series:
    data = savgol_filter(series.values, window_length, polyorder)
    return pd.Series(data = data, index = series.index)

def pandas_gradient(series:pd.Series, strike_level = -1)->pd.Series:
    price = series.values
    strikes = series.index.get_level_values(strike_level)
    data = np.gradient(price)/np.gradient(strikes)
    return pd.Series(data=data, index = series.index)

def pd_GaussKernelSmoother(series:pd.Series, sigma=3, order = 0)->pd.Series:
    smoothed = gaussian_filter1d(series.dropna(), sigma, order)
    smoothed_series = pd.Series(index = series.index)
    smoothed_series[series.isna()] = np.nan
    smoothed_series[~series.isna()] = smoothed
    return smoothed_series