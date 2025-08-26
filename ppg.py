import pandas as pd
from scipy.stats import skew, kurtosis

# This file contains functions which process additional PPG features.

def features(ppg):
    '''
    Extracts additional features (skeweness and kurtosis) from PPG pulses.

    :param ppg: PPG signal to be processed
    :return: A dataframe of skeweness and kurtosis by pulse
    '''
    dict = {'Skewness': [],
            'Kurtosis': []}
    for pulse in ppg:
        dict['Skewness'].append(skew(pulse))
        dict['Kurtosis'].append(kurtosis(pulse))
    return pd.DataFrame(dict)

def ppg_stats(ppg_features):
    '''
    Calculates statistics for additional PPG features.

    :param ppg_features: A dataframe of PPG features
    :return: Statistics for additional PPG features across all pulses
    '''
    index = ['mean', 'median', 'std', 'percentile_25', 'percentile_75', 'iqr', 'skew', 'kurtosis', 'mad']
    dict = {"Skewness": [],
            "Kurtosis": []}
    for key, value in ppg_features.mean().items():
        dict[key].append(value)
    for key, value in ppg_features.median().items():
        dict[key].append(value)
    for key, value in ppg_features.std().items():
        dict[key].append(value)
    for key, value in ppg_features.quantile(q=0.25).items():
        dict[key].append(value)
    for key, value in ppg_features.quantile(q=0.75).items():
        dict[key].append(value)
    # Subtract 75th (last item) from 25th percentile (2nd to last item) to get IQR
    for key in dict:
        dict[key].append(dict[key][-1] - dict[key][-2])
    for key, value in ppg_features.skew().items():
        dict[key].append(value)
    for key, value in ppg_features.kurtosis().items():
        dict[key].append(value)
    # Mean absolute deviation
    for key in dict:
        dict[key].append((ppg_features[key] - ppg_features[key].mean()).abs().mean())

    return pd.DataFrame(dict, index=index)
