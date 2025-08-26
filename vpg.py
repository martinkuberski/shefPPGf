import numpy as np
import pandas as pd

# This file contains functions which process additional VPG features

def sys_dia(vpg, fp):
    '''
    Splits VPG pulses into systolic and diastolic phases

    :param vpg: VPG signal to be split
    :param fp: Fiducials object containing the fiducial points data
    :return: Two arrays, one containing parts of VPG pulses in systolic phase and the other in diastolic phase
    '''
    sys_vpg = []
    dia_vpg = []
    for i in range(len(fp.on)):
        sys_vpg.append(vpg[fp.on[i] : fp.dn[i]])
        if i == len(fp.on) - 1:
            dia_vpg.append(vpg[fp.dn[i]:])
        else:
            dia_vpg.append(vpg[fp.dn[i] : fp.on[i + 1]])
    return sys_vpg, dia_vpg

def features(vpg, fp):
    '''
    Extracts additional features (mean and variance in systolic/diastolic phases) from VPG signal

    :param vpg: VPG signal to be processed
    :param fp: Fiducials object containing the fiducial points data
    :return: A dataframe containing mean and variance of VPG pulses in systolic and diastolic phases
    '''
    dict = {'sys_mean':[],
            'sys_var':[],
            'dia_mean':[],
            'dia_var':[],}
    sys_vpg, dia_vpg = sys_dia(vpg, fp)
    for pulse in sys_vpg:
        dict['sys_mean'].append(np.mean(pulse))
        dict['sys_var'].append(np.var(pulse))
    for pulse in dia_vpg:
        dict['dia_mean'].append(np.mean(pulse))
        dict['dia_var'].append(np.var(pulse))
    stats = pd.DataFrame(dict)
    return stats

def stats(features):
    '''
    Calculates statistics for VPG features

    :param features: A dataframe of VPG features
    :return: A dataframe of VPG features statistics across all pulses
    '''
    index = ['mean', 'median', 'std', 'percentile_25', 'percentile_75', 'iqr', 'skew', 'kurtosis', 'mad']
    dict = {"sys_mean": [],
            "sys_var": [],
            "dia_mean": [],
            "dia_var": []}
    for key, value in features.mean().items():
        dict[key].append(value)
    for key, value in features.median().items():
        dict[key].append(value)
    for key, value in features.std().items():
        dict[key].append(value)
    for key, value in features.quantile(q=0.25).items():
        dict[key].append(value)
    for key, value in features.quantile(q=0.75).items():
        dict[key].append(value)
    # Subtract 75th (last item) from 25th percentile (2nd to last item) to get IQR
    for key in dict:
        dict[key].append(dict[key][-1] - dict[key][-2])
    for key, value in features.skew().items():
        dict[key].append(value)
    for key, value in features.kurtosis().items():
        dict[key].append(value)
    # Mean absolute deviation
    for key in dict:
        dict[key].append((features[key] - features[key].mean()).abs().mean())

    return pd.DataFrame(dict, index=index)