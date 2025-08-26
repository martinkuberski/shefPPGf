# Import PPG
from pyPPG import PPG, Fiducials, Biomarkers
from pyPPG.datahandling import load_data, save_data
import pyPPG.preproc as PP
import pyPPG.fiducials as FP
import pyPPG.biomarkers as BM
import pyPPG.ppg_sqi as SQI
# Import other
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# Import internal
import gaussian
import custom_save
import vpg
import ppg
import skewed

def process_signal(path="",
                   fs=200,
                   start=0,
                   end=-1,
                   fL=0.5,
                   fH=12,
                   order=4,
                   sm_wins={'ppg':50, 'vpg':10, 'apg':10, 'jpg':10},
                   correction=pd.DataFrame(),
                   enable_gauss=True,
                   gauss_live_plot=False,
                   g_values=[0.9, 0.2, 10, 2/3, 0.4, 10, 0.5, 0.6, 10, 1/3, 0.8, 10],
                   enable_skewed=True,
                   skewed_live_plot=False,
                   s_values=[0.08, 0.2, 1/8, 1, 0.04, 0.4, 1/8, 1, 0.04, 0.6, 1/8, 1, 0.02, 0.8, 1/8, 1],
                   savingformat='csv',
                   savingfolder='results'):

    # Load a PPG signal
    signal = load_data(data_path=path, fs=fs, start_sig=start, end_sig=end)

    # Pre-processing - filter the signal and obtain derivatives
    prep = PP.Preprocess(fL=fL, fH=fH, order=order)
    signal.filtering = True
    signal.fL = fL
    signal.fH = fH
    signal.order = order
    signal.sm_wins = sm_wins
    signal.ppg, signal.vpg, signal.apg, signal.jpg = prep.get_signals(s=signal)

    # Initialise correction
    corr_on = ['on', 'dn', 'dp', 'v', 'w', 'f']
    correction.loc[0, corr_on] = True
    signal.correction = correction

    # Create PPG class
    s = PPG(s=signal)

    # Raw data plot (debug)
    # fig, (ax1, ax2) = plt.subplots(2, 1)
    # ax1.plot(signal.v)
    # ax2.plot(s.ppg)
    # plt.show()

    # Get fiducial points
    fpex = FP.FpCollection(s=s)
    fiducials = fpex.get_fiducials(s=s)
    fiducials = fiducials.applymap(lambda x: np.nan if pd.isna(x) else x)
    fp = Fiducials(fp=fiducials)

    # Split into pulses
    ppgPulses, vpgPulses, apgPulses, jpgPulses = get_pulses(s, fp)

    if enable_gauss:
        # Gaussian decomposition
        gauss = get_gaussians(ppgPulses, live_plot=gauss_live_plot, g_values=g_values)
        gauss_stats = gaussian_stats(gauss)
        gauss_additional = additional_gauss(gauss)
    else:
        gauss = None
        gauss_stats = None
        gauss_additional = None

    if enable_skewed:
        # Skewed Gaussian decomposition
        skew = get_skewed(ppgPulses, live_plot=skewed_live_plot, initials=s_values)
        skew_stats = skewed_stats(skew)
    else:
        skew = None
        skew_stats = None

    # Get VPG stats
    vpg_features = vpg.features(s.vpg, fp)
    vpg_stats = vpg.stats(vpg_features)

    # Get additional PPG stats
    ppg_features = ppg.features(ppgPulses)
    ppg_stats = ppg.ppg_stats(ppg_features)

    # Calculate SQI
    ppgSQI = round(np.mean(SQI.get_ppgSQI(ppg=s.ppg, fs=s.fs, annotation=fp.sp)) * 100, 2)
    print(f"ppgSQI: {ppgSQI}")

    # Get biomarkers
    bmex = BM.BmCollection(s=s, fp=fp)
    bm_defs, bm_vals, bm_stats = bmex.get_biomarkers()
    bm = Biomarkers(bm_defs=bm_defs, bm_vals=bm_vals, bm_stats=bm_stats)

    # Save data
    fp_new = Fiducials(fp=fp.get_fp() + s.start_sig)
    custom_save.save_data(savingformat=savingformat, savingfolder=savingfolder,s=s, fp=fp_new, bm=bm, gauss=gauss, gauss_stats=gauss_stats, gauss_additional=gauss_additional, skewed=skew, skewed_stats=skew_stats, vpg=vpg_features, vpg_stats=vpg_stats, ppg_extra=ppg_features, ppg_extra_stats=ppg_stats)

def get_pulses(s, fp):
    onsets = fp.get_fp().on[1:]
    ppgPulses = np.split(s.ppg, onsets)
    vpgPulses = np.split(s.vpg, onsets)
    apgPulses = np.split(s.apg, onsets)
    jpgPulses = np.split(s.jpg, onsets)
    return ppgPulses, vpgPulses, apgPulses, jpgPulses

def make_positive(pulse):
    if np.min(pulse) < 0:
        pulse = pulse - np.min(pulse)
    return pulse

def normalise_amplitude(pulse):
    pulse = pulse / np.max(pulse)
    return pulse

def linear_correction(pulse):
    # Get gradient and intercept
    gradient = (pulse[-1] - pulse[0]) / (len(pulse) - 1)
    intercept = pulse[0]

    # DEBUG: PLOT
    # lin = np.zeros(len(pulse))
    # for i in range(len(lin)):
    #     lin[i] = gradient * i + intercept
    # plt.plot(range(len(pulse)), pulse, color='r')
    # plt.plot(range(len(pulse)), lin, color='b')
    # plt.show()

    # Apply linear correction
    for i in range(len(pulse)):
        pulse[i] = pulse[i] - (gradient * i + intercept)
    return pulse

def get_gaussians(ppgPulses, live_plot=False, g_values=[0.9, 0.2, 0.01, 2/3, 0.4, 0.01, 0.5, 0.6, 10, 1/3, 0.8, 0.01]):
    dict = {"a1": [],
            "m1": [],
            "sd1": [],
            "a2": [],
            "m2": [],
            "sd2": [],
            "a3": [],
            "m3": [],
            "sd3": [],
            "a4": [],
            "m4": [],
            "sd4": []}

    for pulse in ppgPulses:
        # Pre-process pulse
        pulse = make_positive(pulse)
        pulse = linear_correction(pulse)
        pulse = normalise_amplitude(pulse)

        # Add parameters to dictionary
        gauss_array = gaussian.find_gaussians(pulse, g_values)
        dict['a1'].append(gauss_array[0])
        dict['m1'].append(gauss_array[1])
        dict['sd1'].append(gauss_array[2])
        dict['a2'].append(gauss_array[3])
        dict['m2'].append(gauss_array[4])
        dict['sd2'].append(gauss_array[5])
        dict['a3'].append(gauss_array[6])
        dict['m3'].append(gauss_array[7])
        dict['sd3'].append(gauss_array[8])
        dict['a4'].append(gauss_array[9])
        dict['m4'].append(gauss_array[10])
        dict['sd4'].append(gauss_array[11])

        if live_plot:
            # Get normalised time
            t = np.arange(len(pulse)) / len(pulse)
            # Plot the pulse
            plt.plot(t, pulse, color='k', label='pulse')
            # Get Gaussians for the pulse
            gauss_plot1 = np.array([gaussian.gaussian(x, dict['a1'][-1], dict['m1'][-1], dict['sd1'][-1]) for x in t])
            gauss_plot2 = np.array([gaussian.gaussian(x, dict['a2'][-1], dict['m2'][-1], dict['sd2'][-1]) for x in t])
            gauss_plot3 = np.array([gaussian.gaussian(x, dict['a3'][-1], dict['m3'][-1], dict['sd3'][-1]) for x in t])
            gauss_plot4 = np.array([gaussian.gaussian(x, dict['a4'][-1], dict['m4'][-1], dict['sd4'][-1]) for x in t])
            gauss_plot_c = gauss_plot1 + gauss_plot2 + gauss_plot3 + gauss_plot4
            # Plot Gaussians
            plt.plot(t, gauss_plot1, color='r', label='gaussian1')
            plt.plot(t, gauss_plot2, color='g', label='gaussian2')
            plt.plot(t, gauss_plot3, color='b', label='gaussian3')
            plt.plot(t, gauss_plot4, color='y', label='gaussian4')
            plt.plot(t, gauss_plot_c, color='r', linestyle='--', label='gaussian (combined)')
            plt.legend(loc='upper right')
            plt.show()

    # Convert dictionary to dataframe
    gaussians = pd.DataFrame(dict)
    gaussians = gaussians.rename_axis("Pulse")
    return gaussians

def gaussian_stats(gauss):
    index = ['mean', 'median', 'std', 'percentile_25', 'percentile_75', 'iqr', 'skew', 'kurtosis', 'mad']
    dict = {"a1": [],
            "m1": [],
            "sd1": [],
            "a2": [],
            "m2": [],
            "sd2": [],
            "a3": [],
            "m3": [],
            "sd3": [],
            "a4": [],
            "m4": [],
            "sd4": []}
    for key, value in gauss.mean().items():
        dict[key].append(value)
    for key, value in gauss.median().items():
        dict[key].append(value)
    for key, value in gauss.std().items():
        dict[key].append(value)
    for key, value in gauss.quantile(q=0.25).items():
        dict[key].append(value)
    for key, value in gauss.quantile(q=0.75).items():
        dict[key].append(value)
    # Subtract 75th (last item) from 25th percentile (2nd to last item) to get IQR
    for key in dict:
        dict[key].append(dict[key][-1] - dict[key][-2])
    for key, value in gauss.skew().items():
        dict[key].append(value)
    for key, value in gauss.kurtosis().items():
        dict[key].append(value)
    # Mean absolute deviation
    for key in dict:
        dict[key].append((gauss[key] - gauss[key].mean()).abs().mean())

    return pd.DataFrame(dict, index=index)

def additional_gauss(gauss):
    dict = {"AI": [],
            "RI": [],
            "RTT": [],
            "AIr": [],
            "RIr": [],
            "Sys/Dia": [],
            "A4/A1": [],
            "sd4/A1": []}

    for index, row in gauss.iterrows():
        dict['AI'].append(gaussian.augmentation_index(row['a1'], row['m1'], row['sd1'], row['a2'], row['m2'], row['sd2'], row['a3']))
        dict['RI'].append(gaussian.reflection_index(row['a1'], row['m1'], row['sd1'], row['a2'], row['m2'], row['sd2'], row['a3'], row['m3'], row['sd3']))
        dict['RTT'].append(np.mean(gaussian.gaussian_array(row['a3'], row['m3'], row['sd3'])) - np.mean(gaussian.gaussian_array(row['a1'], row['m1'], row['sd1'])))
        dict['AIr'].append((row['a1'] - row['a2'])/row['a1'])
        dict['RIr'].append((row['a3'] / row['a1']))
        dict['Sys/Dia'].append(gaussian.sys_dia(row['a1'], row['m1'], row['sd1'], row['a2'], row['m2'], row['sd2'], row['a3'], row['m3'], row['sd3'], row['a4'], row['m4'], row['sd4']))
        dict['A4/A1'].append(row['a4'] / row['a1'])
        dict['sd4/A1'].append(row['sd4'] / row['a1'])

    additional = pd.DataFrame(dict)
    additional.rename_axis("Pulse")
    return additional

def get_skewed(ppgPulses, live_plot=False, initials=[0.08, 0.2, 1/8, 1, 0.04, 0.4, 1/8, 1, 0.04, 0.6, 1/8, 1, 0.02, 0.8, 1/8, 1]):
    dict = {"a1": [],
            "loc1": [],
            "scale1": [],
            "shape1": [],
            "a2": [],
            "loc2": [],
            "scale2": [],
            "shape2": [],
            "a3": [],
            "loc3": [],
            "scale3": [],
            "shape3": [],
            "a4":[],
            "loc4": [],
            "scale4": [],
            "shape4": []}

    for pulse in ppgPulses:
        # Pre-process pulse
        pulse = make_positive(pulse)
        pulse = linear_correction(pulse)
        pulse = normalise_amplitude(pulse)
        # Add parameters to dictionary
        skewed_array = skewed.fit(pulse, initials)
        dict['a1'].append(skewed_array[0])
        dict['loc1'].append(skewed_array[1])
        dict['scale1'].append(skewed_array[2])
        dict['shape1'].append(skewed_array[3])
        dict['a2'].append(skewed_array[4])
        dict['loc2'].append(skewed_array[5])
        dict['scale2'].append(skewed_array[6])
        dict['shape2'].append(skewed_array[7])
        dict['a3'].append(skewed_array[8])
        dict['loc3'].append(skewed_array[9])
        dict['scale3'].append(skewed_array[10])
        dict['shape3'].append(skewed_array[11])
        dict['a4'].append(skewed_array[12])
        dict['loc4'].append(skewed_array[13])
        dict['scale4'].append(skewed_array[14])
        dict['shape4'].append(skewed_array[15])

        if live_plot:
            # Get normalised time
            t = np.arange(len(pulse)) / len(pulse)
            # Plot the pulse
            plt.plot(t, pulse, color='k', label='pulse')
            # Get Gaussians for the pulse
            skewed_plot1 = np.array([skewed.skewed_gaussian(x, dict['a1'][-1], dict['loc1'][-1], dict['scale1'][-1], dict['shape1'][-1]) for x in t])
            skewed_plot2 = np.array([skewed.skewed_gaussian(x, dict['a2'][-1], dict['loc2'][-1], dict['scale2'][-1], dict['shape2'][-1]) for x in t])
            skewed_plot3 = np.array([skewed.skewed_gaussian(x, dict['a3'][-1], dict['loc3'][-1], dict['scale3'][-1], dict['shape3'][-1]) for x in t])
            skewed_plot4 = np.array([skewed.skewed_gaussian(x, dict['a4'][-1], dict['loc4'][-1], dict['scale4'][-1], dict['shape4'][-1]) for x in t])

            skewed_plot_c = skewed_plot1 + skewed_plot2 + skewed_plot3 + skewed_plot4
            # Plot Gaussians
            plt.plot(t, skewed_plot1, color='r', label='skewed1')
            plt.plot(t, skewed_plot2, color='g', label='skewed2')
            plt.plot(t, skewed_plot3, color='b', label='skewed3')
            plt.plot(t, skewed_plot4, color='y', label='skewed4')
            plt.plot(t, skewed_plot_c, color='r', linestyle='--', label='skewed (combined)')
            plt.legend(loc='upper right')
            plt.show()

    # Convert dictionary to dataframe
    skew = pd.DataFrame(dict)
    skew = skew.rename_axis("Pulse")
    return skew

def skewed_stats(skew):
    index = ['mean', 'median', 'std', 'percentile_25', 'percentile_75', 'iqr', 'skew', 'kurtosis', 'mad']
    dict = {"a1": [],
            "loc1": [],
            "scale1": [],
            "shape1": [],
            "a2": [],
            "loc2": [],
            "scale2": [],
            "shape2": [],
            "a3": [],
            "loc3": [],
            "scale3": [],
            "shape3": [],
            "a4": [],
            "loc4": [],
            "scale4": [],
            "shape4": []}
    for key, value in skew.mean().items():
        dict[key].append(value)
    for key, value in skew.median().items():
        dict[key].append(value)
    for key, value in skew.std().items():
        dict[key].append(value)
    for key, value in skew.quantile(q=0.25).items():
        dict[key].append(value)
    for key, value in skew.quantile(q=0.75).items():
        dict[key].append(value)
    # Subtract 75th (last item) from 25th percentile (2nd to last item) to get IQR
    for key in dict:
        dict[key].append(dict[key][-1] - dict[key][-2])
    for key, value in skew.skew().items():
        dict[key].append(value)
    for key, value in skew.kurtosis().items():
        dict[key].append(value)
    # Mean absolute deviation
    for key in dict:
        dict[key].append((skew[key] - skew[key].mean()).abs().mean())

    return pd.DataFrame(dict, index=index)