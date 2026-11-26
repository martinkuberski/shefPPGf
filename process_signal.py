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

# This file contains functions encompassing the processing pipeline of a PPG signal, extracting the features

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
                   savingfolder='results',
                   apply_filter=True,
                   use_best_window=False,
                   best_window_sec=30.0,
                   step_sec=1.0,
                   min_beats_in_window=15,
                   min_valid_sqi_fraction=0.5,
                   min_mean_sqi=0.86,
                   screen_pulse_lengths=False,
                   ):
    '''
    This function is responsible for the PPG signal processing pipeline, taking signal path and configuration options as
    parameters and saving the output to the specified folder.

    :param path: Path of the input signal
    :param fs: Sampling frequency
    :param start: Beginning of the signal in sample
    :param end: End of the signal in sample
    :param fL: Lower cutoff frequency in Hz
    :param fH: Higher cutoff frequency in Hz
    :param order: Order of the filter to be applied
    :param sm_wins: Smoothing windows in ms for each derivative (dictionary)
    :param correction: DataFrame where the key is the name of the fiducial points and the value is bool
    :param enable_gauss: Boolean value to enable Gaussian decomposition (True to enable, False to disable)
    :param gauss_live_plot: Boolean value to display a plot of Gaussian decomposition for each pulse during processing
    :param g_values: Initial values for Gaussian decomposition parameters
    :param enable_skewed: Boolean value to enable Skewed Gaussian decomposition (True to enable, False to disable)
    :param skewed_live_plot: Boolean value to display a plot of Gaussian decomposition for each pulse during processing
    :param s_values: Initial values for Skewed Gaussian decomposition parameters
    :param savingformat: File format for the output ('csv', 'mat', or 'both')
    :param savingfolder: Folder in which the output will be saved
    :param apply_filter: Boolean value to apply filtering in preprocessing (True to apply, False to skip)
    :param use_best_window: Boolean value to enable best window extraction based on SQI (True to enable, False to disable)
    :param best_window_sec: Length of the window to be extracted in seconds (used if use_best_window is True)
    :param step_sec: Step size for sliding the window in seconds (used if use_best_window is True)
    :param min_beats_in_window: Minimum number of beats required in a window for it to be considered valid (used if use_best_window is True)
    :param min_valid_sqi_fraction: Minimum fraction of valid SQI values required in a window for it to be considered valid (used if use_best_window is True)
    :param min_mean_sqi: Minimum mean SQI value required in a window for it to be considered valid (used if use_best_window is True)
    :param screen_pulse_lengths: Keep only 0.40–1.80 s pulses for pulse-level features and fits
    :return: A feature dictionary and arrays of PPG onsets, systolic peaks and first-derivative fiducials.
        The fiducial arrays are in samples relative to the analysed signal.
    '''

    # Load a PPG signal
    signal = load_data(data_path=path, fs=fs, start_sig=start, end_sig=end)

    # Pre-processing - filter the signal and obtain derivatives
    prep = PP.Preprocess(fL=fL, fH=fH, order=order)
    signal.filtering = apply_filter
    signal.fL = fL
    signal.fH = fH
    signal.order = order
    signal.sm_wins = sm_wins
    signal.ppg, signal.vpg, signal.apg, signal.jpg = prep.get_signals(s=signal)

    # Initialise correction
    corr_on = ['on', 'dn', 'dp', 'v', 'w', 'f']
    correction.loc[0, corr_on] = True
    signal.correction = correction

    # ------------------------------------
    # Find best SQI window and crop signal
    # ------------------------------------
    if use_best_window:
        best_start, best_end, best_score = find_best_window_by_sqi(
            signal=signal,
            fs=signal.fs,
            best_window_sec=best_window_sec,
            step_sec=step_sec,
            min_beats_in_window=min_beats_in_window,
            min_valid_sqi_fraction=min_valid_sqi_fraction,
            min_mean_sqi=min_mean_sqi
        )
        # Crop the signal to the best window
        signal.start_sig += best_start
        signal.end_sig = signal.start_sig + (best_end - best_start)
        # Update signal arrays to the best window
        signal.v = signal.v[best_start:best_end]
        signal.ppg = signal.ppg[best_start:best_end]
        signal.vpg = signal.vpg[best_start:best_end]
        signal.apg = signal.apg[best_start:best_end]
        signal.jpg = signal.jpg[best_start:best_end]

    # Create PPG class
    s = PPG(s=signal)

    # Get fiducial points
    fpex = FP.FpCollection(s=s)
    fiducials = fpex.get_fiducials(s=s)
    fiducials = fiducials.applymap(lambda x: np.nan if pd.isna(x) else x)
    fp = Fiducials(fp=fiducials)

    # Split into pulses
    ppgPulses, vpgPulses, apgPulses, jpgPulses = get_pulses(s, fp, fs=s.fs if screen_pulse_lengths else None)

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
        try:
            # Skewed Gaussian decomposition
            skew = get_skewed(ppgPulses, live_plot=skewed_live_plot, initials=s_values)
            skew_stats = skewed_stats(skew)
        except Exception as e:
            print(f"Error in skewed Gaussian decomposition: {e}")
            skew = None
            skew_stats = None
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
    signal_data = flatten_biomarker_stats(
        bm_stats, stats_to_extract=('mean', 'std', 'iqr', 'mad', 'percentile_25', 'percentile_75', 'skew', 'kurtosis')
    )
    signal_data['ppgSQI'] = ppgSQI
    signal_data['Best_Window_Start'] = s.start_sig
    signal_data['Best_Window_End'] = signal.end_sig
    signal_data['Best_Window_Score'] = best_score if use_best_window else np.nan

    ppg_onsets = fp.get_fp().get('on').values
    ppg_peaks = fp.get_fp().get('sp').values
    upg_onsets = fp.get_fp().get('u').values

    for prefix, stats in (("gauss", gauss_stats), ("skewed", skew_stats),
                          ("vpg", vpg_stats), ("ppg", ppg_stats)):
        if stats is not None:
            signal_data.update({f"{prefix}_{col}_{stat}": stats.loc[stat, col]
                                for col in stats.columns for stat in stats.index})

    # Save data
    fp_new = Fiducials(fp=fp.get_fp() + s.start_sig)
    custom_save.save_data(savingformat=savingformat, savingfolder=savingfolder,s=s, fp=fp_new, bm=bm, gauss=gauss, gauss_stats=gauss_stats, gauss_additional=gauss_additional, skewed=skew, skewed_stats=skew_stats, vpg=vpg_features, vpg_stats=vpg_stats, ppg_extra=ppg_features, ppg_extra_stats=ppg_stats)
    return signal_data, ppg_onsets, ppg_peaks, upg_onsets


def find_best_window_by_sqi(
        signal,
        fs,
        best_window_sec=30.0,
        step_sec=1.0,
        min_beats_in_window=15,
        min_valid_sqi_fraction=0.5,
        min_mean_sqi=0.86,  # https://pubmed.ncbi.nlm.nih.gov/25069129/
):
    """Rank windows using beat SQI calculated on the full preprocessed signal.

    Return start and end sample indices and the winning score. The score is
    the mean of the valid beat SQIs minus 0.05 times their standard deviation.
    """
    window_samples = int(round(best_window_sec * fs))
    step_samples = max(1, int(round(step_sec * fs)))
    if window_samples < 1 or window_samples > len(signal.v):
        raise ValueError("The requested SQI window must be at least one sample and fit within the signal")

    s_full = PPG(s=signal)
    fiducials = FP.FpCollection(s=s_full).get_fiducials(s=s_full)
    peaks = fiducials["sp"].dropna().to_numpy(dtype=int)
    if len(peaks) < 2:
        raise RuntimeError("Not enough systolic peaks found to compute SQI")

    beat_sqi = np.asarray(SQI.get_ppgSQI(ppg=s_full.ppg, fs=s_full.fs, annotation=peaks), dtype=float)
    beat_starts = peaks[:-1]
    best_start = None
    best_score = -np.inf
    for start in range(0, len(signal.v) - window_samples + 1, step_samples):
        end = start + window_samples
        scores = beat_sqi[(beat_starts >= start) & (beat_starts < end)]
        if len(scores) < min_beats_in_window:
            continue
        valid_scores = scores[np.isfinite(scores)]
        if not len(valid_scores) or len(valid_scores) / len(scores) < min_valid_sqi_fraction:
            continue
        mean_sqi = float(np.mean(valid_scores))
        if mean_sqi < min_mean_sqi:
            continue
        score = float(mean_sqi - 0.05 * np.std(valid_scores))
        if score > best_score:
            best_start = start
            best_score = score

    if best_start is None:
        raise RuntimeError("No SQI window meets the minimum beat count, valid fraction and mean SQI")

    return best_start, best_start + window_samples, best_score


def get_pulses(s, fp, fs=None):
    '''
    Splits the signal into arrays of individual pulses.

    :param s: PPG object containing the signal data
    :param fp: Fiducials object containing the fiducial points data
    :param fs: Sampling frequency in Hz to screen pulse lengths, or None to keep all pulses
    :return: 4 arrays of individual pulses (as arrays) - PPG, VPG, APG, and JPG
    '''
    onsets = fp.get_fp().on[1:]
    ppgPulses = np.split(s.ppg, onsets)
    vpgPulses = np.split(s.vpg, onsets)
    apgPulses = np.split(s.apg, onsets)
    jpgPulses = np.split(s.jpg, onsets)
    if fs is not None:
        # Keep pulse lengths from 0.40 to 1.80 seconds.
        min_len = int(0.40 * fs)   # 0.40s
        max_len = int(1.80 * fs)   # 1.80s
        keep = [min_len <= len(p) <= max_len for p in ppgPulses]

        ppgPulses = [p for p, k in zip(ppgPulses, keep) if k]
        vpgPulses = [p for p, k in zip(vpgPulses, keep) if k]
        apgPulses = [p for p, k in zip(apgPulses, keep) if k]
        jpgPulses = [p for p, k in zip(jpgPulses, keep) if k]
    return ppgPulses, vpgPulses, apgPulses, jpgPulses


def make_positive(pulse):
    '''
    Shifts the pulse up to remove negative values.

    :param pulse: Pulse to be made positive
    :return: Pulse with negative values removed
    '''
    if np.min(pulse) < 0:
        pulse = pulse - np.min(pulse)
    return pulse

def normalise_amplitude(pulse):
    '''
    Normalises the amplitude of the pulse so that the maximum amplitude is 1.

    :param pulse: Pulse to be normalized
    :return: Normalized pulse
    '''
    pulse = pulse / np.max(pulse)
    return pulse

def linear_correction(pulse):
    '''
    Applies linear correction on the pulse to remove skew (equalises the initial and end amplitudes of the pulse).
    :param pulse: Pulse to be corrected
    :return: Corrected pulse
    '''
    # Get gradient and intercept
    gradient = (pulse[-1] - pulse[0]) / (len(pulse) - 1)
    intercept = pulse[0]

    # Apply linear correction
    for i in range(len(pulse)):
        pulse[i] = pulse[i] - (gradient * i + intercept)
    return pulse

def get_gaussians(ppgPulses, live_plot=False, g_values=[0.9, 0.2, 0.01, 2/3, 0.4, 0.01, 0.5, 0.6, 10, 1/3, 0.8, 0.01]):
    '''
    Performs Gaussian decomposition on PPG pulses.

    :param ppgPulses: An array of PPG pulses to be decomposed
    :param live_plot: A boolean to enable (True) or disable (False) plot display for every pulse
    :param g_values: Initial values of the parameters (amplitude, mean, standard deviation for 4 Gaussian functions in an array)
    :return: A DataFrame of Gaussian parameters by pulse
    '''
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
        pulse = np.asarray(pulse, dtype=np.float64)
        gauss_array = [np.nan] * 12
        if len(pulse) >= 20 and np.isfinite(pulse).all():
            pulse = linear_correction(make_positive(pulse))
            amplitude = np.max(pulse)
            if np.isfinite(amplitude) and amplitude >= 1e-8:
                pulse = normalise_amplitude(pulse)
                try:
                    gauss_array = gaussian.find_gaussians(pulse, g_values)
                except Exception as exc:
                    print(f"Gaussian fitting failed for a pulse: {exc}. Filling with NaN.")
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
    '''
    Calculates statistics of Gaussian decomposition parameters across all pulses.

    :param gauss: A dataframe of Gaussian parameters by pulse
    :return: A dataframe of statistics of Gaussian parameters across all pulses
    '''
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
    '''
    Calculates derived features from Gaussian decomposition.

    :param gauss: A dataframe of Gaussian parameters by pulse
    :return: A dataframe of derived features from Gaussian parameters
    '''
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
    '''
    Performes skewed Gaussian Decomposition on PPG pulses.

    :param ppgPulses: An array of PPG pulses to be decomposed
    :param live_plot: A boolean to enable (True) or disable (False) plot display for every pulse
    :param initials: Initial values of the parameters (amplitude, location, scale, shape for 4 skewed Gaussian functions in an array)
    :return: A dataframe of skewed Gaussian parameters by pulse
    '''
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
        pulse = np.array(pulse, dtype=float)
        skewed_array = [np.nan] * 16
        if len(pulse) >= 20 and np.isfinite(pulse).all():
            pulse = linear_correction(make_positive(pulse))
            amplitude = np.max(pulse)
            if np.isfinite(amplitude) and amplitude >= 1e-8:
                pulse = normalise_amplitude(pulse)
                try:
                    skewed_array = skewed.fit(pulse, initials)
                except Exception as exc:
                    print(f"Skewed Gaussian fitting failed for a pulse: {exc}. Filling with NaN.")
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


def flatten_biomarker_stats(bm_stats, stats_to_extract=('mean', 'std')):
    signal_data = {}

    for _, df in bm_stats.items():
        for stat in stats_to_extract:
            if stat in df.index:
                stat_row = df.loc[stat]
                for metric_name, value in stat_row.items():
                    signal_data[f"{metric_name}_{stat}"] = value

        if 'mean' in df.index and 'std' in df.index:
            for metric_name in df.columns:
                mean_val = df.loc['mean', metric_name]
                std_val = df.loc['std', metric_name]

                if (
                    pd.notna(mean_val)
                    and pd.notna(std_val)
                    and mean_val != 0
                    and np.isfinite(mean_val)
                ):
                    signal_data[f"{metric_name}_CV"] = std_val / mean_val
                else:
                    signal_data[f"{metric_name}_CV"] = np.nan

    return signal_data


def skewed_stats(skew):
    '''
    Calculates statistics of Gaussian decomposition parameters across all pulses.

    :param skew: A dataframe of skewed Gaussian parameters by pulse
    :return: A dataframe of statistics of skewed Gaussian parameters across all pulses
    '''
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
