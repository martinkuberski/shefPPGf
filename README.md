# shefPPGf
### A Python toolbox for extracting PPG features

## Introduction

This is a ready-to-use Python toolbox for extracting features from PPG signal into .csv or .mat files, built upon the [pyPPG](https://pypi.org/project/pyPPG/) package.

It extends it by adding a [Cutie](https://pypi.org/project/cutie/) based command line interface and adding several new features.

## User interface
The user interface allows the user to adjust parameters of feature extraction without changing the code. It provides the following options:

* Signal path: path of the file containing the signal. The accepted files are the same as in the pyPPG package: .mat, .txt, .csv, or .edf. Leave empty for GUI file selection.
* Sampling frequency: set the sampling frequency of the signal in Hz. If a .mat or .edf file already contains the sampling frequency, that value will be used instead.
* Start of signal: offset for the start of signal.
* End of signal: offset for the end of signal (-1 for the whole signal).
* Filter options: cutoff frequencies and order of Chebyshev Type II filter.
* Smoothing windows: the window of moving average filtering, set individualy for derivatives of the signal.
* Gaussian and skewed Gaussian options: Intitial parameters are displayed in the order of amplitude, mean, and standard deviation for Gaussian, and amplitude, location, scale, and shape for skewed Gaussian (repeating for each function).
* Output filetype: .csv, .mat, or both.
* Output folder: folder path which will contain the output files.

## New features
This program contains all the features of pyPPG with the following additional ones:

1. Gaussian decomposition: Decomposition of each PPG beat into four Gaussian functions and features derived from it. Used by [Finnegan et al. (2023)](https://doi.org/10.1038/s41598-022-27170-2).
2. Skewed Gaussian decomposition: Decomposition of each PPG beat into three skewed Gaussian functions. Used by [Basso et al. (2024)](https://doi.org/10.1088/1361-6579/ad9662).
3. Additional PPG and VPG features: Kurtosis and skeweness of individual PPG beats and mean and variance of VPG signal in systolic and diastolic phases. Used by [Finnegan et al. (2023)](https://doi.org/10.1038/s41598-022-27170-2).

## Usage
To use the program, download the source code, [create a Python virtual environment and install packages from requirements.txt with pip](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/). Then, with venv activated, run \_\_main\_\_.py with the Python interpreter. 

## Python processing options

Call `process_signal()` with `use_best_window=True` to select a subwindow by beat SQI before extracting features. The search raises an error when no window meets the configured length and quality criteria. `screen_pulse_lengths=True` limits pulse-level features and fits to pulses from 0.40 to 1.80 seconds. The interactive launcher keeps the full requested range and its original pulse selection.

Use `apply_filter=False` for input that is already filtered. The function saves its usual CSV and/or MAT outputs and returns a feature dictionary and three fiducial arrays (PPG onsets, systolic peaks and first-derivative fiducials).

## Additional notes
Original toolbox by Martin Kuberski. Edited by Kacper Sikorski for the PhysioCGM adaptations.

This program was made for the 2025 Insignio/BRC summer placement at the University of Sheffield.

Special thanks to the authors of the following articles, upon which this work was based:
* Goda, M. A., Charlton, P. H., & Behar, J. A. (2023). **pyPPG: A Python toolbox for comprehensive photoplethysmography signal analysis.** DOI 10.1088/1361-6579/ad33a2, [(THE ACCEPTED MANUSCRIPT IS AVAILABLE HERE)](https://iopscience.iop.org/article/10.1088/1361-6579/ad33a2)
* Finnegan, E., Davidson, S., Harford, M. et al. (2023). **Features from the photoplethysmogram and the electrocardiogram for estimating changes in blood pressure.** *Sci Rep* **13**, 986. DOI 10.1038/s41598-022-27170-2
* Basso, G. et al. (2024). **A skewed-Gaussian model for pulse decomposition analysis of photoplethysmography signals** *Physiol. Meas.* **45** 115006 DOI 10.1088/1361-6579/ad9662

