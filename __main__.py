# Import backend
import skewed
from process_signal import process_signal
# Import other
import cutie
import os

# This file contains the user interface of the application and servers as its starting point

# Global variables (options)
path = "" # Path of a file containing the PPG signal to be analysed
fs = 200 # Sampling frequency of the PPG signal
start = 0 # Start of the signal
end = -1 # End of the signal (-1 for the whole signal)
fL = 0.5 # Lower filter cutoff frequency in Hz
fH = 12 # Higher filter cutoff frequency in Hz
order = 4 # Filter order
sm_ppg=50 # PPG smoothing window in ms
sm_vpg=10 # VPG smoothing window in ms
sm_apg=10 # APG smoothing window in ms
sm_jpg=10 # JPG smoothing window in ms
saving_format='csv' # Output file format ('csv', 'mat', 'both')
saving_folder='results' # Output folder
gauss = True # Perform Gaussian decomposition
gauss_live_plot = False # Plot Gaussian decomposition for each pulse
g_values = [0.9, 0.2, 0.01, 2/3, 0.4, 0.01, 0.5, 0.6, 0.01, 1/3, 0.8, 0.01] # Initial Gaussian parameters
skewed = True # Perform skewed Gaussian decomposition
skewed_live_plot = False # Plot skewed Gaussian decomposition for each pulse
s_values = [0.08, 0.2, 1/8, 1, 0.04, 0.4, 1/8, 1, 0.04, 0.6, 1/8, 1, 0.02, 0.8, 1/8, 1] # Initial skewed Gaussian parameters

def main():
    '''
    Displays a command-line interface enabling the user to adjust options.
    '''
    # Global declarations
    global path
    global fs
    global start
    global end
    global fL
    global fH
    global order
    global sm_ppg
    global sm_vpg
    global sm_apg
    global sm_jpg
    global gauss
    global gauss_live_plot
    global g_values
    global skewed
    global skewed_live_plot
    global s_values
    global saving_format
    global saving_folder

    # Create a menu
    print("shefPPGf - a Python tool for PPG feature extraction.")
    print("====================================================")
    print("Please select a parameter to modify or 'PROCEED' to continue.")
    print("======")
    options = [
        f"Signal path: {path}",
        "===SIGNAL OPTIONS===",
        f"Sampling frequency: {fs} Hz",
        f"Start of signal: {start}",
        f"End of signal: {end}",
        "===FILTER OPTIONS===",
        f"Lower cutoff frequency: {fL} Hz",
        f"Upper cutoff frequency: {fH} Hz",
        f"Filter order: {order}",
        "===SMOOTHING WINDOWS===",
        f"PPG: {sm_ppg} ms",
        f"VPG: {sm_vpg} ms",
        f"APG: {sm_apg} ms",
        f"JPG: {sm_jpg} ms",
        "===GAUSSIAN===",
        f"Apply Gaussian decomposition: {gauss}",
        f"Enable live plotting: {gauss_live_plot}",
        f"Initial parameters: {g_values}",
        "===SKEWED GAUSSIAN===",
        f"Apply skewed Gaussian decomposition: {skewed}",
        f"Enable live plotting: {skewed_live_plot}",
        f"Initial parameters: {s_values}",
        "===OUTPUT OPTIONS===",
        f"Output filetype: {saving_format}",
        f"Output folder: {saving_folder}",
        "======",
        "PROCEED"
    ]
    captions = [1,5,9,14,18,22,25]
    option = cutie.select(options, captions)

    # Handle options
    match option:
        case 0:
            path = input("Please input the signal path, or leave empty to select the file with GUI later: ")
            path = path.strip()
        case 2:
            fs = cutie.get_number("Please input the sampling frequency in Hz: ", 0, None, True)
        case 3:
            start = cutie.get_number("Please input the start of the signal: ", 0, None, True)
        case 4:
            end = cutie.get_number("Please input the end of the signal (-1 for none): ", None, None, True)
        case 6:
            fL = cutie.get_number("Please input the lower filter cutoff frequency in Hz: ", 0, fH, True)
        case 7:
            fH = cutie.get_number("Please input the higher filter cutoff frequency in Hz: ", fL, None, True)
        case 8:
            order = cutie.get_number("Please input the order of the filter: ", 0, None, False)
        case 10:
            sm_ppg = cutie.get_number("Please input the smoothing window of PPG signal in ms: ", 0, None, True)
        case 11:
            sm_vpg = cutie.get_number("Please input the smoothing window of VPG signal in ms: ", 0, None, True)
        case 12:
            sm_apg = cutie.get_number("Please input the smoothing window of APG signal in ms: ", 0, None, True)
        case 13:
            sm_jpg = cutie.get_number("Please input the smoothing window of JPG signal in ms: ", 0, None, True)
        case 15:
            gauss = not gauss
        case 16:
            gauss_live_plot = not gauss_live_plot
        case 17:
            os.system('cls')
            gaussian_params()
        case 19:
            skewed = not skewed
        case 20:
            skewed_live_plot = not skewed_live_plot
        case 21:
            os.system('cls')
            skewed_params()
        case 23:
            print("Please select the output format:")
            format_options = ["csv", "mat", "both"]
            format_index = cutie.select(format_options)
            saving_format = format_options[format_index]
        case 24:
            savingfolder = input("Please input the output folder in which PPG features will be saved (or leave empty for the default folder): ")
            if not savingfolder.strip():
                saving_folder = "results"
        case 26:
            process_signal(path=path, fs=fs, start=start, end=end, fL=fL, fH=fH, order=order, sm_wins={'ppg':sm_ppg, 'vpg':sm_vpg, 'apg':sm_apg, 'jpg':sm_jpg}, enable_gauss=gauss, gauss_live_plot=gauss_live_plot, g_values=g_values, enable_skewed=skewed, skewed_live_plot=skewed_live_plot, s_values=s_values, savingformat=saving_format, savingfolder=saving_folder)
            quit(0)

    # Reset
    os.system('cls')
    main()

def gaussian_params():
    options = [
        "GAUSSIAN PARAMETERS:",
        "======",
        f"Initial 1st amplitude (% of max amplitude): {g_values[0] * 100 :.2f}",
        f"Initial 1st mean (% of duration): {g_values[1] * 100 :.2f}",
        f"Initial 1st standard deviation: {g_values[2] :.2f}",
        f"Initial 2nd amplitude (% of max amplitude): {g_values[3] * 100 :.2f}",
        f"Initial 2nd mean (% of duration): {g_values[4] * 100 :.2f}",
        f"Initial 2nd standard deviation: {g_values[5] :.2f}",
        f"Initial 3rd amplitude (% of max amplitude): {g_values[6] * 100 :.2f}",
        f"Initial 3rd mean (% of duration): {g_values[7] * 100 :.2f}",
        f"Initial 3rd standard deviation: {g_values[8] :.2f}",
        f"Initial 4th amplitude (% of max amplitude): {g_values[9] * 100 :.2f}",
        f"Initial 4th mean (% of duration): {g_values[10] * 100 :.2f}",
        f"Initial 4th standard deviation: {g_values[11] :.2f}",
        "======",
        "BACK TO THE MAIN MENU"]
    captions = [0, 1, 14]
    option = cutie.select(options, captions)
    match option:
        case 2:
            g_values[0] = cutie.get_number("Please input the 1st amplitude as % of max amplitude: ", 0, None,
                                           True) / 100.0
            os.system('cls')
            gaussian_params()
        case 3:
            g_values[1] = cutie.get_number("Please input the 1st mean as % of duration: ", 0, None, True) / 100.0
            os.system('cls')
            gaussian_params()
        case 4:
            g_values[2] = cutie.get_number("Please input the 1st standard deviation: ", None, None, True)
            os.system('cls')
            gaussian_params()
        case 5:
            g_values[3] = cutie.get_number("Please input the 2nd amplitude as % of max amplitude: ", 0, None,
                                           True) / 100.0
            os.system('cls')
            gaussian_params()
        case 6:
            g_values[4] = cutie.get_number("Please input the 2nd mean as % of duration: ", 0, None, True) / 100.0
            os.system('cls')
            gaussian_params()
        case 7:
            g_values[5] = cutie.get_number("Please input the 2nd standard deviation: ", None, None, True)
            os.system('cls')
            gaussian_params()
        case 8:
            g_values[6] = cutie.get_number("Please input the 3rd amplitude as % of max amplitude: ", 0, None,
                                           True) / 100.0
            os.system('cls')
            gaussian_params()
        case 9:
            g_values[7] = cutie.get_number("Please input the 3rd mean as % of duration: ", 0, None, True) / 100.0
            os.system('cls')
            gaussian_params()
        case 10:
            g_values[8] = cutie.get_number("Please input the 3rd standard deviation: ", None, None, True)
            os.system('cls')
            gaussian_params()
        case 11:
            g_values[9] = cutie.get_number("Please input the 4th amplitude as % of max amplitude: ", 0, None,
                                           True) / 100.0
            os.system('cls')
            gaussian_params()
        case 12:
            g_values[10] = cutie.get_number("Please input the 4th mean as % of duration: ", 0, None, True) / 100.0
            os.system('cls')
            gaussian_params()
        case 13:
            g_values[11] = cutie.get_number("Please input the 4th standard deviation: ", None, None, True)
            os.system('cls')
            gaussian_params()
        case 15:
            os.system('cls')
            return

def skewed_params():
    options = [
        "SKEWED GAUSSIAN PARAMETERS:",
        "======",
        f"Initial 1st amplitude: {s_values[0] :.2f}",
        f"Initial 1st location: {s_values[1] :.2f}",
        f"Initial 1st scale: {s_values[2] :.2f}",
        f"Initial 1st shape: {s_values[3] :.2f}",
        f"Initial 2nd amplitude: {s_values[4] :.2f}",
        f"Initial 2nd location: {s_values[5] :.2f}",
        f"Initial 2nd scale: {s_values[6] :.2f}",
        f"Initial 2nd shape: {s_values[7] :.2f}",
        f"Initial 3rd amplitude: {s_values[8] :.2f}",
        f"Initial 3rd location: {s_values[9] :.2f}",
        f"Initial 3rd scale: {s_values[10] :.2f}",
        f"Initial 3rd shape: {s_values[11] :.2f}",
        f"Initial 4th amplitude: {s_values[12] :.2f}",
        f"Initial 4th location: {s_values[13] :.2f}",
        f"Initial 4th scale: {s_values[14] :.2f}",
        f"Initial 4th shape: {s_values[15] :.2f}",
        "======",
        "BACK TO THE MAIN MENU"]
    captions = [0, 1, 18]
    option = cutie.select(options, captions)
    match option:
        case 2:
            s_values[0] = cutie.get_number("Please input the 1st amplitude: ", 0, None, True)
            os.system('cls')
            skewed_params()
        case 3:
            s_values[1] = cutie.get_number("Please input the 1st location: ", 0, None, True)
            os.system('cls')
            skewed_params()
        case 4:
            s_values[2] = cutie.get_number("Please input the 1st scale: ", 0, None, True)#
            os.system('cls')
            skewed_params()
        case 5:
            s_values[3] = cutie.get_number("Please input the 1st shape: ", 0, None, True)
            os.system('cls')
            skewed_params()
        case 6:
            s_values[4] = cutie.get_number("Please input the 2nd amplitude: ", 0, None, True)
            os.system('cls')
            skewed_params()
        case 7:
            s_values[5] = cutie.get_number("Please input the 2nd location: ", 0, None, True)
            os.system('cls')
            skewed_params()
        case 8:
            s_values[6] = cutie.get_number("Please input the 2nd scale: ", 0, None, True)
            os.system('cls')
            skewed_params()
        case 9:
            s_values[7] = cutie.get_number("Please input the 2nd shape: ", 0, None, True)
            os.system('cls')
            skewed_params()
        case 10:
            s_values[8] = cutie.get_number("Please input the 3rd amplitude: ", 0, None, True)
            os.system('cls')
            skewed_params()
        case 11:
            s_values[9] = cutie.get_number("Please input the 3rd location: ", 0, None, True)
            os.system('cls')
            skewed_params()
        case 12:
            s_values[10] = cutie.get_number("Please input the 3rd scale: ", 0, None, True)
            os.system('cls')
            skewed_params()
        case 13:
            s_values[11] = cutie.get_number("Please input the 3rd shape: ", 0, None, True)
            os.system('cls')
            skewed_params()
        case 14:
            s_values[12] = cutie.get_number("Please input the 4th amplitude: ", 0, None, True)
            os.system('cls')
            skewed_params()
        case 15:
            s_values[13] = cutie.get_number("Please input the 4th location: ", 0, None, True)
            os.system('cls')
            skewed_params()
        case 16:
            s_values[14] = cutie.get_number("Please input the 4th scale: ", 0, None, True)
            os.system('cls')
            skewed_params()
        case 17:
            s_values[15] = cutie.get_number("Please input the 4th shape: ", 0, None, True)
            os.system('cls')
            skewed_params()
        case 19:
            os.system('cls')
            return

if __name__ == '__main__':
    os.system('cls')
    main()