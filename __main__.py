# Import backend
from process_signal import process_signal
# Import other
import cutie
import os

# Variables
path = ""
fs = 200
start = 0
end = -1
fL = 0.5
fH = 12
order = 4
sm_ppg=50
sm_vpg=10
sm_apg=10
sm_jpg=10
saving_format='csv'
saving_folder='results'

# Menu
def main():
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
        "===OUTPUT OPTIONS===",
        f"Output filetype: {saving_format}",
        f"Output folder: {saving_folder}",
        "======",
        "PROCEED"
    ]
    captions = [1,5,9,14,17]
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
            print("Please select the output format:")
            format_options = ["csv", "mat"]
            format_index = cutie.select(format_options)
            saving_format = format_options[format_index]
        case 16:
            savingfolder = input("Please input the output folder in which PPG features will be saved (or leave empty for the default folder): ")
            if not savingfolder.strip():
                saving_folder = "results"
        case 18:
            process_signal(path=path, fs=fs, start=start, end=end, fL=fL, fH=fH, order=order, sm_wins={'ppg':sm_ppg, 'vpg':sm_vpg, 'apg':sm_apg, 'jpg':sm_jpg}, savingformat=saving_format, savingfolder=saving_folder)
            quit(0)

    # Reset
    os.system('cls')
    main()

if __name__ == '__main__':
    main()