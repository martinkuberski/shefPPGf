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

def process_signal(path="",
                   fs=200,
                   start=0,
                   end=-1,
                   fL=0.5,
                   fH=12,
                   order=4,
                   sm_wins={'ppg':50, 'vpg':10, 'apg':10, 'jpg':10},
                   correction=pd.DataFrame(),
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

    # Get fiducial points
    fpex = FP.FpCollection(s=s)
    fiducials = fpex.get_fiducials(s=s)
    fiducials = fiducials.applymap(lambda x: np.nan if pd.isna(x) else x)
    fp = Fiducials(fp=fiducials)

    # Calculate SQI
    ppgSQI = round(np.mean(SQI.get_ppgSQI(ppg=s.ppg, fs=s.fs, annotation=fp.sp)) * 100, 2)

    # Get biomarkers
    bmex = BM.BmCollection(s=s, fp=fp)
    bm_defs, bm_vals, bm_stats = bmex.get_biomarkers()
    bm = Biomarkers(bm_defs=bm_defs, bm_vals=bm_vals, bm_stats=bm_stats)

    # Save data
    fp_new = Fiducials(fp=fp.get_fp() + s.start_sig)
    save_data(savingformat=savingformat, savingfolder=savingfolder,s=s, fp=fp_new, bm=bm)