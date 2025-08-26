import pyPPG

import scipy.io
import numpy as np
import pandas as pd
from dotmap import DotMap
import os
from scipy.io import savemat


def save_data(savingformat: str, savingfolder: str, print_flag=True, s={}, fp=pd.DataFrame(), bm=pd.DataFrame(), gauss=None, gauss_stats=None, gauss_additional=None, skewed=None, skewed_stats=None, vpg=pd.DataFrame(), vpg_stats=pd.DataFrame() ,ppg_extra=pd.DataFrame(), ppg_extra_stats=pd.DataFrame()):
    """
    Save the results of the filtered PPG analysis.

    :param savingformat: file format of the saved date, the provided file formats .mat and .csv
    :type savingformat: str
    :param savingfolder: location of the saved data
    :type savingfolder: str
    :param print_flag: a bool for print message
    :type print_flag: bool
    :param s: a struct of PPG signal
    :type s: pyPPG.PPG object
    :type s: DotMap
    :param fp: a struct of fiducial points
    :type fp: pyPPG.Fiducial object
    :param bm: a dictionary of biomarkers
    :type bm: pyPPG.Biomarkers object
    :param gauss: a DataFrame of Gaussians per pulse
    :type gauss: pd.DataFrame
    :param gauss_stats: a DataFrame of Gaussian statistics
    :type gauss_stats: pd.DataFrame

    :return: file_names: dictionary of the saved file names
    """

    # Set up directories
    savingfolder = savingfolder.replace('/', '\\')

    if not(':' in savingfolder) and savingfolder[0]!='/':
        relative_path=r'.'+os.sep
    else:
        relative_path=''

    tmp_dir = savingfolder
    os.makedirs(tmp_dir, exist_ok=True)

    temp_dirs = ['Fiducial_points', 'Biomarker_vals', 'Biomarker_stats', 'Biomarker_defs', 'PPG_struct', 'Biomarker_defs_and_stats', 'Additional']
    for i in temp_dirs:
        temp_dir = tmp_dir + os.sep + i + os.sep
        os.makedirs(temp_dir, exist_ok=True)

    keys=s.__dict__.keys()
    keys_list = list(keys)
    sc=DotMap()
    for i in keys_list:
        exec('sc.'+i+' = s.'+i)

    file_names = {}
    file_name = (relative_path + tmp_dir + os.sep + temp_dirs[4] + os.sep + s.name + '_data_btwn_%s-%s.mat')%(s.start_sig,s.end_sig)
    file_names ['data_struct_mat']= file_name
    scipy.io.savemat(file_name, sc)

    try:
        BM_keys = bm.bm_vals.keys()
    except:
        BM_keys={}

    if savingformat=="csv" or savingformat=="both":
        file_name = (relative_path+tmp_dir+os.sep+temp_dirs[0]+os.sep+s.name+'_'+'Fiducials_btwn_%s-%s.csv')%(s.start_sig,s.end_sig)
        file_names ['fiducials_csv']= file_name
        tmp_fp = fp.get_fp()
        tmp_fp.index = tmp_fp.index + 1
        tmp_fp.to_csv(file_name)

        for key in BM_keys:
            file_name = (relative_path+tmp_dir+os.sep+temp_dirs[1]+os.sep+'%s_btwn_%s-%s.csv')%(s.name+'_'+key,s.start_sig,s.end_sig)
            file_names [key + '_vals_csv']= file_name
            bm.bm_vals[key].index = bm.bm_vals[key].index + 1
            bm.bm_vals[key].to_csv(file_name,index=True,header=True)

            file_name = (relative_path+tmp_dir+os.sep+temp_dirs[2]+os.sep+'%s_btwn_%s-%s.csv')%(s.name+'_'+key,s.start_sig,s.end_sig)
            file_names [key + '_stats_csv']= file_name
            bm.bm_stats[key].to_csv(file_name, index=True, header=True)

            file_name = (relative_path+tmp_dir+os.sep+temp_dirs[3]+os.sep+'%s_btwn_%s-%s.csv')%(s.name+'_'+key,s.start_sig,s.end_sig)
            file_names [key + '_defs_csv']= file_name
            bm.bm_defs[key].index = bm.bm_defs[key].index + 1
            bm.bm_defs[key].to_csv(file_name, index=True, header=True)

        # Gaussians
        if gauss is not None:
            file_name = (relative_path+tmp_dir+os.sep+temp_dirs[6]+os.sep+s.name+'_'+'Gaussians_btwn_%s-%s.csv')%(s.start_sig,s.end_sig)
            file_names ['gaussians_csv']= file_name
            gauss.to_csv(file_name)
        if gauss_stats is not None:
            file_name = (relative_path+tmp_dir+os.sep+temp_dirs[6]+os.sep+s.name+'_'+'Gaussian_stats_btwn_%s-%s.csv')%(s.start_sig,s.end_sig)
            file_names['gaussian_stats_csv'] = file_name
            gauss_stats.to_csv(file_name)
        if gauss_additional is not None:
            file_name = (relative_path+tmp_dir+os.sep+temp_dirs[6]+os.sep+s.name+'_'+'Gaussian_additional_btwn_%s-%s.csv')%(s.start_sig,s.end_sig)
            file_names['gaussian_additional_csv'] = file_name
            gauss_additional.to_csv(file_name)
        # Skewed Gaussians
        if skewed is not None:
            file_name = (relative_path+tmp_dir+os.sep+temp_dirs[6]+os.sep+s.name+'_'+'Skewed_btwn_%s-%s.csv')%(s.start_sig,s.end_sig)
            file_names ['skewed_csv']= file_name
            skewed.to_csv(file_name)
        if skewed_stats is not None:
            file_name = (relative_path+tmp_dir+os.sep+temp_dirs[6]+os.sep+s.name+'_'+'Skewed_stats_btwn_%s-%s.csv')%(s.start_sig,s.end_sig)
            file_names['skewed_stats_csv'] = file_name
            skewed_stats.to_csv(file_name)
        # VPG
        file_name = (relative_path+tmp_dir+os.sep+temp_dirs[6]+os.sep+s.name+'_'+'VPG_btwn_%s-%s.csv')%(s.start_sig,s.end_sig)
        file_names['vpg_csv'] = file_name
        vpg.to_csv(file_name)
        file_name = (relative_path+tmp_dir+os.sep+temp_dirs[6]+os.sep+s.name+'_'+'VPG_stats_btwn_%s-%s.csv')%(s.start_sig,s.end_sig)
        file_names['vpg_stats_csv'] = file_name
        vpg_stats.to_csv(file_name)
        # PPG
        file_name = (relative_path+tmp_dir+os.sep+temp_dirs[6]+os.sep+s.name+'_'+'PPG_extra_btwn_%s-%s.csv')%(s.start_sig,s.end_sig)
        file_names['ppg_extra_csv'] = file_name
        ppg_extra.to_csv(file_name)
        file_name = (relative_path+tmp_dir+os.sep+temp_dirs[6]+os.sep+s.name+'_'+'PPG_extra_stats_btwn_%s-%s.csv')%(s.start_sig,s.end_sig)
        file_names['ppg_extra_stats_csv'] = file_name
        ppg_extra_stats.to_csv(file_name)

    if savingformat=="mat"  or savingformat=="both":
        file_name = (relative_path+tmp_dir+os.sep+temp_dirs[0]+os.sep+s.name+'_'+'Fiducials_btwn_%s-%s.mat')%(s.start_sig,s.end_sig)
        file_names['fiducials_mat']=file_name
        tmp_fp = fp.get_fp()
        tmp_fp.index = tmp_fp.index + 1
        savemat(file_name, {'PPG_fiducials': tmp_fp.to_records(index=True)})

        for key in BM_keys:

            file_name = (relative_path+tmp_dir+os.sep+temp_dirs[1]+os.sep+'%s_btwn_%s-%s.mat')%(s.name+'_'+key,s.start_sig,s.end_sig)
            file_names[key+'_vals_mat']=file_name
            tmp_df_vals=bm.bm_vals[key]
            matlab_struct = tmp_df_vals.to_dict(orient='list')
            savemat(file_name, {'PPG_vals': tmp_df_vals.to_records(index=True)})

            if len(bm.bm_stats)>0:
                file_name = (relative_path+tmp_dir+os.sep+temp_dirs[2]+os.sep+'%s_btwn_%s-%s.mat')%(s.name+'_'+key,s.start_sig,s.end_sig)
                file_names[key + '_stats_mat']=file_name
                tmp_df_stat=bm.bm_stats[key]
                savemat(file_name, {'PPG_stats': tmp_df_stat.to_records(index=True)})

            file_name = (relative_path+tmp_dir+os.sep+temp_dirs[3]+os.sep+'%s_btwn_%s-%s.mat')%(s.name+'_'+key,s.start_sig,s.end_sig)
            file_names[key + '_defs_mat']=file_name
            tmp_df_defs=bm.bm_defs[key]
            matlab_struct = tmp_df_defs.to_dict(orient='list')
            savemat(file_name, {'PPG_defs': tmp_df_defs.to_records(index=True)})

            file_name = (relative_path+tmp_dir+os.sep+temp_dirs[5]+os.sep+'%s_btwn_%s-%s.mat')%(s.name+'_'+key,s.start_sig,s.end_sig)
            file_names[key + '_defs_stats_mat']=file_name
            tmp_df_defs2 = tmp_df_defs.drop('name', axis=1)
            tmp_df_defs2.index =tmp_df_defs['name']
            tmp_df_stat2=tmp_df_stat.transpose()
            tmp_df_defs_and_stats=pd.concat([tmp_df_defs2, tmp_df_stat2], axis=1)
            savemat(file_name, {key: tmp_df_defs_and_stats.to_records(index=True)})

        # Gaussians
        if gauss is not None:
            file_name = (relative_path+tmp_dir+os.sep+temp_dirs[6]+os.sep+s.name+'_'+'Gaussians_btwn_%s-%s.mat')%(s.start_sig,s.end_sig)
            file_names['gaussians_mat']=file_name
            savemat(file_name, {'Gaussians': gauss.to_records(index=True)})
        if gauss_stats is not None:
            file_name = (relative_path+tmp_dir+os.sep+temp_dirs[6]+os.sep+s.name+'_'+'Gaussian_stats_btwn_%s-%s.mat')%(s.start_sig,s.end_sig)
            file_names['gaussian_stats_mat']=file_name
            savemat(file_name, {'Gaussian_stats': gauss_stats.to_records(index=True)})
        if gauss_additional is not None:
            file_name = (relative_path+tmp_dir+os.sep+temp_dirs[6]+os.sep+s.name+'_'+'Gaussian_additional_btwn_%s-%s.mat')%(s.start_sig,s.end_sig)
            file_names['gaussian_additional_mat']=file_name
            savemat(file_name, {'Gaussian_additional': ppg_extra.to_records(index=True)})
        # Skewed Gaussians
        if skewed is not None:
            file_name = (relative_path+tmp_dir+os.sep+temp_dirs[6]+os.sep+s.name+'_'+'Skewed_btwn_%s-%s.mat')%(s.start_sig,s.end_sig)
            file_names['skewed_mat']=file_name
            savemat(file_name, {'Skewed': skewed.to_records(index=True)})
        if skewed_stats is not None:
            file_name = (relative_path+tmp_dir+os.sep+temp_dirs[6]+os.sep+s.name+'_'+'Skewed_stats_btwn_%s-%s.mat')%(s.start_sig,s.end_sig)
            file_names['skewed_stats_mat']=file_name
            savemat(file_name, {'Skewed_stats': gauss_stats.to_records(index=True)})
        # VPG
        file_name = (relative_path+tmp_dir+os.sep+temp_dirs[6]+os.sep+s.name+'_'+'VPG_btwn_%s-%s.mat')%(s.start_sig,s.end_sig)
        file_names['vpg_mat']=file_name
        savemat(file_name, {'VPG': vpg.to_records(index=True)})
        file_name = (relative_path+tmp_dir+os.sep+temp_dirs[6]+os.sep+s.name+'_'+'VPG_stats_btwn_%s-%s.mat')%(s.start_sig,s.end_sig)
        file_names['vpg_stats_mat']=file_name
        savemat(file_name, {'VPG stats': vpg_stats.to_records(index=True)})
        # PPG
        file_name = (relative_path+tmp_dir+os.sep+temp_dirs[6]+os.sep+s.name+'_'+'PPG_extra_btwn_%s-%s.mat')%(s.start_sig,s.end_sig)
        file_names['ppg_extra_mat']=file_name
        savemat(file_name, {'ppg_extra': ppg_extra.to_records(index=True)})
        file_name = (relative_path+tmp_dir+os.sep+temp_dirs[6]+os.sep+s.name+'_'+'PPG_extra_stats_btwn_%s-%s.mat')%(s.start_sig,s.end_sig)
        file_names['ppg_extra_stats_mat']=file_name
        savemat(file_name, {'ppg_extra_stats': ppg_extra_stats.to_records(index=True)})
    if savingformat != "csv" and savingformat != "mat" and savingformat != "both" and savingformat!="none":
        raise('The file format is not suported for data saving! You can use "mat" or "csv" file formats.')

    if print_flag: print('Results have been saved into the "'+tmp_dir+'".')

    return file_names