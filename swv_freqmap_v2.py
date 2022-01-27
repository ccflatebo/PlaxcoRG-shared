# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 08:51:31 2021

@author: sukik
"""

import eab_functions as ef
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib as mpl

from scipy.optimize import curve_fit, least_squares
from scipy.signal import find_peaks, savgol_filter

#%% Directory Setup
maindir = '../Data/EChem/'
expdate = input('Experiment Date: ')#210914'
analysisdate = input('Analysis Date: ')
analysisdir = maindir + expdate + '/' + analysisdate + '/'
figdir = maindir + expdate + '/Figures/'
ef.check_dir_exist(figdir)

#%%
electrodes_ignore = input('Electrodes to ignore? (ie 3, 4, 6): ')
# if electrodes_ignore:
#     electrodes_ignore = electrodes_ignore.split(',')
#     electrodes_ignore = [str(x.strip()) for x in electrodes_ignore]
exclude_freq = []# input('Frequencies to ignore? (ie 5, 10, 800): ')
# if exclude_freq:
#     exclude_freq = exclude_freq.split(',')
#     exclude_freq = [str(x.strip()) for x in exclude_freq]

#%% Load Data
swv_list = 'swvdata.txt'
filetxt = open(maindir + expdate + '/' + swv_list,'r')
filenames = filetxt.read().splitlines()
filetxt.close()
filestart = filenames[0].split('_')[0]
current_conv = 10**6
current_label = 'Current ($\mu$A)'
hill = False
on_freq = input('Signal On Frequencies? (ie 5, 10, 800): ')
off_freq = input('Signal Off Frequencies? (ie 5, 10, 800): ')
if on_freq:
    on_freq = on_freq.split(',')
    on_freq = [x.strip() for x in on_freq]
if off_freq:
    off_freq = off_freq.split(',')
    off_freq = [x.strip() for x in off_freq]
titration = input('Titration? (y/n) ')
if titration == 'y':
    titration = True
    conc_datafile = ef.getfile(maindir + expdate,'Select Concentration Excel Sheet',filetypestoshow=('Excel Files','*.xls *.xlsx'))
    conc_data = pd.read_excel(conc_datafile)
    concentrations = np.array(conc_data['Final [M]'])
    label = 'Concentration (M)'
else:
    titration = False
    label = 'Frames'


#%%
data = np.load(analysisdir + expdate + '_' + filestart + '_combined.npy', allow_pickle = True).item()
# data = pd.read_pickle(analysisdir + expdate + '_' + filestart + '_amplitude-analysis.pkl')

freqs = list(data.keys())
# for freq in exclude_freq:
#     freqs.remove(freq)
freqs_int = [int(x) for x in freqs]
num_freq = len(freqs)
num_frames = len(data[freqs[0]])

num_electrodes = np.shape(data[freqs[0]][str(num_frames)]['params'])[0]
# num_electrodes_short = num_electrodes - len(electrodes_ignore)

frame_data = np.zeros((num_electrodes,num_frames,num_freq))
frame_e0 = np.zeros_like(frame_data)
electrodes_include = []
electrodes_all = []
for i in range(0,num_electrodes):
    for j in range(0,num_frames):
        for k, freq in enumerate(freqs):
            frame_data[i,j,k] = data[freq][str(j+1)]['params']['amplitude'][i]
            frame_e0[i,j,k] = data[freq][str(j+1)]['params']['mean'][i]
    if str(i+1) not in electrodes_ignore:
        electrodes_include.append(str(i))
    electrodes_all.append(str(i))
if not titration:
    concentrations = np.linspace(0,num_frames,num_frames)
num_electrodes_short = np.shape(electrodes_include)[0]
if num_electrodes_short > 3:
    row_plots = 2
    column_plots = int(np.ceil(num_electrodes_short/row_plots))
else:
    row_plots = 1
    column_plots = num_electrodes_short
#%% Calculations
frame_data_norm = np.divide(frame_data,frame_data[:,0,:].reshape((num_electrodes,1,num_freq)))
if titration:
    kdm_dict = {}
    for i in range(0,num_electrodes):
        kdm_dict[i] = dict()
        for j in range(0,num_freq):
            kdm_dict[i][freqs[j]] = {}
            # Normalized Data
            kd_fit_params_kdm_norm = pd.DataFrame(columns=['b','Kd','rsquare'])
            kd_fit_params_kdm_num_norm = pd.DataFrame(columns=['b','Kd','rsquare'])
            kdm_calc_norm = pd.DataFrame()
            kdm_num_calc_norm = pd.DataFrame()
            kd_fit_kdm_norm = pd.DataFrame()
            kd_fit_kdm_num_norm = pd.DataFrame()
            # kdm_dict[freqs[j]][i] = dict()
            for k in range(0,num_freq):
                if k != j:
                    if hill:
                        # kdm_num_calc[freqs[k]], kd_fit_kdm_num[freqs[k]], kd_fit_params_kdm_num.loc[freqs[k]] = ef.langmuir_fitting(frame_data_average[:,j],frame_data_average[:,k],concentrations,numerator=True, hill=True)
                        kdm_num_calc_norm[freqs[k]], kd_fit_kdm_num_norm[i], kd_fit_params_kdm_num_norm.loc[i] = ef.langmuir_fitting(frame_data_norm[i,:,j],frame_data_norm[i,:,k],concentrations,numerator=True,norm=True, hill=True)
                        # KDM
                        # kdm_calc[freqs[k]], kd_fit_kdm[freqs[k]], kd_fit_params_kdm.loc[freqs[k]] = ef.langmuir_fitting(frame_data_average[:,j],frame_data_average[:,k],concentrations, hill=True)
                        kdm_calc_norm[i], kd_fit_kdm_norm[i], kd_fit_params_kdm_norm.loc[i] = ef.langmuir_fitting(frame_data_norm[i,:,j],frame_data_norm[i,:,k],concentrations,norm=True, hill=True)
                    else:
                        # kdm_num_calc[freqs[k]], kd_fit_kdm_num[freqs[k]], kd_fit_params_kdm_num.loc[freqs[k]] = ef.langmuir_fitting(frame_data_average[:,j],frame_data_average[:,k],concentrations,numerator=True)
                        kdm_num_calc_norm[freqs[k]], kd_fit_kdm_num_norm[freqs[k]], kd_fit_params_kdm_num_norm.loc[freqs[k]] = ef.langmuir_fitting(frame_data_norm[i,:,j],frame_data_norm[i,:,k],concentrations,numerator=True,norm=True)
                        # KDM
                        # kdm_calc[freqs[k]], kd_fit_kdm[freqs[k]], kd_fit_params_kdm.loc[freqs[k]] = ef.langmuir_fitting(frame_data_average[:,j],frame_data_average[:,k],concentrations)
                        kdm_calc_norm[freqs[k]], kd_fit_kdm_norm[freqs[k]], kd_fit_params_kdm_norm.loc[freqs[k]] = ef.langmuir_fitting(frame_data_norm[i,:,j],frame_data_norm[i,:,k],concentrations,norm=True)
            # kdm_dict[freqs[j]]['params-kdm'] = kd_fit_params_kdm
            # kdm_dict[freqs[j]]['params-kdm_num'] = kd_fit_params_kdm_num
            # kdm_dict[freqs[j]]['calc-kdm'] = kdm_calc
            # kdm_dict[freqs[j]]['calc-kdm_num'] = kdm_num_calc
            # kdm_dict[freqs[j]]['fit-kdm'] = kd_fit_kdm
            # kdm_dict[freqs[j]]['fit-kdm_num'] = kd_fit_kdm_num
            kdm_dict[i][freqs[j]]['params-kdm'] = kd_fit_params_kdm_norm
            kdm_dict[i][freqs[j]]['params-kdm_num'] = kd_fit_params_kdm_num_norm
            kdm_dict[i][freqs[j]]['calc-kdm'] = kdm_calc_norm
            kdm_dict[i][freqs[j]]['calc-kdm_num'] = kdm_num_calc_norm
            kdm_dict[i][freqs[j]]['fit-kdm'] = kd_fit_kdm_norm
            kdm_dict[i][freqs[j]]['fit-kdm_num'] = kd_fit_kdm_num_norm
# Averages
frame_data_include = np.zeros((len(electrodes_include),num_frames,num_freq))
frame_e0_include = np.zeros_like(frame_data_include)
if titration:
    kdm_dict_include = dict()
for i, val in enumerate(electrodes_include):
    frame_data_include[i,:,:] = frame_data[int(val),:,:]
    frame_e0_include[i,:,:] = frame_data[int(val),:,:]
    if titration:
        kdm_dict_include[i] = kdm_dict[int(val)]
frame_data_average = frame_data_include.mean(axis = 0)
frame_e0_average = frame_e0_include.mean(axis = 0)
frame_data_std = frame_data_include.std(axis = 0)
frame_e0_std = frame_data_include.std(axis = 0)

if titration:
    kdm_dict_avg = {}
    # kdm_dict_norm = {}
    for j in range(0,num_freq):
        kd_fit_params_kdm_norm = pd.DataFrame(columns=['b','Kd','rsquare'])
        kd_fit_params_kdm_num_norm = pd.DataFrame(columns=['b','Kd','rsquare'])
        kdm_calc_norm = pd.DataFrame()
        kdm_num_calc_norm = pd.DataFrame()
        kd_fit_kdm_norm = pd.DataFrame()
        kd_fit_kdm_num_norm = pd.DataFrame()
        kdm_dict_avg[freqs[j]] = dict()
        for k in range(0,num_freq):
            if k != j:
                # Non Normalized
                # Numerator only (signal_on - signal_off)
                if hill:
                    # kdm_num_calc[freqs[k]], kd_fit_kdm_num[freqs[k]], kd_fit_params_kdm_num.loc[freqs[k]] = ef.langmuir_fitting(frame_data_average[:,j],frame_data_average[:,k],concentrations,numerator=True, hill=True)
                    kdm_num_calc_norm[freqs[k]], kd_fit_kdm_num_norm[freqs[k]], kd_fit_params_kdm_num_norm.loc[freqs[k]] = ef.langmuir_fitting(frame_data_average[:,j]/frame_data_average[0,j],frame_data_average[:,k]/frame_data_average[0,k],concentrations,numerator=True,norm=True, hill=True)
                    # KDM
                    # kdm_calc[freqs[k]], kd_fit_kdm[freqs[k]], kd_fit_params_kdm.loc[freqs[k]] = ef.langmuir_fitting(frame_data_average[:,j],frame_data_average[:,k],concentrations, hill=True)
                    kdm_calc_norm[freqs[k]], kd_fit_kdm_norm[freqs[k]], kd_fit_params_kdm_norm.loc[freqs[k]] = ef.langmuir_fitting(frame_data_average[:,j]/frame_data_average[0,j],frame_data_average[:,k]/frame_data_average[0,k],concentrations,norm=True, hill=True)
                else:
                    # kdm_num_calc[freqs[k]], kd_fit_kdm_num[freqs[k]], kd_fit_params_kdm_num.loc[freqs[k]] = ef.langmuir_fitting(frame_data_average[:,j],frame_data_average[:,k],concentrations,numerator=True)
                    kdm_num_calc_norm[freqs[k]], kd_fit_kdm_num_norm[freqs[k]], kd_fit_params_kdm_num_norm.loc[freqs[k]] = ef.langmuir_fitting(frame_data_average[:,j]/frame_data_average[0,j],frame_data_average[:,k]/frame_data_average[0,k],concentrations,numerator=True,norm=True)
                    # KDM
                    # kdm_calc[freqs[k]], kd_fit_kdm[freqs[k]], kd_fit_params_kdm.loc[freqs[k]] = ef.langmuir_fitting(frame_data_average[:,j],frame_data_average[:,k],concentrations)
                    kdm_calc_norm[freqs[k]], kd_fit_kdm_norm[freqs[k]], kd_fit_params_kdm_norm.loc[freqs[k]] = ef.langmuir_fitting(frame_data_average[:,j]/frame_data_average[0,j],frame_data_average[:,k]/frame_data_average[0,k],concentrations,norm=True)
        kdm_dict_avg[freqs[j]]['params-kdm'] = kd_fit_params_kdm_norm
        kdm_dict_avg[freqs[j]]['params-kdm_num'] = kd_fit_params_kdm_num_norm
        kdm_dict_avg[freqs[j]]['calc-kdm'] = kdm_calc_norm
        kdm_dict_avg[freqs[j]]['calc-kdm_num'] = kdm_num_calc_norm
        kdm_dict_avg[freqs[j]]['fit-kdm'] = kd_fit_kdm_norm
        kdm_dict_avg[freqs[j]]['fit-kdm_num'] = kd_fit_kdm_num_norm
#%% Plotting
# Colors
color_freq = plt.cm.coolwarm(np.linspace(0,1,num_freq+1))
color_frame = plt.cm.copper_r(np.linspace(0,1,num_frames))
color_electrode = plt.cm.get_cmap('Paired').colors
cmap_frame = mpl.cm.copper_r
norm_frame = mpl.colors.Normalize(vmin = 0,vmax=np.amax(concentrations))
cmap_freq = mpl.cm.coolwarm
norm_freq = mpl.colors.Normalize(vmin = np.amin(freqs_int),vmax=np.amax(freqs_int))
# cmap_frame = mpl.cm.copper_r
# norm_frame = mpl.colors.Normalize(vmin = 0,vmax=np.amax(concentrations))
# Gain
fig_gain, ax = plt.subplots(row_plots,column_plots,figsize = (10,6),sharex = True, sharey = True)
ax_list = ax.reshape(-1)
fig_gain.suptitle('Gain at each Frequency')

for i in range(0,num_electrodes_short):
    ax_list[i].set_title('Electrode #' + str(int(electrodes_include[i]) + 1))
    for j in range(0,num_frames):
        colorchoose = color_frame[j]
        ax_list[i].semilogx(freqs_int,(frame_data_include[i,j,:]-frame_data_include[i,0,:])/frame_data_include[i,0,:]*100,marker='.',color = colorchoose)

# ax.annotate(r'$\dfrac{C_n - C_0}{C_0} \times 100$',xy = (.25,.99),xycoords = 'axes fraction',va = 'top', ha = 'left')
fig_gain.text(0.5,0,'Frequency (Hz)',ha='center')
fig_gain.text(0,0.5,'% Signal Change',va='center',rotation='90')
plt.colorbar(mpl.cm.ScalarMappable(norm=norm_frame,cmap=cmap_frame),label = label, ax = ax_list[-1])
plt.colorbar(mpl.cm.ScalarMappable(norm=norm_frame,cmap=cmap_frame),label = label, ax = ax_list[len(ax_list)//2-1])
plt.tight_layout()
plt.savefig(figdir + filestart + '_gain-indelectrodes.png',pady = 10)
# Lovric
fig_lovric, ax = plt.subplots(row_plots,column_plots,figsize = (10,6),sharex = True, sharey = True)
ax_list = ax.reshape(-1)
fig_lovric.suptitle('Lovric at each Frequency')

for i in range(0,num_electrodes_short):
    ax_list[i].set_title('Electrode #' + str(int(electrodes_include[i]) + 1))
    for j in range(0,num_frames):
        colorchoose = color_frame[j]
        ax_list[i].semilogx(freqs_int,(frame_data_include[i,j,:])/freqs_int,marker='.',color = colorchoose, linestyle='none')

# ax.annotate(r'$\dfrac{C_n - C_0}{C_0} \times 100$',xy = (.25,.99),xycoords = 'axes fraction',va = 'top', ha = 'left')
fig_lovric.text(0.5,0,'Frequency (Hz)',ha='center')
fig_lovric.text(0,0.5,'Coulombs',va='center',rotation='90')
plt.colorbar(mpl.cm.ScalarMappable(norm=norm_frame,cmap=cmap_frame),label = 'Frames', ax = ax_list[-1])
plt.colorbar(mpl.cm.ScalarMappable(norm=norm_frame,cmap=cmap_frame),label = 'Frames', ax = ax_list[len(ax_list)//2-1])
plt.tight_layout()
plt.savefig(figdir + filestart + '_lovric-indelectrodes.png',pady = 10)

# Avg Lovric and Gain
fig_lovgain, ax = plt.subplots(1,2,figsize = (8,4),sharex = True, sharey = False)

for j in range(0,num_frames):
    colorchoose = color_frame[j]
    ax[0].semilogx(freqs_int,(frame_data_average[j,:]-frame_data_average[0,:])/frame_data_average[0,:]*100,color = colorchoose,marker='.',label = str(np.round(concentrations[j]*10**9,decimals=3)) + ' nM')
    ax[1].semilogx(freqs_int,(frame_data_average[j,:])/freqs_int,color = colorchoose,marker='.',linestyle='none',label = str(np.round(concentrations[j]*10**9,decimals=3)) + ' nM')
    
# ax[0,0].set_ylim([np.nanmean(frame_e0_average)-np.nanstd(frame_e0_average)*2,np.nanmean(frame_e0_average)+np.nanstd(frame_e0_average)*2])
# ax.annotate(r'$\dfrac{C_0 - C_n}{C_0} \times 100$',xy = (.25,.99),xycoords = 'axes fraction',va = 'top', ha = 'left')
fig_lovgain.text(0.5,0,'Frequency (Hz)',ha='center')
ax[0].set_ylabel('% Signal Change')
ax[1].set_ylabel('Coulombs')
ax[0].set_title('Average Gain')
ax[1].set_title('Average Lovric')
# ax.legend()
# ax.set_xlim(0,100)
# ax[1].set_ylabel(current_label)
plt.tight_layout()
plt.savefig(figdir + filestart + '_avg_lovgain.png',pady = 10)
#%%

#%%
if titration:
    fig_amplitude_diff, ax = plt.subplots(len(off_freq),len(on_freq),figsize = (10,6),sharex = True, sharey = False)
    fig_amplitude_diff.suptitle('Signal On - Signal Off Langmuir Isotherm for ' + filestart)
    for j, on in enumerate(on_freq):
        for k, off in enumerate(off_freq):
            if len(on_freq) > 1:
                if len(off_freq) > 1:
                    ax_val = ax[k,j]
                else:
                    ax_val = ax[j]
            else:
                if len(off_freq) > 1:
                    ax_val = ax[k]
                else:
                    ax_val = ax
            for i, val in enumerate(electrodes_include):
                ax_val.semilogx(concentrations,kdm_dict_include[i][on]['calc-kdm_num'][off]*100,color = color_electrode[int(val)],linestyle='none', marker = '.')
                ax_val.semilogx(concentrations,kdm_dict_include[i][on]['fit-kdm_num'][off]*100,color = color_electrode[int(val)],label=str(int(val) + 1))
            ax_val.set_title('Signal On = ' + on + ' Hz')
            ax_val.semilogx(concentrations,kdm_dict_avg[on]['calc-kdm_num'][off]*100,color = 'k',linestyle='none', marker = 'd',markersize = 10)
            ax_val.semilogx(concentrations,kdm_dict_avg[on]['fit-kdm_num'][off]*100,color = 'k',linewidth = 4)
            # ax_val.fill_between(concentrations,-frame_data_std.T,frame_data_std.T)
            ax_val.annotate('Signal Off: ' + off +'Hz\n' + #'a = ' + str(np.format_float_scientific(kdm_dict_norm[on]['params-kdm_num'].loc[off,'a'],precision = 3)) + '\n' + 
                         # 'b = ' + str(np.format_float_scientific(kdm_dict_norm[on]['params-kdm_num'].loc[off,'b'],precision = 3)) + '\n' + 
                         'K$_D$ = ' + str(np.format_float_scientific(kdm_dict_avg[on]['params-kdm_num'].loc[off,'Kd'],precision = 3)),
                         xy = (0.01,.98), 
                         xycoords = 'axes fraction',
                         va = 'top')
    plt.legend(bbox_to_anchor=(1,1.25), loc='upper left', ncol=1)
    fig_amplitude_diff.text(0.5,0.04,'Concentration (M)',ha='center')
    fig_amplitude_diff.text(0.04,0.5,'% Signal Change',va='center',rotation='vertical')
    # ax_val.set_ylim(top=20)
    fig_amplitude_diff.savefig(figdir + filestart + '_KDM_num-selectedfreq_' + str(num_electrodes_short) + '.png')
    
    fig_amplitude_diff, ax = plt.subplots(len(off_freq),len(on_freq),figsize = (10,6),sharex = True, sharey = False)
    fig_amplitude_diff.suptitle('KDM Langmuir Isotherm for ' + filestart)
    for j, on in enumerate(on_freq):
        for k, off in enumerate(off_freq):
            if len(on_freq) > 1:
                if len(off_freq) > 1:
                    ax_val = ax[k,j]
                else:
                    ax_val = ax[j]
            else:
                if len(off_freq) > 1:
                    ax_val = ax[k]
                else:
                    ax_val = ax
            for i, val in enumerate(electrodes_include):
                ax_val.semilogx(concentrations,kdm_dict_include[i][on]['calc-kdm'][off]*100,color = color_electrode[int(val)],linestyle='none', marker = '.')
                ax_val.semilogx(concentrations,kdm_dict_include[i][on]['fit-kdm'][off]*100,color = color_electrode[int(val)],label=str(int(val) + 1))
            ax_val.set_title('Signal On = ' + on + ' Hz')
            ax_val.semilogx(concentrations,kdm_dict_avg[on]['calc-kdm'][off]*100,color = 'k',linestyle='none', marker = 'd')
            ax_val.semilogx(concentrations,kdm_dict_avg[on]['fit-kdm'][off]*100,color = 'k',linewidth = 4)
            # ax_val.fill_between(concentrations,-frame_data_std.T,frame_data_std.T)
            ax_val.annotate('Signal Off: ' + off +'Hz\n' + #'a = ' + str(np.format_float_scientific(kdm_dict_norm[on]['params-kdm_num'].loc[off,'a'],precision = 3)) + '\n' + 
                         # 'b = ' + str(np.format_float_scientific(kdm_dict_norm[on]['params-kdm_num'].loc[off,'b'],precision = 3)) + '\n' + 
                         'K$_D$ = ' + str(np.format_float_scientific(kdm_dict_avg[on]['params-kdm'].loc[off,'Kd'],precision = 3)),
                         xy = (0.01,.98), 
                         xycoords = 'axes fraction',
                         va = 'top')
    plt.legend(bbox_to_anchor=(1,1.25), loc='upper left', ncol=1)
    fig_amplitude_diff.text(0.5,0.04,'Concentration (M)',ha='center')
    fig_amplitude_diff.text(0.04,0.5,'% Signal Change',va='center',rotation='vertical')
    # ax_val.set_ylim(top=20)
    fig_amplitude_diff.savefig(figdir + filestart + '_KDM-selectedfreq_' + str(num_electrodes_short) + '.png')
    for j in range(0,num_freq):
        fig_amplitude_diff, ax = plt.subplots(2,2,figsize = (10,6),sharex = False, sharey = False)
        fig_amplitude_diff.suptitle('Langmuir Isotherm: ' + freqs[j] + ' Hz',color = color_freq[j])
        for k in range(0,num_freq):
            if k != j:
                colorchoose = color_freq[k]
                # Non Normalized
                # Numerator only (signal_on - signal_off)
                ax[0,0].plot(concentrations,kdm_dict_avg[freqs[j]]['calc-kdm_num'][freqs[k]]*100,color = colorchoose,linestyle = 'none', marker = '.')
                ax[0,0].plot(concentrations,kdm_dict_avg[freqs[j]]['fit-kdm_num'][freqs[k]]*100,color = colorchoose)
                ax[1,0].semilogx(concentrations,kdm_dict_avg[freqs[j]]['calc-kdm_num'][freqs[k]]*100,color = colorchoose,linestyle = 'none', marker = '.')
                ax[1,0].semilogx(concentrations,kdm_dict_avg[freqs[j]]['fit-kdm_num'][freqs[k]]*100,color = colorchoose)
                # KDM
                ax[0,1].plot(concentrations,kdm_dict_avg[freqs[j]]['calc-kdm'][freqs[k]]*100,color = colorchoose,linestyle = 'none', marker = '.')
                ax[0,1].plot(concentrations,kdm_dict_avg[freqs[j]]['fit-kdm'][freqs[k]]*100,color = colorchoose)
                ax[1,1].semilogx(concentrations,kdm_dict_avg[freqs[j]]['calc-kdm'][freqs[k]]*100,color = colorchoose,linestyle = 'none', marker = '.')
                ax[1,1].semilogx(concentrations,kdm_dict_avg[freqs[j]]['fit-kdm'][freqs[k]]*100,color = colorchoose)
                
        # ax[1,0].set_ylim([np.amin(kdm_dict[freqs[j]]['fit-kdm'])*100 - 10,200])
        # ax[0,0].set_ylim([-0.5,.5])
        fig_amplitude_diff.text(0.5,0.04,label,ha='center')
        ax[0,0].set_ylabel('Signal Change (%)')
        ax[1,0].set_ylabel('Signal Change (%)')
        ax[0,0].set_title('KDM Numerator')
        ax[0,1].set_title('KDM')
        # if clinical_low:
        #     for idx, axis in enumerate(ax.reshape(-1)):
        #         clinical = patches.Rectangle((clinical_low,axis.get_ylim()[0]), clinical_high-clinical_low, axis.get_ylim()[1]-axis.get_ylim()[0], color = 'r',alpha=0.2)
        #         axis.add_patch(clinical)
        fig_amplitude_diff.savefig(figdir + filestart + '_KDM-avg_' + freqs[j] + 'Hz.png')