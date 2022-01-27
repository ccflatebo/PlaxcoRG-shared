# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 11:50:27 2021

@author: ccflatebo
"""

from tkinter import filedialog
import tkinter as Tk 

import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl

import pandas as pd
# from scipy import sparse
# from scipy.linalg import cholesky
# from scipy.sparse.linalg import spsolve
# from scipy.optimize import curve_fit, least_squares
# from scipy.signal import find_peaks, savgol_filter, peak_widths
from scipy import fft
import eab_functions as ef
import os
import warnings
warnings.filterwarnings('ignore')

filedir = r'../Data/EChem/'
date_exp = input('Experiment Date: ') #'210914'

#%% Pick electrodes
electrodes_ignore = input('Electrodes to ignore? (ie 3, 4, 6): ')
if electrodes_ignore:
    electrodes_ignore = electrodes_ignore.split(',')
    electrodes_ignore = [int(x.strip()) for x in electrodes_ignore]

smoothing_bool = True
#%%
datadir = filedir + date_exp + '/'
rawdir = ef.select_dir(datadir,'Raw Data Files')
# rawdir = datadir + 'Raw/'
convertdir = datadir + 'Converted/'
analysisdir = datadir + datetime.datetime.today().strftime('%y%m%d') + '/'

figdir = filedir + date_exp + '/Figures/SWV/'
ef.check_dir_exist(figdir)
#%% Initialize Paths
ef.check_dir_exist(convertdir)
ef.check_dir_exist(analysisdir)

# Data Selection
swv_list = os.path.dirname(datadir) + '/swvdata.txt'
if not os.path.exists(swv_list): # If swvdata doesn't exist, autoselects for you
    filenames = list(ef.scan_dir(rawdir)) # autoscans
    if len(filenames) == 0:
        print('No Files')
    else: # autoreads from text file
        for i in range(0,len(filenames)):
            filenames[i] = filenames[i].rstrip('.txt')
            # filenames[i] = filenames[i].split('/')[-1]
        filetxt = open(swv_list,'w')
        filetxt.write('\n'.join(map(str,filenames)))
        filetxt.close()
else: # loads swvdata to create a list of filenames
    print('Loading SWV Data')
    filetxt = open(swv_list,'r')
    filenames = filetxt.read().splitlines()
    filetxt.close()
numfiles = len(filenames)
#%% Convert Files from text to pandas dataframes
# filenames = ['CF-03-002_5Hz_1','CF-03-002_5Hz_2','CF-03-002_5Hz_3']

for filename in filenames: # 51 seconds for 1728 files with headers
    if not os.path.exists(convertdir + filename + '.pkl'):
        if filename == filenames[0]:
            print('Converting Data')
        output = ef.convert_swv(rawdir + filename + '.txt',convertdir)
    
#%%

def bounded_fit():
    pass

def fft_swv(potentials, data, freq):
    freq = int(new_freq)
    total_time = np.abs(np.amax(potentials) - np.amin(potentials))/(freq * np.abs(potentials[1] - potentials[0]))
    n = len(potentials)
    # t = np.linspace(0,total_time,n,endpoint=False)
    dt = total_time/n
    freqs = (1/(dt*n)) * np.arange(n)
    # freq_idx = ef.find_nearest(freqs, freq)
    data_fft = fft.fft(np.array(swv_difference.iloc[:,0]))
    data_fft_abs = np.abs(data_fft)**2
    # plt.figure()
    # plt.plot(freqs,data_fft_abs)
    height = np.sum(data_fft_abs)/len(data_fft_abs)
    peak_freqs = find_peaks(data_fft_abs,height = height)
    # heights = peak_heights(data_fft_abs,peak_freqs[0])
    # peak_freqs = np.array(peak_freqs)
    # peak_freqs = [0,np.array(peak_freqs[0])]
    peak_freqs = np.insert(peak_freqs[0],0,0)
    # plt.plot(freqs[peak_freqs],data_fft_abs[peak_freqs],'x',linestyle='none')
    for i in range(0,len(peak_freqs)):
        data_fft[peak_freqs[i]] = data_fft[peak_freqs[i]]*False
        # lowlim = data_fft[peak_freqs[0][i]] - widths[0][i]
        # highlim = data_fft[peak_freqs[0][i]] + widths[0][i]
        # for j in range(0,len(data_fft)):
        #     if j >= lowlim and j <= highlim:
        #         data_fft[j] = data_fft[j]*False
    # for i in range(0,len(data_fft)):
    #     if i >= peak_freqs[0]:
    #         data_fft[i] = data_fft[i] * True
    #     else:
    #         data_fft[i] = data_fft[i] * False
    data_filt = np.real(fft.ifft(data_fft))
    return data_filt



# Set up empty datasets

current_freq = 0
dict_freq = {}
for i, filename in enumerate(filenames):
    # Determine the frequency of the file !!Potential Bug!! requires it to be in filename
    freq = filename.split('_')
    
    if os.path.exists(convertdir + filename + '_echem-params.pkl'):
        echem_saved = True
        echem_params = pd.read_pickle(convertdir + filename + '_echem-params.pkl')
        echem_params = echem_params.set_index(echem_params.iloc[:,0]).iloc[:,1]
        new_freq = str(int(echem_params['Frequency (Hz) ']))
    else:
        echem_saved = False
        for x in freq:
            if 'Hz' in x:
                new_freq = x.rstrip('Hz')
                break
    # Load converted data and metadata (if it exists)
    data = pd.read_pickle(convertdir + filename + '.pkl')
    # Logic that allows for the saving of the data corresponding to each frequency
    if current_freq != new_freq and i != 0: # i != 0 makes sure you don't try to save an empty dataset or assign an empty dataset to the dictionary
        dict_freq[current_freq] = dict_freq_frames
        np.save(analysisdir + freq[0] + '_' + current_freq + 'Hz_fanalysis',dict_freq_frames)
    if current_freq != new_freq: # creates a new dictionary if it's a new frequency
        dict_freq_frames = {}
        peakidx_avg = np.shape(data)[0]/2 # initial guess of what E0 should be
        print('Currently processing ' + str(new_freq) + ' Hz, File ' + str(i + 1) + ' of ' + str(numfiles)) # not necessary but gives you an idea of where you are in the analysis
    potentials, swv_difference = ef.select_swv_data(data, 'd',remove = electrodes_ignore) # selects difference data (forward and reverse are available)
    swv_difference_bgcorr = pd.DataFrame().reindex_like(swv_difference) # makes an empty dataset that has same headers as the raw data
    # potentials = swv_difference.iloc[:,0] # assigns applied potentials to a variable
    # swv_difference_bgcorr.iloc[:,0] = swv_difference.iloc[:,0] # makes sure data has applied potentials when you save it
    yprime = pd.DataFrame() # empty dataframe to store the fits
    params = pd.DataFrame(columns = ['amplitude','mean','width','rsquare']) # empty dataframe to store the fit parameters
    col_headers = swv_difference.columns # grabs column headers so we can get the number of electrodes
    numelectrodes = len(col_headers) 
    colors = plt.cm.jet(np.linspace(0,1,numelectrodes)) # lets us give each electrode a difference color
    # fig = plt.figure()
    # plt.title(filename)
    if numelectrodes == 1:
        base_distance = 10
        base_width = 5
        
        swv_difference_bgcorr.iloc[:,0] = fft_swv(potentials,swv_difference.iloc[:,0], int(new_freq))
        swv_difference_bgcorr.iloc[:,0] = ef.baseline_removal(swv_difference_bgcorr.iloc[:,0])
        
        # swv_difference_bgcorr.iloc[:,0] = swv_difference.iloc[:,0]
        yprime_temp, params_temp = ef.fit_swv_trace(potentials, swv_difference_bgcorr.iloc[:,0], peakidx_avg)
        yprime = yprime_temp
        params = params.append(params_temp, ignore_index = True)
        # plt.plot(potentials,yprime,color=colors[0])
        # plt.plot(potentials,swv_difference_bgcorr.iloc[:,0],marker = '.')
        # plt.plot(params.loc[0,'mean'],params.loc[0,'amplitude'],color=colors[0],marker = 'x',markersize = 10)
        # plt.vlines([lowbound,highbound],0,peak_deets['width_heights'][0],'gray',linestyle = 'dashed')
        if current_freq != new_freq:
            peakidx_avg = ef.find_nearest(potentials, params.loc[0,'mean'])
        else:
            peakidx_avg = (peakidx_avg + ef.find_nearest(potentials, params.loc[0,'mean']))/2
    else:
        k = 0
        for j in range(0,len(col_headers)):
            # electrode_num = j - 1
            # if j+1 in electrodes_ignore:
            #     # print('Ignore Electrode ' + str(k+1))
            #     continue
            base_distance = 10
            base_width = 5
            # swv_difference_bgcorr.iloc[:,j] = fft_swv(potentials,swv_difference.iloc[:,j], int(new_freq))
            # swv_difference_bgcorr.iloc[:,j] = ef.baseline_removal(swv_difference_bgcorr.iloc[:,j])
            swv_difference_bgcorr.iloc[:,j] = ef.baseline_removal(swv_difference.iloc[:,j])
            # swv_difference_bgcorr.iloc[:,j] = swv_difference.iloc[:,j]
            yprime_temp, params_temp = ef.fit_swv_trace(potentials, swv_difference_bgcorr.iloc[:,j], peakidx_avg)
            yprime[j] = yprime_temp
            params = params.append(params_temp, ignore_index = True)
            # plt.plot(potentials,yprime_temp,color=colors[j])
            # plt.plot(potentials,swv_difference_bgcorr.loc[:,j],color=colors[j],marker = '.', linestyle = 'none')
            # plt.plot(params.loc[j,'mean'],params.loc[j,'amplitude'],color=colors[j],marker = 'x',markersize = 10)
            # plt.vlines([lowbound,highbound],0,peak_deets['width_heights'][0],'gray',linestyle = 'dashed')
            if current_freq != new_freq:
                peakidx_avg = ef.find_nearest(potentials, params.loc[j,'mean'])
            else:
                peakidx_avg = (peakidx_avg + ef.find_nearest(potentials, params.loc[j,'mean']))/2
            k += 1
    dict_freq_frames[freq[-1]] = dict()
    dict_freq_frames[freq[-1]]['yprime'] = yprime
    dict_freq_frames[freq[-1]]['raw'] = swv_difference
    dict_freq_frames[freq[-1]]['bgcorr'] = swv_difference_bgcorr
    dict_freq_frames[freq[-1]]['params'] = params
    dict_freq_frames[freq[-1]]['potentials'] = potentials
    dict_freq_frames[freq[-1]]['mtime'] = os.path.getmtime(rawdir+filename+'.bin')
    dict_freq_frames[freq[-1]]['times'] = np.linspace(0, np.abs(potentials.iloc[-1] - potentials.iloc[0])/(float(new_freq)*(potentials.iloc[0]-potentials.iloc[1])),len(potentials))
    if echem_saved:
        dict_freq_frames[freq[-1]]['echem_params'] = echem_params
        
    current_freq = new_freq
    # ax = swv_difference.plot(0,color = colors,marker = '.',linestyle='none')
    # swv_difference_bgcorr.plot(0,color = colors,marker = 'x',linestyle='none',ax=ax)
    # for j in range(1,len(col_headers)):
    #     plt.plot(swv_difference.iloc[:,0],yprime.loc[:,j],color=colors[j-1])
    # plt.title(filename)
# baseline_corr = arpls(swv_difference[swv_difference.columns[1]])
dict_freq[current_freq] = dict_freq_frames
np.save(analysisdir + freq[0] + '_' + current_freq + 'Hz_fanalysis',dict_freq_frames)
np.save(analysisdir + date_exp + '_' + freq[0] + '_combined',dict_freq)
del(dict_freq_frames)
#%%
frequencies = list(dict_freq.keys())
for i in frequencies:
    data_freq = dict_freq[i]
    frames = list(data_freq.keys())
    frame_colors = plt.cm.copper_r(np.linspace(0,1,len(frames)))
    cmap = mpl.cm.copper_r
    norm = mpl.colors.Normalize(vmin = 0,vmax=len(frames))
    if numelectrodes > 3:
        row_plots = 2
        column_plots = int(np.ceil(numelectrodes/row_plots))
    else:
        row_plots = 1
        column_plots = numelectrodes
    fig_avg, ax = plt.subplots(1,1)
    plt.title(filename.split('_')[0] + ': ' + i + ' Hz Average')
    fig_electrodes, axes = plt.subplots(row_plots,column_plots,sharey=True,figsize = (16,10))
    fig_electrodes.suptitle(filename.split('_')[0] + ': ' + i + ' Hz')
    if numelectrodes != 1:
        axreshape = axes.reshape(-1)
    else:
        axreshape = axes
    for j, frame in enumerate(frames):
        norm_data = data_freq[frame]['bgcorr']/data_freq['1']['yprime'].max()
        norm_yprime = data_freq[frame]['yprime']/data_freq['1']['yprime'].max()
        ax.plot(data_freq[frame]['potentials'],norm_data.mean(axis=1),marker = '.',markersize=1,alpha=.25, color =frame_colors[j])
        ax.plot(data_freq[frame]['params']['mean'].mean(),data_freq[frame]['params']['amplitude'].mean()/data_freq['1']['params']['amplitude'].mean(),color=frame_colors[j],marker = 'x',markersize = 10)
        if len(col_headers) != 1:
            for k in col_headers:
                axreshape[k].set_title('Electrode #' + str(k+1))
                axreshape[k].plot(data_freq[frame]['potentials'],norm_yprime[k],color=frame_colors[j])
                axreshape[k].plot(data_freq[frame]['potentials'],norm_data[k],marker = '.', color=frame_colors[j],markersize=1,alpha=.25)
                axreshape[k].plot(data_freq[frame]['params']['mean'][k],data_freq[frame]['params']['amplitude'][k]/data_freq['1']['params']['amplitude'][k],color=frame_colors[j],marker = 'x',markersize = 10)
            axreshape[0].set_ylim(0,2)
        else:
            k=0
            axreshape.set_title('Electrode #' + str(k+1))
            axreshape.plot(data_freq[frame]['potentials'],norm_yprime,color=frame_colors[j])
            axreshape.plot(data_freq[frame]['potentials'],norm_data,marker = '.', color=frame_colors[j],markersize=1,alpha=.25)
            axreshape.plot(data_freq[frame]['params']['mean'],data_freq[frame]['params']['amplitude']/data_freq['1']['params']['amplitude'],color=frame_colors[j],marker = 'x',markersize = 10)
            axreshape.set_ylim([0,None])
    fig_electrodes.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap),label = 'Frames', ax = axreshape[len(axreshape)//2-1])
    fig_electrodes.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap),label = 'Frames', ax = axreshape[-1])
    fig_avg.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap),label = 'Frames')
    ax.set_ylim([0,None])
    fig_electrodes.tight_layout()
    
    fig_electrodes.text(0.5,0,'Potential (V)',ha='center')
    fig_electrodes.text(0,0.5,'Normalized Current (a.u.)',va='center',rotation='90')
    fig_avg.tight_layout()
    
    ax.set_xlabel('Potential (V)')
    ax.set_ylabel('Normalized Current (a.u.)')
    fig_avg.savefig(figdir + filename.split('_')[0] + '_' + i + 'Hz_avg.png')
    # fig_electrodes.tight_layout()
    fig_electrodes.savefig(figdir + filename.split('_')[0] + '_' + i + 'Hz_individual.png')
    # plt.close('all')
del(dict_freq)
#%% Plotting
# Colors
data = np.load(analysisdir + date_exp + '_' + filename.split('_')[0] + '_combined.npy', allow_pickle = True).item()
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
concentrations = np.linspace(0,num_frames,num_frames)
num_electrodes_short = np.shape(electrodes_include)[0]
if num_electrodes_short > 3:
    row_plots = 2
    column_plots = int(np.ceil(num_electrodes_short/row_plots))
else:
    row_plots = 1
    column_plots = num_electrodes_short
frame_data_include = np.zeros((len(electrodes_include),num_frames,num_freq))
frame_e0_include = np.zeros_like(frame_data_include)
for i, val in enumerate(electrodes_include):
    frame_data_include[i,:,:] = frame_data[int(val),:,:]
    frame_e0_include[i,:,:] = frame_data[int(val),:,:]
color_frame = plt.cm.copper_r(np.linspace(0,1,len(frames)))
color_electrode = plt.cm.get_cmap('Paired').colors
cmap_frame = mpl.cm.copper_r
norm_frame = mpl.colors.Normalize(vmin = 0,vmax=len(frames))
label = 'Frames'
# cmap_frame = mpl.cm.copper_r
# norm_frame = mpl.colors.Normalize(vmin = 0,vmax=np.amax(concentrations))
# Gain
fig_gain, ax = plt.subplots(row_plots,column_plots,figsize = (10,6),sharex = True, sharey = False)
ax_list = ax.reshape(-1)
fig_gain.suptitle('Gain at each Frequency')

for i in range(0,numelectrodes):
    ax_list[i].set_title('Electrode #' + str(i + 1))
    for j in range(0,len(frames)):
        colorchoose = color_frame[j]
        ax_list[i].semilogx(freqs_int,(frame_data_include[i,j,:]-frame_data_include[i,0,:])/frame_data_include[i,0,:]*100,marker='.',color = colorchoose)

# ax.annotate(r'$\dfrac{C_n - C_0}{C_0} \times 100$',xy = (.25,.99),xycoords = 'axes fraction',va = 'top', ha = 'left')
fig_gain.text(0.5,0,'Frequency (Hz)',ha='center')
fig_gain.text(0,0.5,'% Signal Change',va='center',rotation='90')
plt.colorbar(mpl.cm.ScalarMappable(norm=norm_frame,cmap=cmap_frame),label = label, ax = ax_list[-1])
plt.colorbar(mpl.cm.ScalarMappable(norm=norm_frame,cmap=cmap_frame),label = label, ax = ax_list[len(ax_list)//2-1])
plt.tight_layout()
plt.savefig(figdir + filename.split('_')[0] + '_gain-indelectrodes.png',pady = 10)
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
plt.savefig(figdir + filename.split('_')[0] + '_lovric-indelectrodes.png',pady = 10)