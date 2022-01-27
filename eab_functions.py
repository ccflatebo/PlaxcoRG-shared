# -*- coding: utf-8 -*- 
""" 
Created on Wed Sep 15 18:12:43 2021 
 
@author: ccflatebo 
""" 
import tkinter as Tk 
from tkinter import filedialog as tkFileDialog 
import numpy as np
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from scipy import sparse
from scipy.linalg import cholesky
from scipy.sparse.linalg import spsolve
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter
from scipy import fft
import eab_functions as ef
import os
import re

import warnings
warnings.filterwarnings('ignore')
 
# File Selection 
def getfile(directory, name, filetypestoshow = None): # gets a single file for analysis 
    root = Tk.Tk() 
    if filetypestoshow:
        filetypes_list = [filetypestoshow,('All files','*.*')]
        files = tkFileDialog.askopenfilename(initialdir = directory, title = name, filetypes = filetypes_list) 
    else:
        files = tkFileDialog.askopenfilename(initialdir = directory, title = name)
    root.destroy() 
    return files 
 
def getfiles(directory, name, filetypestoshow = None): # gets multiple files for analysis 
    root = Tk.Tk() 
    if filetypestoshow:
        if type(filetypestoshow) == tuple:
            filetypes_list = [filetypestoshow]
        else:
            filetypes_list = []
            for types in filetypestoshow:
                filetypes_list.append(types)
        filetypes_list.append(('All files','*.*'))
        files = tkFileDialog.askopenfilenames(initialdir = directory, title = name, filetypes = filetypes_list) 
    else:
        files = tkFileDialog.askopenfilenames(initialdir = directory, title = name)
    # files = files[1:] + tuple([files[0]]) # corrects for ordering 
    root.destroy() 
    return files 

def natural_sort(names):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)',key)]
    return sorted(names,key=alphanum_key)

def select_dir(directory, name): # gets a single file for analysis 
    root = Tk.Tk() 
    selected_directory = tkFileDialog.askdirectory(title=name,initialdir = directory) + '/'
    root.destroy() 
    return selected_directory

def scan_dir(directory):
    filenames = []
    for _,_, files in os.walk(directory):
        # path = root.split(os.sep)
        # print((len(path) - 1) * '---', os.path.basename(root))
        for file in files:
            if not file.endswith('.bin'):
                filenames.append(file)
    filenames = ef.natural_sort(filenames)
    return filenames

def check_dir_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
 
def read_file(file): 
    file_read = open(file,'r') 
    lines = file_read.readlines() 
    line_breaks = [] 
    for idx, line in enumerate(lines): 
        if line == '\n': 
            line_breaks.append(idx) 
    file_read.close() 
    # print(line_breaks) 
    if len(line_breaks) > 3: 
        # print('someone saved the results section') 
        # instrument_info = pd.read_csv(file,delimiter = '\t', 
        #                               nrows=line_breaks[0],header=None) 
        # experiment_info = pd.read_csv(file,delimiter = '=', 
        #                               skiprows=line_breaks[0], 
        #                               nrows=line_breaks[1]-1 - line_breaks[0], 
        #                               header=None) 
        data = pd.read_csv(file,delimiter = ',',skiprows=line_breaks[-2]) 
        return data#, experiment_info, instrument_info
    elif len(line_breaks) == 3: 
        # print('3 breaks')
        instrument_info = pd.read_csv(file,delimiter = '\t', 
                                      nrows=line_breaks[0],header=None) 
        experiment_info = pd.read_csv(file,delimiter = '=', 
                                      skiprows=line_breaks[0], 
                                      nrows=line_breaks[1]-1 - line_breaks[0], 
                                      header=None) 
        data = pd.read_csv(file,delimiter = ',',skiprows=line_breaks[-2]) 
        return data, experiment_info, instrument_info 
    elif len(line_breaks) == 2: 
        instrument_info = pd.read_csv(file,delimiter = '\t', 
                                      nrows=line_breaks[0],header=None) 
        data = pd.read_csv(file,delimiter = ',',skiprows=line_breaks[-1]) 
        return data, instrument_info 
    else: 
        # print('No Headers') 
        data = pd.read_csv(file,delimiter = ',',header=None) 
        return data 
 
def select_swv_data(data,string,remove=[]): 
    if remove:
        remove = remove
    else:
        remove = []
    potentials = data.iloc[:,0]
    data_d = pd.DataFrame()
    # data_d[data.columns[0]] = data[data.columns[0]] 
    num_electrodes = (len(data.columns)-1)//3
    # print(num_electrodes)
    if string == 'd':
        col_num = 1
    elif string == 'f':
        col_num = 2
    else:
        col_num = 3
    elecnum = 0
    for i in range(0,num_electrodes):
        if i+1 not in remove:
            data_d[elecnum] = data.iloc[:,col_num+3*(i)]
            elecnum += 1
    # for column in data.columns: 
    #     if string in column.lower(): 
    #         data_d[column] = data[column] 
    return potentials, data_d 

def convert_swv(file,convertdir):
    filename = file.split('/')[-1].rstrip('.txt')
    if not os.path.exists(convertdir + filename + '.pkl'):
        output = read_file(file)
        num_outputs = np.shape(output)[0]
        # print(num_outputs)
        # print(num_outputs)
        if type(output) == tuple:
            output[0].to_pickle(convertdir + filename + '.pkl')
            for i in range(1,num_outputs):
                if output[i].loc[:,0].str.contains('Quiet').any():
                    output[i].to_pickle(convertdir + filename + '_echem-params.pkl')
                elif output[i].loc[:,0].str.contains('File').any():
                    output[i].to_pickle(convertdir + filename + '_metadata.pkl')
        else:
            output.to_pickle(convertdir + filename + '.pkl')
        return output

def fit_swvcurrent(potentials,data,guesses, maxiter = 10000):
    c, _ = curve_fit(gauss,potentials,
                      data,
                      p0=guesses,
                      maxfev = maxiter)
    fit = gauss(potentials,*c)
    residuals = ef.r2_calc(data, fit)
    return c, fit, residuals

def fit_swv_trace(potentials,smoothed_data,peakidx_avg, base_distance = 10, base_width = 5):
    params = dict()
    peaks, peak_deets = find_peaks(smoothed_data,width = base_width,distance = base_distance)
    if np.shape(peaks)[0] == 1: # checks for multiple peaks selected
        peak_range = [int(peaks[0] - peak_deets['widths'][0]/2),
                      int(peaks[0] + peak_deets['widths'][0]/2)]
    elif np.shape(peaks)[0] > 1:
        peak_idx = find_nearest(peaks, peakidx_avg)
        peaks = np.asarray([peaks[ef.find_nearest(peaks, peakidx_avg)]])
        peak_range = [int(peaks[0] - peak_deets['widths'][peak_idx]/2),
                      int(peaks[0] + peak_deets['widths'][peak_idx]/2)]
    else:
        peaks = np.asarray([int(peakidx_avg)])
        peak_range = [peaks[0] - 15,
                      peaks[0] + 15]
    peak_range.sort() # makes sure that the range is in ascending order
    if peak_range[-1] > np.shape(potentials)[0]: # correction to make sure peak_range isn't bigger than array size
        peak_range[-1] = np.shape(potentials)[0]
    if peak_range[0] < 0: # correction to make sure peak_range isn't negative
        peak_range[0] = 0
    guesses = [smoothed_data[peaks[0]], # a
               potentials[peaks[0]], # b
               potentials[peak_range[-1]] - potentials[peak_range[0]]] # c
    try: # iteration of fitting
        c, yprime, residuals = fit_swvcurrent(potentials, smoothed_data, guesses)
        params['amplitude'] = c[0]
        params['mean'] = c[1]
        params['width'] = c[2]
        params['rsquare'] = residuals[0]
        destroy_file = 0
        while c[1] > np.amax(potentials) or c[1] < np.amin(potentials): # makes sure b is within the potential range
            if destroy_file >= 10: # limits this process to 10 iterations
                print('Unable to fit mean within potential window')
                yprime = np.full_like(smoothed_data,np.nan)
                params['amplitude'] = np.nan
                params['mean'] = np.nan
                params['width'] = np.nan
                params['rsquare'] = np.nan
                return yprime, params
            else:
                guesses[1] = potentials.iloc[int(np.shape(smoothed_data)[0]/2)]
                guess_width = 0.1
                peak_range = [find_nearest(potentials,
                                        guesses[1]
                                        - guess_width) + destroy_file,
                              find_nearest(potentials,
                                        guesses[1]
                                        + guess_width) - destroy_file]
                peak_range.sort()
                if peak_range[-1] > np.shape(potentials)[0]:
                    peak_range[-1] = np.shape(potentials)[0]
                if peak_range[0] < 0:
                    peak_range[0] = 0
                c, yprime, residuals = fit_swvcurrent(potentials.iloc[peak_range[0]:peak_range[-1]], 
                                smoothed_data[peak_range[0]:peak_range[-1]], 
                                guesses)
                yprime = gauss(potentials,*c)
                if peak_range[0] + 1 == peak_range[-1] - 1:
                    destroy_file = 10
                else:
                    destroy_file += 1
                params['amplitude'] = c[0]
                params['mean'] = c[1]
                params['width'] = c[2]
                params['rsquare'] = residuals[0]
    except RuntimeError:
        try:
            c, yprime, residuals = fit_swvcurrent(potentials.iloc[peak_range[0]:peak_range[-1]], 
                               smoothed_data[peak_range[0]:peak_range[-1]], 
                               guesses)
            yprime = gauss(potentials,*c)
            params['amplitude'] = c[0]
            params['mean'] = c[1]
            params['width'] = c[2]
            params['rsquare'] = residuals[0]
        except RuntimeError:
            # print('No distinguishable signal for File ' + freq[-1] + ', Electrode ' + str(j))
            yprime = np.full_like(smoothed_data,np.nan)
            params['amplitude'] = np.nan
            params['mean'] = np.nan
            params['width'] = np.nan
            params['rsquare'] = np.nan
    return yprime, params

# Curve 
def gauss(x,a,b,c): 
    return a*np.exp(-(x-b)**2/(2*c**2)) 

def r2_calc(raw,fit):
    residuals = raw - fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((raw - np.mean(raw))**2)
    r_squared = 1 - (ss_res/ss_tot)
    return r_squared, ss_res, ss_tot, residuals

def langmuir_isotherm(x,k,b,a):
    return a + (x*b)/(x + k)

def langmuir_isotherm_norm(x,k,b):
    return (x*b)/(x + k)

def hilllangmuir_isotherm(x,k,b,c,a):
    return a + (x**c*b)/(x**c + k)

def hilllangmuir_isotherm_norm(x,k,b,c):
    return (x**c*b)/(x**c + k)

def langmuir_fitting(signal_on, signal_off, concentrations,numerator = False,norm=False,hill=False):
    # kdm = kdm_calc(signal_on, signal_off)
    kdm = kdm_calc_true(signal_on, signal_off,signal_on[0],signal_off[0])
    kdm_parameters = dict()
    if numerator:
        data_grab = 1
    else:
        data_grab = 0
    if hill:
        if norm:
            c, _ = curve_fit(hilllangmuir_isotherm_norm, concentrations, kdm[data_grab], maxfev = 100000)
            fit_kdm = hilllangmuir_isotherm_norm(concentrations, *c)
            residuals = r2_calc(kdm[data_grab], fit_kdm)
            kdm_parameters['c'] = c[2]
        else:
            c, _ = curve_fit(hilllangmuir_isotherm, concentrations, kdm[data_grab], maxfev = 10000)
            fit_kdm = hilllangmuir_isotherm(concentrations, *c)
            residuals = r2_calc(kdm[data_grab], fit_kdm)
            kdm_parameters['a'] = c[3]
            kdm_parameters['c'] = c[2]
    else:
        if norm:
            c, _ = curve_fit(langmuir_isotherm_norm, concentrations, kdm[data_grab], maxfev = 10000)
            fit_kdm = langmuir_isotherm_norm(concentrations, *c)
            residuals = r2_calc(kdm[data_grab], fit_kdm)
        else:
            c, _ = curve_fit(langmuir_isotherm, concentrations, kdm[data_grab], maxfev = 10000)
            fit_kdm = langmuir_isotherm(concentrations, *c)
            residuals = r2_calc(kdm[data_grab], fit_kdm)
            kdm_parameters['a'] = c[2]
    kdm_parameters['b'] = c[1]
    kdm_parameters['Kd'] = c[0]
    kdm_parameters['rsquare'] = residuals[0]
    return kdm[data_grab], fit_kdm, kdm_parameters
 
# BG correction 
def kdm_calc(signal_on, signal_off): 
    numerator = signal_on-signal_off 
    denominator = (signal_on + signal_off)/2.0 
    return numerator/denominator, numerator, denominator 
def kdm_calc_true(signal_on, signal_off,signal_on_i,signal_off_i): 
    numerator = signal_on/signal_on_i-signal_off/signal_off_i
    denominator = (signal_on/signal_on_i + signal_off/signal_off_i)/2.0 
    return numerator/denominator, numerator, denominator 


def find_nearest(array, value): 
    array = np.asarray(array) 
    idx = (np.abs(array - value)).argmin() 
    return idx

def arpls(y, lam=1e7, ratio=0.05, itermax=100):
    r"""
    Baseline correction using asymmetrically
    reweighted penalized least squares smoothing
    Sung-June Baek, Aaron Park, Young-Jin Ahna and Jaebum Choo,
    Analyst, 2015, 140, 250 (2015)
    """
    N = len(y)
    D = sparse.eye(N, format='csc')
    D = D[1:] - D[:-1]  # numpy.diff( ,2) does not work with sparse matrix. This is a workaround.
    D = D[1:] - D[:-1]
    H = lam * D.T * D
    w = np.ones(N)
    for i in range(itermax):
        W = sparse.diags(w, 0, shape=(N, N))
        WH = sparse.csc_matrix(W + H)
        C = sparse.csc_matrix(cholesky(WH.todense()))
        z = spsolve(C, spsolve(C.T, w * y))
        d = y - z
        dn = d[d < 0]
        m = np.mean(dn)
        s = np.std(dn)
        wt = 1. / (1 + np.exp(2 * (d - (2 * s - m)) / s))
        if np.linalg.norm(w - wt) / np.linalg.norm(w) < ratio:
            break
        w = wt
    return z

def baseline_removal(data,print_baseline=False):
    baseline = arpls(data)
    data_corr = data - baseline
    if print_baseline:
        plt.figure()
        plt.plot(data,'k')
        plt.plot(baseline,'r',linestyle = 'dashed')
        plt.plot(data_corr,'r',marker='.',linestyle = 'none')
    return data_corr