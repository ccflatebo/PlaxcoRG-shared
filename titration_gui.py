# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 16:48:38 2021

@author: ccflatebo
"""
import os
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
import pandas as pd

class params:
    def __init__(self):
        self.datadir = '../Data/Test/' # where you want your file saved
        self.startfilename = 'CF-01-001' # name of your saved files
        """ Concentration Parameters """
        self.base = 10
        # Choose 'u' for micromolar, 'n' for nanomolar and 'IU' for hormones FML
        self.stock_conc_input = 0.02 # desired stock target molecule concentration (M) or IU for hormones
        self.unit_stock = 0 # Unit for highest concentration desired
        self.conc_min_input = 1 # this is the lowest concentration to use and is 10^-3 of chosen unit
        self.conc_max_input = 1000 # this is highest concentration to use and matches chosen unit
        self.unit_min = 3 # Unit for highest concentration desired
        self.unit_max = 2 # Unit for highest concentration desired
        """ Shot Glass Parameters """
        self.start_vol = 0.025 # Starting volume (L) of buffer in shot glass
        self.remove = 0 # 'y' for removing volume, 'n' for only adding, 'a' for autotitrator
        self.end_vol = 0 # size of shot glass volume (L) allowed (throws an error if you go too high)
        ## This is the range of volumes you want to add. I guarantee the smallest volume added will always be the lower limit
        ## *TO PREVENT AN INFINITE LOOP*  although everything I've test so far works...I don't guarantee the largest volume added
        self.smallest_vol_added = 2 # uL Smallest volume you think the pipette is accurate for
        self.largest_vol_added = 1000 # uL Largest volume you'd like to add to the shot glass
        """ Titration Parameters """
        self.num_measurements = 50 # total number of measurements you want to take
        ## Ratio frac_low:frac_mid:frac_high - Default 1:1:1 ##
        self.frac_low = 1 # this is the ratio of measurements in the lower range
        self.frac_mid = 1 # this is the ratio of measurements in the mid range
        self.frac_high = 1 # this is the ratio of measurements in the upper range
        
        """ Optional Kd Parameters """
        ## Do you know your Kd? if not please don't touch me ##
        self.k_d_known = False
        self.k_d_input = 60 # proposed Kd, use [] if unknown and numbers (like 60 uM) if known. This value uses the units you chose above at "unit"
        self.unit_kd = 3
        self.k_d_spread = 50 # spread of points surrounding Kd i.e. +/- 50 uM 
        #* I highly advise keeping k_d_spread < k_d if you want to use this functionality
        self.header_array = ['Desired Conc (M)', 'Stock Conc (M)', 'Added Volume (uL)',
                        'Removed Volume (uL)','Final Conc after Added Volume (M)']
        # if you do hormone stuff, this is useful, if not, ignore
        """ Hormone Annoyance """
        self.hormone = False
        self.conv_IU_input = 5.6 # only used for hormones
        self.conv_unit = 3 # only used for hormone IU
        self.run_calc()
    def run_calc(self):
        self.conc_conv()
        self.dist_points()
        self.calc_conc()
    def conc_conv(self):
        self.conv_IU = self.conv_IU_input * self.base ** (self.conv_unit*-3)
        if self.hormone:
            self.conc_min = self.conc_min_input * self.base ** (self.unit_min*-3)*self.conv_IU
            self.conc_max = self.conc_max_input * self.base ** (self.unit_max*-3)*self.conv_IU
            self.stock_conc = self.stock_conc_input * self.base ** (self.unit_stock*-3)*self.conv_IU
        else:
            self.conc_min = self.conc_min_input * self.base ** (self.unit_min*-3)
            self.conc_max = self.conc_max_input * self.base ** (self.unit_max*-3)
            self.stock_conc = self.stock_conc_input * self.base ** (self.unit_stock*-3)
        self.k_d = self.k_d_input * self.base ** (self.unit_kd * -3)
    def dist_points(self):
        # turns inputs into fraction form
        frac_denominator = float(self.frac_low + self.frac_mid + self.frac_high)
        self.num_measurements_low = int(np.round(self.frac_low/frac_denominator * self.num_measurements))
        self.num_measurements_mid = int(np.round(self.frac_mid/frac_denominator * self.num_measurements))
        self.num_measurements_high = int(np.round(self.frac_high/frac_denominator * self.num_measurements))
        measurements_total = self.num_measurements_low + self.num_measurements_mid + self.num_measurements_high
        measurements_diff = measurements_total - self.num_measurements
        # makes sure you have the right number of measurements
        if measurements_diff > 0: 
            # print('Too Many Measurements')
            if (measurements_diff)%2 == 0:
                self.num_measurements_low-=measurements_diff//2
                self.num_measurements_high-=measurements_diff//2
            else:
                self.num_measurements_low-=measurements_diff%2
                self.num_measurements_high-=measurements_diff//2
        elif measurements_diff < 0:
            # print('Too few measurements')
            measurements_diff = np.abs(measurements_diff)
            # print((measurements_diff)%2)
            if (measurements_diff)%2 == 0:
                self.num_measurements_low-=measurements_diff//2
                self.num_measurements_high-=measurements_diff//2
            else:
                self.num_measurements_low+=measurements_diff%2
                self.num_measurements_high+=measurements_diff//2
    
    def calc_conc(self):
        if not self.k_d_known:
            if self.num_measurements%3 == 0: # deals with odd number of measurements chosen
                low_half = self.num_measurements//3
                upper_half = self.num_measurements//3
            else:
                low_half = self.num_measurements//3
                upper_half = self.num_measurements//3+1
            conc_all_temp = np.geomspace(self.conc_min,self.conc_max,self.num_measurements)
            conc_low = np.geomspace(self.conc_min,conc_all_temp[low_half],self.num_measurements_low,endpoint=False)
            conc_high = np.geomspace(conc_all_temp[-upper_half],self.conc_max,self.num_measurements_high,endpoint=True)
            conc_mid = np.geomspace(conc_all_temp[low_half],np.amin(conc_high),self.num_measurements_mid,endpoint=False)
            self.conc_all = np.unique(np.concatenate((conc_low,conc_mid,conc_high),axis = 0))
        else:
            if self.num_measurements%2 == 0: # deals with odd number of measurements chosen
                low_half = self.num_measurements//2
                upper_half = self.num_measurements//2
            else:
                low_half = self.num_measurements//2 + 1
                upper_half = self.num_measurements//2
            if self.num_measurements_mid%2 == 0: # deals with odd number of measurements chosen
                low_half_mid = self.num_measurements_mid//2
            else:
                low_half_mid = self.num_measurements_mid//2 + 1
            # conc_all_temp = np.geomspace(self.conc_min,self.conc_max,self.num_measurements)
            conc_low_temp = np.geomspace(self.conc_min,self.k_d,low_half,endpoint=False)
            conc_high_temp = np.geomspace(self.k_d,self.conc_max,upper_half)
            
            conc_low = np.geomspace(self.conc_min,np.mean(conc_low_temp),self.num_measurements_low,endpoint=False)
            conc_high = np.geomspace(np.mean(conc_high_temp),self.conc_max,self.num_measurements_high)
            conc_mid_low = np.geomspace(np.mean(conc_low_temp),self.k_d,low_half_mid,endpoint=False)
            conc_mid_high = np.geomspace(self.k_d,np.mean(conc_high_temp),self.num_measurements_mid//2,endpoint=False)
            self.conc_all = np.unique(np.concatenate((conc_low,conc_mid_low,conc_mid_high,conc_high),axis = 0))
            # self.conc_all = np.unique(np.concatenate((conc_low_temp,conc_high_temp),axis = 0))

    def plot_conc(self, ax):
        for ax_num in ax.figure.axes:
            if ax_num is ax:
                ax_num.clear()
            else:
                ax_num.remove()
        ax.set_ylabel('[M]')
        if self.hormone:
            axy = ax.twinx()
            axy.semilogy(self.conc_all / (self.conv_IU),'o',color = 'm',fillstyle= 'none')
            axy.set_ylabel('[IU]')
        if self.k_d_known:
            ax.hlines(self.k_d,0,self.num_measurements,colors='r',linestyle = 'dashed')
            ax.annotate('Selected $K_d$',(0,(self.k_d*1.05)),xycoords='data',color = 'r')
        ax.set_xlabel('Measurement #')
        ax.annotate('R1 Points',(0,np.amax(self.conc_all)*.99),xycoords = 'data')
        ax.annotate('R2 Points',(self.num_measurements_low+ 1,np.amax(self.conc_all)*.99),xycoords = 'data')
        ax.annotate('R3 Points',(self.num_measurements_low + self.num_measurements_mid+1,np.amax(self.conc_all)*.99),xycoords = 'data')
        ax.vlines([self.num_measurements_low,self.num_measurements_low+
                   self.num_measurements_mid],0,np.amax(self.conc_all),colors='k',linestyle = 'dotted')
        ax.semilogy(self.conc_all,'x')

    def create_array(self):
        self.df = pd.DataFrame()
        self.df['Desired Conc (M)'] = self.conc_all
        self.df['Stock [M]'] = np.nan
        self.df['Added Volume (uL)'] = np.nan
        self.df['Final [M]'] = np.nan
        if self.remove != 0:
            self.df['Remove Volume (uL)'] = np.nan
        prefix = 0
        conc_start = 0
        rounding = 1
        conv = 6
        added_vol = []
        vol = self.start_vol
        for idx, x in enumerate(self.conc_all):
            # Calculates what added volume you would use for the stock concentration
            add_vol_L = c2v2(self.stock_conc,conc_start,x,vol)
            add_vol = np.round(add_vol_L * self.base ** conv,decimals = rounding)
            # Limits added volume to the range you provided
            while add_vol > float(self.largest_vol_added):
                prefix+=1
                add_vol = np.round(c2v2(self.stock_conc*self.base**prefix,conc_start,x,vol)* self.base ** conv,decimals = rounding)
            while add_vol < float(self.smallest_vol_added):
                prefix-=1
                add_vol = np.round(c2v2(self.stock_conc*self.base**prefix,conc_start,x,vol)*  self.base ** conv,decimals = rounding)
            if add_vol >= 200:
                add_vol = np.round(add_vol)
            # Logs the data in an array so you know what to add
            self.df['Stock [M]'].iloc[idx] = self.stock_conc * self.base**prefix
            self.df['Added Volume (uL)'].iloc[idx] = add_vol
            vol+=add_vol*self.base **(-conv) # makes sure the volume is updated accordingly
            self.df['Final [M]'].iloc[idx] = c2v2_verify(self.df['Stock [M]'].iloc[idx],conc_start, self.df['Added Volume (uL)'].iloc[idx]*self.base**(-conv), vol) # Calculates true concentration at the titration point
            # If you're using the autotitrator, removes for every point
            if self.remove == 2:
                total_vol_added = vol - self.start_vol
                self.df['Remove Volume (uL)'].iloc[idx] = total_vol_added*10**6
                vol-=total_vol_added
            # if you want to remove volume but not do it at every point
            elif self.remove == 1 and vol/self.start_vol > 1.01: # only tells you to remove volume if you've gone above 1% of the total volume
                total_vol_added = vol - self.start_vol
                self.df['Remove Volume (uL)'].iloc[idx] = total_vol_added*10**6 # logs the amount needed to be removed
                vol-=total_vol_added
            # Only for hormone calculations
            conc_start = x
            prefix = 0
        if self.hormone:
            self.df['[IU]'] = self.df['Final [M]'] / (self.conv_IU)
            self.df['Stock [IU]'] = self.df['Stock [M]'] / (self.conv_IU)
        unique_stocks = np.unique(self.df['Stock [M]'])
        if self.hormone:
            unique_IU_stocks = np.unique(self.df['Stock [IU]'])
        total_vol_perstock = np.zeros_like(unique_stocks)
        for item in self.df.index.values.tolist():
            idx = np.where(unique_stocks==self.df['Stock [M]'][item])
            total_vol_perstock[idx] = total_vol_perstock[idx] + self.df['Added Volume (uL)'][item]
            
        output_txtfile = open(self.datadir + '/' + self.startfilename + '_stocks.csv','w')
        if self.hormone:
            output_txtfile.write('Stock Concentrations (M), Stock Concentrations (IU), Total Vol (uL)\n')
            for idx, val in enumerate(unique_stocks):
                output_txtfile.write(str(val) + ', ' + str(unique_IU_stocks[idx]) + ', ' + str(total_vol_perstock[idx]) + '\n')
        else:
            output_txtfile.write('Stock Concentrations (M), Total Vol (uL)\n')
            for idx, val in enumerate(unique_stocks):
                output_txtfile.write(str(val) + ', ' + str(total_vol_perstock[idx]) + '\n')
        
        
        output_txtfile.close()
        df1 = pd.DataFrame(np.zeros((1,len(self.df.columns))),columns = self.df.columns)
        self.df = df1.append(self.df,ignore_index = True)
        self.df.set_index(np.linspace(1,self.num_measurements + 1,self.num_measurements + 1))
        
    

    def write_txtfile(self):
        self.create_array()
        self.df.to_excel(self.datadir + '/' + self.startfilename +'_titration.xlsx',index = True,header = True)

# Functions for calulating concentrations and what to add
# def c2v2(stock_conc,start_conc,wanted_conc,vol):
#     final_conc = wanted_conc
#     return (final_conc * vol)/(stock_conc - final_conc)
# def c2v2_verify(stock_conc,start_conc,add_vol,total_vol):
#     final_conc = (stock_conc * add_vol - start_conc*total_vol)/total_vol
#     return (stock_conc * add_vol)/total_vol
# def c2v2(stock_conc,start_conc,wanted_conc,vol):
#     final_conc = wanted_conc - start_conc
#     return (final_conc * vol)/(stock_conc - final_conc)
# def c2v2_verify(stock_conc,start_conc,add_vol,total_vol):
#     final_conc = (stock_conc * add_vol - start_conc*total_vol)/total_vol
#     return (stock_conc * add_vol + start_conc * (total_vol - add_vol))/total_vol

##### THESE ARE THE CORRECT FUNCTIONS FOR CALCULATING CONCENTRATIONS
def c2v2(stock_conc,start_conc,final_conc,vol):
    return vol * (final_conc - start_conc)/(stock_conc - final_conc)
def c2v2_verify(stock_conc,start_conc,add_vol,total_vol):
    final_conc = (stock_conc * add_vol - start_conc*total_vol)/total_vol
    return (stock_conc * add_vol + start_conc * (total_vol - add_vol))/total_vol



def clicked():
    expdir = filedialog.askdirectory()
    directory.delete(0,'end')
    directory.insert(END,expdir)
    
def print_macro():
    global new_params
    new_params.datadir = directory.get()
    new_params.startfilename = ent_file.get()
    get_concsettings()
    new_params.write_txtfile()
    messagebox.showinfo(title = 'File Saved', message = 'File Location: ' + new_params.datadir + '\nFile Name: ' + new_params.startfilename)
    
def close_program():
    root.destroy()

def reset():
    set_defaults()
    
def set_defaults():
    global default_params, new_params
    ent_start_conc.delete(0,END)
    ent_start_conc.insert(0,default_params.stock_conc_input)
    ent_concmin.delete(0,END)
    ent_concmin.insert(0,default_params.conc_min_input)
    ent_concmax.delete(0,END)
    ent_concmax.insert(0,default_params.conc_max_input)
    ent_conv.delete(0,END)
    ent_conv.insert(0,default_params.conv_IU_input)
    ent_conv.configure(state='disabled')
    ent_start_vol.delete(0,END)
    ent_start_vol.insert(0,default_params.start_vol * 10**3)
    ent_numpoints.delete(0,END)
    ent_numpoints.insert(0, default_params.num_measurements)
    ent_numpoints_1.delete(0,END)
    ent_numpoints_1.insert(0, default_params.frac_low)
    ent_numpoints_2.delete(0,END)
    ent_numpoints_2.insert(0, default_params.frac_mid)
    ent_numpoints_3.delete(0,END)
    ent_numpoints_3.insert(0, default_params.frac_high)
    ent_kd.delete(0,END)
    ent_kd.insert(0,default_params.k_d_input)
    ent_kd.configure(state = 'disabled')
    ent_small_vol.delete(0,END)
    ent_small_vol.insert(0, default_params.smallest_vol_added)
    ent_large_vol.delete(0,END)
    ent_large_vol.insert(0, default_params.largest_vol_added)
    hormone.set(0)
    kd_known.set(0)
    drop_unitskd.configure(state = 'disabled')
    drop_units.set_menu(unit_options[0][2],*unit_options[0])
    drop_unitsstock.set_menu(unit_options[0][0],*unit_options[0])
    drop_unitslow.set_menu(unit_options[0][3],*unit_options[0])
    drop_unitsconv.configure(state='disabled')
    ent_conv.configure(state='disabled')
    # get_concsettings()
    new_params = params()
    new_params.plot_conc(ax)
    canvas.draw()

def numpoints_change():
    global new_params
    new_params.num_measurements = int(ent_numpoints.get())
    new_params.dist_points()
    new_params.calc_conc()
    new_params.plot_conc(ax)
    canvas.draw()
        
def numpoints_change_enter(event):
    numpoints_change()
    
def dist_change_1():
    global new_params
    new_params.frac_low = int(ent_numpoints_1.get())
    new_params.dist_points()
    new_params.calc_conc()
    new_params.plot_conc(ax)
    canvas.draw()
        
def dist_change_1_enter(event):
    dist_change_1()
    
def dist_change_2():
    global new_params
    new_params.frac_mid = int(ent_numpoints_2.get())
    new_params.dist_points()
    new_params.calc_conc()
    new_params.plot_conc(ax)
    canvas.draw()
        
def dist_change_2_enter(event):
    dist_change_2()
    
def dist_change_3():
    global new_params
    new_params.frac_high = int(ent_numpoints_3.get())
    new_params.dist_points()
    new_params.calc_conc()
    new_params.plot_conc(ax)
    canvas.draw()
        
def dist_change_3_enter(event):
    dist_change_3()

def hormone_isChecked():
    if hormone.get() == 0:
        new_params.hormone = False
        drop_units.set_menu(unit_options[0][2],*unit_options[0])
        drop_unitsstock.set_menu(unit_options[0][0],*unit_options[0])
        drop_unitslow.set_menu(unit_options[0][3],*unit_options[0])
        drop_unitsconv.configure(state='disabled')
        ent_conv.configure(state='disabled')
        get_concsettings()
        new_params.plot_conc(ax)
        canvas.draw()
    else:
        new_params.hormone = True
        drop_units.set_menu(unit_options[1][0],*unit_options[0])
        drop_unitsstock.set_menu(unit_options[1][0],*unit_options[0])
        drop_unitslow.set_menu(unit_options[1][1],*unit_options[0])
        drop_unitsconv.configure(state='normal')
        ent_conv.configure(state='normal')
        get_concsettings()
        new_params.plot_conc(ax)
        canvas.draw()

def kd_isChecked():
    if kd_known.get() == 0:
        new_params.k_d_known = False
        drop_unitskd.configure(state='disabled')
        ent_kd.configure(state='disabled')
        get_concsettings()
        new_params.plot_conc(ax)
        canvas.draw()
    else:
        new_params.k_d_known = True
        drop_unitskd.configure(state='normal')
        ent_kd.configure(state='normal')
        get_concsettings()
        new_params.plot_conc(ax)
        canvas.draw()
        
        
def get_concsettings():
    new_params.conc_max_input = float(ent_concmax.get())
    new_params.conc_min_input = float(ent_concmin.get())
    new_params.stock_conc_input = float(ent_start_conc.get())
    new_params.conv_IU_input = float(ent_conv.get())
    new_params.k_d_input = float(ent_kd.get())
    new_params.remove = remove_options.index(remove.get())
    new_params.start_vol = float(ent_start_vol.get())*10**-3
    new_params.smallest_vol_added = float(ent_small_vol.get())
    new_params.largest_vol_added = float(ent_large_vol.get())
    if new_params.hormone:
        dict_ind = 1
        new_params.conv_unit = unit_options[0].index(units_conv.get())
    else:
        dict_ind = 0
    new_params.unit_max = unit_options[dict_ind].index(units.get())
    new_params.unit_min = unit_options[dict_ind].index(units_low.get())
    new_params.unit_stock = unit_options[dict_ind].index(unitsstock.get())
    new_params.unit_kd = unit_options[0].index(unitskd.get())
    new_params.run_calc()
    
def enter(event):
    print('Enter felt')
    get_concsettings()
    new_params.plot_conc(ax)
    canvas.draw()

# Constants
width_textbox = 5   
width_frame = 100 
default_params = params()
new_params = params()

root = Tk()
root.title('Titration Curve Calculator')
outputframe = ttk.LabelFrame(master = root, text = 'Output',
                            relief='raised',padding = '20 0 80 0')
root.columnconfigure(0, weight=2)
root.columnconfigure(1, weight=1)

root.rowconfigure(0, weight = 1)
# Write to File
# # Set up variables
units = StringVar()
units_low = StringVar()
unitsstock = StringVar()
units_conv = StringVar()
unitskd = StringVar()

hormone = BooleanVar()
kd_known = BooleanVar()

k_d = DoubleVar()
k_d_spread = DoubleVar()
conc_min = DoubleVar()
conc_max = DoubleVar()
num_measurements = IntVar()
frac_low = IntVar()
frac_mid = IntVar()
frac_high = IntVar()
remove = StringVar()
end_vol = IntVar()
start_conc = DoubleVar()
start_vol = DoubleVar()
unit_options = {0:['M','mM','\u03bcM','nM','pM'],
                1: ['IU','mIU','\u03bcIU','nIU','pIU']}

shot_options = ['30 mL', '10 mL']
remove_options = ['no','yes','each point']
#%% Optional Defaults
# menubar = Menu(root)
# menu_defaults = Menu(menubar,tearoff=0)
# menu_defaults.add_command(label = default_options[0],command = default_cleanNaOH)
# menu_defaults.add_command(label = default_options[1],command = default_cleanH2SO4)
# menu_defaults.add_command(label = default_options[2],command = default_roughen)
# menu_defaults.add_command(label = default_options[3],command = default_check)
# menu_defaults.add_command(label = default_options[4],command = default_freqmap_50)
# menu_defaults.add_command(label = default_options[5],command = default_freqmap_20)
# menu_defaults.add_command(label = default_options[6],command = default_titrationcurve)

# menubar.add_cascade(label = 'Default Macros',menu=menu_defaults)

#%% File Shit
frame_file = ttk.Frame(root)
frame_root = frame_file
frame_root.grid(row=0,column=0,columnspan = 2)
directory = ttk.Entry(frame_root,width='60')
directory.grid(row=0,column=0,sticky=W,columnspan=3)
btn = ttk.Button(frame_root,text='Select Directory >>',command=clicked)
btn.grid(row=0,column = 3)
lbl_file = ttk.Label(frame_root,text='File Start:').grid(row=0,column = 4)
ent_file = ttk.Entry(frame_root,width = '20')
ent_file.grid(row=0,column = 5)
# check_save = Checkbutton(frame_root,variable = save_scans,text='Save Scans?')
# check_save.grid(row = 1, column = 4)
# Defaults for File Stuff
directory.insert(0,default_params.datadir)
ent_file.insert(0,default_params.startfilename)
# check_save.select()
#%% Target Molecule
# Frame
frame_conc = ttk.LabelFrame(master = root, text = 'Target Molecule Settings',
                            relief='raised',padding = '20 0 20 0')
frame_conc.grid(row = 1, column = 0, sticky=W,padx=20,pady=10)
frame_root = frame_conc
# Widgets
lbl_start_conc = ttk.Label(master = frame_root,text = 'Target Stock: ').grid(row=0,column=0,sticky = E)
ent_start_conc = ttk.Entry(master=frame_root, width = width_textbox)
ent_start_conc.grid(row = 0,column = 1)
ent_start_conc.bind('<Return>',enter)
drop_unitsstock = ttk.OptionMenu(frame_root, unitsstock, unit_options[0][0],*unit_options[0])
drop_unitsstock.grid(row=0,column=2,sticky = E)
lbl_conc = ttk.Label(master=frame_root,text = 'Min Conc: ').grid(row=1,column = 0,sticky = E)
ent_concmin = ttk.Entry(master=frame_root, width = width_textbox)
ent_concmin.grid(row = 1,column = 1)
ent_concmin.bind('<Return>',enter)

drop_unitslow = ttk.OptionMenu(frame_root, units_low, unit_options[0][3],*unit_options[0],command=enter)
drop_unitslow.grid(row=1,column=2,sticky = E)

# lbl_units = ttk.Label(master = frame_conc,text = 'Units').grid(row=gr_units,column=0)
lbl_concmax = ttk.Label(master=frame_root,text = 'Max Conc: ').grid(row=2,column = 0,sticky = E)
ent_concmax = ttk.Entry(master=frame_root,width = width_textbox)
ent_concmax.grid(row = 2,column = 1)
ent_concmax.bind('<Return>',enter)

drop_units = ttk.OptionMenu(frame_root, units, unit_options[0][2],*unit_options[0],command=enter)
drop_units.grid(row=2,column=2,sticky = E)

lbl_conv = ttk.Label(master=frame_root,text = 'IU Conversion: ').grid(row=3,column = 0,sticky = E)
ent_conv = ttk.Entry(master=frame_root,width = width_textbox)
ent_conv.grid(row = 3,column = 1)
ent_conv.bind('<Return>',enter)

drop_unitsconv = ttk.OptionMenu(frame_root, units_conv, unit_options[0][3],*unit_options[0],command=enter)
drop_unitsconv.configure(state='disabled')
drop_unitsconv.grid(row=3,column=2,sticky = E)

check_hormone = ttk.Checkbutton(master = frame_root,text = 'Hormone?', variable = hormone, onvalue = 1, offvalue = 0,command = hormone_isChecked)
check_hormone.grid(row=4,column=0)
#%% Shotglass ##
frame_lowerleft = ttk.LabelFrame(master = root, text = 'Concentration Calculation Parameters',
                            relief='raised',padding = '20 0 20 0')
frame_lowerleft.grid(row = 2, column = 0, sticky = W,padx=20,pady=10)
frame_root = frame_lowerleft

# frame_shot = ttk.LabelFrame(master = frame_root, text = 'Volume Settings',
#                             relief='raised',padding = '20 0 20 0')
# frame_shot.grid(row = 1, column = 0, sticky=W,padx=20,pady=10)
# frame_root = frame_shot
lbl_start_vol = ttk.Label(master = frame_root,text = 'Starting Volume (mL):').grid(row=0,column=0, sticky = E)
ent_start_vol = ttk.Entry(master=frame_root,width = width_textbox)
ent_start_vol.grid(row = 0,column = 1)
lbl_glass = ttk.Label(master = frame_root,text = 'Max volume: ').grid(row=1,column=0,sticky = E)
drop_shot = ttk.OptionMenu(frame_root, end_vol, shot_options[0], *shot_options)
drop_shot.grid(row = 1,column = 1)

lbl_small_vol = ttk.Label(master = frame_root,text = 'Smallest added volume (\u03bcL):').grid(row=2,column=0, sticky = E)
ent_small_vol = ttk.Entry(master=frame_root,width = width_textbox)
ent_small_vol.grid(row = 2,column = 1)

lbl_large_vol = ttk.Label(master = frame_root,text = 'Largest Added Volume (\u03bcL):').grid(row=3,column=0, sticky = E)
ent_large_vol = ttk.Entry(master=frame_root,width = width_textbox)
ent_large_vol.grid(row = 3,column = 1)

lbl_remove = ttk.Label(master = frame_root,text = 'Remove volume? ').grid(row=4,column=0,sticky = E)
drop_remove = ttk.OptionMenu(frame_root, remove, remove_options[0], *remove_options)
drop_remove.grid(row = 4,column = 1)

#%% Titration
frame_titration = ttk.LabelFrame(master = root, text = 'Titration Settings',relief='raised',borderwidth=10,width = width_frame)
frame_titration.grid(row = 3, column = 0,sticky=W,padx=20,pady=10)
frame_root = frame_titration
lbl_meas = ttk.Label(master = frame_root,text = '# Points: ').grid(row=0,column=0)
ent_numpoints = ttk.Spinbox(master = frame_root,from_=1,to = 100,width = width_textbox,command=numpoints_change)
ent_numpoints.grid(row = 0,column = 1, sticky=W)
ent_numpoints.bind('<Return>',numpoints_change_enter)

lbl_meas = ttk.Label(master = frame_root,text = 'Fractional Distribution').grid(row=1,column=0, columnspan = 2, sticky = W)
lbl_meas = ttk.Label(master = frame_root,text = 'R1: ').grid(row=2,column=0)
ent_numpoints_1 = ttk.Spinbox(master = frame_root,from_=1,to = 100,width = width_textbox,command=dist_change_1)
ent_numpoints_1.grid(row = 2,column = 1, sticky=W)
ent_numpoints_1.bind('<Return>',dist_change_1_enter)

lbl_meas = ttk.Label(master = frame_root,text = 'R2: ').grid(row=3,column=0)
ent_numpoints_2 = ttk.Spinbox(master = frame_root,from_=1,to = 100,width = width_textbox,command=dist_change_2)
ent_numpoints_2.grid(row = 3,column = 1, sticky=W)
ent_numpoints_2.bind('<Return>',dist_change_2_enter)

lbl_meas = ttk.Label(master = frame_root,text = 'R3: ').grid(row=4,column=0)
ent_numpoints_3 = ttk.Spinbox(master = frame_root,from_=1,to = 100,width = width_textbox,command=dist_change_3)
ent_numpoints_3.grid(row = 4,column = 1, sticky=W)
ent_numpoints_3.bind('<Return>',dist_change_3_enter)

check_kd = ttk.Checkbutton(master = frame_root,text = 'Kd Known?', variable = kd_known, onvalue = 1, offvalue = 0,command = kd_isChecked)
check_kd.grid(row=5,column=0)

ent_kd = ttk.Entry(master=frame_root,width = width_textbox)
ent_kd.grid(row = 5,column = 1)
ent_kd.bind('<Return>',enter)
drop_unitskd = ttk.OptionMenu(frame_root, unitskd, unit_options[0][3],*unit_options[0],command=enter)
drop_unitskd.configure(state='disabled')
drop_unitskd.grid(row=5,column=2,sticky = E)

#%% Figure
frame_canvas = ttk.Frame(root)
frame_root = frame_canvas
frame_root.grid(row = 1,column = 1,rowspan = 3)
fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig,master = frame_root)
canvas.draw()
canvas.get_tk_widget().grid(row=0,column=0)

#%% Buttons
# frame_buttons = ttk.Frame(master = root)
# frame_buttons.grid(row = 4, column = 1)
# b_reset = ttk.Button(master=frame_buttons,text = 'Reset',command = reset).grid(row = 0,column=0)
# b_calculate = ttk.Button(master=frame_buttons,text = 'Calculate',command = calculate).grid(row = 0,column=3)
#%% Buttons
frame_buttons = ttk.Frame(master = root)
frame_buttons.grid(row = 3, column = 1,sticky=S,pady = 20)
b_reset = ttk.Button(master=frame_buttons,text = 'Reset',command = reset).grid(row = 0,column=0)
b_calculate = ttk.Button(master=frame_buttons,text = 'Print Titration Curve',command = print_macro).grid(row = 0,column=1)
b_close = ttk.Button(master=frame_buttons,text = 'Quit',command = close_program).grid(row = 0,column=2)
set_defaults()
# root.config(menu=menubar)
root.mainloop()
