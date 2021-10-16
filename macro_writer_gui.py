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
import datetime

class params:
    def __init__(self):
        # file labeling stuff
        self.datadir = '../Data/'
        self.savedir = 'F:/Plaxco_Labs/CFlatebo/210920/Raw/'
        self.macroname = 'Macro_File'
        self.startfilename = 'CF-01-001'
        self.save_scans = True
        # technique Parameters
        self.tech = 'CV'
        # potentials
        self.ei = 0
        self.eh = 1.8
        self.el = 0
        self.ef = 0
        # potentiostat
        self.simultaneous = True
        self.onecell = True
        self.efon = True # CV
        self.total_electrodes = 6
        self.quiet_time = 0
        self.sens = '1e-3'
        self.sens_same = True
        self.sens_list = []
        # scans
        self.direction = 'p' # CV, CA
        self.scanrate = 1 # CV
        self.sweeps = 320 # CV sweep segments, CA num steps
        self.sample_interval = 0.001 # CV, CA; SWV increment
        self.pulse_width = 0.02 # CA
        self.amp = 0.025 # SWV
        self.num_freq = 2
        self.freq_list = np.array([10,200]) # SWV
        self.incr_list = np.array([0.003,0.001])
        # run multiple times
        self.num_repeat = 0
        self.delay = 0

    def num_electrodes(self):
        self.electrodelist = []
        self.electrodelist.append('e1on')
        self.electrodelist.append('sens = ' + self.sens)
        for x in range(2,self.total_electrodes+1):
            self.electrodelist.append('e' + str(x) + 'on')
            self.electrodelist.append('e' + str(x) + 'scan')
            if self.sens_same:
                self.electrodelist.append('sens' + str(x) + ' = ' + self.sens)
            else:
                self.electrodelist.append('sens' + str(x) + ' = ' + self.sens_list[x-2])
    def write_txtfile(self):
        file=open(self.datadir + '/' + self.macroname + '.txt','w+')
        file.write('#\n# ' + self.macroname + '\n#\n')
        if self.save_scans:
            file.write('\nfolder = ' + self.datadir + '\n\n')
        file.write('tech = ' + self.tech + '\n')
        # Figure out how long the macro takes to run #
        if self.tech == 'SWV':
            freq_time = 0
            for idx in range(0,self.num_freq):
                freq_time += np.abs(self.ef - self.ei)/(self.freq_list[idx]*self.incr_list[idx])
            total_time_per_run = float(self.quiet_time) + freq_time
            if self.num_repeat != 0:
                total_time = (self.num_repeat + 1)*total_time_per_run
                if self.delay != 0:
                    total_time_withdelay = total_time + float(self.delay)*(self.num_repeat+1)
                    file.write('#\n# Macro run time with delays: ' + str(datetime.timedelta(seconds=total_time_withdelay)))
            else:
                total_time = total_time_per_run
        elif self.tech == 'CV':
            if self.direction == 'n':
                sweeps_i = np.abs(self.ei - np.amin([self.eh,self.el]))/float(self.scanrate)
                if self.sweeps % 2 > 0: # if odd
                    sweeps_f = np.abs(self.ef - self.eh)/float(self.scanrate)
                else:
                    sweeps_f = np.abs(self.ef - self.el)/float(self.scanrate)
            else:
                sweeps_i = np.abs(self.ei - np.amax([self.eh,self.el]))/float(self.scanrate)
                if self.sweeps % 2 > 0: # if odd
                    sweeps_f = np.abs(self.ef - self.el)/float(self.scanrate)
                else:
                    sweeps_f = np.abs(self.ef - self.eh)/float(self.scanrate)
            sweeps_n = (self.sweeps-2)*np.abs(self.eh - self.el)/float(self.scanrate)
            total_time_per_run = float(self.quiet_time) + sweeps_n + sweeps_i + sweeps_f
            if self.num_repeat != 0:
                total_time = (self.num_repeat + 1)*total_time_per_run
                if self.delay != 0:
                    total_time_withdelay = total_time + float(self.delay)*(self.num_repeat+1)
                    file.write('#\n# Macro run time with delays: ' + str(datetime.timedelta(seconds=total_time_withdelay)))
            else:
                total_time = total_time_per_run
        elif self.tech == 'CA' or self.tech == 'CC':
            total_time_per_run = float(self.quiet_time) + float(self.pulse_width)*float(self.sweeps)
            if self.num_repeat != 0:
                total_time = (self.num_repeat + 1)*total_time_per_run
                if self.delay != 0:
                    total_time_withdelay = total_time + float(self.delay)*(self.num_repeat+1)
                    file.write('#\n# Macro run time with delays: ' + str(datetime.timedelta(seconds=total_time_withdelay)))
            else:
                total_time = total_time_per_run
        file.write('\n# Macro run time: ' + str(datetime.timedelta(seconds=total_time)) + '\n')
        file.write('#\n# Experiment Params\n#\n\n')
        if self.simultaneous:
            file.write('simultaneous\n')
        else:
            file.write('sequential\n')
        if self.onecell:
            file.write('onecell\n')
        else:
            file.write('indcell\n')
        file.write('\n#\n# Electrodes to Turn On\n#\n\n')
        self.num_electrodes()
        for item in self.electrodelist:
            file.write(item + '\n')
        file.write('\n#\n# Technique Params\n#\n\n')
        file.write('qt = ' + str(self.quiet_time) + '\n')
        if self.num_repeat != 0:
            file.write('\nfor = ' + str(self.num_repeat) + '\n')
        file.write('ei = ' + str(self.ei) + '\n')
        if self.tech == 'CV':
            if self.efon:
                file.write('efon\n')
                file.write('ef = ' + str(self.ef) + '\n')
            file.write('eh = ' + str(self.eh) + '\n')
            file.write('el = ' + str(self.el) + '\n')
            file.write('pn = ' + str(self.direction) + ' # scan direction\n')
            file.write('v = ' + str(self.scanrate) + ' # scan rate\n')
            file.write('cl = ' + str(self.sweeps) + ' # segments\n')
            file.write('si = ' + str(self.sample_interval) + ' # sample interval\n')
            file.write('run\n')
            if self.save_scans:
                file.write('save = ' + self.startfilename + '\ntsave = ' + self.startfilename + '\n')
        elif self.tech == 'CA':
            file.write('eh = ' + str(self.eh) + '\n')
            file.write('el = ' + str(self.el) + '\n')
            file.write('pn = ' + str(self.direction) + ' # scan direction\n')
            file.write('cl = ' + str(self.sweeps) + ' # number of steps\n')
            file.write('pw = ' + str(self.pulse_width) + ' # pulse width\n')
            file.write('si = ' + str(self.sample_interval) + ' # sample interval\n')
            file.write('run\n')
            if self.save_scans:
                file.write('save = ' + self.startfilename + '\ntsave = ' + self.startfilename + '\n')
        elif self.tech == 'CC':
            file.write('ef = ' + str(self.ef) + '\n')
            file.write('cl = ' + str(self.sweeps) + ' # number of steps\n')
            file.write('pw = ' + str(self.pulse_width) + ' # pulse width\n')
            file.write('si = ' + str(self.sample_interval) + ' # sample interval\n')
            file.write('run\n')
            if self.save_scans:
                file.write('save = ' + self.startfilename + '\ntsave = ' + self.startfilename + '\n')
        elif self.tech == 'SWV':
            file.write('ef = ' + str(self.ef) + '\n')
            file.write('amp = ' + str(self.amp) + ' # amplitude\n\n')
            for idx in range(0,self.num_freq):
                file.write('freq = ' + str(self.freq_list[idx]) + ' # frequency\n')
                file.write('incre = ' + str(self.incr_list[idx]) + ' # increment\n')
                file.write('run\n')
                if self.save_scans:
                    file.write('save = ' + self.startfilename + '_' + str(self.freq_list[idx]) + 'Hz\ntsave = ' + self.startfilename + '_' + str(self.freq_list[idx]) + 'Hz\n\n')
                
        else:
            print('Macro Writer not able to write ' + self.tech + ' yet')
        if self.num_repeat != 0:
            if int(self.delay) != 0:
                file.write('delay = ' + str(self.delay) + ' # number of seconds to delay between runs\n')
            file.write('next\n')
        gene_mcr_file(file)
        file.close()

def gene_mcr_file(file):
    if os.path.isfile(file.name[:-3] + 'mcr'):
        os.remove(file.name[:-3] + 'mcr')
    file.seek(0, 0)
    mcr_file=open(file.name[:-3] + 'mcr','w')
    mcr_file.write('Hh\x00\x00'+ file.read())
    mcr_file.close()
    file.seek(0, 2)

def clicked():
    expdir = filedialog.askdirectory()
    directory.delete(0,'end')
    directory.insert(END,expdir)
    
def print_macro():
    global new_params
    new_params.datadir = directory.get()
    new_params.macroname = ent_macro.get()
    new_params.startfilename = ent_file.get()
    new_params.save_scans = bool(save_scans.get())
    new_params.simultaneous = bool(simul.get())
    new_params.onecell = bool(samecell.get())
    tech = drop_technique.index('current')
    new_params.quiet_time = ent_qt.get()
    new_params.sens = sens.get()
    new_params.ei = ei.get()
    new_params.sample_interval = interval.get()
    new_params.total_electrodes = electrode.get()
    new_params.num_repeat = int(ent_repeat.get())
    new_params.delay = int(ent_delay.get())
    if tech == 0:
        new_params.tech = 'CV'
        new_params.eh = eh.get()
        new_params.el = el.get()
        try:
            new_params.ef = ef.get()
        except TclError:
            print('Final V not selected')
        new_params.scanrate = ent_scan.get()
        new_params.sweeps = sweeps.get()
        new_params.direction = pol.get()
    elif tech == 1:
        new_params.tech = 'SWV'
        new_params.ef = ef.get()
        new_params.amp = ent_amp.get()
        new_params.num_freq = int(ent_freqnum.get())
        new_params.freq_list = np.zeros(new_params.num_freq,dtype=int)
        new_params.incr_list = np.zeros_like(new_params.freq_list,dtype=float)
        for x, label in enumerate(reversed(swv_freq.grid_slaves(column=1))):
            new_params.freq_list[x] = label.get()
        for x, label in enumerate(reversed(swv_freq.grid_slaves(column=2))):
            new_params.incr_list[x] = label.get()
    elif tech == 2:
        new_params.tech = 'CA'
        new_params.eh = eh.get()
        new_params.pulse_width = ent_pw.get()
        new_params.sweeps = sweeps.get()
        new_params.direction = pol.get()
    elif tech == 3:
        new_params.tech = 'CC'
        new_params.ef = ef.get()
        new_params.pulse_width = ent_pw_cc.get()
        new_params.sweeps = sweeps.get()
    if not os.path.isdir(new_params.datadir):
        os.makedirs(new_params.datadir)
    new_params.write_txtfile()
    messagebox.showinfo(title = 'File Saved', message = 'File Location: ' + new_params.datadir + '\nFile Name: ' + new_params.macroname)
    
def close_program():
    root.destroy()

def reset():
    set_defaults()
    
def default_cleanNaOH():
    ent_macro.delete(0,END)
    ent_macro.insert(0,'Cleaning_NaOH_6elec')
    drop_technique.select(0)
    electrode.set(6)
    ei.set(-1)
    ef.set(-1)
    eh.set(-1.8)
    el.set(-1)
    pol.set('n')
    sweeps.set(250)
    interval.set(0.001)
    sens.set(sens_options[0])
    ent_scan.delete(0,END)
    ent_scan.insert(0,1)
    ent_repeat.delete(0,END)
    ent_repeat.insert(0,4)
    ent_delay.delete(0,END)
    ent_delay.insert(0,0)
    check_save.deselect()

def default_cleanH2SO4():
    ent_macro.delete(0,END)
    ent_macro.insert(0,'Cleaning_H2SO4_6elec')
    drop_technique.select(0)
    electrode.set(6)
    ei.set(0)
    ef.set(0)
    eh.set(1.8)
    el.set(0)
    pol.set('p')
    sweeps.set(10)
    interval.set(0.001)
    sens.set(sens_options[1])
    ent_scan.delete(0,END)
    ent_scan.insert(0,1)
    ent_repeat.delete(0,END)
    ent_repeat.insert(0,10)
    ent_delay.delete(0,END)
    ent_delay.insert(0,0)
    check_save.deselect()
    
def default_roughen():
    ent_macro.delete(0,END)
    ent_macro.insert(0,'Roughen_H2SO4_6elec')
    drop_technique.select(2)
    electrode.set(6)
    ei.set(0)
    eh.set(2.2)
    el.set(0)
    pol.set('p')
    sweeps.set(320)
    interval.set(0.001)
    sens.set(sens_options[0])
    ent_pw.delete(0,END)
    ent_pw.insert(0,0.02)
    ent_repeat.delete(0,END)
    ent_repeat.insert(0,100)
    ent_delay.delete(0,END)
    ent_delay.insert(0,0)
    check_save.deselect()
    
def default_check():
    ent_macro.delete(0,END)
    ent_macro.insert(0,'Check_H2SO4_6elec')
    drop_technique.select(0)
    electrode.set(6)
    ei.set(0)
    ef.set(0)
    eh.set(1.8)
    el.set(0)
    pol.set('p')
    sweeps.set(10)
    interval.set(0.001)
    sens.set(sens_options[0])
    ent_scan.delete(0,END)
    ent_scan.insert(0,1)
    ent_repeat.delete(0,END)
    ent_repeat.insert(0,0)
    ent_delay.delete(0,END)
    ent_delay.insert(0,0)
    check_save.select()
    
def default_freqmap_20():
    default_freqmap(20)

def default_freqmap_50():
    default_freqmap(50)
def default_freqmap(val):
    global swv_freq
    ent_macro.delete(0,END)
    ent_macro.insert(0,'FreqMap_6elec')
    drop_technique.select(1)
    electrode.set(6)
    ent_repeat.delete(0,END)
    ent_repeat.insert(0,51)
    ei.set(-0.18)
    ef.set(-0.48)
    ent_delay.delete(0,END)
    ent_delay.insert(0,200)
    sens.set(sens_options[2])
    new_params.freq_list = np.concatenate([np.array([5,8,10,15,20,25]),
                            np.arange(30,110,10),
                            np.arange(150,500 + val,val),
                            np.arange(600,1100,100)])
    new_params.num_freq = int(np.shape(new_params.freq_list)[0])
    new_params.incr_list = np.zeros_like(new_params.freq_list,dtype=float)
    # print(type(new_params.incr_list))
    ent_freqnum.set(new_params.num_freq)
    swv_freq.destroy()
    swv_freq = ttk.Frame(swv_freq_header)
    swv_freq.grid(row=1,column=0,columnspan = 3, sticky=NW)
    for x in range(0,new_params.num_freq):
        lbl_freq = ttk.Label(swv_freq,text = str(x+1)).grid(row=x,column=0)
        ent_freq = ttk.Entry(swv_freq,width=width_textbox)
        ent_freq.grid(row = x, column = 1)
        ent_freq.insert(0,new_params.freq_list[x])
        if new_params.freq_list[x] <= 25:
            new_params.incr_list[x] = 0.003
        elif new_params.freq_list[x] <= 90:
            new_params.incr_list[x] = 0.002
        else:
            new_params.incr_list[x] = 0.001
        ent_incr = ttk.Entry(swv_freq,width=width_textbox)
        ent_incr.grid(row = x, column = 2)
        ent_incr.insert(0,new_params.incr_list[x])

def default_titrationcurve():
    global swv_freq
    ent_macro.delete(0,END)
    ent_macro.insert(0,'Titration_6elec')
    drop_technique.select(1)
    electrode.set(6)
    ei.set(-0.18)
    ef.set(-0.48)
    sens.set(sens_options[2])
    new_params.freq_list = np.array([10,200])
    new_params.num_freq = int(np.shape(new_params.freq_list)[0])
    new_params.incr_list = np.zeros_like(new_params.freq_list,dtype=float)
    # print(type(new_params.incr_list))
    ent_freqnum.set(new_params.num_freq)
    ent_repeat.delete(0,END)
    ent_repeat.insert(0,51)
    ent_delay.delete(0,END)
    ent_delay.insert(0,200)
    swv_freq.destroy()
    swv_freq = ttk.Frame(swv_freq_header)
    swv_freq.grid(row=1,column=0,columnspan = 3, sticky=NW)
    for x in range(0,new_params.num_freq):
        lbl_freq = ttk.Label(swv_freq,text = str(x+1)).grid(row=x,column=0)
        ent_freq = ttk.Entry(swv_freq,width=width_textbox)
        ent_freq.grid(row = x, column = 1)
        ent_freq.insert(0,new_params.freq_list[x])
        if new_params.freq_list[x] <= 25:
            # print('less than 25')
            new_params.incr_list[x] = 0.003
        elif new_params.freq_list[x] <= 90:
            new_params.incr_list[x] = 0.002
        else:
            new_params.incr_list[x] = 0.001
        ent_incr = ttk.Entry(swv_freq,width=width_textbox)
        ent_incr.grid(row = x, column = 2)
        ent_incr.insert(0,new_params.incr_list[x])
    check_save.select()

def set_defaults():
    global default_params
    ei.set(default_params.ei)
    ef.set(default_params.ef)
    eh.set(default_params.eh)
    interval.set(default_params.sample_interval)
    sweeps.set(default_params.sweeps)
    el.set(default_params.el)
    ent_scan.delete(0)
    ent_qt.delete(0)
    ent_freqnum.delete(0)
    ent_incre.delete(0,END)
    ent_pw.delete(0,END)
    ent_pw_cc.delete(0,END)
    ent_amp.delete(0,END)
    ent_repeat.delete(0,END)
    ent_delay.delete(0,END)
    ent_delay.insert(0, default_params.delay)
    ent_repeat.insert(0, default_params.num_repeat)
    electrode.set(default_params.total_electrodes)
    sens.set(default_params.sens)
    pol.set(default_params.direction)
    ent_pw.insert(0, default_params.pulse_width)
    ent_pw_cc.insert(0, default_params.pulse_width)
    ent_scan.insert(0,default_params.scanrate)
    ent_qt.insert(0,default_params.quiet_time)
    ent_incre.insert(0,default_params.sample_interval)
    ent_freqnum.set(default_params.num_freq)
    ent_amp.insert(0,default_params.amp)
    for x in range(0,default_params.num_freq):
        lbl_freq = ttk.Label(swv_freq,text = str(x+1)).grid(row=x,column=0)
        ent_freq = ttk.Entry(swv_freq,width=width_textbox)
        ent_freq.grid(row = x, column = 1)
        ent_freq.insert(0,default_params.freq_list[x])
        ent_freq = ttk.Entry(swv_freq,width=width_textbox)
        ent_freq.grid(row = x, column = 2)
        ent_freq.insert(0,default_params.incr_list[x])
    for label in swv_freq.grid_slaves():
        if int(label.grid_info()['row']) > default_params.num_freq-1:
            label.grid_forget()
    check_save.select()

def freq_change():
    global new_params, swv_freq
    new_params.sample_interval = ent_incre.get()
    addsub_freq(swv_freq,new_params)
        
def freq_change_enter(event):
    freq_change()

def addsub_freq(swv_freq,new_params):
    new_freq_num = int(ent_freqnum.get())
    # print(new_freq_num)
    if new_freq_num > new_params.num_freq:
        for x in range(new_params.num_freq,new_freq_num):
            lbl_freq = ttk.Label(swv_freq,text = str(x+1)).grid(row=x,column=0)
            ent_freq = ttk.Entry(swv_freq,width=width_textbox)
            ent_freq.grid(row = x, column = 1)
            # ent_freq.insert(0,new_params.freq_list[x][0])
            np.append(new_params.freq_list,np.nan)
            ent_freq = ttk.Entry(swv_freq,width=width_textbox)
            ent_freq.grid(row = x, column = 2)
            ent_freq.insert(0,new_params.sample_interval)
            np.append(new_params.incr_list,ent_freq.get())
    elif new_freq_num < new_params.num_freq:
        for label in swv_freq.grid_slaves():
            if int(label.grid_info()['row']) > new_freq_num-1:
                label.grid_forget()
    new_params.num_freq = new_freq_num
# Constants
width_textbox = 5   
width_frame = 100 
default_params = params()
new_params = params()

root = Tk()
root.title('Macro Printer')
outputframe = ttk.LabelFrame(master = root, text = 'Output',
                            relief='raised',padding = '20 0 80 0')
root.columnconfigure(0, weight=2)
root.columnconfigure(1, weight=1)

root.rowconfigure(0, weight = 1)
# Write to File
# # Set up variables
ei = DoubleVar()
ef = DoubleVar()
eh = DoubleVar()
el = DoubleVar()
pol = StringVar()
sens = StringVar()
electrode = IntVar()
interval = DoubleVar()
sweeps = IntVar()
simul = IntVar()
save_scans = IntVar()
samecell = IntVar()
tech_options = ['Cyclic Voltammetry', 'Square Wave Voltammetry', 'Chronoamperometry', 'Chronocoulometry']
default_options = ['Cleaning NaOH','Cleaning H2SO4','Roughening H2SO4','Check H2SO4','Frequency Map (few)','Frequency Map (many)','Titration Curve']
pol_options = ['n', 'p']
sens_options = ['1e-3','1e-4','1e-5','1e-6','1e-7','1e-8','1e-9']
electrode_options = [1,2,3,4,5,6,7,8]
#%% Optional Defaults
menubar = Menu(root)
menu_defaults = Menu(menubar,tearoff=0)
menu_defaults.add_command(label = default_options[0],command = default_cleanNaOH)
menu_defaults.add_command(label = default_options[1],command = default_cleanH2SO4)
menu_defaults.add_command(label = default_options[2],command = default_roughen)
menu_defaults.add_command(label = default_options[3],command = default_check)
menu_defaults.add_command(label = default_options[4],command = default_freqmap_50)
menu_defaults.add_command(label = default_options[5],command = default_freqmap_20)
menu_defaults.add_command(label = default_options[6],command = default_titrationcurve)

menubar.add_cascade(label = 'Default Macros',menu=menu_defaults)

#%% File Shit
frame_file = ttk.Frame(root)
frame_root = frame_file
frame_root.grid(row=0,column=0)
directory = ttk.Entry(frame_root,width='60')
directory.grid(row=0,column=0,sticky=W,columnspan=3)
btn = ttk.Button(frame_root,text='Select Directory >>',command=clicked)
btn.grid(row=0,column = 3)
lbl_macro = ttk.Label(frame_root,text='Macro Name:').grid(row=1,column = 0)
ent_macro = ttk.Entry(frame_root,width = '30')
ent_macro.grid(row=1,column = 1)
lbl_file = ttk.Label(frame_root,text='File Start:').grid(row=1,column = 2)
ent_file = ttk.Entry(frame_root,width = '20')
ent_file.grid(row=1,column = 3)
check_save = Checkbutton(frame_root,variable = save_scans,text='Save Scans?')
check_save.grid(row = 1, column = 4)
# Defaults for File Stuff
directory.insert(0,default_params.datadir)
ent_macro.insert(0,default_params.macroname)
ent_file.insert(0,default_params.startfilename)
check_save.select()
#%% Technique
frame_technique = ttk.LabelFrame(master = root, text = 'Technique',
                                 relief='raised',padding = '20 0 20 0')
frame_root = frame_technique
frame_root.grid(row = 1, column = 0,padx=10,pady=10)
lbl_qt = ttk.Label(master = frame_root,text = 'Quiet Time: ').grid(row=1,column=0,sticky=E)
ent_qt = ttk.Entry(master = frame_root, width = width_textbox)
ent_qt.grid(row = 1,column = 1)

lbl_elec = ttk.Label(master = frame_root,text = '# Electrodes: ').grid(row=0,column=0,sticky=E)
drop_elec = ttk.OptionMenu(frame_root, electrode, electrode_options[5],*electrode_options)
drop_elec.grid(row = 0,column = 1)
lbl_repeat = ttk.Label(master = frame_root,text = '# Repeat Cycles: ').grid(row=0,column=2,sticky=E)
ent_repeat = ttk.Entry(master = frame_root, width = width_textbox)
ent_repeat.grid(row = 0,column = 3)
lbl_delay = ttk.Label(master = frame_root,text = 'Delay: ').grid(row=0,column=4,sticky=E)
ent_delay = ttk.Entry(master = frame_root, width = width_textbox)
ent_delay.grid(row = 0,column = 5)
lbl_sens = ttk.Label(master = frame_root,text = 'Sensitivity: ').grid(row=1,column=2,sticky=E)
drop_sens = ttk.OptionMenu(frame_root, sens, sens_options[0],*sens_options)
drop_sens.grid(row = 1,column = 3)
check_simul = Checkbutton(frame_root,variable = simul,text='Simultaneous')
check_simul.grid(row = 1, column = 4)
check_simul.select()
check_onecell = Checkbutton(frame_root,variable = samecell, text='Same RE/CE')
check_onecell.grid(row = 1, column = 5)
check_onecell.select()
# Widgets
# lbl_technique = ttk.Label(master = frame_root,text = 'Technique: ').grid(row=0,column=0,sticky=W)
drop_technique = ttk.Notebook(frame_root)
drop_technique.grid(row = 2,column = 0,columnspan = 6,sticky=W)

cv_tab = ttk.Frame(drop_technique)
swv_tab = ttk.Frame(drop_technique)
ca_tab = ttk.Frame(drop_technique)
cc_tab = ttk.Frame(drop_technique)
drop_technique.add(cv_tab,text = 'Cyclic Voltammetry')
drop_technique.add(swv_tab,text = 'Square Wave Voltammetry')
drop_technique.add(ca_tab,text = 'Chronoamperometry')
drop_technique.add(cc_tab,text = tech_options[3])
# frame_technique_params = ttk.Frame(master = frame_root, padding = '20 0 20 0')
frame_root = cv_tab
# frame_root.grid(row = 1, column = 0,columnspan = 2, sticky=W)
lbl_ei = ttk.Label(master = frame_root,text = 'Init E (V): ').grid(row=0,column=0,sticky=E)
ent_ei = ttk.Entry(master = frame_root,textvariable = ei, width = width_textbox).grid(row = 0,column = 1)
lbl_ef = ttk.Label(master = frame_root,text = 'Final E (V): ').grid(row=3,column=0,sticky=E)
ent_ef = ttk.Entry(master = frame_root,textvariable = ef, width = width_textbox).grid(row = 3,column = 1)
lbl_eh = ttk.Label(master = frame_root,text = 'High E (V): ').grid(row=1,column=0,sticky=E)
ent_eh = ttk.Entry(master = frame_root,textvariable = eh, width = width_textbox).grid(row = 1,column = 1)
lbl_el = ttk.Label(master = frame_root,text = 'Low E (V): ').grid(row=2,column=0,sticky=E)
ent_el = ttk.Entry(master = frame_root,textvariable = el, width = width_textbox).grid(row = 2,column = 1)

lbl_scan = ttk.Label(master = frame_root,text = 'Scan Rate: ').grid(row=0,column=2,sticky=E)
ent_scan = ttk.Entry(master = frame_root, width = width_textbox)
ent_scan.grid(row = 0,column = 3, sticky=W)
lbl_sweeps = ttk.Label(master = frame_root,text = 'Sweep Segments: ').grid(row=1,column=2,sticky=E)
ent_sweeps = ttk.Entry(master = frame_root,textvariable = sweeps, width = width_textbox).grid(row = 1,column = 3, sticky=W)
lbl_interval = ttk.Label(master = frame_root,text = 'Sample Interval: ').grid(row=2,column=2,sticky=E)
ent_interval = ttk.Entry(master = frame_root,textvariable = interval, width = width_textbox).grid(row = 2,column = 3, sticky=W)

lbl_pol = ttk.Label(master = frame_root,text = 'Polarity: ').grid(row=3,column=2,sticky=E)
drop_pol = ttk.OptionMenu(frame_root, pol,pol_options[0],*pol_options)
drop_pol.grid(row = 3,column = 3)

frame_root = swv_tab
# frame_root.grid(row = 1, column = 0,columnspan = 2, sticky=W)
swv_params = ttk.Frame(swv_tab)
swv_params.grid(row=0,column=0,padx = 5,sticky=NW)
frame_root = swv_params
lbl_ei = ttk.Label(master = frame_root,text = 'Init E (V): ').grid(row=0,column=0,sticky=E)
ent_ei = ttk.Entry(master = frame_root,textvariable = ei, width = width_textbox).grid(row = 0,column = 1)
lbl_ef = ttk.Label(master = frame_root,text = 'Final E (V): ').grid(row=1,column=0,sticky=E)
ent_ef = ttk.Entry(master = frame_root,textvariable = ef, width = width_textbox).grid(row = 1,column = 1)
lbl_freqnum = ttk.Label(master = frame_root,text = '# Frequencies: ').grid(row=1,column=2,sticky=E)
ent_freqnum = ttk.Spinbox(master = frame_root,from_=1,to = 100,width = width_textbox,command=freq_change)
ent_freqnum.grid(row = 1,column = 3, sticky=W)
ent_freqnum.bind('<Return>',freq_change_enter)
lbl_amp = ttk.Label(master = frame_root,text = 'Amplitude: ').grid(row=0,column=2,sticky=E)
ent_amp = ttk.Entry(master = frame_root, width = width_textbox)
ent_amp.grid(row = 0,column = 3, sticky=W)
lbl_incre = ttk.Label(master = frame_root,text = 'Default Increment: ').grid(row=2,column=2,sticky=E)
ent_incre = ttk.Entry(master = frame_root,width = width_textbox)
ent_incre.grid(row = 2,column = 3, sticky=W)
swv_freq_header = ttk.Frame(swv_tab)
swv_freq_header.grid(row=0,column=1,sticky=NE,padx = 10)
frame_root = swv_freq_header
lbl_col1 = ttk.Label(frame_root,text = '#').grid(row = 0, column = 0)
lbl_col2 = ttk.Label(frame_root,text = 'Frequency').grid(row = 0, column = 1)
lbl_col3 = ttk.Label(frame_root,text = 'Increment').grid(row = 0, column = 2)
swv_freq = ttk.Frame(swv_freq_header)
swv_freq.grid(row=1,column=0,columnspan = 3, sticky=NW)


frame_root = ca_tab
# frame_root.grid(row = 1, column = 0,columnspan = 2, sticky=W)
lbl_ei = ttk.Label(master = frame_root,text = 'Init E (V): ').grid(row=0,column=0,sticky=E)
ent_ei = ttk.Entry(master = frame_root,textvariable = ei, width = width_textbox).grid(row = 0,column = 1)
lbl_eh = ttk.Label(master = frame_root,text = 'High E (V): ').grid(row=1,column=0,sticky=E)
ent_eh = ttk.Entry(master = frame_root,textvariable = eh, width = width_textbox).grid(row = 1,column = 1)
lbl_el = ttk.Label(master = frame_root,text = 'Low E (V): ').grid(row=2,column=0,sticky=E)
ent_el = ttk.Entry(master = frame_root,textvariable = el, width = width_textbox).grid(row = 2,column = 1)

lbl_pw = ttk.Label(master = frame_root,text = 'Pulse Width: ').grid(row=0,column=2,sticky=E)
ent_pw = ttk.Entry(master = frame_root,width = width_textbox)
ent_pw.grid(row = 0,column = 3, sticky=W)
lbl_sweeps = ttk.Label(master = frame_root,text = '# Steps: ').grid(row=1,column=2,sticky=E)
ent_sweeps = ttk.Entry(master = frame_root,textvariable = sweeps, width = width_textbox).grid(row = 1,column = 3, sticky=W)
lbl_interval = ttk.Label(master = frame_root,text = 'Sample Interval: ').grid(row=2,column=2,sticky=E)
ent_interval = ttk.Entry(master = frame_root,textvariable = interval, width = width_textbox).grid(row = 2,column = 3, sticky=W)

lbl_pol = ttk.Label(master = frame_root,text = 'Polarity: ').grid(row=3,column=2,sticky=E)
drop_pol = ttk.OptionMenu(frame_root, pol, pol_options[0], *pol_options)
drop_pol.grid(row = 3,column = 3)

frame_root = cc_tab
# frame_root.grid(row = 1, column = 0,columnspan = 2, sticky=W)
lbl_ei = ttk.Label(master = frame_root,text = 'Init E (V): ').grid(row=0,column=0,sticky=E)
ent_ei = ttk.Entry(master = frame_root,textvariable = ei, width = width_textbox).grid(row = 0,column = 1)
lbl_ef = ttk.Label(master = frame_root,text = 'Final E (V): ').grid(row=1,column=0,sticky=E)
ent_ef = ttk.Entry(master = frame_root,textvariable = ef, width = width_textbox).grid(row = 1,column = 1)

lbl_sweeps = ttk.Label(master = frame_root,text = '# Steps: ').grid(row=1,column=2,sticky=E)
ent_sweeps = ttk.Entry(master = frame_root,textvariable = sweeps, width = width_textbox).grid(row = 1,column = 3, sticky=W)
lbl_interval = ttk.Label(master = frame_root,text = 'Sample Interval: ').grid(row=2,column=2,sticky=E)
ent_interval = ttk.Entry(master = frame_root,textvariable = interval, width = width_textbox).grid(row = 2,column = 3, sticky=W)
lbl_pw = ttk.Label(master = frame_root,text = 'Pulse Width: ').grid(row=0,column=2,sticky=E)
ent_pw_cc = ttk.Entry(master = frame_root,width = width_textbox)
ent_pw_cc.grid(row = 0,column = 3, sticky=W)
#%% Buttons
frame_buttons = ttk.Frame(master = root)
frame_buttons.grid(row = 3, column = 0, columnspan = 2)
b_reset = ttk.Button(master=frame_buttons,text = 'Reset Technique Parameters',command = reset).grid(row = 0,column=0)
b_calculate = ttk.Button(master=frame_buttons,text = 'Print Macro',command = print_macro).grid(row = 0,column=1)
b_close = ttk.Button(master=frame_buttons,text = 'Quit',command = close_program).grid(row = 0,column=2)
set_defaults()
root.config(menu=menubar)
root.mainloop()
