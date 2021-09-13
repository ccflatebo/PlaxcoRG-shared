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

class params:
    def __init__(self):
        # file labeling stuff
        self.datadir = 'C:/Users/ccflatebo/Documents/Data/Test'
        self.macroname = 'Macro_File'
        self.startfilename = 'E1_'
        self.save_scans = True
        # technique Parameters
        self.tech = 'CV'
        # potentials
        self.ei = 0
        self.eh = 1.8
        self.el = 0
        self.ef = []
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
        self.freq_list = [10,200] # SWV
        self.incr_list = [0.003,0.001]
        # run multiple times
        self.repeat = True
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
        file.write('tech = ' + self.tech + '\n')
        file.write('qt = ' + str(self.quiet_time) + '\n')
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
        elif self.tech == 'CA':
            file.write('eh = ' + str(self.eh) + '\n')
            file.write('el = ' + str(self.el) + '\n')
            file.write('pn = ' + str(self.direction) + ' # scan direction\n')
            file.write('cl = ' + str(self.sweeps) + ' # number of steps\n')
            file.write('pw = ' + str(self.pulse_width) + ' # pulse width\n')
            file.write('si = ' + str(self.sample_interval) + ' # sample interval\n')
        elif self.tech == 'CC':
            file.write('ef = ' + str(self.ef) + '\n')
            file.write('cl = ' + str(self.sweeps) + ' # number of steps\n')
            file.write('pw = ' + str(self.pulse_width) + ' # pulse width\n')
            file.write('si = ' + str(self.sample_interval) + ' # sample interval\n')
        elif self.tech == 'SWV':
            file.write('ef = ' + str(self.ef) + '\n')
            file.write('amp = ' + str(self.amp) + ' # amplitude\n')
            if np.shape(self.freq_list)[0] == 1:
                file.write('freq = ' + str(self.freq_list) + ' # frequency\n')
                file.write('incre = ' + str(self.incre_list) + ' # increment\n')
        else:
            print('Macro Writer not able to write ' + self.tech + ' yet')

        if self.repeat:
            file.write('\nfor = ' + str(self.num_repeat) + '\n')
        file.write('run\n')
        if self.save_scans:
            file.write('save = ' + self.startfilename + '\ntsave = ' + self.startfilename + '\n')
        if self.repeat:
            file.write('next')
        gene_mcr_file(file)
        file.close()

def gene_mcr_file(file):
    if os.path.isfile(file.name[:-3] + 'mcr'):
        os.remove(file.name[:-3] + 'mcr')
    file.seek(0, 0)
    mcr_file=open(file.name[:-3] + 'mcr','w')
    mcr_file.write('Hh\x00\x00'+ file.read())
    file.seek(0, 2)
    pass

def clicked():
    expdir = filedialog.askdirectory()
#    directory.configure(text=expdir)
    directory.delete(0,'end')
    directory.insert(END,expdir)
    
def print_macro():
    global new_params
    new_params.datadir = directory.get()
    new_params.macroname = ent_macro.get()
    new_params.startfilename = ent_file.get()
    tech = drop_technique.index('current')
    new_params.quiet_time = ent_qt.get()
    new_params.sens = sens.get()
    new_params.ei = ei.get()
    new_params.sample_interval = interval.get()
    new_params.total_electrodes = electrode.get()
    if tech == 0:
        new_params.tech = 'CV'
        new_params.eh = eh.get()
        try:
            new_params.ef = ef.get()
        except TclError:
            print('Final V not selected')
        new_params.scanrate = ent_scan.get()
        new_params.sweeps = sweeps.get()
        new_params.direction = pol.get()
    elif tech == 1:
        new_params.tech = 'SWV'
    elif tech == 2:
        new_params.tech = 'CA'
        new_params.eh = eh.get()
        new_params.pulse_width = ent_pw.get()
        new_params.sweeps = sweeps.get()
        new_params.direction = pol.get()
    elif tech == 3:
        new_params.tech = 'CC'
        new_params.ef = ef.get()
        new_params.pulse_width = ent_pw.get()
        new_params.sweeps = sweeps.get()
    if not os.path.isdir(new_params.datadir):
        os.makedirs(new_params.datadir)
    new_params.write_txtfile()
    messagebox.showinfo(title = 'File Saved', message = 'File Location: ' + new_params.datadir + '\nFile Name: ' + new_params.macroname)
    
def close_program():
    root.destroy()

def reset():
    set_defaults()
    
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
    ent_amp.delete(0,END)
    ent_repeat.delete(0,END)
    ent_repeat.insert(0, default_params.num_repeat)
    electrode.set(default_params.total_electrodes)
    sens.set(default_params.sens)
    pol.set(default_params.direction)
    ent_pw.insert(0, default_params.pulse_width)
    ent_scan.insert(0,default_params.scanrate)
    ent_qt.insert(0,default_params.quiet_time)
    ent_incre.insert(0,default_params.sample_interval)
    ent_freqnum.set(default_params.num_freq)
    ent_amp.insert(0,default_params.amp)
    for x in range(0,new_params.num_freq):
        lbl_freq = ttk.Label(swv_freq,text = str(x+1)).grid(row=x+1,column=0)
        ent_freq = ttk.Entry(swv_freq,width=width_textbox)
        ent_freq.grid(row = 1 + x, column = 1)
        ent_freq.insert(0,new_params.freq_list[x])
        ent_freq = ttk.Entry(swv_freq,width=width_textbox)
        ent_freq.grid(row = 1 + x, column = 2)
        ent_freq.insert(0,new_params.incr_list[x])

def freq_change():
    global new_params, swv_freq
    new_params.sample_interval = ent_incre.get()
    addsub_freq(swv_freq,new_params)
        
def freq_change_enter(event):
    freq_change()

def addsub_freq(swv_freq,new_params):
    new_freq_num = int(ent_freqnum.get())
    if new_freq_num > new_params.num_freq:
        for x in range(new_params.num_freq,new_freq_num):
            lbl_freq = ttk.Label(swv_freq,text = str(x+1)).grid(row=x+1,column=0)
            ent_freq = ttk.Entry(swv_freq,width=width_textbox)
            ent_freq.grid(row = 1 + x, column = 1)
            # ent_freq.insert(0,new_params.freq_list[x][0])
            new_params.freq_list.append(ent_freq.get())
            ent_freq = ttk.Entry(swv_freq,width=width_textbox)
            ent_freq.grid(row = 1 + x, column = 2)
            ent_freq.insert(0,new_params.sample_interval)
            new_params.incr_list.append(ent_freq.get())
    elif new_freq_num < new_params.num_freq:
        for x in range(0,new_freq_num):
            lbl_freq = ttk.Label(swv_freq,text = str(x+1)).grid(row=x+1,column=0)
            ent_freq = ttk.Entry(swv_freq,width=width_textbox)
            ent_freq.grid(row = 1 + x, column = 1)
            ent_freq.insert(0,new_params.freq_list[x])
            ent_freq = ttk.Entry(swv_freq,width=width_textbox)
            ent_freq.grid(row = 1 + x, column = 2)
            ent_freq.insert(0,new_params.incr_list[x])
        for label in swv_freq.grid_slaves():
            if int(label.grid_info()['row']) > new_freq_num:
                label.grid_forget()
        new_params.freq_list = new_params.freq_list[0:new_freq_num-1]
        new_params.incr_list = new_params.incr_list[0:new_freq_num-1]
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
tech_options = ['Cyclic Voltammetry', 'Square Wave Voltammetry', 'Chronoamperometry', 'Chronocoulometry']
pol_options = ['n', 'p']
sens_options = ['1e-3','1e-4','1e-5','1e-6','1e-7','1e-8','1e-9']
electrode_options = [1,2,3,4,5,6,7,8]

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
check_save = Checkbutton(frame_root,text='Save Scans?')
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
lbl_sens = ttk.Label(master = frame_root,text = 'Sensitivity: ').grid(row=1,column=2,sticky=E)
drop_sens = ttk.OptionMenu(frame_root, sens, sens_options[0],*sens_options)
drop_sens.grid(row = 1,column = 3)
check_simul = Checkbutton(frame_root,text='Simultaneous')
check_simul.grid(row = 1, column = 4)
check_simul.select()
check_onecell = Checkbutton(frame_root,text='Same RE/CE')
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
swv_freq = ttk.Frame(swv_tab)
swv_freq.grid(row=0,column=1,sticky=NE)
frame_root = swv_freq
lbl_col1 = ttk.Label(frame_root,text = '#').grid(row = 0, column = 0)
lbl_col2 = ttk.Label(frame_root,text = 'Frequency').grid(row = 0, column = 1)
lbl_col3 = ttk.Label(frame_root,text = 'Increment').grid(row = 0, column = 2)



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
ent_pw = ttk.Entry(master = frame_root,width = width_textbox)
ent_pw.grid(row = 0,column = 3, sticky=W)
#%% Buttons
frame_buttons = ttk.Frame(master = root)
frame_buttons.grid(row = 3, column = 0, columnspan = 2)
b_reset = ttk.Button(master=frame_buttons,text = 'Reset Technique Parameters',command = reset).grid(row = 0,column=0)
b_calculate = ttk.Button(master=frame_buttons,text = 'Print Macro',command = print_macro).grid(row = 0,column=1)
b_close = ttk.Button(master=frame_buttons,text = 'Quit',command = close_program).grid(row = 0,column=2)
set_defaults()
root.mainloop()
