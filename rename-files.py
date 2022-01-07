# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 15:48:33 2021

@author: ccflatebo
"""

import os
from tkinter import *
from tkinter import ttk
from tkinter import filedialog

def clicked(idx):
    global directories
    expdir = filedialog.askdirectory()
    directories[idx].delete(0,'end')
    directories[idx].insert(END,expdir)
    iteration = 0
    for files in os.listdir(expdir):
        if iteration == 0:
            ent_exp.delete(0,'end')
            ent_exp.insert(END,files.split('_')[0])
            iteration += 1
        else:
            break

def close_program():
    root.destroy()
    
def rename_files():
    rawdir = directories[0].get() + '/'
    root = Tk() 
    files_to_convert = filedialog.askopenfilenames(initialdir = rawdir, title = 'Select Files', filetypes = [("Text Files","*.txt"),('All files','*.*')]) 
    # files = files[1:] + tuple([files[0]]) # corrects for ordering 
    root.destroy() 
    convertdir = directories[1].get() + '/'
    increase = ent_macro.get()
    if not os.path.exists(convertdir):
        os.makedirs(convertdir)
    for x in files_to_convert:
        filename = x.split('/')[-1]
        filename_split = filename.split('_')
        if filename_split[0] != ent_exp.get():
            filename_split[0] = ent_exp.get()
        filename_idx = int(filename_split[-1].split('.')[0])
        filename_start = filename_split[0] + '_' + filename_split[1] + '_' + str(filename_idx + int(increase))
        os.rename(rawdir + filename,convertdir + filename_start + '.txt')
        os.rename(rawdir + filename.split('.')[0] + '.bin',convertdir + filename_start + '.bin')
    messagebox.showinfo(title = 'Renaming', message = 'Renamed files from: ' + rawdir + '\nNew Location: ' + convertdir)


root = Tk()
root.title('Batch Rename')
outputframe = ttk.LabelFrame(master = root, text = 'Output',
                            relief='raised',padding = '20 0 80 0')
root.columnconfigure(0, weight=2)
root.columnconfigure(1, weight=1)

root.rowconfigure(0, weight = 1)

#%% File Shit
frame_file = ttk.Frame(root)
frame_root = frame_file
frame_root.grid(row=0,column=0)
directory = ttk.Entry(frame_root,width='60')
directory.grid(row=0,column=0,sticky=W,columnspan=3)
btn = ttk.Button(frame_root,text='Select Original Files Directory >>',command=lambda:clicked(0))
btn.grid(row=0,column = 3)
directory_conv = ttk.Entry(frame_root,width='60')
directory_conv.grid(row=1,column=0,sticky=W,columnspan=3)
btn = ttk.Button(frame_root,text='Select New Directory >>',command=lambda:clicked(1))
btn.grid(row=1,column = 3)
lbl_macro = ttk.Label(frame_root,text='Number to increase by:').grid(row=2,column = 0)
ent_macro = ttk.Entry(frame_root,width = '30')
ent_macro.grid(row=2,column = 1)
ent_macro.delete(0,'end')
ent_macro.insert(END,0)
lbl_exp = ttk.Label(frame_root,text='Leading label:').grid(row=3,column = 0)
ent_exp = ttk.Entry(frame_root,width = '30')
ent_exp.grid(row=3,column = 1)

directories = [directory,directory_conv]
frame_buttons = ttk.Frame(master = root)
frame_buttons.grid(row = 3, column = 0, columnspan = 2)
b_calculate = ttk.Button(master=frame_buttons,text = 'Rename',command = rename_files).grid(row = 0,column=1)
b_close = ttk.Button(master=frame_buttons,text = 'Quit',command = close_program).grid(row = 0,column=2)

root.mainloop()