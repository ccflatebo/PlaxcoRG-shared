# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 13:44:35 2022

@author: ccflatebo
"""
import os
from tkinter import *
from tkinter import ttk
from tkinter import filedialog


def calculate_enter(event):
    calculate()
    
def calculate():
    for entry in entry_list:
        if not entry.get():
            return
    mass_animal = float(ent_mass.get()) # 660 # g
    vol_heparin = float(ent_volhep.get()) # mL
    molmass_drug = float(ent_mw.get()) # g/mol
    dose = float(ent_dose.get()) # mg/kg
    stock_drug = float(ent_stock.get()) # M
    # num_injections = 3
    
    moles_drug = (dose * mass_animal/1000)/molmass_drug # mol
    vol_inject = moles_drug/stock_drug
    vol_inject_hep = vol_inject + vol_heparin
    # total_vol = vol_inject * num_injections
    # total_vol_hep = vol_heparin * num_injections
    final_vol.set(str(round(vol_inject,3)))
    finalhep_vol.set(str(round(vol_inject_hep,3)))

root = Tk()
root.title('Dosage Volume Calculator')
outputframe = ttk.LabelFrame(master = root, text = 'Output',
                            relief='raised',padding = '20 0 80 0')
root.columnconfigure(0, weight=2)
root.columnconfigure(1, weight=1)

root.rowconfigure(0, weight = 1)

#%% File Shit
ent_width = '10'
frame_input = ttk.LabelFrame(master = root, text = 'Inputs',
                            relief='raised',padding = '20 0 20 0')
frame_root = frame_input
frame_root.grid(row=0,column=0)
ent_mass = ttk.Entry(frame_root,width=ent_width)

lbl_mass = ttk.Label(frame_root,text='Mass Animal')
unit_mass = ttk.Label(frame_root,text = 'g')
ent_dose = ttk.Entry(frame_root,width=ent_width)
lbl_dose = ttk.Label(frame_root,text='Drug Dose')
unit_dose = ttk.Label(frame_root,text='mg/kg')
ent_mw = ttk.Entry(frame_root,width=ent_width)
lbl_mw = ttk.Label(frame_root,text='Molar Mass Drug')
unit_mw = ttk.Label(frame_root,text = 'g/mol')

ent_stock = ttk.Entry(frame_root,width=ent_width)
lbl_stock = ttk.Label(frame_root,text='Stock [Drug]')
unit_stock = ttk.Label(frame_root,text = 'M')
ent_volhep = ttk.Entry(frame_root,width=ent_width)
lbl_volhep = ttk.Label(frame_root,text='Volume of Heparin')
ent_volhep.insert(0,'0.2')
unit_volhep = ttk.Label(frame_root,text = 'mL')

entry_list = [ent_mass,ent_dose,ent_mw,ent_stock]

frame_output = ttk.LabelFrame(master = root, text = 'Outputs',
                            relief='raised',padding = '20 0 20 0')
frame_root = frame_output
lbl_injvol = ttk.Label(frame_root,text = 'Injection Volume: ')
final_vol = StringVar()
text_injvol = ttk.Label(frame_root,textvariable = final_vol)
units_injvol = ttk.Label(frame_root, text = 'mL')
lbl_injvolhep = ttk.Label(frame_root,text = 'With Heparin: ')
finalhep_vol = StringVar()
text_injvolhep = ttk.Label(frame_root,textvariable = finalhep_vol)
units_injvolhep = ttk.Label(frame_root, text= 'mL')
# ent_mass = ttk.Entry(frame_root,width='20')
# lbl_mass = ttk.Label(frame_root,text='Mass Animal')

frame_buttons = ttk.Frame(master = root)
b_calculate = ttk.Button(master=frame_buttons,text = 'Calculate',command = calculate).grid(row = 0,column=1)

# placements
frame_input.grid(row=0,column = 0)
frame_output.grid(row=1, column = 0)
frame_buttons.grid(row=2, column = 0)
ent_mass.grid(row=0,column=1,sticky=W)
lbl_mass.grid(row=0,column = 0,sticky=W)
unit_mass.grid(row = 0, column = 3, sticky = W)
ent_dose.grid(row=1,column = 1,sticky=W)
lbl_dose.grid(row=1,column = 0,sticky=W)
unit_dose.grid(row = 1, column = 3, sticky = W)

ent_mw.grid(row=2,column = 1,sticky=W)
lbl_mw.grid(row=2,column = 0,sticky=W)
unit_mw.grid(row = 2, column = 3, sticky = W)

ent_stock.grid(row=3,column = 1,sticky=W)
lbl_stock.grid(row=3,column = 0,sticky=W)
unit_stock.grid(row=3,column = 3,sticky=W)

ent_volhep.grid(row=4,column = 1,sticky=W)
lbl_volhep.grid(row=4,column = 0,sticky=W)
unit_volhep.grid(row = 4, column = 3, sticky = W)

lbl_injvol.grid(row=0,column =0,sticky = W)
text_injvol.grid(row=0,column=1)
units_injvol.grid(row=0,column = 2,sticky = E)
lbl_injvolhep.grid(row=1,column =0,sticky = W)
text_injvolhep.grid(row=1,column=1)
units_injvolhep.grid(row=1,column=2,sticky = E)

ent_mass.bind('<Return>',calculate_enter)
ent_mw.bind('<Return>',calculate_enter)
ent_stock.bind('<Return>',calculate_enter)
ent_dose.bind('<Return>',calculate_enter)
ent_volhep.bind('<Return>',calculate_enter)


root.mainloop()