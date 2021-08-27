# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

####### You can EDIT the VALUES in this section ##########

class params(object): # Default parameters
    filedir = '../Data/Test/' # where you want your file saved
    filename = 'Titration_removed' # name of your saved files
    save_titrationfig = 'y' # (y/n) this just lets you choose whether or not you want to save the plot of the concentrations
    """ Concentration Parameters """
    unit = 'u' # Unit for highest concentration desired
    # Choose 'u' for micromolar, 'n' for nanomolar and 'IU' for hormones FML
    stock_conc = 0.02 # desired stock target molecule concentration (M) or IU for hormones
    conc_min = 1 # this is the lowest concentration to use and is 10^-3 of chosen unit
    conc_max = 1000 # this is highest concentration to use and matches chosen unit
    """ Shot Glass Parameters """
    start_vol = 0.02 # Starting volume (L) of buffer in shot glass
    remove = 'y' # 'y' for removing volume, 'n' for only adding, 'a' for autotitrator
    end_vol = 0.03 # size of shot glass volume (L) allowed (throws an error if you go too high)
    ## This is the range of volumes you want to add. I guarantee the smallest volume added will always be the lower limit
    ## *TO PREVENT AN INFINITE LOOP*  although everything I've test so far works...I don't guarantee the largest volume added
    smallest_vol_added = 2 # uL Smallest volume you think the pipette is accurate for
    largest_vol_added = 1000 # uL Largest volume you'd like to add to the shot glass
    """ Titration Parameters """
    num_measurements = 50 # total number of measurements you want to take
    ## Ratio frac_low:frac_mid:frac_high - Default 1:1:1 ##
    frac_low = 1 # this is the ratio of measurements in the lower range
    frac_mid = 1 # this is the ratio of measurements in the mid range
    frac_high = 1 # this is the ratio of measurements in the upper range
    """ Optional Kd Parameters """
    ## Do you know your Kd? if not please don't touch me ##
    k_d = [] # proposed Kd, use [] if unknown and numbers (like 60 uM) if known. This value uses the units you chose above at "unit"
    k_d_spread = 50 # spread of points surrounding Kd i.e. +/- 50 uM 
    #* I highly advise keeping k_d_spread < k_d if you want to use this functionality
    header_array = ['Desired Conc (M)', 'Stock Conc (M)', 'Added Volume (uL)',
                    'Removed Volume (uL)','Final Conc after Added Volume (M)']
    # if you do hormone stuff, this is useful, if not, ignore
    """ Hormone Annoyance """
    conv_IU = 5.6 # only used for hormones
    conv_unit = 'n' # only used for hormone IU
    
#%% NO TOUCHY %%#
# Functions for calulating concentrations and what to add
def c2v2(stock_conc,start_conc,wanted_conc,vol):
    final_conc = wanted_conc - start_conc
    return (final_conc * vol)/(stock_conc - final_conc)

def c2v2_verify(stock_conc,start_conc,add_vol,total_vol):
    final_conc = (stock_conc * add_vol - start_conc*total_vol)/total_vol
    return (stock_conc * add_vol + start_conc * (total_vol - add_vol))/total_vol

titration_params = params # initializes the class

# Selects the conversions we want to use based on the user input
base = 10
prefix = 0
if titration_params.unit == 'u':
    lowlim = -9
    conversion = -6
elif titration_params.unit == 'n':
    lowlim = -12
    conversion = -9
elif titration_params.unit == 'IU':
    if titration_params.conv_unit == 'u':
        lowlim = -9
        conversion = -6
    elif titration_params.conv_unit == 'n':
        lowlim = -12
        conversion = -9
    titration_params.stock_conc = titration_params.stock_conc * (titration_params.conv_IU * base**conversion)
    titration_params.header_array.append('IU concentration')
    titration_params.header_array.append('IU Stock')
else:
    print("Sorry, the selected unit doesn't exist yet for this script")

# Determines the range of concentrations to use and how many points per region
conc_min = titration_params.conc_min * base**lowlim 
conc_max = titration_params.conc_max * base**conversion 
frac_denominator = float(titration_params.frac_low + titration_params.frac_mid + titration_params.frac_high)
frac_low = titration_params.frac_low/frac_denominator
frac_mid = titration_params.frac_mid/frac_denominator
frac_high = titration_params.frac_high/frac_denominator

num_measurements_low = int(np.round(frac_low * titration_params.num_measurements))
num_measurements_mid = int(np.round(frac_mid * titration_params.num_measurements))
num_measurements_high = int(np.round(frac_high * titration_params.num_measurements))

measurements_total = num_measurements_low + num_measurements_mid + num_measurements_high
measurements_diff = measurements_total - titration_params.num_measurements

# makes sure you have the right number of measurements
if measurements_diff > 0: 
    # print('Too Many Measurements')
    if (measurements_diff)%2 == 0:
        num_measurements_low-=measurements_diff//2
        num_measurements_high-=measurements_diff//2
    else:
        num_measurements_low-=measurements_diff%2
        num_measurements_high-=measurements_diff//2
elif measurements_diff < 0:
    # print('Too few measurements')
    measurements_diff = np.abs(measurements_diff)
    # print((measurements_diff)%2)
    if (measurements_diff)%2 == 0:
        num_measurements_low-=measurements_diff//2
        num_measurements_high-=measurements_diff//2
    else:
        num_measurements_low+=measurements_diff%2
        num_measurements_high+=measurements_diff//2
measurements_total = num_measurements_low + num_measurements_mid + num_measurements_high

# Create the figure for plotting the concentration at each measurement
fig, ax = plt.subplots()
ax.set_ylabel('Concentration (M)')
ax.set_xlabel('Measurement #')

# Decide what concentrations to use to spread them evenly on the log scale
if not titration_params.k_d: # Kd is Unknown
    if measurements_total%3 == 0: # deals with odd number of measurements chosen
        low_half = measurements_total//3
        upper_half = measurements_total//3
    else:
        low_half = measurements_total//3
        upper_half = measurements_total//3+1
    conc_all_temp = np.geomspace(conc_min,conc_max,titration_params.num_measurements)
    conc_low = np.geomspace(conc_min,conc_all_temp[low_half],num_measurements_low,endpoint=False)
    conc_high = np.geomspace(conc_all_temp[-upper_half],conc_max,num_measurements_high,endpoint=True)
    conc_mid = np.geomspace(conc_all_temp[low_half],np.amin(conc_high),num_measurements_mid,endpoint=False)
    conc_all = np.unique(np.concatenate((conc_low,conc_mid,conc_high),axis = 0))
else: # Kd is known
    if num_measurements_mid%2 == 0: # deals with odd number of measurements chosen
        low_half = num_measurements_mid//2
    else:
        low_half = num_measurements_mid//2 + 1
    if titration_params.k_d_spread >= titration_params.k_d: # deals with the weird spread error
        conc_mid_low = np.geomspace(titration_params.k_d * base**conversion - titration_params.k_d_spread * base**lowlim,
                               titration_params.k_d * base**conversion,
                               low_half,endpoint=False)
        conc_mid_high = np.geomspace(titration_params.k_d * base**conversion,
                                     titration_params.k_d * base**conversion + titration_params.k_d_spread * base**conversion,
                                     num_measurements_mid//2)
    else:
        conc_mid_low = np.geomspace(titration_params.k_d * base**conversion - titration_params.k_d_spread * base**conversion,
                               titration_params.k_d * base**conversion,
                               low_half,endpoint=False)
        conc_mid_high = np.geomspace(titration_params.k_d * base**conversion,
                                     titration_params.k_d * base**conversion + titration_params.k_d_spread * base**conversion,
                                     num_measurements_mid//2)
    conc_low = np.geomspace(conc_min,np.amin(conc_mid_low)-base**conversion,num_measurements_low)
    conc_high = np.geomspace(np.amax(conc_mid_high) + base**conversion,conc_max,num_measurements_high)
    conc_all = np.unique(np.concatenate((conc_low,conc_mid_low,conc_mid_high,conc_high),axis = 0))
    ax.hlines(titration_params.k_d * base**conversion,0,titration_params.num_measurements,colors='r',linestyle = 'dashed')
    ax.annotate('Selected $K_d$',(0,(titration_params.k_d-30) * base**(conversion)),xycoords='data',color = 'r')
ax.vlines([num_measurements_low,num_measurements_low+num_measurements_mid],0,np.amax(conc_all),colors='k',linestyle = 'dotted')
ax.semilogy(conc_all,'x')
if titration_params.unit == 'IU':
    axy = ax.twinx()
    axy.semilogy(conc_all / (titration_params.conv_IU*base**conversion),'o',color = 'm',fillstyle= 'none')
    axy.set_ylabel('Concentration (IU)')

measurements_final = len(conc_all)
vol = titration_params.start_vol
conc_start = 0
added_vol = []
if titration_params.unit == 'IU':
    conc_array = np.zeros((measurements_final,7))
else:
    conc_array = np.zeros((measurements_final,5))

# Determines additions and subtractions and the stock concentrations to use
for idx, x in enumerate(conc_all):
    conc_array[idx,0] = x
    # Calculates what added volume you would use for the stock concentration
    add_vol = np.round(c2v2(titration_params.stock_conc,conc_start,x,vol) * 10**6,decimals = 5)
    # Limits added volume to the range you provided
    while add_vol > float(titration_params.largest_vol_added):
        prefix+=1
        add_vol = np.round(c2v2(titration_params.stock_conc*base**prefix,conc_start,x,vol) * 10**6,decimals = 5)
    while add_vol < float(titration_params.smallest_vol_added):
        prefix-=1
        add_vol = np.round(c2v2(titration_params.stock_conc*base**prefix,conc_start,x,vol) * 10**6,decimals = 5)
    # Logs the data in an array so you know what to add
    conc_array[idx,1] = titration_params.stock_conc * base**prefix
    conc_array[idx,2] = np.round(add_vol,decimals = 1)
    vol+=np.round(add_vol,decimals = 1)*10**-6 # makes sure the volume is updated accordingly
    conc_array[idx,4] = c2v2_verify(conc_array[idx,1],conc_start, np.round(add_vol,decimals = 1)*10**-6, vol) # Calculates true concentration at the titration point
    # If you're using the autotitrator, removes for every point
    if titration_params.remove == 'a':
        total_vol_added = vol - titration_params.start_vol
        conc_array[idx,3] = total_vol_added*10**6
        vol-=total_vol_added
    # if you want to remove volume but not do it at every point
    elif titration_params.remove == 'y' and vol/titration_params.start_vol > 1.01: # only tells you to remove volume if you've gone above 1% of the total volume
        total_vol_added = vol - titration_params.start_vol
        conc_array[idx,3] = total_vol_added*10**6 # logs the amount needed to be removed
        vol-=total_vol_added
    # Only for hormone calculations
    if titration_params.unit == 'IU':
        conc_array[idx,5] = conc_array[idx,4] / (titration_params.conv_IU*base**conversion)
        conc_array[idx,6] = conc_array[idx,1] / (titration_params.conv_IU*base**conversion)
    conc_start = x
    prefix = 0
if vol >= titration_params.end_vol: # this error should only throw if your starting volume == end volume
    print('Overflow, try new stock concentration or change setting to remove volume during the experiment')
    
# Figures out all of the stock concentrations you need and the volumes required
if titration_params.unit == 'IU':
    unique_stocks = np.unique(conc_array[:,-1])
else:
    unique_stocks = np.unique(conc_array[:,1])

total_vol_perstock = np.zeros_like(unique_stocks)
for item in range(0,np.shape(conc_array)[0]):
    if titration_params.unit =='IU':
        idx = np.where(unique_stocks==conc_array[item,-1])
    else:
        idx = np.where(unique_stocks==conc_array[item,1])
    total_vol_perstock[idx] = total_vol_perstock[idx] + conc_array[item,2]
    
#### SAVING STUFF ######
# Saves Text file with Stock Concentrations and corresponding total volumes needed
output_txtfile = open(titration_params.filedir + titration_params.filename + '.csv','w')
output_txtfile.write('Stock Concentrations (M), Total Vol (uL)\n')
for idx, val in enumerate(unique_stocks):
    output_txtfile.write(str(val) + ', ' + str(total_vol_perstock[idx]) + '\n')
output_txtfile.close()
    
# Save to an excel file
df = pd.DataFrame(conc_array,columns=titration_params.header_array)
df.to_excel(titration_params.filedir + titration_params.filename +'.xlsx',index = False,header = True)

# Saves the figure of the selected concentrations
if titration_params.save_titrationfig == 'y':
    plt.tight_layout()
    plt.savefig(titration_params.filedir + titration_params.filename + '.png')