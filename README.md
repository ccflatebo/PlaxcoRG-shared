# PlaxcoRG-shared
## titration-curve.py 
Python script that calculates the titration concentration values, stock concentrations and total volumes needed for additions.
Newest update allows the user to change the smallest and largest added volumes.
The clustering of points near the middle issue for the unknown Kd has been dealt with as well.
### Outputs:
1. Excel file with the concentration at each point, the stock concentration to use for each point and the volumes to add and remove
2. CSV file with the necessary stock concentrations to make and the total volumes used for each of them for the titration*
3. OPTIONAL PNG file that plots the concentrations used for visualization purposes
*the volume does NOT account for the serial dilutions you will make to create each stock
## macro_writer_gui.py:
Python script that prints out macros for CHInstruments software. CV, SWV, CA, and CC are functioning. Unchecking saved scans glitch will be fixed in next update.
1. Text file containing the macro script for easy access for users
2. MCR file to be opened by CHInstruments software
