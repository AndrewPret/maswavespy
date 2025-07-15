# -*- coding: utf-8 -*-
"""
Test MASWavesPy wavefield and dispersion modules (1)

This example covers the following:
- Import multi-channel seismic data from text files or waveform files 
  (one file for each record). Note that when working with large datasets 
  containing multiple shot gathers, the use of Dataset objects is recommended
  (see examples/MASWavesPy_Dispersion_test2).
- Plot imported data.
- Compute and view the dispersion image (phase velocity spectrum) of the 
  imported data.
- Identify/pick dispersion curve based on spectral maxima (GUI application).
- Return identified dispersion curve as a dictionary.

Input file (prepared):
- Data/Oysand_dx_2m_x1_30m_forward.dat (MASW shot gather)

Outputs:
- rec_TestSite: Initialized RecordMC object (type wavefield.RecordMC).
- Plot showing the imported wavefield.
- Plot showing the dispersion image of the recorded wavefield.
- edc_TestSite: Initialized ElementDC object (type dispersion.ElementDC).
- Picked elementary dispersion curve, saved to the ElementDC object edc_TestSite.
- Plot showing the identified elementary dispersion curve.

"""
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
from maswavespy import wavefield
import pickle
import matplotlib.pyplot as plt
import numpy as np

# ------ Dispersion processing: Import seismic data ------
# Import data from a text file, general requirements:      
# - By default, it is assumed that the recorded data is stored in 
#   a whitespace/tab-delimited text file. If a different string is 
#   used to separate values, it should be provided as an additional 
#   keyword argument (e.g., delimiter=',' for a comma-separated textfile).
# - The number of header lines, including comments, is header_lines.
# - Each trace must be stored in a single column.
# - All traces must be of equal length and missing values are not 
#   allowed (each row in the text file must include the same number
#   of values).
#
# Seismic data can also be imported from any waveform file that can be read by 
# the obspy.read() function from the ObsPy library using the from_waveform
# class method. See https://docs.obspy.org/packages/autogen/obspy.core.stream.read.html
# for a list of supported file formats. 

# Initialize a multi-channel record object from a text 

file_name = r"O:\SysEng SW\Projects_General_Access\RD0077 Sensonic\passive-MASW\results_from_7thJuly\AZD_site1\VSG\all_trains_section_74_MidP_chan_1184_VSG_NOT_SEG2.dat"; header_lines = 0
section = "section74"

n = 30                 # Number of receivers
direction = 'forward';   # Direction of measurement
dx = 1.595                 # Receiver spacing [m]
x1 = 2*dx                # Source offset [m]
fs = 200                # Sampling frequency [Hz]
f_pick_min = 4.5       # Only identify the dispersion curve at frequencies higher or equal to f_pick_min

# Create a multi-channel record object
site = 'AZD1'
profile = 'P1'
rec_TestSite = wavefield.RecordMC.import_from_textfile(site, profile, file_name, header_lines, n, direction, dx, x1, fs, f_pick_min)

# Plot the recorded wavefield
rec_TestSite.plot_data(du=0.75, normalized=False, filled=True)
plt.show()
# Print message to user
print('A multi-channel record containing ' + str(n) + ' traces has been imported as a RecordMC object from the file ' + file_name)
print('The imported wavefield has been plotted.')

#%%
# Compute and view dispersion image of recorded wavefield
# Create a elementary dispersion curve object
cT_min = 2; cT_max = 1000; cT_step = 0.5
edc_TestSite = rec_TestSite.element_dc(cT_min, cT_max, cT_step)

f_min = 2; f_max = 50

for i in range(edc_TestSite.A.shape[0]):
    max_val = np.max(edc_TestSite.A[i, :])
    if max_val != 0:
        edc_TestSite.A[i, :] /= max_val
  
edc_TestSite.plot_dispersion_image(f_min, f_max)

# Print message to user
print('An elementary dispersion curve object has been initialized.')
print('The dispersion image of the imported wavefield has been plotted.')

#%%
# Identify elementary dispersion curves (DC) based on spectral maxima.
# The identified DC is saved to the ElementDC object (edc_TestSite in this tutorial).
# Please note that the GUI for dispersion curve identification will open in a new window.
# Instructions on how to use the dispersion curve identification tool are provided.
# within the GUI 
edc_TestSite.pick_dc(f_min, f_max)

#%%
# Return the identified elementary DC as a dictionary. 
edc_TestSite_dict = edc_TestSite.return_to_dict() 

np.savetxt(rf"C:\Users\andrew.pretorius\Documents\azd1_manual picks\f0_{section}.txt", edc_TestSite_dict['f0'])
np.savetxt(rf"C:\Users\andrew.pretorius\Documents\azd1_manual picks\c0_{section}.txt", edc_TestSite_dict['c0'])

#pickle.dump(edc_TestSite,r"O:\SysEng SW\Projects_General_Access\RD0077 Sensonic\passive-MASW\results_from_7thJuly\AZD_site1\DC_picks_manual\all_trains_section_75_MidP_chan_1200.pkl", "wb")

# Plot returned DC
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,1)
#edc_TestSite.plot_dispersion_image(f_min, f_max, fig=fig, ax=ax)
ax.plot(edc_TestSite_dict['f0'], edc_TestSite_dict['c0'],'o') 
ax.grid()
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('Phase velocity [m/s]')
ax.set_xlim([f_min, f_max])
ax.set_ylim([cT_min, cT_max])
plt.savefig(rf"C:\Users\andrew.pretorius\Documents\azd1_manual picks\picks_{section}.png")
plt.show()

# Print message to user
print('The identified dispersion curve has been plotted.')

