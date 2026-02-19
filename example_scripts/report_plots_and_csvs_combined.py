"""
MASW Post-Processing Script for MASWavesPy

This script performs post-processing and reporting for a MASW campaign using MASWavesPy.
It generates and saves:
- Binned and trimmed dispersion curve plots
- Mean dispersion curves
- Median Vs profiles with and without error bars
- Accepted model plots within dispersion curve bounds
- CSV exports of measured and modelled phase velocities and depth profiles

Inputs:
- .dc files (picked dispersion curves)
- .pkl files from previous inversion runs (initial models, inversion results, median profiles)

Ensure all paths and filenames are correctly set before running.
"""

# --- Imports ---
import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cmcrameri import cm
import random

from os import mkdir, path
from maswavespy import cy_theoretical_dc as t_dc
from maswavespy import zetica_utils as zutil


# --- Directories ---
#DC_dir = rf"R:\Projects\Live\P14072-24 SEGL Abingdon Engineering\SurfaceWavePicker\May 2025\exports\2-30 (450-0,300-100)_full spread"
working_dir = rf"R:\Projects\Live\P15628-25 Align JV South Heath C1 (HS2) Eng\MASWavesPy\September_2025"
results_dir = rf"R:\Projects\Live\P15628-25 Align JV South Heath C1 (HS2) Eng\MASWavesPy\September_2025\best_results"

within_bounds_threshold = 1

# --- Parameters ---
site = 'P15628-25_sept25'
pseudo_depth = True
max_depth = 15
figsize = (7.8, 7)
Vr_x_lim_low = 0
Vr_x_lim_up = 1000
Vs_x_lim_low = 0
Vs_x_lim_up = 1200
no_std_files = 1


colours = ['blue', 'black', 'red', 'orange', 'green', 'purple']
# colours = ['black', '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442']
misfit_accept_percentile = 100


inversion_dir = rf"R:\Projects\Live\P15628-25 Align JV South Heath C1 (HS2) Eng\MASWavesPy\September_2025\best_results\inv_P15628-25_sept25_2025-09-23_16-21"

report_plot_dir = rf"{inversion_dir}\report"
if not path.exists(report_plot_dir):
    mkdir(report_plot_dir)
        
# finalised line names: 0-50 = VR-1, 50-100 = VR-2, 100-150 = VR-3
line_list = ["VR - 7"]
title_line_list = ["VR - 7"]
#line_list = ["CT1"]

# Velocity test range
c_test = {
        'min': 200,
        'max': 1200,
        'step': 1,
        'delta_c': 3
    }







########## --- Plot: all median profiles together ---

print("plotting single plot with all lines ...\n")

legend_fontsize = 9

# with error bars
fig, ax = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
up_low = False

# median_legend_list = ["CT1 mean measured", "CT1 median modelled", "CT2 mean measured", "CT2 median modelled", "CT3 mean measured", "CT3 median modelled"]
median_legend_list = []
model_legend_list = title_line_list

for i, line in enumerate(line_list):

    inv_obj = pickle.load(open(rf"{inversion_dir}\inversion_obj_{line}_std{no_std_files}.pkl", 'rb'))
    initial = pickle.load(open(rf"{inversion_dir}\initial_model_{line}_std{no_std_files}.pkl", 'rb'))
    inv_obj.within_boundaries(runs='all', threshold=within_bounds_threshold)

    median_profile = inv_obj.median_profile(q=[10,90], dataset='selected')

    inv_obj.plot_Nprofiles(
        median_profile, line, initial,
        col=colours[i], up_low=up_low,
        fig=fig, ax=ax, return_axes=True
    )

    # lowest_misfit = find_lowest_misfit(inv_obj, initial, max_depth, c_test)
    # median_legend_list.append(f"{line} observed mean")
    # median_legend_list.append(f"{line} median model")
    # median_legend_list.append(f"MM: {round(inv_obj.e_median_profile, 2)} %, LM: {round(lowest_misfit,2)} %")


# Add a single legend after all plots
# ax[0].legend(median_legend_list, fontsize=legend_fontsize, frameon=False, labelspacing=0.4, loc='lower left')
ax[0].legend(frameon=False, loc='lower left')
#ax[1].legend(model_legend_list, fontsize=legend_fontsize, frameon=False, labelspacing=0.4)

ax[0].set_title("Theoretical dispersion curve")
ax[1].set_title("Median profile")
plt.savefig(rf"{report_plot_dir}\{len(title_line_list)}_median_profiles.png")
plt.close()






############### --- Plot: Median Profiles (individual) ---

print("plotting individual line median profiles ...\n")

for i, line in enumerate(line_list):
    
    inv_obj = pickle.load(open(rf"{inversion_dir}\inversion_obj_{line}_std{no_std_files}.pkl", 'rb'))
    initial = pickle.load(open(rf"{inversion_dir}\initial_model_{line}_std{no_std_files}.pkl", 'rb'))
    inv_obj.within_boundaries(runs='all', threshold=within_bounds_threshold)
    median_profile = inv_obj.median_profile(q=[10,90], dataset='selected')

    fig, ax = inv_obj.plot_profile_ReportFigs(median_profile, initial, col='red', up_low=True, return_axes=True)
    fig.suptitle(rf"Median profile, {title_line_list[i]}")
    plt.savefig(rf"{report_plot_dir}\{title_line_list[i]}_median_profile.png")
    plt.close()


################## --- Plot: Accepted Profiles w/ median

print("plotting individual line accepted profiles ...\n")

for i, line in enumerate(line_list):

    initial = pickle.load(open(rf"{inversion_dir}\initial_model_{line}_std{no_std_files}.pkl", 'rb'))
    inv_obj = pickle.load(open(rf"{inversion_dir}\inversion_obj_{line}_std{no_std_files}.pkl", 'rb'))
    # median_profile = pickle.load(open(rf"{inversion_dir}\median_profiles_{line}_std{no_std_files}.pkl", 'rb'))
    inv_obj.within_boundaries(runs='all', threshold=within_bounds_threshold)
    median_profile = inv_obj.median_profile(q=[10,90], dataset='selected')


    fig, ax = inv_obj.plot_within_boundaries_MedianProfiles(median_profile, initial, show_all=False, runs='all', col_map='viridis', 
                                                            colorbar=True, DC_yaxis='linear', return_axes=True, show_legend=True)
    
    fig.suptitle(f'Accepted models, {title_line_list[i]}')
    plt.savefig(rf"{report_plot_dir}\{title_line_list[i]}_accepted_profiles_with_median.png")
    plt.close()



