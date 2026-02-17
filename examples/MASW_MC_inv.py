# -*- coding: utf-8 -*-
"""
MASW Inversion Processing Script
--------------------------------
Requires Zetica's version of MASWavesPy, available from ths Github repo:


--------------------------------
This script performs the following steps for each 1D sounding: 

1. Import picked dispersion curves (DCs) from SurfaceWavePicker output.
2. Create a CombineDCs object to hold the imported DCs.
3. Resample and plot DCs in frequency and wavelength domains.
4. Optionally remove unwanted dispersion curve points from selected frequency-velocity windows.
5. Load the initial model and initialize the inversion object.
6. Run Monte Carlo inversion to generate shear wave velocity profiles.
7. Plot and save the results, including accepted models and the median profile.
8. Save all relevant results (DCs, inversion objects, median models) to disk.

Inputs:
-------
- Dispersion curve files from `SurfaceWavePicker` (.dc format)
- Initial model file for each line (CSV format)

Outputs:
--------
- Figures: Plots of dispersion curves, initial model, sampled models, accepted models, and median profile
- Text files: Resampled and composite DCs
- Pickled objects: Inversion object, initial model, and median profile

Requirements:
-------------
- MASWavesPy installed with `combination` and `inversion` modules
- User-defined helper functions in `functions.py`
"""

import os
import pickle
from maswavespy import combination, inversion
import maswavespy.zetica_utils as zutil
import numpy as np
import time
import json


# ----- USER INPUTS -----

site = 'test' ### TYPED ENTRY ####

# Add line IDs here, only used for labelling outputs
line_list = ["VR - 7"] #### LIST OF TYPED ENTRIES ####

# Main working directory
working_dir = rf"C:\maswavespy\working_dir_test" #### CHOOSE DIRECTORY ####


# Main directory containing picked dispersion curves (.dc files)
data_dir_list = [rf"\maswavespy\data_dir_test"] #### CHOOSE DIRECTORY ####


# List of dispersion curves to ignore if needed
ignore_files = [] #### CHOOSE MULTIPLE FILES ####

# initial velocity model file
# Script automatically generates additional initial model csv files, so this should be in it's own directory
main_initial_file = rf"{working_dir}\initial_models\initial_model_const_vel.csv" #### CHOOSE FILE ####

# Set up output directories
overwrite_dir = False # If False, each run creates a timestamped results folder to prevent overwriting
results_dir, inversion_dir, initial_model_dir = zutil.results_directories(working_dir, overwrite_dir, site)
print(rf'Inversion directory created: {inversion_dir}')



# All settings

max_depth = 15

settings = {
    # general plotting
    'figsize': (18, 15),    # Output figure size (width, height)
    'max_depth': max_depth,       # Maximum depth shown in output figures
    'pseudo_depth': True,  # Plot y-axis as pseudo-depth (1/3 wavelength assumed) if true, or wavelength if wavelength
    'dc_axis_limits': [(0,1000),(0,max_depth)], # [(x min, x max), (y min, y max)]
    'models_axes_lims': [(0,1000), # [(dc x min, dc x max),
                    (0,max_depth),   #  (dc y min, dc y max), 
                    (0,1000), #  (model x min, model x max), 
                    (0,max_depth)],   #  (model y min, model y max)]                    
    
    # dispersion curves
    'no_std': 1, # Number of standard deviations for the upper/lower boundary curves of the composite dispersion curve
    'resample_n': 30, # number of points to resample comp dc at log a intervals
    'resamp_max_pdepth': None, # cut off depth for resampled curve if needed
    'dc_resamp_smoothing': True, # smooth resampled dispersion curve
    'a_values': {'VR - 7': 2},     # resampling density of disp curves in Log-a spaced wavelength bands (2-5 usually works well). Add dictionary entry with line name if variable values needed
    'only_save_accepted': False,
    'c_test': {'min': 100,        # Dispersion curve testing range min and max (Rayleigh wave vels m/s)
                'max': 1100,
                'step': 0.2, # step between test velocities and
                'delta_c': 3}, # Zero search initiation parameter [m/s] TODO better explanation of delta_c

    # iterations and model perturbation settings
    'run': 10,                      # Number of Monte Carlo runs, will be divided equally between initial models
    'N_max': 100,                  # Max iterations per run
    'repeat_run_if_fail': False,    # if no accepted models found during run, repeat run (all runs)
    'run_success_minimum': 0,       # runs will repeat until minimum successful runs completed
    'bs': 20,                       # Max % perturbation of Vs per iteration
    'bh': 10,                       # Max % perturbation of layer thickness per iteration
    'bs_min': 4,                    # minimum % perturbation after decay
    'bh_min': 2,                    # minimum % perturbation after decay
    'b_decay_iterations': 200,      # iterations from start to decay for bs and bh to decay to min
    'bs_min_halfspace': None,
    'N_stall_max': 0,               # how many time to increase bh and bs if misfit stall in st/lt window
    'st_lt_window': (200, 1000),

    # misfit settings
    'misfit_mode': 'average',       # 'average', 'average_weighted', 'max', 'max_weighted', see misfit method in inversion.py
    'uncertainty_weighting': False,
    'depth_weight_offset_const': 0,         # 1 = depth weighting function ranges from 1 to 0.5, 0 = 1 to 0, 2 = 1 to 1 to 0.666, etc 
    'plot_misfit_weights': False,  
    'within_bounds_thresh':  1,              # 1 = theoretical DC must fit within 100% of uncertainty bounds, 0.9 = 90% must fit, etc

    # regularisation
    'regularisation': 'model',                  # Penalises curvature in model ('model') or dispersion curve ('dc')
    'regularisation_ranges': None,            # list of (min, max) tuples,  1/3 wavelength (dc) or depth (model) range over which regularisation is applied.

    # misfit penalty weights (all min 0 to max 1)
    'uncert_bounds_misfit_weight': 20,        # misfit penalty for any points outside measured DC uncertainty
    'rep_explore_pen_weight': 0.2,             # repulsion based exploration penalty weight
    'regularisation_weights': [1e-7],

    # velocity reversal settings
    'rev_min_depth': None,       # reversals allowance settings
    'rev_max_depth': None,
    'rev_min_layer': None,
    'rev_max_layer': None,
    'max_retries': 200,            # max retries to find valid profile if using min or max reversal parameters
    'force_monotonic': False,         # ensure normal dispersive velocity structure (Vs increasing with depth)
    'monotonic_tol': 100,              # if reversals are constrained, max allowable tolerance for change in Vs in m/s e.g in a layer where reversal aren't allowed, 10 will allow a -10 m/s chnage in Vs between layers
    
    # initial model settings
    'vel_shift': 150,                 # shift velocity to create multiple starting models
    'n_initial_models': 5,            # must be odd, e.g. if 5 will create 5 new models shifted by -2*vel_shift, -vel_shift, 0, vel_shift, 2*vel_shift
    'new_run_prev_best_fit': False,   # use the lowest misfit model from the previous run as initial model
    'runs_initial': 1                 # number of runs before initial model used again
}


# Optional: Remove selected regions (freq-vel space) from DCs (set to None to skip)
# Each tuple is (f_min, f_max, c_min, c_max) c = rayleigh wave vel
points_to_remove = None


# ----- MAIN PROCESSING LOOP -----
print("\n")
print("Inversion process started")
print("\n")
for i, line in enumerate(line_list):
    print(rf"Processing line {line}")

    start_time = time.time()

    # === 1. line specific directories
    line_data_dir = data_dir_list[i]

    # === 2. Import and Prepare Dispersion Curves ===
    f_vec, c_vec = zutil.data_import(line, line_data_dir, inversion_dir, points_to_remove, ignore_files)
    combDC_obj = combination.CombineDCs(site, line, settings, freq=f_vec, c=c_vec)
    print('Data imported and CombineDCs object initialized.')

    # === 3. Resample and Plot Dispersion Curves ===
    dc_freq_plot_file = os.path.join(inversion_dir, f"1_dc_cov_{line}.png")
    dc_wavelength_plot_file = os.path.join(inversion_dir, f"2_combined_dc_{line}.png")
    dc_resamp_plot_file = os.path.join(inversion_dir, f"3_resampled_dc_{line}.png")

    zutil.dc_resamp_and_plot(line, combDC_obj, dc_freq_plot_file, dc_wavelength_plot_file, dc_resamp_plot_file)
    print(f'Dispersion curve plotted with {settings['no_std']} standard deviation uncertainty bounds')
    print('Dispersion curves resampled and plotted.')

    # Save composite DC to file
    zutil.save_combined_dc(combDC_obj, save_file=os.path.join(inversion_dir, f"{line}_combined_dc.txt"))

    # === 4. Set Up Inversion ===
    c_mean = combDC_obj.resampled['c_mean']
    c_low = combDC_obj.resampled['c_low']
    c_up = combDC_obj.resampled['c_up']
    wavelengths = combDC_obj.resampled['wavelength']

    # new inversion object for the line
    inv_obj = inversion.InvertDC(site, line, c_mean, c_low, c_up, wavelengths, settings)
    print('Initial model imported and inversion object initialized.')

    # Save settings to JSON file
    with open(rf"{inversion_dir}\settings.json", "w") as f:
        json.dump(settings, f, indent=4)
    
    # create new additional initial models with velocity shifts here instead manually in excel

    vel_shift_array = np.linspace(0, settings["n_initial_models"]-1, settings["n_initial_models"])*settings["vel_shift"]
    middle_index = len(vel_shift_array) // 2
    vel_shift_array = vel_shift_array-vel_shift_array[middle_index]

    initial_file_list = []

    for j in range(settings["n_initial_models"]):
        output_path = rf"{working_dir}\initial_models\initial_model_const_vel_{j+1}.csv"
        new_initial_model = zutil.new_initial_models(main_initial_file, output_path, "Vs [m/s]", vel_shift_array[j])
        initial_file_list.append(new_initial_model)

    initial_list = []

    for j, initial_file in enumerate(initial_file_list):

        # === 5. Load Initial Model ===
        initial = zutil.import_initial_model(initial_file, settings)
        
        initial_list.append(initial)

        # === 6. Plot Initial Model Fit ===
        zutil.plot_initial_model(line, inv_obj, initial, save_file=os.path.join(initial_model_dir, f"4_initial_model_{line}_{j+1}.png"))
        
    print('Initial model and associated dispersion curve plotted.') 

    # === 7. Run Inversion ===
    print(f'Inversion of profile {line} initiated.')
    inv_obj.mc_inversion_MultInitModels(initial_list, settings)
    print('All runs completed.')

    # ### plot misfit, bh and bs evolution ###
    # plot_misfit_bh_bs(inv_obj, inversion_dir, line)

    end_time = time.time()
    elapsed_time = round(end_time - start_time) 

    print(f"Line {line} inversion time: {elapsed_time} seconds")


    # === 8. Plot Sampled Vs Models ===
    zutil.plot_sampled_models(line, inv_obj, save_file=os.path.join(inversion_dir, f"5_sampled_model_{line}.png"))
                        
    print('Sampled Vs profiles plotted.')


    # === 9. Print Inversion Info ===
    zutil.inv_info(line, inv_obj, settings["within_bounds_thresh"])

    # check if any profiles accepted
    inv_obj.within_boundaries(runs='all')
    if len(inv_obj.selected['c_t']) == 0:
        print("No profiles fit within measured uncertainty bounds")
        continue

    # === 10. Plot Accepted Models ===
    zutil.plot_accepted_models(line, inv_obj, inversion_dir, save_file=os.path.join(inversion_dir, f"6_accepted_Vs_profiles_{line}.png"))

    # === 11. Compute and Plot Median Profile ===
    median_profile = zutil.plot_median_profile(line, inv_obj, initial, save_file=os.path.join(inversion_dir, f"7_median_profile_{line}.png"))
    print('Median Vs profile computed and plotted.')


    # === 12. Save Results as Pickle Files ===
    pickle.dump(inv_obj, open(os.path.join(inversion_dir, f"inversion_obj_{line}.pkl"), "wb"))
    pickle.dump(initial, open(os.path.join(inversion_dir, f"initial_model_{line}.pkl"), "wb"))
    pickle.dump(median_profile, open(os.path.join(inversion_dir, f"median_profiles_{line}.pkl"), "wb"))

# Save settings to JSON file
with open(rf"{inversion_dir}\settings.json", "w") as f:
    json.dump(settings, f, indent=4)


