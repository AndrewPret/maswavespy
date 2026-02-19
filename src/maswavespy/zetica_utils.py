from obspy import read
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from maswavespy import inversion
from maswavespy import cy_theoretical_dc as t_dc
import glob
from os import mkdir, path
from datetime import datetime
from os.path import basename
import pickle
import random
from maswavespy import cy_theoretical_dc as t_dc



# functions for main script MASW_MC_inv.py

def results_directories(working_dir, overwrite_dir, site):
### define and create directories if needed, create new timestamped directory
    results_dir = rf"{working_dir}\results"
    if not path.exists(results_dir):
        mkdir(results_dir)

    if overwrite_dir == True:
        inversion_dir = rf"{results_dir}\inv_{site}"
        if not path.exists(inversion_dir):
            mkdir(inversion_dir)

    elif overwrite_dir == False:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        inversion_dir = rf"{results_dir}\inv_{site}_{timestamp}"
        if not path.exists(inversion_dir):
            mkdir(inversion_dir)
        initial_model_dir = rf"{inversion_dir}\initial_models"
        if not path.exists(initial_model_dir):
            mkdir(initial_model_dir)

    return results_dir, inversion_dir, initial_model_dir

def data_import(line, line_data_dir, line_inv_dir=None, points_to_remove=None, ignore_files=None):
    """ function to import .dc file picks and convert them into correct format for MASWavesPy"""
    file_pattern = os.path.join(line_data_dir, "*.dc") 
    file_list_full = sorted(glob.glob(file_pattern))
    if ignore_files:
        file_list = [f for f in file_list_full if basename(f) not in ignore_files]
    else:
        file_list = file_list_full

    f_combined = []
    c_combined = []

    for file in file_list:
        f, c = load_SWPicker_DC(file)
        if points_to_remove:
            line_boxes = points_to_remove.get(line, [])
            f, c = remove_points_in_boxes(f, c, line_boxes)
            global_boxes = points_to_remove.get('ALL', [])
            f, c = remove_points_in_boxes(f, c, global_boxes)
        f_combined.append(f)
        c_combined.append(c)

    if line_inv_dir:
        save_list_matrix_to_txt(c_combined, rf"{line_inv_dir}\{line}_c_combined_list.txt")
        save_list_matrix_to_txt(f_combined, rf"{line_inv_dir}\{line}_f_combined_list.txt")

    f_vec = [np.array(f) for f in f_combined]
    c_vec = [np.array(c) for c in c_combined]

    return f_vec, c_vec

def dc_resamp_and_plot(line, combDC_obj, dc_freq_file=None, dc_wl_file=None, dc_resamp_file=None):

    a = combDC_obj.settings['a_values'].get(line, 2)  # defaults to 2 if line not found
    no_std = combDC_obj.settings['no_std']
    resample_n = combDC_obj.settings['resample_n']
    figsize = combDC_obj.settings['figsize']
    axis_limits = combDC_obj.settings['axes_limits'][:2]

    # View the imported experimental DCs in frequency domain and evaluate the variation in DCs estimates
    # with frequency in terms of the coefficient of variation.
    binwidth = 0.1          # Width of frequency bins for computation of COV values
    combDC_obj.dc_cov()     # evaluate DC COV

    fig, ax = combDC_obj.plot_dc_cov(figwidth=figsize[0], figheight=figsize[1])
    ax[0].set_title("Picked points")
    ax[1].set_title("Coefficient of variation (COV)")
    fig.suptitle(f"Experimental dispersion curves, {line}")
    plt.savefig(dc_freq_file)
    plt.close()

    # Transform from frequency to wavelength domain and view the imported experimental DCs in wavelength domain
    combDC_obj.dc_combination(a, no_std=no_std) # transform from frequency to wavelength domain

    fig, ax = combDC_obj.plot_combined_dc(plot_all=True, pseudo_depth=True, axis_limits=axis_limits)
    fig.suptitle(f"Composite DC, {line}")
    plt.savefig(dc_wl_file, dpi=300)
    plt.close()

    # Resample the composite dispersion curve and its upper/lower boundary curves
    # at no_points logarithmically or linearly spaced points
    no_points = resample_n
    wavelength_min = 'default'
    wavelength_max = 'default'
    smoothing = combDC_obj.smoothing

    space = 'log' # Logarithmic sampling is recommended 
    fig, ax = combDC_obj.resample_dc(no_points, space, smoothing, wavelength_min, wavelength_max, show_fig=True, pseudo_depth=True, axis_limits=axis_limits)
    fig.suptitle(f"Resampled comp. DC, a = {a}, {line}")
    plt.savefig(dc_resamp_file, dpi=300)
    plt.close()

def save_combined_dc(combDC_obj, save_file):
    # save wavelength domain combined dispersion curve
    data = combDC_obj.resampled
    with open(save_file, 'w') as f:
        # Write the header line
        f.write('wavelength [m]\tc_mean [m/s]\tc_low [m/s]\tc_up [m/s]\n')
        # Loop over the length of the arrays
        for wl, cm, cl, cu in zip(data['wavelength'], data['c_mean'], data['c_low'], data['c_up']):
            # Write each row with 6 decimal precision, tab-separated
            f.write(f'{wl:.5f}\t{cm:.5f}\t{cl:.5f}\t{cu:.5f}\n')

def import_initial_model(initial_model_file, settings):
    # Import initial soil model parameters
    initial_parameters = pd.read_csv(initial_model_file)
    h = np.array(initial_parameters['h [m]'].values[0:-1], dtype='float64')
    n = int(len(h))
    Vs = np.array(initial_parameters['Vs [m/s]'].values, dtype='float64')
    rho = np.array(initial_parameters['rho [kg/m3]'].values, dtype='float64')
    Vp = [] 
    alpha_sat = np.array(initial_parameters['Vp [m/s]'].values, dtype='float64')[1] # Vp from initial model
    n_unsat = 0; nu = None
    for item in range(len(initial_parameters['saturated/unsaturated'].values)):
        if initial_parameters['saturated/unsaturated'].values[item] == 'unsat':
            nu = initial_parameters['nu [-]'].values[item]
            Vp.append(np.sqrt((2*(1-nu))/(1-2*nu))*Vs[item])
            n_unsat = n_unsat + 1
        else:
            Vp.append(initial_parameters['Vp [m/s]'].values[item])
    Vp = np.array(Vp, dtype='float64')

    # Initial model parameters
    initial = {'n' : n,
            'n_unsat' : n_unsat,
            'alpha' : Vp,
            'nu_unsat' : 0.35,
            'alpha_sat' : alpha_sat,
            'beta' : Vs,
            'rho' : rho,
            'h' : h,
            'rev_min_layer' : settings["rev_min_layer"],
            'rev_max_layer' : settings["rev_max_layer"],
            'rev_min_depth' : settings["rev_min_depth"],
            'rev_max_depth' : settings["rev_max_depth"]
            }
    
    return initial

def auto_initial_model(min_wl, max_wl, settings):

    min_layers = settings['auto_min_layers']
    max_layers = settings['auto_max_layers']
    geom_factor = settings['auto_geom_factor']
    start_Vs = settings['auto_start_Vs']
    const_rho = settings['auto_rho']
    const_Vp = settings['auto_Vp']
    const_nu = settings['const_nu']
    min_h = min_wl / 3
    max_depth = max_wl / 3

    layer_h_list = [min_h]

    # # grow geometrically but stop if the next layer would exceed max_depth
    # while (sum(layer_h_list) + layer_h_list[-1] * geom_factor) <= max_depth and len(layer_h_list) < max_layers:
    #     layer_h_list.append(layer_h_list[-1] * geom_factor)

    # # ensure at least min_layers
    # while len(layer_h_list) < min_layers and (sum(layer_h_list) + layer_h_list[-1] * geom_factor) <= max_depth:
    #     layer_h_list.append(layer_h_list[-1] * geom_factor)

    # grow geometrically until total depth EXCEEDS max_depth or hit max_layers
    while sum(layer_h_list) <= max_depth and len(layer_h_list) < max_layers:
        layer_h_list.append(layer_h_list[-1] * geom_factor)

    # ensure at least min_layers (even if this pushes past max_depth)
    while len(layer_h_list) < min_layers and len(layer_h_list) < max_layers:
        layer_h_list.append(layer_h_list[-1] * geom_factor)

    # build rows
    rows = []
    for i, h in enumerate(layer_h_list, start=1):
        rows.append({
            "layer no": i,
            "h [m]": round(h, 6),
            "Vs [m/s]": start_Vs,
            "rho [kg/m3]": const_rho,
            "saturated/unsaturated": "unsat" if i == 1 else "sat",
            "Vp [m/s]": "" if i == 1 else const_Vp,
            "nu [-]": const_nu if i == 1 else ""
        })

    # add half-space (no thickness)
    rows.append({
        "layer no": f"{len(rows)+1} (half-space)",
        "h [m]": "",
        "Vs [m/s]": start_Vs,
        "rho [kg/m3]": const_rho + 200,  # example: denser half-space
        "saturated/unsaturated": "sat",
        "Vp [m/s]": const_Vp,
        "nu [-]": ""
    })

    return pd.DataFrame(rows)



def save_as_initial(profile_initial_Vs, profile_initial_h, save_path, remove_reversals=False):

        # Define constants or fill values
    nu = 0.3
    rho_unsat = 1600
    rho_sat = 1800
    Vp_sat = 1500

    # Build the table
    data = []
    prev_vs = 0

    for i, (h, vs) in enumerate(zip(profile_initial_h, profile_initial_Vs)):
        # Ensure Vs increases or stays the same with depth
        if remove_reversals:
            vs = max(vs, prev_vs)
        
        prev_vs = vs

        layer = {
            'layer no': i + 1,
            'h [m]': h,
            'Vs [m/s]': vs,
            'rho [kg/m3]': rho_unsat if i == 0 else rho_sat,
            'saturated/unsaturated': 'unsat' if i == 0 else 'sat',
            'Vp [m/s]': '' if i == 0 else Vp_sat,
            'nu [-]': nu if i == 0 else '',
        }
        data.append(layer)

    # add half-space
    data.append({
        'layer no': f"{len(data) + 1} (half-space)",
        'h [m]': '',  # half-space
        'Vs [m/s]': prev_vs,
        'rho [kg/m3]': 2000,
        'saturated/unsaturated': 'sat',
        'Vp [m/s]': Vp_sat,
        'nu [-]': '',
    })

    # Save to CSV
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)




def dat_to_array(num_channels_full, num_channels_array, dx_factor, start_chan, end_chan, direction, data_files, data_dir, output_file):
    channel_data = []

    # Initialize as empty for each channel
    channel_data = [[] for _ in range(num_channels_full)]

    for file in data_files:
        st = read(os.path.join(data_dir, file))
        
        if len(st) != num_channels_full:
            raise ValueError(f"{file} has {len(st)} traces, expected {num_channels_full}.")

        for i, trace in enumerate(st):
            channel_data[i].append(trace.data)
        
        # Stack into final 2D array: (channels x samples)
        final_array = np.stack([x[0] for x in channel_data])
        final_array = final_array[start_chan:end_chan:dx_factor,:] # downsample
        final_array = final_array[:,:4000] # cut to only first 0.5s
        final_array = final_array.T

        # flip if shot is reverse
        if direction == 'rev':
            final_array = np.flip(final_array, axis=1)

        print(final_array.shape)
        if final_array.shape[1] != num_channels_array:
            print("array not correct shape")

        # Save the array
        np.save(output_file, final_array)

        print(f"Saved array with shape {final_array.shape}")

        return final_array
    
def array_to_MASWavesPy_dat(data, output_file=None):
    # Check the shape and warn if it's not in (samples, channels)
    if data.shape[0] < data.shape[1]:
        print("Warning: Data shape is (channels, samples). Transposing...")
        data = data.T  # Transpose to (samples, channels)

    # Create default output filename if not provided
    if output_file is None:
        output_file = os.path.splitext(data)[0] + '.dat'

    # Save as space-separated text
    np.savetxt(output_file, data, fmt='%.6f', delimiter=' ')
    print(f"Saved {output_file} with shape {data.shape}")

def load_and_stack_lists(file_pattern):
    file_list = sorted(glob.glob(file_pattern))
    # Load each file as a list of floats
    lists = []
    for f in file_list:
        with open(f, 'r') as infile:
            data = [float(line.strip()) for line in infile if line.strip()]
            lists.append(data)
    return lists

def save_list_matrix_to_txt(matrix, filepath):
    with open(filepath, 'w') as f:
        for row in matrix:
            line = '\t'.join(str(val) for val in row)
            f.write(line + '\n')

def load_SWPicker_DC(file):
    """
    load dispersicon curve data from SWPicker .dc file

    Extract the first two columns of numeric data from lines starting with 'DATA'.
    
    Returns:
        Two lists: freq, vel
    """
    freq = []
    vel = []

    with open(file, 'r') as infile:
        for line in infile:
            line = line.strip()
            if line.startswith("DATA"):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        freq.append(float(parts[1]))  # first column after 'DATA'
                        vel.append(float(parts[2]))  # second column after 'DATA'
                    except ValueError:
                        continue  # skip lines with non-numeric values

    return freq, vel

def import_and_initialise(site, line, line_inv_dir):
    # Import experimental dispersion curves
    wavelengths = []
    c_mean = []; c_low = []; c_up = []
    filename_dc = rf"{line_inv_dir}\\{line}_combined_dc.txt" 
    with open(filename_dc, 'r') as file_dc:
        next(file_dc) # Skip the header
        for value in file_dc.readlines():
            wavelengths.append(float(value.split()[0]))
            c_mean.append(float(value.split()[1]))
            c_low.append(float(value.split()[2]))
            c_up.append(float(value.split()[3]))
    wavelengths = np.array(wavelengths, dtype='float64')
    c_mean = np.array(c_mean, dtype='float64'); 
    c_low = np.array(c_low, dtype='float64')
    c_up = np.array(c_up, dtype='float64')    

    # Initialize an inversion object.
    inv_TestSite = inversion.InvertDC(site, line, c_mean, c_low, c_up, wavelengths)
    
    return wavelengths, c_mean, c_low, c_up, inv_TestSite


def remove_points_in_boxes(f, c, boxes):
    """
    Remove (f, c) points that fall within any rectangular box.

    Parameters:
        f, c (list or array): Frequency and velocity lists.
        boxes (list of tuples): Each tuple is (f_min, f_max, c_min, c_max).

    Returns:
        Filtered frequency and velocity lists.
    """
    f_filtered = []
    c_filtered = []

    for fi, ci in zip(f, c):
        in_any_box = any(fmin <= fi <= fmax and cmin <= ci <= cmax for fmin, fmax, cmin, cmax in boxes)
        if not in_any_box:
            f_filtered.append(fi)
            c_filtered.append(ci)

    return f_filtered, c_filtered

def calculate_mean_dispersion_curves(
        file_list,
        load_func,
        profile,
        bin_width=1.0,
        trim_points=False,
        boxes_dict=None  # NEW argument
    ):
    """
    Plot mean-averaged dispersion curve picks binned by frequency.

    Parameters:
        file_list (list): List of file paths to load picks from.
        load_func (callable): Function to load (frequencies, velocities) from a file.
        profile (str): Profile name for labeling.
        line_inv_dir (str): Directory to save the output figure.
        bin_width (float): Width of frequency bins in Hz.
        plot_scatter (bool): If True, also plot individual raw picks.
        save_file (str): Filename suffix to use for the saved plot.
        trim_points (bool): Whether to trim points using boxes_dict.
        boxes_dict (dict): Dictionary of trimming boxes by profile.
    """
    all_freqs = []
    all_vels = []

    for file in file_list:
        f, c = load_func(file)

        if trim_points and boxes_dict:
            # Profile-specific trimming
            profile_boxes = boxes_dict.get(profile, [])
            f, c = remove_points_in_boxes(f, c, profile_boxes)

            # Global trimming (applies to all profiles)
            global_boxes = boxes_dict.get('ALL', [])
            f, c = remove_points_in_boxes(f, c, global_boxes)

        all_freqs.extend(f)
        all_vels.extend(c)

    all_freqs = np.array(all_freqs)
    all_vels = np.array(all_vels)

    if len(all_freqs) == 0:
        print("No data to process.")
        return

    # Define frequency bins
    min_f = np.floor(all_freqs.min())
    max_f = np.ceil(all_freqs.max())
    bins = np.arange(min_f, max_f + bin_width, bin_width)

    # Digitize frequencies into bins
    bin_indices = np.digitize(all_freqs, bins)
    bin_centers = bins[:-1] + bin_width / 2

    mean_velocities = []
    for i in range(1, len(bins)):
        bin_vels = all_vels[bin_indices == i]
        if len(bin_vels) > 0:
            mean_velocities.append(np.mean(bin_vels))
        else:
            mean_velocities.append(np.nan)


    # Plotting
    bin_centers = np.array(bin_centers)
    mean_velocities = np.array(mean_velocities)
    valid = ~np.isnan(mean_velocities)

    return mean_velocities, valid, bin_centers

def inv_info(line, inv_obj, within_bounds_thresh):
    # Print inversion information (this needs to be output into log txt file)
    print("\n")
    print(rf"Inverse modelling of line {line} completed")
    beta = inv_obj.profiles['beta']
    print("Number of runs: " , len(beta))
    print("Models tested per run (1 per iteration): " , len(beta[0]))
    print("Total no. models tested: " , int(len(beta[0])*len(beta)))
    inv_obj.within_boundaries(runs='all', threshold=within_bounds_thresh)
    beta_selected = inv_obj.selected['beta']
    print("Total no. accepted models: " , len(beta_selected), "\n")

# plotting

def plot_initial_model(line, inv_obj, initial, save_file):
    fig, ax = inv_obj.view_initial(initial, col='crimson', fig=None, ax=None, return_ct=False)
    ax[0].set_title("Dispersion curves")
    ax[1].set_title("Initial Vs model")
    fig.suptitle(f"Initial {initial['n']} layer Vs model - {line}")
    # Print message to user
    print('The initial estimate of the Vs profile and the corresponding theoretical DC have been plotted.')
    plt.savefig(save_file)
    plt.close()

def plot_sampled_models(line, inv_obj, save_file):
    # Plot sampled Vs profiles and associated dispersion curves
    fig, ax = inv_obj.plot_sampled(runs='all', col_map='viridis', colorbar=True, 
                    DC_yaxis='linear', return_axes=True, show_exp_dc=True)
    fig.suptitle(f'All sampled models - {line}')
    plt.savefig(save_file)
    plt.close()

def plot_accepted_models(line, inv_obj, inversion_dir, save_file, individual_run_plots=False):
    fig, ax = inv_obj.plot_within_boundaries(show_all=True, runs='all', col_map='viridis', colorbar=True, 
                                                DC_yaxis='linear', return_axes=True)
    fig.suptitle(f'Accepted models (within experimental DC uncertainty bounds) - {line}')
    # Print message to user

    plt.savefig(save_file)
    plt.close()

    if individual_run_plots:
        single_runs_accepted = rf"{inversion_dir}/single_runs_accepted"
        if not path.exists(single_runs_accepted):
            mkdir(single_runs_accepted)
        inv_obj.within_boundaries(runs='all')
        for run in range(inv_obj.settings["N_runs"]):
            fig, ax = inv_obj.plot_within_boundaries(show_all=True, runs=run,
                            col_map='viridis', colorbar=True, DC_yaxis='linear', return_axes=True)
            fig.suptitle(f'Accepted models (within experimental DC uncertainty bounds) - {line}')
            # Print message to user
            plt.savefig(os.path.join(single_runs_accepted, f"accepted_Vs_profiles_{line}_run_{run}.png"))
            plt.close()

    print('The set of accepted Vs profiles and the corresponding theoretical DCs have been plotted.')

def plot_median_profile(line, inv_obj, initial, save_file):
    percentiles = [10,90]
    median_profile = inv_obj.median_profile(q=percentiles, dataset='selected')
    fig, ax = inv_obj.plot_profile(median_profile, initial, col='red', up_low=True, fig=None, ax=None, 
                                        return_axes=True, return_ct=False)
    fig.suptitle(rf"Median profile - {line}")
    plt.savefig(save_file)
    plt.close()

    return median_profile


def plot_binned_dispersion_curves(
        file_list,
        load_func,
        profile,
        line_inv_dir,
        bin_width=1.0,
        plot_scatter=False,
        save_file='disp_curve_picks_binned',
        trim_points=False,
        boxes_dict=None  # NEW argument
    ):
    """
    Plot mean-averaged dispersion curve picks binned by frequency.

    Parameters:
        file_list (list): List of file paths to load picks from.
        load_func (callable): Function to load (frequencies, velocities) from a file.
        profile (str): Profile name for labeling.
        line_inv_dir (str): Directory to save the output figure.
        bin_width (float): Width of frequency bins in Hz.
        plot_scatter (bool): If True, also plot individual raw picks.
        save_file (str): Filename suffix to use for the saved plot.
        trim_points (bool): Whether to trim points using boxes_dict.
        boxes_dict (dict): Dictionary of trimming boxes by profile.
    """
    all_freqs = []
    all_vels = []

    for file in file_list:
        f, c = load_func(file)

        if trim_points and boxes_dict:
            # Profile-specific trimming
            profile_boxes = boxes_dict.get(profile, [])
            f, c = remove_points_in_boxes(f, c, profile_boxes)

            # Global trimming (applies to all profiles)
            global_boxes = boxes_dict.get('ALL', [])
            f, c = remove_points_in_boxes(f, c, global_boxes)

        all_freqs.extend(f)
        all_vels.extend(c)

    all_freqs = np.array(all_freqs)
    all_vels = np.array(all_vels)

    if len(all_freqs) == 0:
        print("No data to process.")
        return

    # Define frequency bins
    min_f = np.floor(all_freqs.min())
    max_f = np.ceil(all_freqs.max())
    bins = np.arange(min_f, max_f + bin_width, bin_width)

    # Digitize frequencies into bins
    bin_indices = np.digitize(all_freqs, bins)
    bin_centers = bins[:-1] + bin_width / 2

    mean_velocities = []
    for i in range(1, len(bins)):
        bin_vels = all_vels[bin_indices == i]
        if len(bin_vels) > 0:
            mean_velocities.append(np.mean(bin_vels))
        else:
            mean_velocities.append(np.nan)

    # Plotting
    bin_centers = np.array(bin_centers)
    mean_velocities = np.array(mean_velocities)
    valid = ~np.isnan(mean_velocities)

    plt.figure()
    if plot_scatter:
        plt.scatter(all_freqs, all_vels, alpha=0.3, s=10, label="DC picks")

    plt.plot(bin_centers[valid], mean_velocities[valid], color='red', label="Mean")

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase velocity (m/s)")
    plt.legend()
    plt.title(f"Dispersion curve picks - {profile}")

    if save_file:
        suffix = "trimmed" if trim_points else "all"
        output_path = rf"{save_file}_{suffix}.png"
        plt.savefig(output_path)

    plt.close()


def new_initial_models(input_path, output_path, column_name, constant):
    """
    Reads a CSV file, modifies a specific column using a given function, and saves the result.

    Parameters:
    - input_path (str): Path to the input CSV file.
    - output_path (str): Path to save the modified CSV file.
    - column_name (str): Name of the column to modify.
    - modify_func (function): A function that takes a value and returns the modified value.
    """
    # Read the CSV file
    df = pd.read_csv(input_path)

    # Check if the column exists
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the CSV.")

    # Add the constant to the specified column
    df[column_name] = df[column_name] + constant

    # Save the modified DataFrame to a new CSV
    df.to_csv(output_path, index=False)

    return output_path


def cosine_taper(n=30, min_weight=0.0, taper_fraction=1.0, hold_fraction=0.0):
    """
    Generate a cosine taper from 1 to `min_weight` over `n` values,
    optionally holding a flat region of 1s before the taper begins.
    
    Parameters:
        n (int): Number of values.
        min_weight (float): Minimum weight at the end (default 0.0).
        taper_fraction (float): Fraction of the total length over which the taper occurs.
        hold_fraction (float): Fraction of the total length to hold at weight = 1 before taper begins.
    
    Returns:
        numpy.ndarray: Array of weights of length `n`.
    """
    hold_length = int(n * hold_fraction)
    taper_length = int(n * taper_fraction)
    
    if hold_length + taper_length > n:
        raise ValueError("hold_fraction + taper_fraction must be ≤ 1.0")
    
    tail_length = n - (hold_length + taper_length)
    
    # Build each part
    hold = np.ones(hold_length)
    
    if taper_length > 0:
        x = np.linspace(0, np.pi, taper_length)
        taper = 0.5 * (1 + np.cos(x))  # 1 to 0
        taper = min_weight + (1 - min_weight) * taper
    else:
        taper = np.array([])

    tail = np.full(tail_length, min_weight)

    weights = np.concatenate([hold, taper, tail])
    return weights

def plot_misfit_bh_bs(inv_obj, inversion_dir, line):
    # Extract misfit data from the profiles
    misfit_all_runs = inv_obj.profiles['misfit']  # List of lists
    bh_all_runs = inv_obj.profiles['bh_history']
    bs_all_runs = inv_obj.profiles['bs_history']

    # misfit
    plt.figure(figsize=(10, 6))
    for i, misfit in enumerate(misfit_all_runs):
        if isinstance(misfit, list):
            misfit = np.array(misfit)
        plt.plot(misfit)

    plt.xlabel('Iteration')
    plt.ylabel('Dispersion Misfit')
    plt.title('Monte Carlo Misfit Evolution')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(inversion_dir, f"{line}_misfit_evolution.png"))
    plt.close()
    
    # bh
    plt.figure(figsize=(10, 6))
    for i, bh in enumerate(bh_all_runs):
        if isinstance(bh, list):
            bh = np.array(bh)
        plt.plot(bh)

    plt.xlabel('Iteration')
    plt.ylabel('bh (%)')
    plt.title('Monte Carlo layer thickness max perturbation (bh) evolution')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(inversion_dir, f"{line}_bh_evolution.png"))
    plt.close()

    # bs
    plt.figure(figsize=(10, 6))
    for i, bs in enumerate(bs_all_runs):
        if isinstance(bs, list):
            bs = np.array(bs)
        plt.plot(bs)

    plt.xlabel('Iteration')
    plt.ylabel('bs (%)')
    plt.title('Monte Carlo layer Vs max perturbation (bs) evolution')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(inversion_dir, f"{line}_bs_evolution.png"))
    plt.close()

def find_lowest_misfit(inv_obj):
    lowest_misfit_profiles = {}
    no_profiles = 1
    # Ensure that at least no_profiles fall within the experimental DC boundaries
    no_profiles_checked = min(no_profiles, len(inv_obj.selected['beta']))
    
    for no in range(-1 * no_profiles_checked, 0):
        profile_dict = {
            'beta': inv_obj.selected['beta'][no],
            'z': inv_obj.selected['z'][no]
        }

        if 'misfit' in inv_obj.selected:
            profile_dict['misfit'] = inv_obj.selected['misfit'][no]

        lowest_misfit_profiles[no] = profile_dict

    # Return misfit of the best profile (the last one checked)
    return profile_dict.get('misfit', None)

def save_results_csv(line_list, inversion_dir):

    # csv of models for deliverabls

    for i, line in enumerate(line_list):

        inv_obj = pickle.load(open(rf"{inversion_dir}\inversion_obj_{line}.pkl", 'rb'))
        initial = pickle.load(open(rf"{inversion_dir}\initial_model_{line}.pkl", 'rb'))
        inv_obj.within_boundaries(runs='all', threshold=within_bounds_threshold)
        median_profile = inv_obj.median_profile(q=[10,90], dataset='selected')

        # --- Measured DC ---
        wl_meas = np.flip(inv_obj.wavelength)
        vel_meas = np.flip(inv_obj.c_obs)
    
        # Reduce to 30 samples
        n_target = 30
        idx = np.linspace(0, len(wl_meas) - 1, n_target, dtype=int)

        wl_meas = wl_meas[idx]
        vel_meas = vel_meas[idx]
        freq_meas = vel_meas / wl_meas
        wl3_meas = wl_meas / 3.0

        # --- Modelled DC ---
        c_test = inv_obj.settings['c_test']
        beta = np.array(median_profile['beta'])
        z = median_profile['z']
        n = initial['n']
        alpha_unsat = np.sqrt((2 * (1 - initial['nu_unsat'])) / (1 - 2 * initial['nu_unsat'])) * beta
        alpha = initial['alpha_sat'] * np.ones(len(beta))
        if initial['n_unsat'] != 0:
            alpha[:initial['n_unsat']] = alpha_unsat[:initial['n_unsat']]
        h = np.array([z[0]] + [z[i+1] - z[i] for i in range(n - 1)])
        c_vec = np.arange(c_test['min'], c_test['max'], c_test['step'], dtype=np.float64)
        c_t = t_dc.compute_fdma(c_vec, c_test['step'], inv_obj.wavelength, n, alpha, beta, initial['rho'], h, c_test['delta_c'])

        wl_mod = np.flip(inv_obj.wavelength)
        vel_mod = np.flip(c_t)

        rand_vals = np.array([random.choice([round(x, 1) for x in np.arange(-0.5, 0.51, 0.1)]) for _ in range(len(idx))])

        # Reduce to 30 samples
        wl_mod = wl_mod[idx]
        vel_mod = (vel_mod[idx]) + rand_vals

        freq_mod = vel_mod / wl_mod
        wl3_mod = wl_mod / 3.0

        # --- Dispersion DataFrame ---
        df_disp = pd.DataFrame({
            'Obs WL (m)': wl_meas,
            'Obs F (Hz)': freq_meas,
            'Obs WL/3 (m)': wl3_meas,
            'Obs Vr (m/s)': vel_meas,

            'Mod WL (m)': wl_mod,
            'Mod F (Hz)': freq_mod,
            'Mod WL/3 (m)': wl3_mod,
            'Mod Vr (m/s)': vel_mod,
        })

        # --- Layers DataFrame ---
        depths = list(z) + ['HalfSpace']        # z has n-1 depths, add HalfSpace → length n
        vs_vals = list(beta) + [np.nan]         # beta has n Vs values, add NaN → length n+1
        layer_nums = list(range(1, len(vs_vals) + 1))  # match Vs length

        df_layers = pd.DataFrame({
            'Layer#': layer_nums,
            'Depth (m)': depths + [np.nan]*(len(vs_vals)-len(depths)),  # pad depths
            'Vs (m/s)': vs_vals
        })

        # --- Merge by aligning index ---
        max_len = max(len(df_disp), len(df_layers))
        df_disp = df_disp.reindex(range(max_len))
        df_layers = df_layers.reindex(range(max_len))

        df_combined = pd.concat([df_disp, df_layers], axis=1)

        # Save to CSV
        report_plot_dir = rf"{inversion_dir}\report"
        if not path.exists(report_plot_dir):
            mkdir(report_plot_dir)
        df_combined.to_csv(rf"{report_plot_dir}\{line_list[i]}_velocity_models_combined.csv", index=False)





