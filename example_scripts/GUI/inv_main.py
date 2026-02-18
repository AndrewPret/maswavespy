"""
MASW Inversion Main Processing Function
Refactored to accept config dictionary from GUI instead of global variables

This function receives a config dictionary from the GUI and performs
all the MASW inversion processing steps.
"""
 
import os
import pickle
import numpy as np
import time
import json
from maswavespy import combination, inversion
import maswavespy.zetica_utils as zutil


def MASW_inv_main(config):
    """
    Main MASW inversion processing function.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary containing all parameters from the GUI:
        - site: str - Survey site name
        - line_list: list - List of line IDs to process
        - working_dir: str - Main working directory
        - main_data_dir: str - Parent directory containing data for all lines
        - data_dir_list: list - List of specific data directories (one per line)
        - main_initial_file: str - Path to initial velocity model file
        - ignore_files: list - List of filenames to ignore
        - overwrite_dir: bool - Whether to overwrite existing results
        - max_depth: int - Maximum depth for visualization
        - points_to_remove: dict or None - Regions to remove from dispersion curves
        - settings: dict - All inversion settings
    
    Returns
    -------
    None
        Results are saved to disk as pickle files and images
    """
    
    # ===== EXTRACT CONFIG VARIABLES =====
    site = config['site']
    line_list = config['line_list']
    working_dir = config['working_dir']
    main_data_dir = config['main_data_dir']
    data_dir_list = config['data_dir_list']
    main_initial_file = config['main_initial_file']
    ignore_files = config['ignore_files']
    overwrite_dir = config['overwrite_dir']
    points_to_remove = config['points_to_remove']
    settings = config['settings']
    
    # ===== CREATE OUTPUT DIRECTORIES =====
    print("\n" + "="*70)
    print("MASW INVERSION PROCESSING")
    print("="*70)
    
    # Create results directories using zutil function
    # Assuming zutil.results_directories returns (results_dir, inversion_dir, initial_model_dir)
    results_dir, inversion_dir, initial_model_dir = zutil.results_directories(
        working_dir, overwrite_dir, site
    )
    
    print(f'\nInversion directory created: {inversion_dir}')
    
    # Store these in config for reference
    config['results_dir'] = results_dir
    config['inversion_dir'] = inversion_dir
    config['initial_model_dir'] = initial_model_dir
    
    # ===== MAIN PROCESSING LOOP =====
    print("\n" + "-"*70)
    print("Inversion process started")
    print("-"*70 + "\n")
    
    for i, line in enumerate(line_list):
        print(f"\nProcessing line: {line}")
        print("-" * 50)
        
        start_time = time.time()
        
        # === 1. Get line-specific data directory ===
        line_data_dir = data_dir_list[i]
        
        # === 2. Import and Prepare Dispersion Curves ===
        print("  [1] Importing dispersion curves...")
        f_vec, c_vec = zutil.data_import(
            line, line_data_dir, inversion_dir, 
            points_to_remove, ignore_files
        )
        
        combDC_obj = combination.CombineDCs(
            site, line, settings, freq=f_vec, c=c_vec
        )
        print('      Data imported and CombineDCs object initialized.')
        
        # === 3. Resample and Plot Dispersion Curves ===
        print("  [2] Resampling and plotting dispersion curves...")
        dc_freq_plot_file = os.path.join(inversion_dir, f"1_dc_cov_{line}.png")
        dc_wavelength_plot_file = os.path.join(inversion_dir, f"2_combined_dc_{line}.png")
        dc_resamp_plot_file = os.path.join(inversion_dir, f"3_resampled_dc_{line}.png")
        
        zutil.dc_resamp_and_plot(
            line, combDC_obj, dc_freq_plot_file, 
            dc_wavelength_plot_file, dc_resamp_plot_file
        )
        print(f'      Plotted with {settings["no_std"]} standard deviation bounds')
        print('      Dispersion curves resampled and plotted.')
        
        # Save composite DC to file
        zutil.save_combined_dc(
            combDC_obj, 
            save_file=os.path.join(inversion_dir, f"{line}_combined_dc.txt")
        )
        
        # === 4. Set Up Inversion ===
        print("  [3] Setting up inversion...")
        c_mean = combDC_obj.resampled['c_mean']
        c_low = combDC_obj.resampled['c_low']
        c_up = combDC_obj.resampled['c_up']
        wavelengths = combDC_obj.resampled['wavelength']
        
        # Create new inversion object for the line
        inv_obj = inversion.InvertDC(
            site, line, c_mean, c_low, c_up, wavelengths, settings
        )
        print('      Initial model imported and inversion object initialized.')
        
        # Save settings to JSON file
        with open(os.path.join(inversion_dir, "settings.json"), "w") as f:
            json.dump(settings, f, indent=4)
        
        # === 5. Create Multiple Initial Models with Velocity Shifts ===
        print("  [4] Creating initial velocity models with shifts...")
        vel_shift_array = np.linspace(
            0, settings["n_initial_models"] - 1, settings["n_initial_models"]
        ) * settings["vel_shift"]
        middle_index = len(vel_shift_array) // 2
        vel_shift_array = vel_shift_array - vel_shift_array[middle_index]
        
        initial_file_list = []
        
        for j in range(settings["n_initial_models"]):
            output_path = os.path.join(
                working_dir, 'initial_models', 
                f'initial_model_const_vel_{j+1}.csv'
            )
            new_initial_model = zutil.new_initial_models(
                main_initial_file, output_path, 
                "Vs [m/s]", vel_shift_array[j]
            )
            initial_file_list.append(new_initial_model)
        
        initial_list = []
        
        # === 6. Load and Plot Each Initial Model ===
        print("  [5] Loading and plotting initial models...")
        for j, initial_file in enumerate(initial_file_list):
            
            # Load initial model
            initial = zutil.import_initial_model(initial_file, settings)
            initial_list.append(initial)
            
            # Plot initial model fit
            zutil.plot_initial_model(
                line, inv_obj, initial,
                save_file=os.path.join(
                    initial_model_dir, 
                    f"4_initial_model_{line}_{j+1}.png"
                )
            )
        
        print('      Initial models loaded and plotted.')
        
        # === 7. Run Inversion ===
        print("  [6] Running Monte Carlo inversion...")
        print(f'      Inversion of profile {line} initiated.')
        
        inv_obj.mc_inversion_MultInitModels(initial_list, settings)
        
        print('      All runs completed.')
        
        end_time = time.time()
        elapsed_time = round(end_time - start_time)
        print(f"      Processing time: {elapsed_time} seconds")
        
        # === 8. Plot Sampled Vs Models ===
        print("  [7] Plotting sampled velocity profiles...")
        zutil.plot_sampled_models(
            line, inv_obj,
            save_file=os.path.join(inversion_dir, f"5_sampled_model_{line}.png")
        )
        print('      Sampled Vs profiles plotted.')
        
        # === 9. Print Inversion Info ===
        print("  [8] Inversion statistics...")
        zutil.inv_info(line, inv_obj, settings["within_bounds_thresh"])
        
        # Check if any profiles accepted
        inv_obj.within_boundaries(runs='all')
        if len(inv_obj.selected['c_t']) == 0:
            print(f"      WARNING: No profiles fit within measured uncertainty bounds for {line}")
            continue
        
        # === 10. Plot Accepted Models ===
        print("  [9] Plotting accepted models...")
        zutil.plot_accepted_models(
            line, inv_obj, inversion_dir,
            save_file=os.path.join(inversion_dir, f"6_accepted_Vs_profiles_{line}.png")
        )
        
        # === 11. Compute and Plot Median Profile ===
        print("  [10] Computing median velocity profile...")
        median_profile = zutil.plot_median_profile(
            line, inv_obj, initial,
            save_file=os.path.join(inversion_dir, f"7_median_profile_{line}.png")
        )
        print('      Median Vs profile computed and plotted.')
        
        # === 12. Save Results as Pickle Files ===
        print("  [11] Saving results...")
        pickle.dump(
            inv_obj,
            open(os.path.join(inversion_dir, f"inversion_obj_{line}.pkl"), "wb")
        )
        pickle.dump(
            initial,
            open(os.path.join(inversion_dir, f"initial_model_{line}.pkl"), "wb")
        )
        pickle.dump(
            median_profile,
            open(os.path.join(inversion_dir, f"median_profiles_{line}.pkl"), "wb")
        )
        print('      Results saved.')
        
        print(f"\n✓ Line {line} processing complete\n")
    
    # ===== SAVE FINAL SETTINGS =====
    print("\n" + "-"*70)
    print("Saving final configuration...")
    with open(os.path.join(inversion_dir, "settings.json"), "w") as f:
        json.dump(settings, f, indent=4)
    
    print("="*70)
    print("MASW INVERSION COMPLETE")
    print("="*70)
    print(f"Results directory: {inversion_dir}\n")


# ============================================================================
# HELPER FUNCTION: Load and run inversion from saved config
# ============================================================================

def load_and_run_from_json(json_config_file):
    """
    Load configuration from a JSON file and run inversion.
    
    Useful for re-running inversions with the same parameters
    or for batch processing.
    
    Parameters
    ----------
    json_config_file : str
        Path to JSON file containing configuration
    """
    import json
    
    with open(json_config_file, 'r') as f:
        config = json.load(f)
    
    MASW_inv_main(config)


# ============================================================================
# EXAMPLE: How to call this function from the GUI
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of MASW_inv_main with a sample config.
    This would be called from the GUI like this:
    
    from masw_inversion_main import MASW_inv_main
    MASW_inv_main(config)  # where config is the validated GUI config dict
    """
    
    # Example config (this would come from the GUI)
    example_config = {
        'site': 'test',
        'line_list': ['VR - 7'],
        'working_dir': r'C:\maswavespy\working_dir_test',
        'main_data_dir': r'C:\maswavespy\data_dir_test',
        'data_dir_list': [r'C:\maswavespy\data_dir_test\VR - 7'],
        'main_initial_file': r'C:\maswavespy\working_dir_test\initial_models\initial_model_const_vel.csv',
        'ignore_files': [],
        'overwrite_dir': False,
        'points_to_remove': None,
        'max_depth': 15,
        'settings': {
            'figsize': (18, 15),
            'max_depth': 15,
            'pseudo_depth': True,
            'dc_axis_limits': [(0, 1000), (0, 15)],
            'models_axes_lims': [(0, 1000), (0, 15), (0, 1000), (0, 15)],
            'no_std': 1,
            'resample_n': 30,
            'resamp_max_pdepth': None,
            'dc_resamp_smoothing': True,
            'a_values': {'VR - 7': 2},
            'only_save_accepted': False,
            'c_test': {'min': 100, 'max': 1100, 'step': 0.2, 'delta_c': 3},
            'run': 10,
            'N_max': 100,
            'repeat_run_if_fail': False,
            'run_success_minimum': 0,
            'bs': 20,
            'bh': 10,
            'bs_min': 4,
            'bh_min': 2,
            'b_decay_iterations': 200,
            'bs_min_halfspace': None,
            'N_stall_max': 0,
            'st_lt_window': (200, 1000),
            'misfit_mode': 'average',
            'uncertainty_weighting': False,
            'depth_weight_offset_const': 0,
            'plot_misfit_weights': False,
            'within_bounds_thresh': 1,
            'regularisation': 'model',
            'regularisation_ranges': None,
            'uncert_bounds_misfit_weight': 20,
            'rep_explore_pen_weight': 0.2,
            'regularisation_weights': [1e-7],
            'rev_min_depth': None,
            'rev_max_depth': None,
            'rev_min_layer': None,
            'rev_max_layer': None,
            'max_retries': 200,
            'force_monotonic': False,
            'monotonic_tol': 100,
            'vel_shift': 150,
            'n_initial_models': 5,
            'new_run_prev_best_fit': False,
            'runs_initial': 1
        }
    }
    
    # Uncomment to run with example config:
    # MASW_inv_main(example_config)