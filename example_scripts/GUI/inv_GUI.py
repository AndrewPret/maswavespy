import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import threading
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive, thread-safe backend


# -*- coding: utf-8 -*-
"""
GUI for running MASWavesPy inversion
--------------------------------
Requires Zetica's version of MASWavesPy, available here: https://github.com/AndrewPret/maswavespy
--------------------------------
This script performs the following steps for each 1D sounding: 

1. Import picked dispersion curves from SurfaceWavePicker .dc file output.
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


"""


class MAASWInversionGUI:
    """
    MASW Inversion Parameter Configuration GUI.
    
    This GUI collects all parameters, validates them, and then calls
    MASW_inv_main(config) to start the inversion processing.
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("MASW Inversion - Configuration Manager")
        self.root.geometry("1000x800")
        
        # Central config storage - this gets passed to MASW_inv_main()
        self.config = {
            'site': '',
            'line_list': [],
            'working_dir': '',
            'main_data_dir': '',
            'data_dir_list': [],
            'initial_models': [],
            'ignore_files': [],
            'main_initial_file': '',
            'overwrite_dir': False,
            'max_depth': 15,
            'points_to_remove': None,
            'results_dir': '',
            'inversion_dir': '',
            'initial_model_dir': '',
            'settings': {}
        }
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.add_basic_tab()
        self.add_inversion_tab()
        self.add_constraints_tab()
        self.add_advanced_tab()
        
        # Status bar
        self.create_button_frame()
    
    def create_button_frame(self):
        """Create bottom button frame with control buttons."""
        bf = ttk.Frame(self.root)
        bf.pack(fill=tk.X, padx=5, pady=5)
        
        # Main processing button
        ttk.Button(bf, text="START INVERSION", command=self.start_inversion,
                    ).pack(side=tk.RIGHT, padx=5)
        
        # Utility buttons
        ttk.Button(bf, text="Validate Only", command=self.validate_only).pack(side=tk.RIGHT, padx=5)
        ttk.Button(bf, text="Preview Config", command=self.show_config).pack(side=tk.RIGHT, padx=5)
        ttk.Button(bf, text="Reset", command=self.reset_form).pack(side=tk.RIGHT, padx=5)
        
        self.status = ttk.Label(bf, text="Ready", relief=tk.SUNKEN)
        self.status.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    
    def make_scrollable_frame(self, parent):
        """Create a scrollable frame."""
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        frame = ttk.Frame(canvas)
        
        frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        canvas.create_window((0, 0), window=frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        return frame
    
    def add_entry(self, parent, label, default="", width=50, row=0):
        """Add labeled entry field."""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        entry = ttk.Entry(parent, width=width)
        entry.insert(0, str(default))
        entry.grid(row=row, column=1, sticky=tk.EW, padx=5, pady=5)
        return entry
    
    def add_spinbox(self, parent, label, default, from_val, to_val, row=0):
        """Add spinbox field."""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        spinbox = ttk.Spinbox(parent, from_=from_val, to=to_val, width=20)
        spinbox.set(default)
        spinbox.grid(row=row, column=1, sticky=tk.W, padx=5, pady=5)
        return spinbox
    
    def add_file_browse(self, parent, label, is_dir=False, row=0):
        """Add file/directory browser."""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=1, sticky=tk.EW, padx=5, pady=5)
        
        entry = ttk.Entry(frame, width=40)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        def browse():
            if is_dir:
                path = filedialog.askdirectory()
            else:
                path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All", "*.*")])
            if path:
                entry.delete(0, tk.END)
                entry.insert(0, path)

        ttk.Button(frame, text="Browse", width=10, command=browse).pack(side=tk.LEFT)
        return entry
    
    
    def add_model_file_browse(self, parent, label, is_dir=False, row=0):
        """Add file/directory browser."""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=1, sticky=tk.EW, padx=5, pady=5)
        
        entry = ttk.Entry(frame, width=40)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        # ✅ Set default path to relative location here
        default_path = rf"initial_models\initial_model_const_vel.csv"
        entry.insert(0, default_path)
        
        def browse():
            if is_dir:
                path = filedialog.askdirectory()
            else:
                path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All", "*.*")])
            if path:
                entry.delete(0, tk.END)
                entry.insert(0, path)

        rf"C:\default_file"
        
        ttk.Button(frame, text="Browse", width=10, command=browse).pack(side=tk.LEFT)
        return entry
    
    # ============================================================================
    # TAB 1: BASIC SETTINGS
    # ============================================================================
    
    def add_basic_tab(self):
        """Add input and output directory tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Input/Output")
        scroll_frame = self.make_scrollable_frame(frame)
        
        row = 0
        ttk.Label(scroll_frame, text="Project Information", 
                font=("Arial", 10, "bold")).grid(row=row, column=0, columnspan=2, 
                sticky=tk.W, padx=5, pady=(10, 5))
        row += 1

        self.site_entry = self.add_entry(scroll_frame, "Site Name:", "test", row=row)
        row += 1
        
        self.lines_entry = self.add_entry(scroll_frame, "Line IDs (comma-separated):" \
        "\n   - These must match names of" \
        "\n     directories containing line data", "VR - 7", row=row)
        row += 1
        
        self.working_dir = self.add_file_browse(scroll_frame, "Working Directory:" \
        "\n   - Results directory will be created here", is_dir=True, row=row)
        row += 1
        
        self.data_dir = self.add_file_browse(scroll_frame, "Main Data Directory:" \
        "\n   - Select parent data directory which " \
        "\n     must contain 1 directory per Line ID", is_dir=True, row=row)
        row += 1
        
        self.initial_model = self.add_model_file_browse(scroll_frame, "Initial Model File (.csv):" \
        "\n   - Default relative path:" \
        "\n     initial_models\initial_model_const_vel.csv", is_dir=False, row=row)
        row += 1
        
        self.ignore_files_entry = self.add_entry(scroll_frame, "Files to Ignore (comma-separated):" \
        "\n   - Specify .dc files by name" \
        "\n     e.g. 1-18(Cx7p5)(Sx-2p5)(fn100)R.dc", "", row=row)
        row += 1
        
        ttk.Label(scroll_frame, text="Overwrite Results:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        self.overwrite_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(scroll_frame, variable=self.overwrite_var,
                    text="If unchecked, creates new timestamped folder in\nresults directory each time inversion is run").grid(row=row, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(scroll_frame, text="Visualisation", 
                font=("Arial", 10, "bold")).grid(row=row, column=0, columnspan=2, 
                sticky=tk.W, padx=5, pady=(10, 5))
        row += 1

        self.max_depth_spinbox = self.add_spinbox(scroll_frame, "Max model depth (m):", 15, 5, 100, row=row)
        row += 1
        self.figsize_w = self.add_spinbox(scroll_frame, "Figure Width:", 18, 5, 30, row=row)
        row += 1
        self.figsize_h = self.add_spinbox(scroll_frame, "Figure Height:", 15, 5, 30, row=row)
        row += 1

        
    
    # ============================================================================
    # TAB 2: INVERSION SETTINGS
    # ============================================================================
    
    def add_inversion_tab(self):
        """Add inversion settings tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Inversion")
        scroll_frame = self.make_scrollable_frame(frame)
        
        row = 0
        ttk.Label(scroll_frame, text="Monte Carlo Inversion Parameters:", 
                font=("Arial", 10, "bold")).grid(row=row, column=0, columnspan=2, 
                sticky=tk.W, padx=5, pady=(10, 5))
        row += 1

        ttk.Label(scroll_frame, text="All run starts without information on previous runs\n" \
        "Iteration 1 applies random pertuabtions to the initial model\n" \
        "Further iterations apply random perturbations to the previous\n" \
        "iteration's model", 
                font=("Arial", 9, "bold")).grid(row=row, column=0, columnspan=2, 
                sticky=tk.W, padx=5, pady=(10, 5))
        row += 1


        self.num_runs = self.add_spinbox(scroll_frame, "Number of runs:", 10, 1, 100, row=row)
        row += 1
        self.n_max = self.add_spinbox(scroll_frame, "Max Iterations per run:", 100, 10, 500, row=row)
        row += 1
        
        ttk.Label(scroll_frame, text="Change the maximum random perturbation of model layer" \
        "\nthickness and Vs per iteration and number of iterations" \
        "\n to decay from max to min value." , 
                font=("Arial", 9, "bold")).grid(row=row, column=0, columnspan=2, 
                sticky=tk.W, padx=5, pady=(10, 5))
        row += 1
        
        self.bs = self.add_spinbox(scroll_frame, "Max Vs Perturbation (%):", 20, 1, 50, row=row)
        row += 1
        self.bs_min = self.add_spinbox(scroll_frame, "Min Vs Perturbation (%):", 4, 1, 20, row=row)
        row += 1
        self.bh = self.add_spinbox(scroll_frame, "Max Thickness Perturbation (%):", 10, 1, 50, row=row)
        row += 1
        self.bh_min = self.add_spinbox(scroll_frame, "Min Thickness Perturbation (%):", 2, 1, 20, row=row)
        row += 1
        self.b_decay = self.add_spinbox(scroll_frame, "Decay Iterations:", 200, 50, 1000, row=row)
        row += 1

        ttk.Label(scroll_frame, text="DC Smoothing:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        self.dc_smooth_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(scroll_frame, variable=self.dc_smooth_var).grid(row=row, column=1, sticky=tk.W, padx=5, pady=5)
        row += 1
        
        self.lines_entry = self.add_entry(scroll_frame, "Resampling Density (a-value):"\
        "These must match order of entry of directories " \
        "containing line data and order of specified Line IDs", 2, row=row)
        row += 1

        
        ttk.Label(scroll_frame, text="Forward Modelling (Vr)", 
                font=("Arial", 10, "bold")).grid(row=row, column=0, columnspan=2, 
                sticky=tk.W, padx=5, pady=(10, 5))
        row += 1
        
        self.c_min = self.add_spinbox(scroll_frame, "Min Test Velocity (m/s):", 100, 50, 500, row=row)
        row += 1
        self.c_max = self.add_spinbox(scroll_frame, "Max Test Velocity (m/s):", 1100, 500, 2000, row=row)
        row += 1
        self.c_step = self.add_spinbox(scroll_frame, "Velocity Step:" \
        "\n    -Trade off between processing time and sampling", 0.2, 0.1, 1.0, row=row)
        row += 1
        self.delta_c = self.add_spinbox(scroll_frame, "Delta-C (m/s):", 3, 1, 10, row=row)

        ttk.Label(scroll_frame, text="Initial Models", 
                font=("Arial", 10, "bold")).grid(row=row, column=0, columnspan=2, 
                sticky=tk.W, padx=5, pady=(10, 5))
        row += 1
        
        self.n_initial = self.add_spinbox(scroll_frame, "Number of Initial Models (odd):" \
        "\n   -If 1, only provided initial model will be used," \
        "\n    Otherwise each model will be used as the ", 5, 1, 15, row=row)
        row += 1
        self.vel_shift = self.add_spinbox(scroll_frame, "Velocity Shift (m/s):" \
        "\n   -Creates additional starting models " \
        "\n    with +shift and -shift velocities ", 150, 10, 500, row=row)
        row += 1

    

    
    # ============================================================================
    # TAB 5: CONSTRAINTS
    # ============================================================================
    
    def add_constraints_tab(self):
        """Add constraints tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Constraints")
        scroll_frame = self.make_scrollable_frame(frame)
        
        row = 0
        ttk.Label(scroll_frame, text="Limit velocity reversals", 
                 font=("Arial", 10, "bold")).grid(row=row, column=0, columnspan=2, 
                 sticky=tk.W, padx=5, pady=(10, 5))
        row += 1
        
        self.rev_min_depth = self.add_entry(scroll_frame, "Min reversal Depth:", "None", row=row)
        row += 1
        self.rev_max_depth = self.add_entry(scroll_frame, "Max reversal Depth:", "None", row=row)
        row += 1
        self.max_retries = self.add_spinbox(scroll_frame, "Max Retries:", 200, 50, 500, row=row)
        row += 1
        
        ttk.Label(scroll_frame, text="Force Monotonic:\n" \
        "   -Force modelled Vs to increase with depth").grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        self.force_monotonic_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(scroll_frame, variable=self.force_monotonic_var).grid(row=row, column=1, sticky=tk.W, padx=5, pady=5)
        row += 1
        
        self.monotonic_tol = self.add_spinbox(scroll_frame, "Monotonic Tolerance (m/s):" \
        "\n    -If Force Monotonic is enable, tolerate decrease in velocity between adjacent layers", 100, 0, 500, row=row)
        row += 1

        ttk.Label(scroll_frame, text="MISFIT", 
                 font=("Arial", 10, "bold")).grid(row=row, column=0, columnspan=2, 
                 sticky=tk.W, padx=5, pady=(10, 5))
        row += 1
        
        ttk.Label(scroll_frame, text="Uncertainty Weighting:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        self.uncert_weight_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(scroll_frame, variable=self.uncert_weight_var).grid(row=row, column=1, sticky=tk.W, padx=5, pady=5)
        row += 1
        
        self.depth_weight = self.add_spinbox(scroll_frame, "Depth Weight Offset:", 0, 0, 5, row=row)
        row += 1
        self.within_bounds = self.add_spinbox(scroll_frame, "Within Bounds Threshold:", 1.0, 0.5, 2.0, row=row)
        row += 1
        self.uncert_bounds_weight = self.add_spinbox(scroll_frame, "Uncert Bounds Weight:", 20, 1, 100, row=row)
        row += 1
        self.rep_explore_weight = self.add_spinbox(scroll_frame, "Repulsion/Exploration Weight:", 0.2, 0.0, 1.0, row=row)
        row += 1
        
        ttk.Label(scroll_frame, text="REGULARISATION", 
                 font=("Arial", 10, "bold")).grid(row=row, column=0, columnspan=2, 
                 sticky=tk.W, padx=5, pady=(10, 5))
        row += 1
        
        ttk.Label(scroll_frame, text="Type:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        self.regularisation = ttk.Combobox(scroll_frame, values=["model", "dc"], width=47)
        self.regularisation.current(0)
        self.regularisation.grid(row=row, column=1, sticky=tk.EW, padx=5, pady=5)
        row += 1
        
        self.reg_weights = self.add_entry(scroll_frame, "Regularisation Weights:", "1e-7", row=row)
        


        
        # ttk.Label(scroll_frame, text="Use Previous Best Fit (this do):").grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        # self.use_prev_best_var = tk.BooleanVar(value=False)
        # ttk.Checkbutton(scroll_frame, variable=self.use_prev_best_var).grid(row=row, column=1, sticky=tk.W, padx=5, pady=5)
        # row += 1
        
        # self.runs_initial = self.add_spinbox(scroll_frame, "Runs Before Reuse:", 1, 1, 10, row=row)

    # ============================================================================
    # TAB 6: ADVANCED SETTINGS TAB
    # ============================================================================

    def add_advanced_tab(self):
        """Add advanced tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Advanced settings")
        scroll_frame = self.make_scrollable_frame(frame)
        
        row = 0
        ttk.Label(scroll_frame, text="Repeat failed iterations", 
                 font=("Arial", 10, "bold")).grid(row=row, column=0, columnspan=2, 
                 sticky=tk.W, padx=5, pady=(10, 5))
        row += 1

        ttk.Label(scroll_frame, text="Repeat If No Success:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        self.repeat_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(scroll_frame, variable=self.repeat_var).grid(row=row, column=1, sticky=tk.W, padx=5, pady=5)
        row += 1
        
        self.run_success_min = self.add_spinbox(scroll_frame, "Minimum Successful Runs:", 0, 0, 50, row=row)
        row += 1

        ttk.Label(scroll_frame, text="Misfit Mode:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        self.misfit_mode = ttk.Combobox(scroll_frame, 
            values=["average", "average_weighted", "max", "max_weighted"], width=47)
        self.misfit_mode.current(0)
        self.misfit_mode.grid(row=row, column=1, sticky=tk.EW, padx=5, pady=5)
        row += 1

        ttk.Label(scroll_frame, text="DISPERSION CURVE", 
            font=("Arial", 10, "bold")).grid(row=row, column=0, columnspan=2, 
            sticky=tk.W, padx=5, pady=(10, 5))
        row += 1
        
        self.no_std = self.add_spinbox(scroll_frame, "Std Dev for Bounds:", 1, 0.5, 5, row=row)
        row += 1
        self.resample_n = self.add_spinbox(scroll_frame, "Resample Points:", 30, 10, 100, row=row)
        row += 1

        ttk.Label(scroll_frame, text="In case of stalling misfit (not recommended)", 
                font=("Arial", 10, "bold")).grid(row=row, column=0, columnspan=2, 
                sticky=tk.W, padx=5, pady=(10, 5))
        row += 1
        
        row += 1
        self.stall_max = self.add_spinbox(scroll_frame, "Max Stall Iterations:", 0, 0, 20, row=row)
        row += 1
        self.st_window = self.add_spinbox(scroll_frame, "Short-term Window:", 200, 50, 500, row=row)
        row += 1
        self.lt_window = self.add_spinbox(scroll_frame, "Long-term Window:", 1000, 500, 2000, row=row)
    
        
    
    # ============================================================================
    # CONFIGURATION MANAGEMENT
    # ============================================================================
    
    def populate_config(self):
        """
        Validate all inputs and populate self.config.
        This is the crucial step before calling MASW_inv_main().
        """
        # Extract and validate basic settings
        site = self.site_entry.get().strip()
        if not site:
            raise ValueError("Site name cannot be empty")
        
        lines = [l.strip() for l in self.lines_entry.get().split(',') if l.strip()]
        if not lines:
            raise ValueError("At least one line ID is required")
        
        working_dir = self.working_dir.get().strip()
        if not working_dir:
            raise ValueError("Working directory is required")
        if not os.path.exists(working_dir):
            raise ValueError(f"Working directory does not exist: {working_dir}")
        
        main_data_dir = self.data_dir.get().strip()
        if not main_data_dir:
            raise ValueError("Main data directory is required")
        
        main_initial_file = self.initial_model.get().strip()
        if not main_initial_file:
            raise ValueError("Initial model file is required")
        if not os.path.exists(main_initial_file):
            raise ValueError(f"Initial model file does not exist: {main_initial_file}")
        
        max_depth = int(self.max_depth_spinbox.get())
        if max_depth <= 0:
            raise ValueError("Max depth must be positive")
        
        n_initial = int(self.n_initial.get())
        if n_initial % 2 == 0:
            raise ValueError("Number of initial models must be odd")
        
        # Create data directory list for each line
        data_dir_list = [os.path.join(main_data_dir, line) for line in lines]
        
        # Create results directories using the utility function approach
        overwrite_dir = self.overwrite_var.get()
        
        # Store all config in self.config
        self.config['site'] = site
        self.config['line_list'] = lines
        self.config['working_dir'] = working_dir
        self.config['main_data_dir'] = main_data_dir
        self.config['data_dir_list'] = data_dir_list
        self.config['main_initial_file'] = main_initial_file
        self.config['ignore_files'] = [f.strip() for f in self.ignore_files_entry.get().split(',')] if self.ignore_files_entry.get().strip() else []
        self.config['overwrite_dir'] = overwrite_dir
        self.config['max_depth'] = max_depth
        self.config['points_to_remove'] = None
        
        # Store processing settings in settings dictionary
        self.config['settings'] = {
            'figsize': (int(self.figsize_w.get()), int(self.figsize_h.get())),
            'max_depth': max_depth,
            'pseudo_depth': True,
            'dc_axis_limits': [(0, 1000), (0, max_depth)],
            'models_axes_lims': [(0, 1000), (0, max_depth), (0, 1000), (0, max_depth)],
            'no_std': float(self.no_std.get()),
            'resample_n': int(self.resample_n.get()),
            'resamp_max_pdepth': None,
            'dc_resamp_smoothing': self.dc_smooth_var.get(),
            'a_values': {line: int(self.a_values.get()) for line in lines},
            'only_save_accepted': False,
            'c_test': {
                'min': int(self.c_min.get()),
                'max': int(self.c_max.get()),
                'step': float(self.c_step.get()),
                'delta_c': int(self.delta_c.get())
            },
            'run': int(self.num_runs.get()),
            'N_max': int(self.n_max.get()),
            'repeat_run_if_fail': self.repeat_var.get(),
            'run_success_minimum': int(self.run_success_min.get()),
            'bs': int(self.bs.get()),
            'bh': int(self.bh.get()),
            'bs_min': int(self.bs_min.get()),
            'bh_min': int(self.bh_min.get()),
            'b_decay_iterations': int(self.b_decay.get()),
            'bs_min_halfspace': None,
            'N_stall_max': int(self.stall_max.get()),
            'st_lt_window': (int(self.st_window.get()), int(self.lt_window.get())),
            'misfit_mode': self.misfit_mode.get(),
            'uncertainty_weighting': self.uncert_weight_var.get(),
            'depth_weight_offset_const': int(self.depth_weight.get()),
            'plot_misfit_weights': False,
            'within_bounds_thresh': float(self.within_bounds.get()),
            'regularisation': self.regularisation.get(),
            'regularisation_ranges': None,
            'uncert_bounds_misfit_weight': int(self.uncert_bounds_weight.get()),
            'rep_explore_pen_weight': float(self.rep_explore_weight.get()),
            'regularisation_weights': [float(w.strip()) for w in self.reg_weights.get().split(',')],
            'rev_min_depth': self._parse_optional(self.rev_min_depth.get()),
            'rev_max_depth': self._parse_optional(self.rev_max_depth.get()),
            'rev_min_layer': None,
            'rev_max_layer': None,
            'max_retries': int(self.max_retries.get()),
            'force_monotonic': self.force_monotonic_var.get(),
            'monotonic_tol': int(self.monotonic_tol.get()),
            'vel_shift': int(self.vel_shift.get()),
            'n_initial_models': int(self.n_initial.get()),
            'new_run_prev_best_fit': self.use_prev_best_var.get(),
            'runs_initial': int(self.runs_initial.get())
        }
    
    def _parse_optional(self, value):
        """Parse None or integer."""
        value = value.strip()
        return None if value.lower() == "none" else int(value)
    
    # ============================================================================
    # PROCESSING FUNCTIONS
    # ============================================================================
    
    def validate_only(self):
        """Validate configuration without running inversion."""
        try:
            self.populate_config()
            self.show_config()
            self.status.config(text="✓ Validation successful!")
        except ValueError as e:
            messagebox.showerror("Validation Error", str(e))
            self.status.config(text="✗ Validation failed")
    
    def start_inversion(self):
        """
        Main entry point: Validate config, then start MASW_inv_main() in a separate thread.
        This prevents GUI freezing during processing.
        """
        try:
            # Validate and populate config
            self.populate_config()
            self.status.config(text="Configuration validated, preparing inversion...")
            
            # Confirm with user
            confirm = messagebox.askyesno(
                "Start Inversion",
                f"Start MASW inversion for site '{self.config['site']}'?\n\n"
                f"Lines: {', '.join(self.config['line_list'])}\n"
                f"Working directory: {self.config['working_dir']}"
            )
            
            if not confirm:
                self.status.config(text="Cancelled by user")
                return
            
            # Run inversion in a separate thread to keep GUI responsive
            thread = threading.Thread(target=self._run_inversion_thread, daemon=True)
            thread.start()
            
        except ValueError as e:
            messagebox.showerror("Validation Error", str(e))
            self.status.config(text="✗ Validation failed")
    
    def _run_inversion_thread(self):
        """Run MASW_inv_main in a separate thread."""
        try:
            self.status.config(text="⏳ Processing - DO NOT CLOSE")
            
            # Import here to avoid issues if module isn't found during GUI setup
            from inv_main import MASW_inv_main
            
            # Call the main inversion function with the validated config
            MASW_inv_main(self.config)
            
            self.status.config(text="✓ Inversion complete!")
            messagebox.showinfo("Success", 
                f"Inversion complete for site '{self.config['site']}'!\n\n"
                f"Results saved to:\n{self.config['inversion_dir']}"
            )
            
        except Exception as e:
            messagebox.showerror("Inversion Error", str(e))
            self.status.config(text="✗ Inversion failed")
    
    def show_config(self):
        """Display current configuration in a popup."""
        config_str = "Current Configuration:\n\n"
        config_str += f"Site: {self.config['site']}\n"
        config_str += f"Lines: {', '.join(self.config['line_list'])}\n"
        config_str += f"Working Dir: {self.config['working_dir']}\n"
        config_str += f"Data Dir: {self.config['main_data_dir']}\n"
        config_str += f"Max Depth: {self.config['max_depth']} m\n"
        config_str += f"\nInversion Parameters:\n"
        config_str += f"  MC Runs: {self.config['settings']['run']}\n"
        config_str += f"  Max Iterations: {self.config['settings']['N_max']}\n"
        config_str += f"  Vs Perturbation: {self.config['settings']['bs']}%\n"
        
        messagebox.showinfo("Configuration", config_str)
    
    def reset_form(self):
        """Reset form to defaults."""
        if messagebox.askyesno("Confirm", "Reset all fields to defaults?"):
            self.site_entry.delete(0, tk.END)
            self.site_entry.insert(0, "test")
            self.lines_entry.delete(0, tk.END)
            self.lines_entry.insert(0, "VR - 7")
            self.status.config(text="Form reset to defaults")


if __name__ == "__main__":
    root = tk.Tk()
    app = MAASWInversionGUI(root)
    root.mainloop()