import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import os
from pathlib import Path
import os
import pickle
from maswavespy import combination, inversion
import maswavespy.zetica_utils as zutil
import numpy as np
import time
import json

# -*- coding: utf-8 -*-
"""
MASW Inversion Processing Script
--------------------------------
Requires Zetica's version of MASWavesPy, available to clone here: https://github.com/AndrewPret/maswavespy.git

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

class MAASWInversionGUI:
    """GUI for MASW inversion parameter input with validation."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("MASW Inversion Parameter Configuration")
        self.root.geometry("1000x850")
        self.root.minsize(900, 700)
        
        self.validated_data = {}
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add tabs
        self.add_basic_tab()
        self.add_processing_tab()
        self.add_inversion_tab()
        self.add_regularisation_tab()
        self.add_constraints_tab()
        
        # Status bar at bottom
        status_frame = ttk.Frame(root)
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(status_frame, text="Validate & Export", 
                   command=self.validate_all).pack(side=tk.RIGHT, padx=5)
        ttk.Button(status_frame, text="Reset Form", 
                   command=self.reset_form).pack(side=tk.RIGHT, padx=5)
        
        self.status_label = ttk.Label(status_frame, text="Ready", relief=tk.SUNKEN)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    
    def make_scrollable_frame(self, parent):
        """Create a scrollable frame within parent."""
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        return scrollable_frame
    
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
    
    def add_basic_tab(self):
        """Add basic settings tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Basic Settings")
        scroll_frame = self.make_scrollable_frame(frame)
        
        row = 0
        self.site_entry = self.add_entry(scroll_frame, "Site Name:", "test", row=row)
        row += 1
        
        self.lines_entry = self.add_entry(scroll_frame, "Line IDs (comma-separated):", "VR - 7", row=row)
        row += 1
        
        self.working_dir = self.add_file_browse(scroll_frame, "Working Directory:", is_dir=True, row=row)
        row += 1
        
        self.data_dir = self.add_file_browse(scroll_frame, "Data Directory:", is_dir=True, row=row)
        row += 1
        
        self.initial_model = self.add_file_browse(scroll_frame, "Initial Model File (.csv):", is_dir=False, row=row)
        row += 1
        
        self.ignore_files_entry = self.add_entry(scroll_frame, "Files to Ignore (comma-separated):", "", row=row)
        row += 1
        
        ttk.Label(scroll_frame, text="Overwrite Results:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        self.overwrite_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(scroll_frame, variable=self.overwrite_var,
                       text="If False, creates timestamped folders").grid(row=row, column=1, sticky=tk.W, padx=5, pady=5)
        row += 1
        
        self.max_depth_spinbox = self.add_spinbox(scroll_frame, "Maximum Depth (m):", 15, 5, 100, row=row)
    
    def add_processing_tab(self):
        """Add processing settings tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Processing")
        scroll_frame = self.make_scrollable_frame(frame)
        
        row = 0
        
        # Section header
        ttk.Label(scroll_frame, text="FIGURE SETTINGS", font=("Arial", 10, "bold")).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(10, 5))
        row += 1
        
        self.figsize_w = self.add_spinbox(scroll_frame, "Figure Width (inches):", 18, 5, 30, row=row)
        row += 1
        self.figsize_h = self.add_spinbox(scroll_frame, "Figure Height (inches):", 15, 5, 30, row=row)
        row += 1
        
        ttk.Label(scroll_frame, text="DISPERSION CURVE", font=("Arial", 10, "bold")).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(10, 5))
        row += 1
        
        self.no_std = self.add_spinbox(scroll_frame, "Std Dev for Bounds:", 1, 0.5, 5, row=row)
        row += 1
        self.resample_n = self.add_spinbox(scroll_frame, "Resample Points:", 30, 10, 100, row=row)
        row += 1
        
        ttk.Label(scroll_frame, text="DC Smoothing:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        self.dc_smooth_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(scroll_frame, variable=self.dc_smooth_var).grid(row=row, column=1, sticky=tk.W, padx=5, pady=5)
        row += 1
        
        self.resamp_depth = self.add_spinbox(scroll_frame, "Resample Max Depth (0=none):", 0, 0, 100, row=row)
        row += 1
        self.a_values = self.add_spinbox(scroll_frame, "Resampling Density (a-value):", 2, 1, 10, row=row)
        row += 1
        
        ttk.Label(scroll_frame, text="VELOCITY TESTING", font=("Arial", 10, "bold")).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(10, 5))
        row += 1
        
        self.c_min = self.add_spinbox(scroll_frame, "Min Test Velocity (m/s):", 100, 50, 500, row=row)
        row += 1
        self.c_max = self.add_spinbox(scroll_frame, "Max Test Velocity (m/s):", 1100, 500, 2000, row=row)
        row += 1
        self.c_step = self.add_spinbox(scroll_frame, "Velocity Step:", 0.2, 0.1, 1.0, row=row)
        row += 1
        self.delta_c = self.add_spinbox(scroll_frame, "Delta-C (m/s):", 3, 1, 10, row=row)
    
    def add_inversion_tab(self):
        """Add inversion settings tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Inversion")
        scroll_frame = self.make_scrollable_frame(frame)
        
        row = 0
        
        ttk.Label(scroll_frame, text="MONTE CARLO", font=("Arial", 10, "bold")).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(10, 5))
        row += 1
        
        self.num_runs = self.add_spinbox(scroll_frame, "Number of MC Runs:", 10, 1, 100, row=row)
        row += 1
        self.n_max = self.add_spinbox(scroll_frame, "Max Iterations per Run:", 100, 10, 500, row=row)
        row += 1
        
        ttk.Label(scroll_frame, text="Repeat If No Success:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        self.repeat_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(scroll_frame, variable=self.repeat_var).grid(row=row, column=1, sticky=tk.W, padx=5, pady=5)
        row += 1
        
        self.run_success_min = self.add_spinbox(scroll_frame, "Minimum Successful Runs:", 0, 0, 50, row=row)
        row += 1
        
        ttk.Label(scroll_frame, text="PERTURBATION", font=("Arial", 10, "bold")).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(10, 5))
        row += 1
        
        self.bs = self.add_spinbox(scroll_frame, "Max Vs Perturbation (%):", 20, 1, 50, row=row)
        row += 1
        self.bh = self.add_spinbox(scroll_frame, "Max Thickness Pert (%):", 10, 1, 50, row=row)
        row += 1
        self.bs_min = self.add_spinbox(scroll_frame, "Min Vs Perturbation (%):", 4, 1, 20, row=row)
        row += 1
        self.bh_min = self.add_spinbox(scroll_frame, "Min Thickness Pert (%):", 2, 1, 20, row=row)
        row += 1
        self.b_decay = self.add_spinbox(scroll_frame, "Decay Iterations:", 200, 50, 1000, row=row)
        row += 1
        self.stall_max = self.add_spinbox(scroll_frame, "Max Stall Iterations:", 0, 0, 20, row=row)
        row += 1
        self.st_window = self.add_spinbox(scroll_frame, "Short-term Window:", 200, 50, 500, row=row)
        row += 1
        self.lt_window = self.add_spinbox(scroll_frame, "Long-term Window:", 1000, 500, 2000, row=row)
    
    def add_regularisation_tab(self):
        """Add regularisation tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Regularisation & Misfit")
        scroll_frame = self.make_scrollable_frame(frame)
        
        row = 0
        
        ttk.Label(scroll_frame, text="MISFIT", font=("Arial", 10, "bold")).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(10, 5))
        row += 1
        
        ttk.Label(scroll_frame, text="Misfit Mode:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        self.misfit_mode = ttk.Combobox(scroll_frame, values=["average", "average_weighted", "max", "max_weighted"], width=47)
        self.misfit_mode.current(0)
        self.misfit_mode.grid(row=row, column=1, sticky=tk.EW, padx=5, pady=5)
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
        
        ttk.Label(scroll_frame, text="REGULARISATION", font=("Arial", 10, "bold")).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(10, 5))
        row += 1
        
        ttk.Label(scroll_frame, text="Type:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        self.regularisation = ttk.Combobox(scroll_frame, values=["model", "dc"], width=47)
        self.regularisation.current(0)
        self.regularisation.grid(row=row, column=1, sticky=tk.EW, padx=5, pady=5)
        row += 1
        
        self.reg_weights = self.add_entry(scroll_frame, "Regularisation Weights:", "1e-7", row=row)
    
    def add_constraints_tab(self):
        """Add constraints tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Constraints")
        scroll_frame = self.make_scrollable_frame(frame)
        
        row = 0
        
        ttk.Label(scroll_frame, text="VELOCITY REVERSALS", font=("Arial", 10, "bold")).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(10, 5))
        row += 1
        
        self.rev_min_depth = self.add_entry(scroll_frame, "Min Reversal Depth or None:", "None", row=row)
        row += 1
        self.rev_max_depth = self.add_entry(scroll_frame, "Max Reversal Depth or None:", "None", row=row)
        row += 1
        self.max_retries = self.add_spinbox(scroll_frame, "Max Retries:", 200, 50, 500, row=row)
        row += 1
        
        ttk.Label(scroll_frame, text="Force Monotonic:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        self.force_monotonic_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(scroll_frame, variable=self.force_monotonic_var).grid(row=row, column=1, sticky=tk.W, padx=5, pady=5)
        row += 1
        
        self.monotonic_tol = self.add_spinbox(scroll_frame, "Monotonic Tolerance (m/s):", 100, 0, 500, row=row)
        row += 1
        
        ttk.Label(scroll_frame, text="INITIAL MODELS", font=("Arial", 10, "bold")).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(10, 5))
        row += 1
        
        self.vel_shift = self.add_spinbox(scroll_frame, "Velocity Shift (m/s):", 150, 10, 500, row=row)
        row += 1
        self.n_initial = self.add_spinbox(scroll_frame, "Number of Initial Models (odd):", 5, 1, 15, row=row)
        row += 1
        
        ttk.Label(scroll_frame, text="Use Previous Best Fit:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        self.use_prev_best_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(scroll_frame, variable=self.use_prev_best_var).grid(row=row, column=1, sticky=tk.W, padx=5, pady=5)
        row += 1
        
        self.runs_initial = self.add_spinbox(scroll_frame, "Runs Before Reuse:", 1, 1, 10, row=row)
    
    def validate_all(self):
        """Validate inputs and generate output."""
        try:
            # Validation
            site = self.site_entry.get().strip()
            if not site:
                raise ValueError("Site name cannot be empty")
            
            lines = [l.strip() for l in self.lines_entry.get().split(',') if l.strip()]
            if not lines:
                raise ValueError("At least one line ID required")
            
            max_depth = int(self.max_depth_spinbox.get())
            if max_depth <= 0:
                raise ValueError("Max depth must be positive")
            
            n_initial = int(self.n_initial.get())
            if n_initial % 2 == 0:
                raise ValueError("Number of initial models must be odd")
            
            # Generate config code
            config_code = self._generate_config()
            self._show_preview(config_code)
            self.status_label.config(text="✓ Validation successful")
            
        except Exception as e:
            messagebox.showerror("Validation Error", str(e))
            self.status_label.config(text="✗ Validation failed")
    
    def _generate_config(self):
        """Generate Python configuration code."""
        lines = [l.strip() for l in self.lines_entry.get().split(',') if l.strip()]
        max_depth = int(self.max_depth_spinbox.get())
        
        config = f"""# ----- USER INPUTS (Generated by MASW GUI) -----

                site = '{self.site_entry.get().strip()}'
                line_list = {lines}
                working_dir = r"{self.working_dir.get()}"
                main_data_dir = r"{self.data_dir.get()}"
                ignore_files = {[f.strip() for f in self.ignore_files_entry.get().split(',') if f.strip()]}
                main_initial_file = r"{self.initial_model.get()}"
                overwrite_dir = {self.overwrite_var.get()}
                max_depth = {max_depth}

                settings = {{
                    'figsize': ({int(self.figsize_w.get())}, {int(self.figsize_h.get())}),
                    'max_depth': {max_depth},
                    'pseudo_depth': True,
                    'dc_axis_limits': [(0, 1000), (0, {max_depth})],
                    'models_axes_lims': [(0, 1000), (0, {max_depth}), (0, 1000), (0, {max_depth})],
                    'no_std': {float(self.no_std.get())},
                    'resample_n': {int(self.resample_n.get())},
                    'resamp_max_pdepth': {None if self.resamp_depth.get() == '0' else int(self.resamp_depth.get())},
                    'dc_resamp_smoothing': {self.dc_smooth_var.get()},
                    'a_values': {{{', '.join([f"'{l}': {int(self.a_values.get())}" for l in lines])}}},
                    'only_save_accepted': False,
                    'c_test': {{'min': {int(self.c_min.get())}, 'max': {int(self.c_max.get())}, 'step': {float(self.c_step.get())}, 'delta_c': {int(self.delta_c.get())}}},
                    'run': {int(self.num_runs.get())},
                    'N_max': {int(self.n_max.get())},
                    'repeat_run_if_fail': {self.repeat_var.get()},
                    'run_success_minimum': {int(self.run_success_min.get())},
                    'bs': {int(self.bs.get())},
                    'bh': {int(self.bh.get())},
                    'bs_min': {int(self.bs_min.get())},
                    'bh_min': {int(self.bh_min.get())},
                    'b_decay_iterations': {int(self.b_decay.get())},
                    'bs_min_halfspace': None,
                    'N_stall_max': {int(self.stall_max.get())},
                    'st_lt_window': ({int(self.st_window.get())}, {int(self.lt_window.get())}),
                    'misfit_mode': '{self.misfit_mode.get()}',
                    'uncertainty_weighting': {self.uncert_weight_var.get()},
                    'depth_weight_offset_const': {int(self.depth_weight.get())},
                    'plot_misfit_weights': False,
                    'within_bounds_thresh': {float(self.within_bounds.get())},
                    'regularisation': '{self.regularisation.get()}',
                    'regularisation_ranges': None,
                    'uncert_bounds_misfit_weight': {int(self.uncert_bounds_weight.get())},
                    'rep_explore_pen_weight': {float(self.rep_explore_weight.get())},
                    'regularisation_weights': [{self.reg_weights.get()}],
                    'rev_min_depth': {self._parse_optional(self.rev_min_depth.get())},
                    'rev_max_depth': {self._parse_optional(self.rev_max_depth.get())},
                    'rev_min_layer': None,
                    'rev_max_layer': None,
                    'max_retries': {int(self.max_retries.get())},
                    'force_monotonic': {self.force_monotonic_var.get()},
                    'monotonic_tol': {int(self.monotonic_tol.get())},
                    'vel_shift': {int(self.vel_shift.get())},
                    'n_initial_models': {int(self.n_initial.get())},
                    'new_run_prev_best_fit': {self.use_prev_best_var.get()},
                    'runs_initial': {int(self.runs_initial.get())}
                }}

                points_to_remove = None

        return config
    
    def _parse_optional(self, value):
        """Parse None or integer."""
        value = value.strip()
        return "None" if value.lower() == "none" else int(value)
    
    def _show_preview(self, config_code):
        """Show configuration preview window."""
        preview = tk.Toplevel(self.root)
        preview.title("Configuration Preview")
        preview.geometry("700x600")
        
        text = tk.Text(preview, wrap=tk.WORD, font=("Courier", 9))
        text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        text.insert(1.0, config_code)
        text.config(state=tk.DISABLED)
        
        def copy():
            self.root.clipboard_clear()
            self.root.clipboard_append(config_code)
            messagebox.showinfo("Success", "Copied to clipboard!")
        
        def save():
            path = filedialog.asksaveasfilename(defaultextension=".py", filetypes=[("Python", "*.py")])
            if path:
                with open(path, 'w') as f:
                    f.write(config_code)
                messagebox.showinfo("Success", f"Saved to {path}")
        
        bf = ttk.Frame(preview)
        bf.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(bf, text="Copy", command=copy).pack(side=tk.LEFT, padx=5)
        ttk.Button(bf, text="Save", command=save).pack(side=tk.LEFT, padx=5)
    
    def reset_form(self):
        """Reset form."""
        if messagebox.askyesno("Confirm", "Reset all values to defaults?"):
            self.status_label.config(text="Form reset")


if __name__ == "__main__":
    root = tk.Tk()
    app = MAASWInversionGUI(root)
    root.mainloop()


# -------- MAIN PROCESSING LOOP --------
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


