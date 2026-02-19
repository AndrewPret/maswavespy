import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import threading
import os
import matplotlib
import json
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


class MASWInversionGUI:
    """
    MASW Inversion Parameter Configuration GUI.
    
    This GUI collects all parameters, validates them, and then calls
    MASW_inv_main(config) to start the inversion processing using maswavespy
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("MASW Inversion - Configuration Manager")
        self.root.geometry("700x900")

        # # Custom icon
        # icon = tk.PhotoImage(file="icon-16.png")
        # root.iconphoto(True, icon)

        self.wrap_length = 500
        
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
            'points_to_remove': None,
            'results_dir': '',
            'inversion_dir': '',
            'initial_model_dir': '',
            'settings': {}
        }
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.add_data_tab()
        self.add_initial_model_tab()
        self.add_comp_DC_tab()
        self.add_inversion_tab()
        self.add_constraints_tab()
        self.add_advanced_tab()
        
        # Status bar
        self.create_button_frame()

        # define persistent default directory json as same as the script location
        self.settings_file = os.path.join(os.path.expanduser("~"), ".masw_gui_settings.json")
        self.apply_persistent_settings()

        # Bind save to window close:
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    


    ### persistant settings
    def load_persistent_settings(self):
        """Load saved settings from file."""
        if os.path.exists(self.settings_file):
            with open(self.settings_file, 'r') as f:
                return json.load(f)
        return {}

    # After building your GUI widgets, load and apply saved settings
    def _get_widget_map(self):
        """Map of settings keys to their widgets."""
        return {
            # File/dir entries
            "working_dir": self.working_dir,
            "data_dir": self.data_dir,
            "initial_model_file": self.initial_model_file,
            # Text entries
            "site": self.site_entry,
            "line_ids": self.line_ID_entry,
            "ignore_files": self.ignore_files_entry,
            "a_values": self.a_values,
            "figsize_w": self.figsize_w,
            "figsize_h": self.figsize_h,
            #"max_depth": self.max_depth_spinbox,
            "no_std": self.no_std,
            "resample_n": self.resample_n,
            "c_min": self.c_min,
            "c_max": self.c_max,
            "c_step": self.c_step,
            "delta_c": self.delta_c,
            "num_runs": self.num_runs,
            "n_max": self.n_max,
            "run_success_min": self.run_success_min,
            "bs": self.bs,
            "bh": self.bh,
            "bs_min": self.bs_min,
            "bh_min": self.bh_min,
            "b_decay": self.b_decay,
            "stall_max": self.stall_max,
            "st_window": self.st_window,
            "lt_window": self.lt_window,
            "depth_weight": self.depth_weight,
            "within_bounds": self.within_bounds,
            "uncert_bounds_weight": self.uncert_bounds_weight,
            "rep_explore_weight": self.rep_explore_weight,
            "reg_weights": self.reg_weights,
            "rev_min_depth": self.rev_min_depth,
            "rev_max_depth": self.rev_max_depth,
            "max_retries": self.max_retries,
            "monotonic_tol": self.monotonic_tol,
            "vel_shift": self.vel_shift,
            "n_initial": self.n_initial,
            "auto_min_layers": self.auto_min_layers,
            "auto_max_layers": self.auto_max_layers,
            "auto_start_Vs": self.auto_start_Vs,
            "auto_const_rho": self.auto_const_rho,
            "auto_const_Vp": self.auto_const_Vp,
            "auto_const_nu": self.auto_const_nu,
            "models_dc_xmin":  self.models_dc_xmin,
            "models_dc_xmax":  self.models_dc_xmax,
            "models_dc_ymin":  self.models_dc_ymin,
            "models_dc_ymax":  self.models_dc_ymax,
            "models_mod_xmin": self.models_mod_xmin,
            "models_mod_xmax": self.models_mod_xmax,
            "models_mod_ymin": self.models_mod_ymin,
            "models_mod_ymax": self.models_mod_ymax,
            "f_min": self.f_min_entry,
            "f_max": self.f_max_entry,
        }

    def _get_checkvar_map(self):
        """Map of settings keys to their BooleanVars (checkboxes)."""
        return {
            "manual_initial": self.manual_initial_var,
            "overwrite": self.overwrite_var,
            "dc_smooth": self.dc_smooth_var,
            "repeat": self.repeat_var,
            "uncertainty_weighting": self.uncert_weight_var,
            "force_monotonic": self.force_monotonic_var,
        }

    def _get_combovar_map(self):
        """Map of settings keys to their StringVars (dropdowns/radiobuttons)."""
        return {
            "misfit_mode": self.misfit_mode,
            "regularisation": self.regularisation,
        }

    def save_persistent_settings(self):
        settings = {}
        for key, widget in self._get_widget_map().items():
            settings[key] = widget.get()
        for key, var in self._get_checkvar_map().items():
            settings[key] = var.get()
        for key, var in self._get_combovar_map().items():
            settings[key] = var.get()
        with open(self.settings_file, 'w') as f:
            json.dump(settings, f, indent=2)

    def apply_persistent_settings(self):
        if not os.path.exists(self.settings_file):
            return
        with open(self.settings_file, 'r') as f:
            saved = json.load(f)
        for key, widget in self._get_widget_map().items():
            if key in saved:
                widget.delete(0, tk.END)
                widget.insert(0, saved[key])
        for key, var in self._get_checkvar_map().items():
            if key in saved:
                var.set(saved[key])
        for key, var in self._get_combovar_map().items():
            if key in saved:
                var.set(saved[key])

    def save_config_to_file(self):
        """Save current settings to a user-chosen file."""
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All", "*.*")],
            title="Save Configuration"
        )
        if not path:
            return
        settings = {}
        for key, widget in self._get_widget_map().items():
            settings[key] = widget.get()
        for key, var in self._get_checkvar_map().items():
            settings[key] = var.get()
        for key, var in self._get_combovar_map().items():
            settings[key] = var.get()
        with open(path, 'w') as f:
            json.dump(settings, f, indent=2)
        self.status.config(text=f"✓ Config saved to {os.path.basename(path)}")

    def load_config_from_file(self):
        """Load settings from a user-chosen file."""
        path = filedialog.askopenfilename(
            filetypes=[("JSON", "*.json"), ("All", "*.*")],
            title="Load Configuration"
        )
        if not path:
            return
        try:
            with open(path, 'r') as f:
                saved = json.load(f)
            for key, widget in self._get_widget_map().items():
                if key in saved:
                    widget.delete(0, tk.END)
                    widget.insert(0, saved[key])
            for key, var in self._get_checkvar_map().items():
                if key in saved:
                    var.set(saved[key])
            for key, var in self._get_combovar_map().items():
                if key in saved:
                    var.set(saved[key])
            self.status.config(text=f"✓ Config loaded from {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load config:\n{e}")
            self.status.config(text="✗ Config load failed")

    def on_close(self):
        self.save_persistent_settings()
        self.root.destroy()
    
    def create_button_frame(self):
        """Create bottom button frame with control buttons."""
        bf = ttk.Frame(self.root)
        bf.pack(fill=tk.X, padx=5, pady=5)
        
        # Main processing button
        ttk.Button(bf, text="START INVERSION", command=self.start_inversion,
                    ).pack(side=tk.RIGHT, padx=5)
        
        # Utility buttons
        ttk.Button(bf, text="Save Config", command=self.save_config_to_file).pack(side=tk.RIGHT, padx=5)
        ttk.Button(bf, text="Load Config", command=self.load_config_from_file).pack(side=tk.RIGHT, padx=5)
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
    
    def update_ids(self):
        if self.line_ID_entry:
            raw_id_text = self.line_ID_entry.get()
        else:
            raw_id_text = "No line IDs specified"

        ids = [x.strip() for x in raw_id_text.split(",") if x.strip()]

        raw_a_value_text = self.a_values.get()
        a_values = [x.strip() for x in raw_a_value_text.split(",") if x.strip()]

        # Make sure both lists are the same length
        if len(a_values) < len(ids):
            a_values += [""] * (len(ids) - len(a_values))
        elif len(a_values) > len(ids):
            ids += [""] * (len(a_values) - len(ids))

        # Clear previous content
        self.line_ids_listbox.delete(0, tk.END)

        # Insert combined values as table
        for id_, a_value in zip(ids, a_values):
            self.line_ids_listbox.insert(tk.END, f"{id_:<10}   ->  {a_value:>10}")
    
    def add_spinbox(self, parent, label, default, from_val, to_val, row=0, increment=1):
        """Add spinbox field."""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        spinbox = ttk.Spinbox(parent, from_=from_val, to=to_val, increment=increment, width=20)
        spinbox.set(default)
        spinbox.grid(row=row, column=1, sticky=tk.W, padx=5, pady=5)
        return spinbox
    
    def add_file_browse(self, parent, label, is_dir=False, row=0):
        """Add file/directory browser."""
        
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky=tk.W, padx=5, pady=5
        )
        
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=1, sticky=tk.EW, padx=5, pady=5)
        
        entry = ttk.Entry(frame, width=40)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        def browse():
            current = entry.get().strip()
            if is_dir:
                initial_dir = current if os.path.exists(current) else os.getcwd()
                path = filedialog.askdirectory(initialdir=initial_dir)
            else:
                initial_dir = os.path.dirname(current) if os.path.exists(os.path.dirname(current)) else os.getcwd()
                path = filedialog.askopenfilename(
                    initialdir=initial_dir,
                    filetypes=[("CSV", "*.csv"), ("All", "*.*")]
                )

            if path:
                entry.delete(0, tk.END)
                entry.insert(0, path)
                
        ttk.Button(frame, text="Browse", width=10, command=browse).pack(side=tk.LEFT)

        return entry

    # DC preview without running full script
    def preview_dc(self):
        """Load and preview dispersion curves without running full inversion."""
        try:
            self.populate_config()
        except ValueError as e:
            messagebox.showerror("Validation Error", str(e))
            return

        # Let user select which line to preview if multiple
        lines = self.config['line_list']
        if len(lines) > 1:
            win = tk.Toplevel(self.root)
            win.title("Select Line")
            win.geometry("250x100")
            ttk.Label(win, text="Select line to preview:").pack(pady=5)
            line_var = tk.StringVar(value=lines[0])
            ttk.Combobox(win, textvariable=line_var, values=lines, state="readonly").pack(pady=5)
            ttk.Button(win, text="Preview", command=lambda: [win.destroy(), self._run_dc_preview(line_var.get())]).pack(pady=5)
        else:
            self._run_dc_preview(lines[0])

    def _run_dc_preview(self, line):
        """Run DC resampling and display plots in a popup window."""
        import threading
        thread = threading.Thread(target=self._dc_preview_thread, args=(line,), daemon=True)
        thread.start()

    def _dc_preview_thread(self, line):
        try:
            self.status.config(text="⏳ Loading dispersion curves...")

            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
            import matplotlib.pyplot as plt
            from maswavespy.combination import CombineDCs
            from maswavespy.zetica_utils import data_import

            config = self.config
            settings = config['settings']
            line_data_dir = os.path.join(config['main_data_dir'], line)
            
            f_vec, c_vec = data_import(line, line_data_dir,f_min=settings.get('f_min'),f_max=settings.get('f_max'))
            
            pre_combDC_obj = CombineDCs(config['site'], line, settings, freq=f_vec, c=c_vec)

            # Run the processing steps, capturing figures instead of saving
            a = settings['a_values'].get(line, 2)
            no_std = settings['no_std']
            resample_n = settings['resample_n']
            figsize = settings['figsize']
            axis_limits = settings['axes_limits'][:2]

            pre_combDC_obj.dc_cov()
            fig1, ax1 = pre_combDC_obj.plot_dc_cov(figwidth=figsize[0], figheight=figsize[1])
            ax1[0].set_title("Picked points")
            ax1[1].set_title("Coefficient of variation (COV)")
            fig1.suptitle(f"Experimental dispersion curves, {line}")

            pre_combDC_obj.dc_combination(a, no_std=no_std)
            fig2, ax2 = pre_combDC_obj.plot_combined_dc(plot_all=True, pseudo_depth=True, axis_limits=axis_limits)
            fig2.suptitle(f"Composite DC, {line}")

            wavelength_min = 'default'
            wavelength_max = 'default'
            smoothing = pre_combDC_obj.smoothing
            fig3, ax3 = pre_combDC_obj.resample_dc(resample_n, 'log', smoothing, wavelength_min, wavelength_max,
                                                show_fig=True, pseudo_depth=True, axis_limits=axis_limits)
            fig3.suptitle(f"Resampled comp. DC, a = {a}, {line}")

            # Display in tkinter window on main thread
            self.root.after(0, lambda: self._show_dc_preview_window(line, fig1, fig2, fig3))
            self.status.config(text="✓ Preview ready")

        except Exception as e:
            self.status.config(text="✗ Preview failed")
            messagebox.showerror("Preview Error", str(e))

    def _show_dc_preview_window(self, line, fig1, fig2, fig3):
        """Display the 3 DC figures in a single scrollable window."""
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        win = tk.Toplevel(self.root)
        win.title(f"DC Preview — {line}")
        win.geometry("1000x800")

        # Scrollable canvas
        outer = ttk.Frame(win)
        outer.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(outer, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        canvas = tk.Canvas(outer, yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=canvas.yview)

        inner = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=inner, anchor="nw")

        # Embed each figure
        for fig in [fig1, fig2, fig3]:
            embed = FigureCanvasTkAgg(fig, master=inner)
            embed.draw()
            embed.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=5)

        inner.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))

        # Mouse wheel scrolling
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(-1 * (e.delta // 120), "units"))

        ttk.Button(win, text="Close", command=win.destroy).pack(pady=5)
    
    # ============================================================================
    # DIRECTORIES TAB
    # ============================================================================
    
    def add_data_tab(self):
        """Add input and output directory tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Input/Output")
        scroll_frame = self.make_scrollable_frame(frame)
        
        row = 0
        ttk.Label(scroll_frame, text="1D Monte Carlo Dispersion Curve Inversion", 
                font=("Arial", 12, "bold")).grid(row=row, column=0, columnspan=2, 
                sticky=tk.W, padx=5, pady=(10, 5))
        row += 1

        ttk.Label(scroll_frame, text="Project Information", 
                font=("Arial", 12, "bold")).grid(row=row, column=0, columnspan=2, 
                sticky=tk.W, padx=5, pady=(10, 5))
        row += 1


        self.site_entry = self.add_entry(scroll_frame, "Site Name:", "test", row=row)
        row += 1
        
        self.line_ID_entry = self.add_entry(scroll_frame, "Line IDs (comma-separated):" \
        "\n     These must match names of" \
        "\n     directories containing line data", "VR - 7", row=row)
        row += 1
        
        self.working_dir = self.add_file_browse(scroll_frame, "Working Directory:" \
        "\n     Results directory will be created here", is_dir=True, row=row)
        row += 1
        
        self.data_dir = self.add_file_browse(scroll_frame, "Main Data Directory:" \
        "\n     Select parent data directory which " \
        "\n     must contain 1 directory per Line ID", is_dir=True, row=row)
        row += 1

        
        self.ignore_files_entry = self.add_entry(scroll_frame, "Files to Ignore (comma-separated):" \
        "\n     Specify .dc files by name to exclude from processing" \
        "\n     e.g. 1-18(Cx7p5)(Sx-2p5)(fn100)R.dc", "", row=row)
        row += 1
        
        ttk.Label(scroll_frame, text="Overwrite Results:",font=("Arial", 10, "bold")).grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        self.overwrite_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(scroll_frame, variable=self.overwrite_var,
                    text="If unchecked, creates new timestamped folder in\nresults directory each time inversion is run").grid(row=row, column=1, sticky=tk.W, padx=5, pady=5)
        row += 1

        ttk.Label(scroll_frame, text="Plotting", 
                font=("Arial", 12, "bold")).grid(row=row, column=0, columnspan=2, 
                sticky=tk.W, padx=5, pady=(10, 5))
        row += 1

        # self.max_depth_spinbox = self.add_spinbox(scroll_frame, "Max depth (m):", 15, 1, 100, increment=0.1, row=row)
        # row += 1

        ttk.Label(scroll_frame, text="DC axes:", font=("Arial", 9)).grid(
        row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(10, 0))
        ttk.Label(scroll_frame, text="Min:                 Max:", font=("Arial", 9)).grid(
        row=row, column=1, columnspan=2, sticky=tk.W, padx=5, pady=(10, 0))
        row += 1

        ttk.Label(scroll_frame, text="Vr (m/s):").grid(row=row, column=0, sticky=tk.W, padx=5)
        dc_x_frame = ttk.Frame(scroll_frame)
        dc_x_frame.grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        self.models_dc_xmin = ttk.Spinbox(dc_x_frame, from_=0, to=10000, increment=50, width=8)
        self.models_dc_xmin.set(0)
        self.models_dc_xmin.pack(side=tk.LEFT, padx=(0, 5))
        self.models_dc_xmax = ttk.Spinbox(dc_x_frame, from_=0, to=10000, increment=50, width=8)
        self.models_dc_xmax.set(1000)
        self.models_dc_xmax.pack(side=tk.LEFT)
        row += 1

        ttk.Label(scroll_frame, text="Pseudo-depth (m):").grid(row=row, column=0, sticky=tk.W, padx=5)
        dc_y_frame = ttk.Frame(scroll_frame)
        dc_y_frame.grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        self.models_dc_ymin = ttk.Spinbox(dc_y_frame, from_=0, to=10000, increment=1, width=8)
        self.models_dc_ymin.set(0)
        self.models_dc_ymin.pack(side=tk.LEFT, padx=(0, 5))
        self.models_dc_ymax = ttk.Spinbox(dc_y_frame, from_=0, to=10000, increment=1, width=8)
        self.models_dc_ymax.set(15)
        self.models_dc_ymax.pack(side=tk.LEFT)
        row += 1

        ttk.Label(scroll_frame, text="Vs model axes:", font=("Arial", 9)).grid(
        row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(10, 0))
        ttk.Label(scroll_frame, text="Min:                 Max:", font=("Arial", 9)).grid(
        row=row, column=1, columnspan=2, sticky=tk.W, padx=5, pady=(10, 0))
        row += 1

        ttk.Label(scroll_frame, text="Vs (m/s):").grid(row=row, column=0, sticky=tk.W, padx=5)
        mod_x_frame = ttk.Frame(scroll_frame)
        mod_x_frame.grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        self.models_mod_xmin = ttk.Spinbox(mod_x_frame, from_=0, to=10000, increment=50, width=8)
        self.models_mod_xmin.set(0)
        self.models_mod_xmin.pack(side=tk.LEFT, padx=(0, 5))
        self.models_mod_xmax = ttk.Spinbox(mod_x_frame, from_=0, to=10000, increment=50, width=8)
        self.models_mod_xmax.set(1000)
        self.models_mod_xmax.pack(side=tk.LEFT)
        row += 1

        ttk.Label(scroll_frame, text="Depth (m/s):").grid(row=row, column=0, sticky=tk.W, padx=5)
        mod_y_frame = ttk.Frame(scroll_frame)
        mod_y_frame.grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        self.models_mod_ymin = ttk.Spinbox(mod_y_frame, from_=0, to=10000, increment=1, width=8)
        self.models_mod_ymin.set(0)
        self.models_mod_ymin.pack(side=tk.LEFT, padx=(0, 5))
        self.models_mod_ymax = ttk.Spinbox(mod_y_frame, from_=0, to=10000, increment=1, width=8)
        self.models_mod_ymax.set(15)
        self.models_mod_ymax.pack(side=tk.LEFT)
        row += 1

        ttk.Label(scroll_frame, text="Output figure size", font=("Arial", 9)).grid(
        row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(10, 0))
        row += 1

        self.figsize_w = self.add_spinbox(scroll_frame, "Width:", 18, 5, 30, row=row)
        row += 1
        self.figsize_h = self.add_spinbox(scroll_frame, "Height:", 15, 5, 30, row=row)
        row += 1

    # ============================================================================
    # INITIAL MODEL TAB
    # ============================================================================

    def add_initial_model_tab(self):
        """Add inversion settings tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Initial Model")
        scroll_frame = self.make_scrollable_frame(frame)
        
        row = 0

        ttk.Label(scroll_frame, text="Create initial model", 
                font=("Arial", 12, "bold")).grid(row=row, column=0, columnspan=2, 
                sticky=tk.W, padx=5, pady=(10, 5))
        row += 1

        ttk.Label(scroll_frame, text="By default, a constant velocity initial model is used to avoid adding any bias towards the initial model." \
                                    " \nNo. of layers (N) and layer thickness (h) are automatically determined using wavelength range of " \
                                    " \ninput dispersion curves and 1/3 wavelength sampling depth",
                font=("Arial", 10)).grid(row=row, column=0, columnspan=2,
                sticky=tk.W, padx=5, pady=(10, 5))
        row += 1

        self.auto_min_layers = self.add_spinbox(scroll_frame, "Minimum number of layers:", 3, 2, 20, row=row)
        row += 1

        self.auto_max_layers = self.add_spinbox(scroll_frame, "Maximum number of layers:", 10, 3, 50, row=row)
        row += 1

        self.auto_geom_factor = self.add_spinbox(scroll_frame, "Geometric factor:" \
        "\n      1.0 = All layers have equal h" \
        "\n      1.2 = layer thickness increases with " \
        "\n            depth by 1.2x previous layer", 1.1, 1.0, 2.0, increment = 0.05, row=row)
        row += 1

        self.auto_start_Vs = self.add_spinbox(scroll_frame, "Starting Vs (m/s)", 200, 10, 2000, increment=10, row=row)
        row += 1

        self.auto_const_Vp = self.add_spinbox(scroll_frame, "Assumed constant Vp (m/s)", 2000, 10, 2000, increment=10, row=row)
        row += 1


        self.auto_const_rho = self.add_spinbox(scroll_frame, "Assumed rho (density) value (units)", 1800, 100, 8000, row=row)
        row += 1

        self.auto_const_nu = self.add_spinbox(scroll_frame, "Assumed nu (density) value (units)", 0.3, 0.01, 1, increment=0.01, row=row)
        row += 1

        ttk.Label(scroll_frame, text="Import from file", 
                font=("Arial", 12, "bold")).grid(row=row, column=0, columnspan=2, 
                sticky=tk.W, padx=5, pady=(10, 5))
        row += 1

        ttk.Label(scroll_frame, text="Use manually defined initial model:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        self.manual_initial_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(scroll_frame, variable=self.overwrite_var,
                    text="If checked, all above settings are ignored").grid(row=row, column=1, sticky=tk.W, padx=5, pady=5)
        row += 1
        
        if self.manual_initial_var:
            self.initial_model_file = self.add_file_browse(scroll_frame, "Initial model file (.csv):", is_dir=False, row=row)
            row += 1

        ttk.Label(scroll_frame, text="Variable initial model velocity", 
        font=("Arial", 12, "bold")).grid(row=row, column=0, columnspan=2, 
        sticky=tk.W, padx=5, pady=(10, 5))
        row += 1
        
        self.n_initial = self.add_spinbox(scroll_frame, "Number of Initial Models:" \
        "\n    Use multiple starting models with different initial Vs can" \
        "\n    improve exploration of model space and avoid local minima" \
        "\n    If 1, only provided initial model will be used, otherwise" \
        "\n    the number of runs will be divided between starting models", 1, 1, 5, increment = 2, row=row)
        row += 1

        self.vel_shift = self.add_spinbox(scroll_frame, "Velocity Shift (m/s):" \
        "\n    e.g. If no. of initial models = 3 , the init. models will have" \
        "\n     starting velocities of -shift, input Vs, +shift", 100, 10, 500, row=row)
        row += 1

    # ============================================================================
    # COMPOSITE DISPERSION CURVE TAB
    # ============================================================================

    def add_comp_DC_tab(self):
        """Add inversion settings tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Composite DC")
        scroll_frame = self.make_scrollable_frame(frame)
        
        row = 0

        ttk.Label(scroll_frame, text="Composite Dispersion Curve", 
                font=("Arial", 12, "bold")).grid(row=row, column=0, columnspan=2, 
                sticky=tk.W, padx=5, pady=(10, 5))
        row += 1

        ttk.Label(scroll_frame,  text="Before inversion, individual DCs are combined and resampled as a single composite DC for each " \
                                    "line by calculating mean phase velocities and their distribution (std dev) within log(A)-spaced " \
                                    "wavelength bins.",
                font=("Arial", 10, "bold"), wraplength=self.wrap_length).grid(row=row, column=0, columnspan=1, 
                sticky=tk.W, padx=5, pady=(10, 5))
        row += 1

        ttk.Label(scroll_frame, text="Increasing bin width (by decreasing A) gives a smoother composite DC (and vice versa). " \
                                    "Intervals should be narrow enough to accurately capture phase velocity variations while " \
                                    "ensuring that the number data points from individual DCsare still reasonably evenly distributed between bins " \
                                    "\n" \
                                    "\n" \
                                    "An uneven distribution of data points (caused by A being too high) creates a composite DC with  "
                                    "very narrow std dev bounds within intervals containing too few data points. ",
                font=("Arial", 10),wraplength = self.wrap_length).grid(row=row, column=0, columnspan=1, 
                sticky=tk.W, padx=5, pady=(10, 5))
        row += 1

        ttk.Label(scroll_frame,      text="TL/DR: Per line, A should be set as high as possible without producing " \
                                        "sharp 'pinches' in std dev bounds. Values between 2 and 5 usually work well. ",
                font=("Arial", 10, "bold"), wraplength=self.wrap_length).grid(row=row, column=0, columnspan=1, 
                sticky=tk.W, padx=5, pady=(10, 5))
        row += 1
        
        ttk.Label(scroll_frame, text="A values for each line (comma-separated):").grid(
            row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(5, 0))
        row += 1

        self.a_values = ttk.Entry(scroll_frame, width=40)
        self.a_values.insert(0, "2")
        self.a_values.grid(row=row, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=(0, 5))
        row += 1


        ttk.Label(scroll_frame, text="Line IDs and A values:" , 
                font=("Arial", 10, "bold")).grid(row=row, column=0, columnspan=2, 
                sticky=tk.W, padx=5, pady=(10, 5))
        row += 1

        self.line_ids_listbox = tk.Listbox(scroll_frame, width=30, height=10)
        self.line_ids_listbox.grid(row=row, column=0, columnspan=1, sticky="EW", padx=5, pady=5)

        row += 1

        ttk.Button(scroll_frame, text="Update", command=self.update_ids).grid(
            row=row, column=0, columnspan=1, sticky="EW", padx=10, pady=5)
        row += 1

        ttk.Label(scroll_frame, text="Remove picks by frequency :", 
          font=("Arial", 9, "bold")).grid(row=row, column=0, columnspan=2, 
          sticky=tk.W, padx=5, pady=(10, 0))
        row += 1

        ttk.Label(scroll_frame, text="Min frequency (Hz):").grid(row=row, column=0, sticky=tk.W, padx=5)
        row += 1
        f_min_frame = ttk.Frame(scroll_frame)
        f_min_frame.grid(row=row, column=0, sticky=tk.EW, padx=5, pady=2)
        self.f_min_var = tk.StringVar(value="None")
        self.f_min_entry = ttk.Entry(f_min_frame, textvariable=self.f_min_var, width=10)
        self.f_min_entry.pack(side=tk.LEFT)
        ttk.Label(f_min_frame, text='  (enter "None" to disable)').pack(side=tk.LEFT)
        row += 1

        ttk.Label(scroll_frame, text="     Max frequency (Hz):").grid(row=row, column=0, sticky=tk.W, padx=5)
        row += 1
        f_max_frame = ttk.Frame(scroll_frame)
        f_max_frame.grid(row=row, column=0, sticky=tk.EW, padx=5, pady=2)
        self.f_max_var = tk.StringVar(value="None")
        self.f_max_entry = ttk.Entry(f_max_frame, textvariable=self.f_max_var, width=10)
        self.f_max_entry.pack(side=tk.LEFT)
        ttk.Label(f_max_frame, text='  (enter "None" to disable)').pack(side=tk.LEFT)
        row += 1

        ttk.Button(scroll_frame, text="Preview DCs", command=self.preview_dc).grid(
            row=row, column=0, columnspan=1, sticky="EW", padx=10, pady=5)

    
    # ============================================================================
    # INVERSION SETTINGS TAB
    # ============================================================================
    
    def add_inversion_tab(self):
        """Add inversion settings tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Inversion")
        scroll_frame = self.make_scrollable_frame(frame)
        
        row = 0
        ttk.Label(scroll_frame, text="Monte Carlo Inversion Parameters:", 
                font=("Arial", 12, "bold")).grid(row=row, column=0, columnspan=2, 
                sticky=tk.W, padx=5, pady=(10, 5))
        row += 1

        ttk.Label(scroll_frame, text="All run starts without information on previous runs\n" \
        "Iteration 1 applies random pertuabtions to the initial model\n" \
        "Further iterations apply random perturbations to the previous\n" \
        "iteration's model", 
                font=("Arial", 10, "bold")).grid(row=row, column=0, columnspan=2, 
                sticky=tk.W, padx=5, pady=(10, 5))
        row += 1

        self.num_runs       = self.add_spinbox(scroll_frame, "Number of runs:", 10, 1, 100, row=row)
        row += 1
        self.n_max =    self.add_spinbox(scroll_frame, "Max Iterations per run:", 100, 10, 500, row=row)
        row += 1
        
        ttk.Label(scroll_frame, text="Change the maximum random perturbation of model layer" \
        "\nthickness and Vs per iteration and number of iterations" \
        "\n to decay from max to min value." , 
                font=("Arial", 10, "bold")).grid(row=row, column=0, columnspan=2, 
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
        
        ttk.Label(scroll_frame, text="Forward Modelling (Vs Model to Vr DC)", 
                font=("Arial", 12, "bold")).grid(row=row, column=0, columnspan=2, 
                sticky=tk.W, padx=5, pady=(10, 5))
        row += 1
        
        self.c_min = self.add_spinbox(scroll_frame, "Min Test Velocity (m/s):", 100, 50, 500, row=row)
        row += 1
        self.c_max = self.add_spinbox(scroll_frame, "Max Test Velocity (m/s):", 1100, 500, 2000, row=row)
        row += 1
        self.c_step = self.add_spinbox(scroll_frame, "Velocity Step:" \
        "\n     Trade off between processing time and sampling", 0.2, 0.1, 1.0, row=row)
        row += 1
        self.delta_c = self.add_spinbox(scroll_frame, "Delta-C (m/s):", 3, 1, 10, row=row)

    # ============================================================================
    # CONSTRAINTS TAB
    # ============================================================================
    
    def add_constraints_tab(self):
        """Add constraints tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Constraints")
        scroll_frame = self.make_scrollable_frame(frame)
        
        row = 0

        ttk.Label(scroll_frame, text="Limit velocity reversals", 
                font=("Arial", 12, "bold")).grid(row=row, column=0, columnspan=2, 
                sticky=tk.W, padx=5, pady=(10, 5))
        row += 1

        ttk.Label(scroll_frame, text="Force Monotonic:\n" \
        "    Force modelled Vs to increase with depth").grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        self.force_monotonic_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(scroll_frame, variable=self.force_monotonic_var).grid(row=row, column=1, sticky=tk.W, padx=5, pady=5)
        row += 1
        
        self.monotonic_tol = self.add_spinbox(scroll_frame, "Monotonic Tolerance (m/s):" \
        "\n     If Force Monotonic is enable, tolerate decrease in velocity between adjacent layers", 50, 0, 500, row=row)
        row += 1

        
        self.rev_min_depth = self.add_entry(scroll_frame, "Min reversal Depth:", "None", row=row)
        row += 1
        self.rev_max_depth = self.add_entry(scroll_frame, "Max reversal Depth:", "None", row=row)
        row += 1
        self.max_retries = self.add_spinbox(scroll_frame, "Max Retries:", 200, 50, 500, row=row)
        row += 1

        ttk.Label(scroll_frame, text="Misfit and regularisation", 
                font=("Arial", 12, "bold")).grid(row=row, column=0, columnspan=2, 
                sticky=tk.W, padx=5, pady=(10, 5))
        row += 1

        ttk.Label(scroll_frame, text="Adds additional penalty terms to the misfit function. Standard RMS misfit is used if all disabled (set to 0)."\
                                    "\n"
                                    "\nTechnical note if these are used: Because new terms have been added, the model fit is no longer described by the RMS misfit value alone."\
                                    "Correct terminology TBC, instead of 'RMS misfit' the most general term is 'objective function', but 'weighted RMS misfit' mighe be appropriate.",
                font=("Arial", 10, "bold"), wraplength=self.wrap_length).grid(row=row, column=0, columnspan=2, 
                sticky=tk.W, padx=5, pady=(10, 5))
        row += 1
        
        self.uncert_bounds_weight = self.add_spinbox(scroll_frame, "Within Bounds Penalty Weight:", 0, 0, 100, row=row)
        row += 1
        self.rep_explore_weight = self.add_spinbox(scroll_frame, "Model Similarity Penalty Weight:", 0.0, 0.0, 1.0, row=row)
        row += 1
        

        ttk.Label(scroll_frame, text="Use regularisation:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        self.regularisation = tk.BooleanVar(value=False)
        ttk.Checkbutton(scroll_frame, variable=self.regularisation).grid(row=row, column=1, sticky=tk.W, padx=5, pady=5)

        row += 1
        self.reg_weights = self.add_entry(scroll_frame, "Regularisation Weights:", "1e-7", row=row)
        row += 1
        
        ttk.Label(scroll_frame, text="Within bounds penalty is an additional penalty (squared difference in Vr) added if the modelled DC is outside the uncertainty bounds." \
                                    "Useful if composite DC has uncertainty bounds and the inversion struggles to find models that fit within."\
                                    "Before using: Check A values for composite DC, they may too large."\
                                    "\n"\
                                    "\nModel similarity penalty encourages exploration of the model space by penalising models based on their similarity " \
                                    "to models from previous iterations of the same run. Encourages better sampling coverage and exploration of model space."\
                                    "\n"\
                                    "\nRegularisation applies a penalty based on the smoothness of the model. Higher weight = smoother model. " \
                                    "Useful to mitigate very sharp jumps and unrealistic oscillation in velocity between adjacent layers."\
                                    "Before using: This could be a symptom of overfitting, try reducing number of model layers ",
                font=("Arial", 9), wraplength=self.wrap_length).grid(row=row, column=0, columnspan=2, 
                sticky=tk.W, padx=5, pady=(10, 5))


    # ============================================================================
    # ADVANCED SETTINGS TAB
    # ============================================================================

    def add_advanced_tab(self):
        """Add advanced tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Advanced")
        scroll_frame = self.make_scrollable_frame(frame)

        row = 0
        ttk.Label(scroll_frame, text="Only useful in very specific circumstances, change at own risk",
                font=("Arial", 12, "bold")).grid(row=row, column=0, columnspan=2, 
                sticky=tk.W, padx=5, pady=(10, 5))
        row += 1
        
        ttk.Label(scroll_frame, text="Repeat failed iterations", 
                font=("Arial", 12, "bold")).grid(row=row, column=0, columnspan=2, 
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

        ttk.Label(scroll_frame, text="Dispersion curves", 
            font=("Arial", 12, "bold")).grid(row=row, column=0, columnspan=2, 
            sticky=tk.W, padx=5, pady=(10, 5))
        row += 1

        ttk.Label(scroll_frame, text="DC Smoothing:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        self.dc_smooth_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(scroll_frame, variable=self.dc_smooth_var).grid(row=row, column=1, sticky=tk.W, padx=5, pady=5)
        row += 1
        
        self.no_std = self.add_spinbox(scroll_frame, "DC uncertainty bounds (std dev):", 1, 0.5, 5, row=row)
        row += 1
        self.resample_n = self.add_spinbox(scroll_frame, "Resample Points:", 30, 10, 100, row=row)
        row += 1

        ttk.Label(scroll_frame, text="In case of stalling misfit (requires further testing)", 
                font=("Arial", 12, "bold")).grid(row=row, column=0, columnspan=2, 
                sticky=tk.W, padx=5, pady=(10, 5))
        row += 1
        
        row += 1
        self.stall_max = self.add_spinbox(scroll_frame, "Max Stall Iterations:", 0, 0, 20, row=row)
        row += 1
        self.st_window = self.add_spinbox(scroll_frame, "Short-term Window:", 200, 50, 500, row=row)
        row += 1
        self.lt_window = self.add_spinbox(scroll_frame, "Long-term Window:", 1000, 500, 2000, row=row)
        row += 1

        ttk.Label(scroll_frame, text="Additional misfit contraints", 
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
        
        lines = [l.strip() for l in self.line_ID_entry.get().split(',') if l.strip()]
        if not lines:
            raise ValueError("At least one line ID is required")
        
        a_vals = [v.strip() for v in self.a_values.get().split(',')]
        if len(a_vals) != len(lines):
            raise ValueError(f"Number of a_values ({len(a_vals)}) must match number of lines ({len(lines)})")
        
        working_dir = self.working_dir.get().strip()
        if not working_dir:
            raise ValueError("Working directory is required")
        if not os.path.exists(working_dir):
            raise ValueError(f"Working directory does not exist: {working_dir}")
        # add placeholder for results directory
        results_dir = f"{working_dir}\results"
        
        main_data_dir = self.data_dir.get().strip()
        if not main_data_dir:
            raise ValueError("Main data directory is required")
        
        if self.manual_initial_var.get():  # Only validate if box is ticked for import model file
            main_initial_file = self.initial_model_file.get().strip()
            if not main_initial_file:
                raise ValueError("Initial model file is required")
            if not os.path.exists(main_initial_file):
                raise ValueError(f"Initial model file does not exist: {main_initial_file}")
        # if auto intial model is used (default) this placeholder is replaced with the filepath to
        # auto generated model once it's been made
        else: 
            main_initial_file = "auto"
        
        # max_depth = int(self.max_depth_spinbox.get())
        # if max_depth <= 0:
        #     raise ValueError("Max depth must be positive")
        
        n_initial = int(self.n_initial.get())
        if n_initial % 2 == 0:
            raise ValueError("Number of initial models must be odd")
        
        # Create results directories using the utility function approach
        overwrite_dir = self.overwrite_var.get()
        
        # Store all config in self.config
        self.config['site'] = site
        self.config['line_list'] = lines
        self.config['working_dir'] = working_dir
        self.config['results_dir'] = results_dir
        self.config['main_data_dir'] = main_data_dir
        self.config['main_initial_file'] = main_initial_file
        self.config['ignore_files'] = [f.strip() for f in self.ignore_files_entry.get().split(',')] if self.ignore_files_entry.get().strip() else []
        self.config['overwrite_dir'] = overwrite_dir
        self.config['points_to_remove'] = None
        
        # Store processing settings in settings dictionary
        self.config['settings'] = {
            'figsize': (int(self.figsize_w.get()), int(self.figsize_h.get())),
            'pseudo_depth': True,
            'axes_limits': [(0, 1000), (0, 15), (0, 1000), (0, 15)],
            'no_std': float(self.no_std.get()),
            'resample_n': int(self.resample_n.get()),
            'dc_resamp_smoothing': self.dc_smooth_var.get(),
            'a_values': {line: int(v.strip()) for line, v in zip(lines, self.a_values.get().split(','))},
            'only_save_accepted': False,
            'c_test': {'min': int(self.c_min.get()),
                        'max': int(self.c_max.get()),
                        'step': float(self.c_step.get()),
                        'delta_c': int(self.delta_c.get())},
            'N_runs': int(self.num_runs.get()),
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
            'file_initial_model': int(self.manual_initial_var.get()),
            'auto_geom_factor': float(self.auto_geom_factor.get()),
            'auto_min_layers': int(self.auto_min_layers.get()),
            'auto_max_layers': int(self.auto_max_layers.get()),
            'auto_start_Vs': int(self.auto_start_Vs.get()),
            'auto_rho': int(self.auto_const_rho.get()),
            'auto_Vp': int(self.auto_const_Vp.get()),
            'const_nu': float(self.auto_const_nu.get()),
            'n_initial_models': int(self.n_initial.get()),
            'axes_limits': [(int(self.models_dc_xmin.get()),  int(self.models_dc_xmax.get())),
                                (int(self.models_dc_ymin.get()),  int(self.models_dc_ymax.get())),
                                (int(self.models_mod_xmin.get()), int(self.models_mod_xmax.get())),
                                (int(self.models_mod_ymin.get()), int(self.models_mod_ymax.get()))],
            'f_min': self._parse_optional(self.f_min_var.get()),
            'f_max': self._parse_optional(self.f_max_var.get()),
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
            if self.config['overwrite_dir'] == True:
                overwrite_check = messagebox.askyesno(
                    "Overwrite results is enabled, previous results will be lost.\n\n"
                    "Are you sure you want to proceed?"
                )
                if not overwrite_check:
                    self.status.config(text="Cancelled by user")
                    return
            
            # Confirm with user
            confirm = messagebox.askyesno(
                "Start Inversion",
                f"Start MASW inversion for site '{self.config['site']}'?\n\n"
                f"Lines: {', '.join(self.config['line_list'])}\n\n"
                f"Results Dir:\n    {self.config['results_dir']}\n\n"
                f"Data Dir:\n   {self.config['main_data_dir']}\n\n"
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
            self.save_persistent_settings()  # ← save before running
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
        """Display current configuration in a scrollable popup window."""
        try:
            self.populate_config()
        except ValueError as e:
            messagebox.showerror("Validation Error", str(e))
            return

        # Create popup window
        win = tk.Toplevel(self.root)
        win.title("Current Configuration")
        win.geometry("600x700")

        # Scrollable text widget
        frame = ttk.Frame(win)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        scrollbar = ttk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        text = tk.Text(frame, wrap=tk.WORD, yscrollcommand=scrollbar.set, font=("Courier", 10))
        text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=text.yview)

        # Build config string
        c = self.config
        s = c['settings']

        lines = []
        lines.append("=" * 50)
        lines.append("  SITE & DIRECTORIES")
        lines.append("=" * 50)
        lines.append(f"  Site:                 {c['site']}")
        lines.append(f"  Line IDs:             {', '.join(c['line_list'])}")
        lines.append(f"  Results Dir:          {c['results_dir']}")
        lines.append(f"  Data Dir:             {c['main_data_dir']}")
        lines.append(f"  Initial Model:        {c['main_initial_file']}")
        lines.append(f"  Ignore DC Files:      {', '.join(c['ignore_files']) if c['ignore_files'] else 'None'}")
        lines.append(f"  Overwrite Results:    {c['overwrite_dir']}")
        lines.append(f"  Figure Size:          {s['figsize'][0]} x {s['figsize'][1]}")
        lines.append(f"  Axis limits:          {s['axes_limits']}")

        lines.append("")
        lines.append("=" * 50)
        lines.append("  INITIAL MODEL")
        lines.append("=" * 50)
        lines.append(f"  File Initial:      {s['file_initial_model']}")
        lines.append(f"  Auto Geom Factor:  {s['auto_geom_factor']}")
        lines.append(f"  Auto Min Layers:   {s['auto_min_layers']}")
        lines.append(f"  Auto Max Layers:   {s['auto_max_layers']}")
        lines.append(f"  Auto Start Vs:     {s['auto_start_Vs']} m/s")
        lines.append(f"  Auto Vp:           {s['auto_Vp']} m/s")
        lines.append(f"  Auto Rho:          {s['auto_rho']} kg/m^3")
        lines.append(f"  Const Nu:          {s['const_nu']}")
        lines.append(f"  N Initial Models:  {s['n_initial_models']}")
        lines.append(f"  Vel Shift:         {s['vel_shift']}")
        lines.append("")
        lines.append("=" * 50)
        lines.append("  DISPERSION CURVE")
        lines.append("=" * 50)
        lines.append(f"  A Values:          {s['a_values']}")
        lines.append(f"  No. Stdev:         {s['no_std']}")
        lines.append(f"  C Test Min:        {s['c_test']['min']} m/s")
        lines.append(f"  C Test Max:        {s['c_test']['max']} m/s")
        lines.append(f"  C Test Step:       {s['c_test']['step']} m/s")
        lines.append(f"  Delta C:           {s['c_test']['delta_c']} m/s")
        lines.append(f"  Resample N:        {s['resample_n']}")
        lines.append(f"  DC Smoothing:      {s['dc_resamp_smoothing']}")
        lines.append("")
        lines.append("=" * 50)
        lines.append("  INVERSION CONTROL")
        lines.append("=" * 50)
        lines.append(f"  No. of Runs:       {s['N_runs']}")
        lines.append(f"  Iter. per run:     {s['N_max']}")
        lines.append(f"  Repeat if Fail:    {s['repeat_run_if_fail']}")
        lines.append(f"  Run Success Min:   {s['run_success_minimum']}")
        lines.append(f"  Max Retries:       {s['max_retries']}")
        lines.append(f"  Stall Max:         {s['N_stall_max']}")
        lines.append(f"  ST/LT Window:      {s['st_lt_window'][0]} / {s['st_lt_window'][1]}")
        lines.append(f"  Vs Pert (bs):      {s['bs']}%")
        lines.append(f"  H Pert (bh):       {s['bh']}%")
        lines.append(f"  Vs Min (bs_min):   {s['bs_min']}%")
        lines.append(f"  H Min (bh_min):    {s['bh_min']}%")
        lines.append(f"  B Decay Iters:     {s['b_decay_iterations']}")
        lines.append("")
        lines.append("=" * 50)
        lines.append("  MISFIT & WEIGHTING")
        lines.append("=" * 50)
        lines.append(f"  Misfit Mode:       {s['misfit_mode']}")
        lines.append(f"  Uncert Weighting:  {s['uncertainty_weighting']}")
        lines.append(f"  Depth Weight:      {s['depth_weight_offset_const']}")
        lines.append(f"  Within Bounds:     {s['within_bounds_thresh']}")
        lines.append(f"  Uncert Bounds Wt:  {s['uncert_bounds_misfit_weight']}")
        lines.append(f"  Rep Explore Wt:    {s['rep_explore_pen_weight']}")
        lines.append("")
        lines.append("=" * 50)
        lines.append("  CONSTRAINTS AND REGULARISATION")
        lines.append("=" * 50)
        lines.append(f"  Regularisation:    {s['regularisation']}")
        lines.append(f"  Reg Weights:       {s['regularisation_weights']}")
        lines.append(f"  Force Monotonic:   {s['force_monotonic']}")
        lines.append(f"  Monotonic Tol:     {s['monotonic_tol']}")
        lines.append(f"  Rev Min Depth:     {s['rev_min_depth']}")
        lines.append(f"  Rev Max Depth:     {s['rev_max_depth']}")
        lines.append("")
        lines.append("=" * 50)


        text.insert(tk.END, "\n".join(lines))
        text.config(state=tk.DISABLED)

        # Close button
        ttk.Button(win, text="Close", command=win.destroy).pack(pady=5)
    
    def reset_form(self):
        """Reset form to defaults."""
        if messagebox.askyesno("Confirm", "Reset all fields to defaults?"):
            self.site_entry.delete(0, tk.END)
            self.site_entry.insert(0, "test")
            self.line_ID_entry.delete(0, tk.END)
            self.line_ID_entry.insert(0, "VR - 7")
            self.status.config(text="Form reset to defaults")


if __name__ == "__main__":
    root = tk.Tk()
    app = MASWInversionGUI(root)
    root.mainloop()