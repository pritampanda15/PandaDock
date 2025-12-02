import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Dict, Any, Optional, Callable
from Bio.PDB import PDBParser
import logging
from ..gridbox.core import GridBox, GridBoxGenerator


class InteractiveGridBoxSelector:
    """Interactive GUI for grid box selection"""

    def __init__(self, receptor_file: str, callback: Optional[Callable] = None):
        self.receptor_file = receptor_file
        self.callback = callback
        self.logger = logging.getLogger(__name__)

        # Load receptor structure
        self.structure = self._load_structure()
        self.atoms = list(self.structure.get_atoms())
        self.residues = list(self.structure.get_residues())

        # Calculate protein center for initial grid placement
        coords = np.array([atom.get_coord() for atom in self.atoms])
        protein_center = np.mean(coords, axis=0)

        # GUI state
        self.selected_residues = []
        self.grid_boxes = []
        self.ai_suggestions = []  # Store AI-generated suggestions
        self.current_center = protein_center.tolist()  # Start at protein center
        self.current_dimensions = [20.0, 20.0, 20.0]
        self.current_spacing = 0.375

        # Create GUI
        self.root = tk.Tk()
        self.root.title("PandaDock Grid Box Selector")
        self.root.geometry("1200x800")

        self._create_widgets()
        self._update_protein_view()

    def _load_structure(self):
        """Load PDB structure"""
        parser = PDBParser(QUIET=True)
        return parser.get_structure('receptor', self.receptor_file)

    def _create_widgets(self):
        """Create GUI widgets"""

        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel - Controls
        left_panel = ttk.Frame(main_frame, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)

        # Right panel - 3D visualization
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Create control sections
        self._create_mode_selection(left_panel)
        self._create_residue_selection(left_panel)
        self._create_manual_controls(left_panel)
        self._create_ai_suggestions(left_panel)
        self._create_grid_properties(left_panel)
        self._create_action_buttons(left_panel)

        # Create 3D visualization
        self._create_3d_plot(right_panel)

    def _create_mode_selection(self, parent):
        """Create mode selection section"""
        frame = ttk.LabelFrame(parent, text="Selection Mode", padding=10)
        frame.pack(fill=tk.X, pady=(0, 10))

        self.mode_var = tk.StringVar(value="residue")

        ttk.Radiobutton(frame, text="Residue Selection", variable=self.mode_var,
                       value="residue", command=self._on_mode_change).pack(anchor=tk.W)
        ttk.Radiobutton(frame, text="Manual Coordinates", variable=self.mode_var,
                       value="manual", command=self._on_mode_change).pack(anchor=tk.W)
        ttk.Radiobutton(frame, text="AI Suggestions", variable=self.mode_var,
                       value="ai", command=self._on_mode_change).pack(anchor=tk.W)

    def _create_residue_selection(self, parent):
        """Create residue selection section"""
        self.residue_frame = ttk.LabelFrame(parent, text="Residue Selection", padding=10)
        self.residue_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Search entry
        search_frame = ttk.Frame(self.residue_frame)
        search_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        self.search_var.trace_add('write', self._on_search_change)
        ttk.Entry(search_frame, textvariable=self.search_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

        # Residue listbox
        list_frame = ttk.Frame(self.residue_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.residue_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, selectmode=tk.MULTIPLE)
        self.residue_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.residue_listbox.yview)

        # Populate residue list
        self._populate_residue_list()

        # Selection buttons
        button_frame = ttk.Frame(self.residue_frame)
        button_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Button(button_frame, text="Add Selected", command=self._add_selected_residues).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Clear All", command=self._clear_selected_residues).pack(side=tk.LEFT)

        # Selected residues display
        ttk.Label(self.residue_frame, text="Selected Residues:").pack(anchor=tk.W, pady=(10, 0))
        self.selected_text = tk.Text(self.residue_frame, height=4, state=tk.DISABLED)
        self.selected_text.pack(fill=tk.X, pady=(5, 0))

    def _create_manual_controls(self, parent):
        """Create manual coordinate controls"""
        self.manual_frame = ttk.LabelFrame(parent, text="Manual Coordinates", padding=10)
        self.manual_frame.pack(fill=tk.X, pady=(0, 10))

        # Center coordinates
        ttk.Label(self.manual_frame, text="Center (X, Y, Z):").pack(anchor=tk.W)
        center_frame = ttk.Frame(self.manual_frame)
        center_frame.pack(fill=tk.X, pady=(5, 10))

        self.center_vars = [tk.DoubleVar(value=self.current_center[i]) for i in range(3)]
        for i, var in enumerate(self.center_vars):
            var.trace_add('write', self._on_manual_change)
            ttk.Entry(center_frame, textvariable=var, width=8).pack(side=tk.LEFT, padx=(0, 5))

        # Dimensions
        ttk.Label(self.manual_frame, text="Dimensions (X, Y, Z):").pack(anchor=tk.W)
        dim_frame = ttk.Frame(self.manual_frame)
        dim_frame.pack(fill=tk.X, pady=(5, 0))

        self.dim_vars = [tk.DoubleVar(value=20.0) for _ in range(3)]
        for i, var in enumerate(self.dim_vars):
            var.trace_add('write', self._on_manual_change)
            ttk.Entry(dim_frame, textvariable=var, width=8).pack(side=tk.LEFT, padx=(0, 5))

    def _create_ai_suggestions(self, parent):
        """Create AI suggestions section"""
        self.ai_frame = ttk.LabelFrame(parent, text="AI Suggestions", padding=10)
        self.ai_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(self.ai_frame, text="Find Binding Sites", command=self._find_ai_suggestions).pack(fill=tk.X)

        # AI suggestions listbox
        self.ai_listbox = tk.Listbox(self.ai_frame, height=4)
        self.ai_listbox.pack(fill=tk.X, pady=(10, 0))
        self.ai_listbox.bind('<<ListboxSelect>>', self._on_ai_selection)

    def _create_grid_properties(self, parent):
        """Create grid properties section"""
        frame = ttk.LabelFrame(parent, text="Grid Properties", padding=10)
        frame.pack(fill=tk.X, pady=(0, 10))

        # Spacing
        ttk.Label(frame, text="Grid Spacing (Å):").pack(anchor=tk.W)
        self.spacing_var = tk.DoubleVar(value=0.375)
        self.spacing_var.trace_add('write', self._on_properties_change)
        ttk.Entry(frame, textvariable=self.spacing_var, width=10).pack(anchor=tk.W, pady=(5, 10))

        # Padding
        ttk.Label(frame, text="Padding (Å):").pack(anchor=tk.W)
        self.padding_var = tk.DoubleVar(value=5.0)
        ttk.Entry(frame, textvariable=self.padding_var, width=10).pack(anchor=tk.W, pady=(5, 0))

    def _create_action_buttons(self, parent):
        """Create action buttons"""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(frame, text="Generate Grid Box", command=self._generate_grid_box).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(frame, text="Save Configuration", command=self._save_configuration).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(frame, text="Load Configuration", command=self._load_configuration).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(frame, text="Apply & Close", command=self._apply_and_close).pack(fill=tk.X)

    def _create_3d_plot(self, parent):
        """Create 3D protein visualization"""
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add mouse interaction
        self.canvas.mpl_connect('button_press_event', self._on_plot_click)

    def _populate_residue_list(self):
        """Populate the residue listbox"""
        self.residue_data = []

        for residue in self.residues:
            try:
                chain_id = residue.get_parent().id
                res_id = residue.id[1]
                res_name = residue.get_resname()
                display_text = f"{chain_id}:{res_id} {res_name}"

                self.residue_data.append({
                    'display': display_text,
                    'residue': residue,
                    'chain': chain_id,
                    'number': res_id,
                    'name': res_name
                })
                self.residue_listbox.insert(tk.END, display_text)
            except:
                continue

    def _update_protein_view(self):
        """Update the 3D protein visualization"""
        self.ax.clear()

        # Plot protein backbone
        coords = np.array([atom.get_coord() for atom in self.atoms if atom.get_name() == 'CA'])
        if len(coords) > 0:
            self.ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                           c='lightblue', alpha=0.6, s=20, label='Protein')

        # Plot selected residues
        if self.selected_residues:
            selected_coords = []
            for residue in self.selected_residues:
                try:
                    ca_atom = residue['CA']
                    selected_coords.append(ca_atom.get_coord())
                except:
                    continue

            if selected_coords:
                selected_coords = np.array(selected_coords)
                self.ax.scatter(selected_coords[:, 0], selected_coords[:, 1], selected_coords[:, 2],
                               c='red', s=100, label='Selected Residues')

        # Plot current grid box
        if self.current_center and self.current_dimensions:
            self._plot_grid_box()

        self.ax.set_xlabel('X (Å)')
        self.ax.set_ylabel('Y (Å)')
        self.ax.set_zlabel('Z (Å)')

        # Set equal aspect ratio and proper axis limits
        all_coords = coords
        if self.selected_residues:
            selected_coords = []
            for residue in self.selected_residues:
                try:
                    ca_atom = residue['CA']
                    selected_coords.append(ca_atom.get_coord())
                except:
                    continue
            if selected_coords:
                all_coords = np.vstack([coords, selected_coords])

        # Add grid box corners to coordinate calculation
        if self.current_center and self.current_dimensions:
            box_corners = np.array(self._get_box_corners())
            all_coords = np.vstack([all_coords, box_corners])

        # Set axis limits with some padding
        padding = 10.0
        self.ax.set_xlim(np.min(all_coords[:, 0]) - padding, np.max(all_coords[:, 0]) + padding)
        self.ax.set_ylim(np.min(all_coords[:, 1]) - padding, np.max(all_coords[:, 1]) + padding)
        self.ax.set_zlim(np.min(all_coords[:, 2]) - padding, np.max(all_coords[:, 2]) + padding)

        self.ax.legend()
        self.canvas.draw()

    def _get_box_corners(self):
        """Get grid box corners for coordinate calculation"""
        center = self.current_center
        dims = self.current_dimensions

        corners = []
        for x_sign in [-1, 1]:
            for y_sign in [-1, 1]:
                for z_sign in [-1, 1]:
                    corner = [
                        center[0] + x_sign * dims[0]/2,
                        center[1] + y_sign * dims[1]/2,
                        center[2] + z_sign * dims[2]/2
                    ]
                    corners.append(corner)
        return corners

    def _plot_grid_box(self):
        """Plot the current grid box"""
        center = self.current_center
        dims = self.current_dimensions

        # Create box corners
        x_range = [center[0] - dims[0]/2, center[0] + dims[0]/2]
        y_range = [center[1] - dims[1]/2, center[1] + dims[1]/2]
        z_range = [center[2] - dims[2]/2, center[2] + dims[2]/2]

        # Plot box edges
        for x in x_range:
            for y in y_range:
                self.ax.plot([x, x], [y, y], z_range, 'r-', alpha=0.7)
        for x in x_range:
            for z in z_range:
                self.ax.plot([x, x], y_range, [z, z], 'r-', alpha=0.7)
        for y in y_range:
            for z in z_range:
                self.ax.plot(x_range, [y, y], [z, z], 'r-', alpha=0.7)

    def _on_mode_change(self):
        """Handle mode change"""
        mode = self.mode_var.get()

        # Show/hide relevant frames
        if mode == "residue":
            self.residue_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
            self.manual_frame.pack_forget()
            self.ai_frame.pack_forget()
        elif mode == "manual":
            self.residue_frame.pack_forget()
            self.manual_frame.pack(fill=tk.X, pady=(0, 10))
            self.ai_frame.pack_forget()
        elif mode == "ai":
            self.residue_frame.pack_forget()
            self.manual_frame.pack_forget()
            self.ai_frame.pack(fill=tk.X, pady=(0, 10))

    def _on_search_change(self, *args):
        """Handle search text change"""
        search_text = self.search_var.get().lower()
        self.residue_listbox.delete(0, tk.END)

        for data in self.residue_data:
            if search_text in data['display'].lower():
                self.residue_listbox.insert(tk.END, data['display'])

    def _add_selected_residues(self):
        """Add selected residues to the selection"""
        selection = self.residue_listbox.curselection()
        for idx in selection:
            residue_text = self.residue_listbox.get(idx)
            residue_data = next((d for d in self.residue_data if d['display'] == residue_text), None)

            if residue_data and residue_data['residue'] not in self.selected_residues:
                self.selected_residues.append(residue_data['residue'])

        self._update_selected_residues_display()
        self._calculate_residue_center()
        self._update_protein_view()

    def _clear_selected_residues(self):
        """Clear selected residues"""
        self.selected_residues = []
        self._update_selected_residues_display()
        self._update_protein_view()

    def _update_selected_residues_display(self):
        """Update the selected residues text display"""
        self.selected_text.config(state=tk.NORMAL)
        self.selected_text.delete(1.0, tk.END)

        for residue in self.selected_residues:
            try:
                chain_id = residue.get_parent().id
                res_id = residue.id[1]
                res_name = residue.get_resname()
                self.selected_text.insert(tk.END, f"{chain_id}:{res_id} {res_name}\n")
            except:
                continue

        self.selected_text.config(state=tk.DISABLED)

    def _calculate_residue_center(self):
        """Calculate center based on selected residues"""
        if not self.selected_residues:
            return

        coords = []
        for residue in self.selected_residues:
            for atom in residue.get_atoms():
                coords.append(atom.get_coord())

        if coords:
            center = np.mean(coords, axis=0)
            self.current_center = center.tolist()

            # Update manual controls
            for i, var in enumerate(self.center_vars):
                var.set(self.current_center[i])

            # Calculate dimensions with padding
            coords_array = np.array(coords)
            min_coords = np.min(coords_array, axis=0)
            max_coords = np.max(coords_array, axis=0)
            dimensions = max_coords - min_coords + 2 * self.padding_var.get()

            self.current_dimensions = dimensions.tolist()
            for i, var in enumerate(self.dim_vars):
                var.set(self.current_dimensions[i])

    def _on_manual_change(self, *args):
        """Handle manual coordinate changes"""
        try:
            self.current_center = [var.get() for var in self.center_vars]
            self.current_dimensions = [var.get() for var in self.dim_vars]
            self._update_protein_view()
        except:
            pass

    def _on_properties_change(self, *args):
        """Handle grid properties changes"""
        try:
            self.current_spacing = self.spacing_var.get()
        except:
            pass

    def _find_ai_suggestions(self):
        """Find AI-suggested binding sites"""
        self.ai_listbox.delete(0, tk.END)

        try:
            # Import and use the actual AI binding site predictor
            from ..ai.binding_site_prediction import AIBindingSitePredictionGenerator

            ai_generator = AIBindingSitePredictionGenerator()
            grid_boxes = ai_generator.generate(
                self.receptor_file,
                max_sites=3,
                confidence_threshold=0.3,  # Lower threshold for GUI
                spacing=self.current_spacing
            )

            self.ai_suggestions = []

            if grid_boxes:
                for i, box in enumerate(grid_boxes):
                    confidence = box.confidence_score
                    center = box.center
                    suggestion_text = f"Site {i+1}: Confidence {confidence:.2f} (x={center[0]:.1f}, y={center[1]:.1f}, z={center[2]:.1f})"
                    self.ai_listbox.insert(tk.END, suggestion_text)

                    # Store the actual grid box data
                    self.ai_suggestions.append({
                        'center': list(center),
                        'dimensions': list(box.dimensions),
                        'confidence': confidence
                    })
            else:
                self.ai_listbox.insert(tk.END, "No binding sites found")

        except Exception as e:
            self.logger.error(f"AI suggestion failed: {e}")
            self.ai_listbox.insert(tk.END, f"AI analysis failed: {str(e)}")
            self.ai_suggestions = []

    def _on_ai_selection(self, event):
        """Handle AI suggestion selection"""
        selection = self.ai_listbox.curselection()
        if selection and hasattr(self, 'ai_suggestions') and self.ai_suggestions:
            try:
                index = selection[0]
                if index < len(self.ai_suggestions):
                    suggestion = self.ai_suggestions[index]

                    # Set grid box based on AI suggestion
                    self.current_center = suggestion['center']
                    self.current_dimensions = suggestion['dimensions']

                    # Update GUI controls
                    for i, var in enumerate(self.center_vars):
                        var.set(self.current_center[i])
                    for i, var in enumerate(self.dim_vars):
                        var.set(self.current_dimensions[i])

                    self._update_protein_view()

                    self.logger.info(f"Selected AI site with confidence {suggestion['confidence']:.3f}")
            except Exception as e:
                self.logger.error(f"Error selecting AI suggestion: {e}")
                messagebox.showerror("Error", f"Failed to apply AI suggestion: {e}")

    def _on_plot_click(self, event):
        """Handle plot click events"""
        if event.inaxes == self.ax and event.button == 1:  # Left click
            # Set center to clicked position (if in manual mode)
            if self.mode_var.get() == "manual" and event.xdata is not None:
                self.current_center = [event.xdata, event.ydata, 0.0]  # Z needs special handling
                for i, var in enumerate(self.center_vars[:2]):
                    var.set(self.current_center[i])
                self._update_protein_view()

    def _generate_grid_box(self):
        """Generate and display grid box"""
        if not all(self.current_center) or not all(self.current_dimensions):
            messagebox.showerror("Error", "Please specify valid center and dimensions")
            return

        grid_box = GridBox(
            center=tuple(self.current_center),
            dimensions=tuple(self.current_dimensions),
            spacing=self.current_spacing,
            receptor_file=self.receptor_file,
            method_used="interactive_gui"
        )

        self.grid_boxes = [grid_box]
        messagebox.showinfo("Success", f"Grid box generated:\nCenter: {grid_box.center}\nDimensions: {grid_box.dimensions}")

    def _save_configuration(self):
        """Save current configuration"""
        if not self.grid_boxes:
            messagebox.showerror("Error", "No grid box to save")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )

        if filename:
            self.grid_boxes[0].save(filename)
            messagebox.showinfo("Success", f"Configuration saved to {filename}")

    def _load_configuration(self):
        """Load configuration from file"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")]
        )

        if filename:
            try:
                grid_box = GridBox.load(filename)
                self.grid_boxes = [grid_box]
                self.current_center = list(grid_box.center)
                self.current_dimensions = list(grid_box.dimensions)
                self.current_spacing = grid_box.spacing

                # Update GUI
                for i, var in enumerate(self.center_vars):
                    var.set(self.current_center[i])
                for i, var in enumerate(self.dim_vars):
                    var.set(self.current_dimensions[i])
                self.spacing_var.set(self.current_spacing)

                self._update_protein_view()
                messagebox.showinfo("Success", f"Configuration loaded from {filename}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration: {e}")

    def _apply_and_close(self):
        """Apply current grid box and close"""
        if not self.grid_boxes:
            self._generate_grid_box()

        if self.grid_boxes and self.callback:
            self.callback(self.grid_boxes[0])

        self.root.destroy()

    def run(self) -> Optional[GridBox]:
        """Run the GUI and return selected grid box"""
        self.root.mainloop()
        return self.grid_boxes[0] if self.grid_boxes else None


class InteractiveGridBoxGenerator(GridBoxGenerator):
    """Generator that uses interactive GUI for grid box selection"""

    def generate(self, receptor_file: str, **kwargs) -> List[GridBox]:
        """Generate grid box using interactive GUI"""

        def gui_callback(grid_box):
            self.result = grid_box

        self.result = None

        # Launch GUI
        selector = InteractiveGridBoxSelector(receptor_file, gui_callback)
        selected_box = selector.run()

        if selected_box:
            return [selected_box]
        else:
            return []