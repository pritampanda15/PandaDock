"""
Utility functions for PandaDock.
This module provides logging, file management, and other utility functions.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
import numpy as np

def setup_logging(output_dir=None, log_name="pandadock", log_level=logging.INFO):
    """
    Configure logging system for PandaDock.
    
    Parameters:
    -----------
    output_dir : str or Path, optional
        Output directory where log files will be saved
    log_name : str, optional
        Name for the logger and log file (default: 'pandadock')
    log_level : int, optional
        Logging level (default: logging.INFO)
    
    Returns:
    --------
    logging.Logger
        Configured logger object
    """
    # Get or create logger
    logger = logging.getLogger(log_name)
    
    # Only configure if not already configured
    if not logger.handlers:
        # Set level
        logger.setLevel(log_level)
        
        # Create formatters
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_formatter = logging.Formatter("%(levelname)s - %(message)s")
        
        # Add console handler
        console = logging.StreamHandler()
        console.setFormatter(console_formatter)
        logger.addHandler(console)
        
        # Add file handler if output directory is provided
        if output_dir:
            logs_dir = Path(output_dir) / "logs"
            os.makedirs(logs_dir, exist_ok=True)
            
            file_handler = logging.FileHandler(logs_dir / f"{log_name}.log")
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
            # Log file location
            logger.info(f"Log file created at: {logs_dir / f'{log_name}.log'}")
    
    return logger
import numpy as np

def is_within_grid(pose, grid_center, grid_radius):
    """
    Check if the centroid of the pose lies within the spherical grid boundary.
    
    Parameters:
    -----------
    pose : Ligand
        Ligand pose with atomic coordinates in .xyz
    grid_center : np.ndarray
        3D center of the search grid
    grid_radius : float
        Radius of the grid sphere

    Returns:
    --------
    bool
        True if pose is inside grid, False otherwise
    """
    centroid = np.mean(pose.xyz, axis=0)
    distance = np.linalg.norm(centroid - grid_center)
    return distance <= grid_radius


def generate_spherical_grid(center, radius, spacing=1.0):
        """
        Generate grid points within a sphere centered at `center` with a given `radius`.

        Parameters:
        -----------
        center : array-like
            Center of the sphere (3D coordinates).
        radius : float
            Radius of the sphere.
        spacing : float
            Approximate spacing between grid points.

        Returns:
        --------
        np.ndarray
            Array of 3D points within the sphere.
        """
        center = np.array(center)
        r = int(np.ceil(radius / spacing))
        grid = []

        for x in range(-r, r + 1):
            for y in range(-r, r + 1):
                for z in range(-r, r + 1):
                    point = np.array([x, y, z]) * spacing + center
                    if np.linalg.norm(point - center) <= radius:
                        grid.append(point)

        return np.array(grid)

def detect_steric_clash(protein_atoms, ligand_atoms, threshold=1.6):
    """
    Check if any ligand atom is too close to a protein atom (steric clash).
    
    Parameters:
    -----------
    protein_atoms : list
    ligand_atoms : list
    threshold : float
        Minimum allowed distance (Å) between non-bonded atoms
    
    Returns:
    --------
    bool
        True if clash detected, False otherwise
    """
    for p in protein_atoms:
        if 'coords' not in p:
            continue
        for l in ligand_atoms:
            if 'coords' not in l:
                continue
            distance = np.linalg.norm(p['coords'] - l['coords'])
            if distance < threshold:
                return True
    return False

def create_initial_files(output_dir, args):
    """
    Create initial files to confirm program is running.
    
    Parameters:
    -----------
    output_dir : str or Path
        Output directory for docking results
    args : argparse.Namespace
        Command-line arguments
    """
    # Create directory structure
    output_dir = Path(output_dir)
    #for subdir in ["logs", "intermediate", "poses"]:
        #os.makedirs(output_dir / subdir, exist_ok=True)
    
    # Get logger
    logger = setup_logging(output_dir)
    logger.info(f"Creating initial files in {output_dir}")
    
    # Create a status file
    status = {
        "start_time": datetime.now().isoformat(),
        "protein": str(args.protein),
        "ligand": str(args.ligand),
        "algorithm": args.algorithm,
        "status": "running",
        "progress": 0.0,
        "total_iterations": getattr(args, 'iterations', 1000),
        "current_iteration": 0,
        "top_score": None
    }
    
    status_path = output_dir / "status.json"
    with open(status_path, 'w') as f:
        json.dump(status, f, indent=2)
    logger.info(f"Status file created at {status_path}")
    
    # Create a README file
    readme_path = output_dir / "README.txt"
    with open(readme_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("PandaDock Molecular Docking\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Input Files:\n")
        f.write(f"  Protein: {args.protein}\n")
        f.write(f"  Ligand: {args.ligand}\n\n")
        f.write("Parameters:\n")
        f.write(f"  Algorithm: {args.algorithm}\n")
        
        # Add common parameters
        if hasattr(args, 'iterations'):
            f.write(f"  Iterations: {args.iterations}\n")
        if hasattr(args, 'population_size'):
            f.write(f"  Population Size: {args.population_size}\n")
        if hasattr(args, 'physics_based') and args.physics_based:
            f.write(f"  Scoring: Physics-based\n")
        elif hasattr(args, 'enhanced_scoring') and args.enhanced_scoring:
            f.write(f"  Scoring: Enhanced\n")
        else:
            f.write(f"  Scoring: Standard\n")
            
        # Add hardware info
        f.write("\nHardware:\n")
        if hasattr(args, 'use_gpu') and args.use_gpu:
            f.write(f"  Using GPU acceleration\n")
        if hasattr(args, 'cpu_workers'):
            f.write(f"  CPU Workers: {args.cpu_workers}\n")
        
    logger.info(f"README file created at {readme_path}")

def save_intermediate_result(pose, score, iteration, output_dir, total_iterations=None):
    """
    Save an intermediate result during docking.
    
    Parameters:
    -----------
    pose : Ligand
        Ligand pose to save
    score : float
        Docking score
    iteration : int
        Current iteration number
    output_dir : str or Path
        Output directory for docking results
    total_iterations : int, optional
        Total number of iterations (for progress calculation)
    """
    output_dir = Path(output_dir)
    intermediate_dir = output_dir / "intermediate"
    os.makedirs(intermediate_dir, exist_ok=True)
    
    # Save only every 10th pose or best poses to avoid too many files
    is_milestone = (iteration % 10 == 0)
    
    # Get logger
    logger = logging.getLogger("pandadock")
    
    # Update status file
    status_path = output_dir / "status.json"
    try:
        with open(status_path, 'r') as f:
            status = json.load(f)
        
        # Update basic info
        status["current_iteration"] = iteration
        status["last_update"] = datetime.now().isoformat()
        
        # Calculate progress
        if total_iterations is None:
            total_iterations = status.get("total_iterations", 100)
        status["progress"] = min(1.0, iteration / total_iterations)
        
        # Track best score
        if status["top_score"] is None or score < status["top_score"]:
            status["top_score"] = score
            is_milestone = True  # Always save best poses
            
        # Update status file
        with open(status_path, 'w') as f:
            json.dump(status, f, indent=2)
            
    except Exception as e:
        logger.warning(f"Could not update status file: {e}")
    
    # Save PDB file for milestone or best poses
    if is_milestone:
        pdb_path = intermediate_dir / f"pose_iter_{iteration}_score_{score:.2f}.pdb"
        try:
            with open(pdb_path, 'w') as f:
                f.write(f"REMARK   1 Iteration: {iteration}, Score: {score:.4f}\n")
                
                # Write atoms
                for j, atom in enumerate(pose.atoms):
                    coords = atom['coords']
                    symbol = atom.get('symbol', 'C')
                    
                    # PDB ATOM format
                    f.write(f"HETATM{j+1:5d} {symbol:<4}{'':<1}{'LIG':<3} {'A':1}{1:4d}    "
                           f"{coords[0]:8.3f}{coords[1]:8.3f}{coords[2]:8.3f}"
                           f"{1.0:6.2f}{0.0:6.2f}          {symbol:>2}\n")
                
            logger.debug(f"Saved intermediate pose at iteration {iteration} to {pdb_path}")
        except Exception as e:
            logger.warning(f"Could not save intermediate pose: {e}")

def save_complex_to_pdb(protein, ligand, output_path):
    """
    Save the full protein-ligand complex as a single PDB file.
    
    Parameters:
    -----------
    protein : Protein
        Protein object
    ligand : Ligand
        Ligand object
    output_path : str or Path
        File path to save the complex
    """
    with open(output_path, 'w') as f:
        # Write protein atoms
        for i, atom in enumerate(protein.atoms):
            coords = atom['coords']
            name = atom.get('name', 'X')
            resname = atom.get('residue_name', 'UNK')
            chain = atom.get('chain_id', 'A')
            resid = atom.get('residue_id', 1)
            f.write(f"ATOM  {i+1:5d} {name:<4} {resname:<3} {chain}{resid:4d}    "
                    f"{coords[0]:8.3f}{coords[1]:8.3f}{coords[2]:8.3f}  1.00  0.00\n")
        
        # Write ligand atoms
        for j, atom in enumerate(ligand.atoms):
            coords = atom['coords']
            symbol = atom.get('symbol', 'C')
            f.write(f"HETATM{j+1:5d} {symbol:<4} LIG A{1:4d}    "
                    f"{coords[0]:8.3f}{coords[1]:8.3f}{coords[2]:8.3f}  1.00  0.00          {symbol:>2}\n")
        
        f.write("END\n")
        
    print(f"Saved complex to {output_path}")
def update_status(output_dir, **kwargs):
    """
    Update the status.json file with new information.
    
    Parameters:
    -----------
    output_dir : str or Path
        Output directory for docking results
    **kwargs : dict
        Key-value pairs to update in the status file
    """
    output_dir = Path(output_dir)
    status_path = output_dir / "status.json"
    
    # Get logger
    logger = logging.getLogger("pandadock")
    
    try:
        # Read current status
        if status_path.exists():
            with open(status_path, 'r') as f:
                status = json.load(f)
        else:
            status = {"start_time": datetime.now().isoformat()}
        
        # Update with new values
        status.update(kwargs)
        status["last_update"] = datetime.now().isoformat()
        
        # Write updated status
        with open(status_path, 'w') as f:
            json.dump(status, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not update status file: {e}")

def extract_base_filename(file_path):
    """
    Extract base filename without extension.
    
    Parameters:
    -----------
    file_path : str
        Path to file
    
    Returns:
    --------
    str
        Base filename without extension
    """
    return Path(file_path).stem

def create_descriptive_output_dir(args):
    """
    Create a more descriptive output directory name based on inputs.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    
    Returns:
    --------
    str
        Descriptive output directory name
    """
    # Extract base filenames
    protein_base = extract_base_filename(args.protein)
    ligand_base = extract_base_filename(args.ligand)
    
    # Get algorithm name
    algo_name = args.algorithm
    
    # Create readable timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    # Build output directory name
    output_dir = f"{args.output}_{protein_base}_{ligand_base}_{algo_name}_{timestamp}"
    
    return output_dir

def save_docking_results(results, output_dir='docking_results', flexible_residues=None):
    """
    Save docking results to output directory.
    
    Parameters:
    -----------
    results : list
        List of (pose, score) tuples
    output_dir : str
        Output directory
    flexible_residues : list, optional
        List of flexible residue objects (for flexible docking)
    """
    # Check if results is empty
    if not results:
        print("Warning: No docking results to save.")
        return
    # Create output directory
    out_path = Path(output_dir)
    os.makedirs(out_path, exist_ok=True)
    
    # Save top poses
    for i, (pose, score) in enumerate(results[:10]):  # Save top 10 poses
        # Generate PDB file for the ligand pose
        pdb_path = out_path / f"pose_{i+1}_score_{score:.1f}.pdb"
        with open(pdb_path, 'w') as f:
            f.write(f"REMARK   1 Docking score: {score}\n")
            
            for j, atom in enumerate(pose.atoms):
                coords = atom['coords']
                symbol = atom.get('symbol', 'C')
                
                # PDB ATOM format
                f.write(f"HETATM{j+1:5d} {symbol:<4}{'':<1}{'LIG':<3} {'A':1}{1:4d}    "
                        f"{coords[0]:8.3f}{coords[1]:8.3f}{coords[2]:8.3f}"
                        f"{1.0:6.2f}{0.0:6.2f}          {symbol:>2}\n")
        
        # If flexible residues are present, save a complex file with ligand and flexible residues
        if flexible_residues:
            complex_path = out_path / f"complex_{i+1}_score_{score:.1f}.pdb"
            with open(complex_path, 'w') as f:
                f.write(f"REMARK   1 Docking score: {score}\n")
                f.write(f"REMARK   2 Complex with flexible residues\n")
                
                # Write flexible residue atoms first
                atom_index = 1
                for res_index, residue in enumerate(flexible_residues):
                    for atom in residue.atoms:
                        coords = atom['coords']
                        name = atom.get('name', '').ljust(4)
                        res_name = atom.get('residue_name', 'UNK')
                        chain_id = atom.get('chain_id', 'A')
                        res_id = atom.get('residue_id', res_index+1)
                        element = atom.get('element', atom.get('name', 'C'))[0]
                        
                        f.write(f"ATOM  {atom_index:5d} {name} {res_name:3s} {chain_id:1s}{res_id:4d}    "
                                f"{coords[0]:8.3f}{coords[1]:8.3f}{coords[2]:8.3f}"
                                f"{1.0:6.2f}{0.0:6.2f}          {element:>2}\n")
                        atom_index += 1
                
                # Write TER record to separate protein from ligand
                f.write("TER\n")
                
                # Write ligand atoms
                for j, atom in enumerate(pose.atoms):
                    coords = atom['coords']
                    symbol = atom.get('symbol', 'C')
                    
                    f.write(f"HETATM{atom_index:5d} {symbol:<4}{'':<1}{'LIG':<3} {'A':1}{1:4d}    "
                            f"{coords[0]:8.3f}{coords[1]:8.3f}{coords[2]:8.3f}"
                            f"{1.0:6.2f}{0.0:6.2f}          {symbol:>2}\n")
                    atom_index += 1
                
                # End of PDB file
                f.write("END\n")
    
    # Create a score plot with non-GUI backend
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-GUI backend
        import matplotlib.pyplot as plt
        
        scores = [score for _, score in results]
        plt.figure(figsize=(12, 8), facecolor='#f8f9fa')
        ax = plt.subplot(111)

        # Apply grid in background with light color
        ax.grid(True, linestyle='--', alpha=0.7, color='#cccccc')
        ax.set_axisbelow(True)  # Place grid behind the data

        # Plot data with better styling
        plt.plot(range(1, len(scores) + 1), scores, 
                marker='o', markersize=8, color='#2077B4', 
                linewidth=2.5, linestyle='-', alpha=0.8)

        # Fill area under curve
        plt.fill_between(range(1, len(scores) + 1), scores, 
                        alpha=0.3, color='#2077B4')

        # Highlight best score point
        plt.scatter(1, scores[0], s=120, color='#e63946', zorder=5, 
                    edgecolor='white', linewidth=1.5, 
                    label=f'Best Score: {scores[0]:.2f}')

        # Improve axis labels and title
        plt.xlabel('Pose Rank', fontsize=14, fontweight='bold', labelpad=10)
        plt.ylabel('Docking Score', fontsize=14, fontweight='bold', labelpad=10)
        plt.title('PandaDock Results - Score Distribution', 
                fontsize=16, fontweight='bold', pad=20)

        # Add score annotation for the best score
        plt.annotate(f'{scores[0]:.2f}', xy=(1, scores[0]), 
                    xytext=(10, -20), textcoords='offset points',
                    fontsize=11, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2',
                                    color='#555555'))

        # Improve tick parameters
        plt.tick_params(axis='both', which='major', labelsize=11, width=1.5)

        # Set plot limits with some padding
        y_min = min(scores) - (max(scores) - min(scores)) * 0.1
        y_max = max(scores) + (max(scores) - min(scores)) * 0.1
        plt.ylim(y_min, y_max)

        # Add legend
        plt.legend(loc='best', frameon=True, framealpha=0.95, fontsize=12)

        # Add a subtle box around the plot
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('#555555')

        # Add score statistics as text
        stats_text = (f"Total Poses: {len(scores)}\n"
                    f"Best Score: {min(scores):.2f}\n"
                    f"Worst Score: {max(scores):.2f}\n"
                    f"Average: {sum(scores)/len(scores):.2f}")
        plt.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                        alpha=0.8, edgecolor='#cccccc'))

        plt.tight_layout()
        plot_path = out_path / "score_plot.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved to {plot_path}")
    except Exception as e:
        print(f"Warning: Could not create score plot: {e}")
        print("Continuing without plot generation.")
    
    print(f"Saved {len(results)} docking results to {output_dir}")
    print(f"Best docking score: {results[0][1]}")
    if flexible_residues:
        print(f"Complex PDB files with flexible residues also saved")


def calculate_rmsd(coords1, coords2):
    """
    Calculate RMSD between two sets of coordinates.
    
    Parameters:
    -----------
    coords1 : array-like
        First set of coordinates (N x 3)
    coords2 : array-like
        Second set of coordinates (N x 3)
    
    Returns:
    --------
    float
        RMSD value in same units as input coordinates
    """
    coords1 = np.array(coords1)
    coords2 = np.array(coords2)
    
    if coords1.shape != coords2.shape:
        raise ValueError(f"Coordinate mismatch: set 1 has shape {coords1.shape}, but set 2 has shape {coords2.shape}")
    
    # For 3D molecular coordinates (Nx3 array)
    # Sum squared differences for each atom's x,y,z components
    squared_diff = np.sum((coords1 - coords2) ** 2, axis=1)
    
    # Calculate mean of the squared differences and take square root
    rmsd = np.sqrt(np.mean(squared_diff))
    
    return rmsd
