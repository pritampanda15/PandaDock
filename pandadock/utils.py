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
import subprocess
import platform
import pkg_resources
from datetime import datetime

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


def generate_spherical_grid(center, radius, spacing=0.375):
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

def create_initial_files(output_dir, args, status="running"):
    """
    Create initial files to confirm PandaDock run and document configuration.

    Parameters:
    -----------
    output_dir : str or Path
        Output directory for docking results
    args : argparse.Namespace
        Command-line arguments
    """
    from datetime import datetime
    import json
    from .utils import setup_logging

    # Create directory structure
    output_dir = Path(output_dir)
    
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
    
    # Create a detailed README file
    readme_path = output_dir / "logs.txt"
    # Gather metadata
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    python_version = platform.python_version()
    os_info = platform.system() + " " + platform.release()
    
    try:
        pandadock_version = pkg_resources.get_distribution("pandadock").version
    except Exception:
        pandadock_version = "dev"  # if not installed via pip
    
    # Try getting git commit ID if running from a repo
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=os.getcwd()).decode().strip()
    except Exception:
        git_commit = "N/A"

    with open(readme_path, 'w') as f:
        f.write(r"""
════════════════════════════════════════════════════════════════════════════════
   ██████╗  █████╗ ███╗   ██╗██████╗  █████╗ ██████╗  ██████╗  ██████╗██╗  ██╗
    ██╔══██╗██╔══██╗████╗  ██║██╔══██╗██╔══██╗██╔══██╗██╔═══██╗██╔════╝██║ ██╔╝
    ██████╔╝███████║██╔██╗ ██║██║  ██║███████║██║  ██║██║   ██║██║     █████╔╝ 
    ██╔═══╝ ██╔══██║██║╚██╗██║██║  ██║██╔══██║██║  ██║██║   ██║██║     ██╔═██╗ 
    ██║     ██║  ██║██║ ╚████║██████╔╝██║  ██║██████╔╝╚██████╔╝╚██████╗██║  ██╗
    ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝ ╚═╝  ╚═╝╚═════╝  ╚═════╝  ╚═════╝╚═╝  ╚═╝
                                                                                                                                                                                                                                  
               PandaDock - Python Molecular Docking Tool                             
               https://github.com/pritampanda15/PandaDock                   
════════════════════════════════════════════════════════════════════════════════
    """)
        
        f.write(f"Run Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"📅 Timestamp: {timestamp}\n")
        f.write(f"🐍 Python Version: {python_version} on {os_info}\n")
        f.write(f"🐼 PandaDock Version: {pandadock_version}\n")
        f.write(f"🔖 Git Commit: {git_commit}\n\n")
        
        if hasattr(args, 'full_command'):
            f.write("📜 Command-line used:\n")
            f.write(f"    {args.full_command}\n\n")
        
        f.write(f"🎯 Docking Status: {str(status).upper()}\n")
        f.write(f"⏱️ Elapsed Time: [pending at start]\n\n")
        
        f.write("INPUT FILES\n")
        f.write("-----------\n")
        f.write(f"Protein File: {args.protein}\n")
        f.write(f"Ligand File: {args.ligand}\n\n")
        
        f.write("DOCKING PARAMETERS\n")
        f.write("------------------\n")
        f.write(f"Algorithm: {args.algorithm}\n")
        f.write(f"Iterations / Generations: {getattr(args, 'iterations', 'N/A')}\n")
        if args.algorithm == 'genetic':
            if hasattr(args, 'population_size'):
                f.write(f"Population Size: {args.population_size}\n")
            if hasattr(args, 'mutation_rate'):
                f.write(f"Mutation Rate: {args.mutation_rate}\n")
            if hasattr(args, 'crossover_rate'):
                f.write(f"Crossover Rate: {args.crossover_rate}\n")
            if hasattr(args, 'selection_method'):
                f.write(f"Selection Method: {args.selection_method}\n")

        elif args.algorithm == 'monte-carlo':
            if hasattr(args, 'mc_steps'):
                f.write(f"Monte Carlo Steps: {args.mc_steps}\n")
            if hasattr(args, 'temperature'):
                f.write(f"Monte Carlo Temperature: {args.temperature} K\n")
            if hasattr(args, 'cooling_rate'):
                f.write(f"Cooling Rate: {args.cooling_rate}\n")   
        
        elif args.algorithm == 'pandadock':
            if hasattr(args, 'pandadock_steps'):
                f.write(f"PandaDock Steps: {args.pandadock_steps}\n")
            if hasattr(args, 'pandadock_temperature'):
                f.write(f"PandaDock Temperature: {args.pandadock_temperature} K\n")
            if hasattr(args, 'pandadock_cooling_rate'):
                f.write(f"PandaDock Cooling Rate: {args.pandadock_cooling_rate}\n")
            if hasattr(args, 'pandadock_mutation_rate'):
                f.write(f"PandaDock Mutation Rate: {args.pandadock_mutation_rate}\n")
            if hasattr(args, 'pandadock_crossover_rate'):
                f.write(f"PandaDock Crossover Rate: {args.pandadock_crossover_rate}\n")
            if hasattr(args, 'pandadock_selection_method'):
                f.write(f"PandaDock Selection Method: {args.pandadock_selection_method}\n")
        elif args.algorithm == 'random':
            if hasattr(args, 'random_steps'):
                f.write(f"Random Steps: {args.random_steps}\n")
            if hasattr(args, 'random_temperature'):
                f.write(f"Random Temperature: {args.random_temperature} K\n")
            if hasattr(args, 'random_cooling_rate'):
                f.write(f"Random Cooling Rate: {args.random_cooling_rate}\n")
        elif args.algorithm == 'default':
            if hasattr(args, 'default_steps'):
                f.write(f"Default Steps: {args.default_steps}\n")
            if hasattr(args, 'default_temperature'):
                f.write(f"Default Temperature: {args.default_temperature} K\n")
            if hasattr(args, 'default_cooling_rate'):
                f.write(f"Default Cooling Rate: {args.default_cooling_rate}\n")
        
        if hasattr(args, 'flexible_residues'):
            f.write(f"Flexible Residues: {args.flexible_residues}\n")
        if hasattr(args, 'fixed_residues'):
            f.write(f"Fixed Residues: {args.fixed_residues}\n")
        if hasattr(args, 'flexible_ligand'):
            f.write(f"Flexible Ligand: {args.flexible_ligand}\n")
        if hasattr(args, 'fixed_ligand'):
            f.write(f"Fixed Ligand: {args.fixed_ligand}\n")

            #################################################
        f.write("\nSCORING PARAMETERS\n")
        f.write("-------------------\n")
        if hasattr(args, 'physics_based') and args.physics_based:
            f.write("Scoring Function: Physics-based\n")
        elif hasattr(args, 'enhanced_scoring') and args.enhanced_scoring:
            f.write("Scoring Function: Enhanced\n")
        else:
            f.write("Scoring Function: Standard\n")
        
        if hasattr(args, 'grid_center'):
            f.write(f"Grid Center: {args.grid_center}\n")
        if hasattr(args, 'grid_spacing'):
            f.write(f"Grid Spacing: {args.grid_spacing} Å\n")
        if hasattr(args, 'grid_size'):
            f.write(f"Grid Size: {args.grid_size} Å³\n")
        if hasattr(args, 'spherical_sampling') and args.spherical_sampling:
            f.write(f"Spherical Sampling Enabled (Radius: {getattr(args, 'sampling_radius', 'default')})\n")
        
        f.write("\nHARDWARE SETTINGS\n")
        f.write("-----------------\n")
        if hasattr(args, 'use_gpu') and args.use_gpu:
            f.write("GPU Acceleration: Enabled\n")
            if hasattr(args, 'gpu_id'):
                f.write(f"GPU ID: {args.gpu_id}\n")
            if hasattr(args, 'gpu_precision'):
                f.write(f"GPU Precision: {args.gpu_precision}\n")
        else:
            f.write("GPU Acceleration: Disabled (CPU-only)\n")
        if hasattr(args, 'cpu_workers'):
            f.write(f"CPU Workers: {args.cpu_workers}\n")
        
        f.write("\nOUTPUT STRUCTURE\n")
        f.write("----------------\n")
        f.write("The following files and directories will be generated:\n")
        f.write("- docking_report.txt\n")
        f.write("- docking_report.html\n")
        f.write("- docking_results.csv\n")
        f.write("- energy_breakdown.csv\n")
        f.write("- status.json\n")
        f.write("- poses/ : Top ranked poses\n")
        f.write("- plots/ : Score distribution and energy component plots\n\n")
        f.write("\n\n")
        f.write("COMMAND USED TO RUN DOCKING\n")
        f.write("----------------------------\n")
        #f.write(f"{args.full_command}\n")
        f.write("=" * 60 + "\n")
        f.write("          PandaDock - Python Molecular Docking Tool\n")
        f.write("=" * 60 + "\n")
    
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
    if args.monte_carlo:
        algo_name = "monte-carlo"
    elif args.genetic_algorithm:
        algo_name = "genetic-algorithm"
    elif args.pandadock:
        algo_name = "pandadock"
    elif args.random:
        algo_name = "random"
    elif args.default:
        algo_name = "default-algorithm"
    elif args.enhanced_scoring:
        algo_name = "enhanced-scoring"
    elif args.physics_based:
        algo_name = "physics-based"
    elif args.standard_scoring:
        algo_name = "standard-scoring"
    elif args.docking:
        algo_name = "docking"
    else:
        algo_name = "default-algorithm"
    # Check if algorithm is valid
    if algo_name not in ["default-algorithm", "genetic-algorithm", "monte-carlo"]:
        raise ValueError(f"Invalid algorithm: {algo_name}")
    # Check if output directory is provided
    if args.output is None:
        raise ValueError("Output directory is not provided")
    # Check if output directory is valid
    if not os.path.exists(args.output):
        raise ValueError(f"Output directory does not exist: {args.output}")
    
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

def generate_valid_random_pose(protein, ligand, center, radius, max_attempts=20):
    """
    Generate a random valid pose inside the sphere with clash checking.
    Retries if clash is detected or outside sphere.
    """
    from .utils import detect_steric_clash
    import copy
    from scipy.spatial.transform import Rotation
    import numpy as np
    import random

    for attempt in range(max_attempts):
        pose = copy.deepcopy(ligand)

        # Sample random point within the sphere
        r = radius * random.betavariate(2, 5) ** (1/3)
        theta = random.uniform(0, 2 * np.pi)
        phi = random.uniform(0, np.pi)

        x = center[0] + r * np.sin(phi) * np.cos(theta)
        y = center[1] + r * np.sin(phi) * np.sin(theta)
        z = center[2] + r * np.cos(phi)

        centroid = np.mean(pose.xyz, axis=0)
        translation = np.array([x, y, z]) - centroid
        pose.translate(translation)

        # Random rotation
        centroid = np.mean(pose.xyz, axis=0)
        pose.translate(-centroid)
        pose.rotate(Rotation.random().as_matrix())
        pose.translate(centroid)

        # Check inside sphere
        distance = np.linalg.norm(np.mean(pose.xyz, axis=0) - center)
        if distance > radius:
            continue  # Retry

        # Check for steric clash
        if detect_steric_clash(protein.atoms, pose.atoms):
            continue  # Retry

        return pose  # ✅ Valid pose found

    return None  # ❌ Failed after retries
