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
import platform
import subprocess
import importlib.util
import random
import json

from pathlib import Path
import json

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

def generate_cartesian_grid(min_corner, max_corner, spacing=1.0):
    """
    Generate Cartesian grid points within a bounding box.
    """
    x_range = np.arange(min_corner[0], max_corner[0], spacing)
    y_range = np.arange(min_corner[1], max_corner[1], spacing)
    z_range = np.arange(min_corner[2], max_corner[2], spacing)

    grid = []
    for x in x_range:
        for y in y_range:
            for z in z_range:
                grid.append(np.array([x, y, z]))

    return grid

def local_optimize_pose(pose, protein, scoring_function, max_steps=20, step_size=0.5):
    """
    Perform basic greedy local optimization by perturbing pose.
    """
    import copy
    import numpy as np

    best_pose = copy.deepcopy(pose)
    best_score = scoring_function.score(protein, best_pose)

    for _ in range(max_steps):
        trial_pose = copy.deepcopy(best_pose)
        trial_pose.translate(np.random.normal(0, step_size, 3))
        trial_score = scoring_function.score(protein, trial_pose)

        if trial_score < best_score:
            best_pose, best_score = trial_pose, trial_score

    return best_pose, best_score

def create_initial_files(output_dir, args, status="running"):
    """
    Create premium aesthetic initial files documenting PandaDock run.
    """

    output_dir = Path(output_dir)
    logger = setup_logging(output_dir)
    logger.info(f"📁 Creating initial docking files in: {output_dir}")

    # Save status.json
    status_data = {
        "start_time": datetime.now().isoformat(),
        "protein": str(args.protein),
        "ligand": str(args.ligand),
        "algorithm": args.algorithm,
        "status": status,
        "progress": 0.0,
        "total_iterations": getattr(args, 'iterations', 1000),
        "current_iteration": 0,
        "top_score": None
    }
    status_path = output_dir / "status.json"
    with open(status_path, 'w') as f:
        json.dump(status_data, f, indent=2)
    logger.info(f"✅ Status file created: {status_path}")

    # Detect environment info
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    python_version = platform.python_version()
    os_info = f"{platform.system()} {platform.release()}"
    cpu_info = platform.processor() or "Unknown CPU"
    cpu_cores = os.cpu_count() or "Unknown cores"

    try:
        pandadock_version = pkg_resources.get_distribution("pandadock").version
    except Exception:
        pandadock_version = "development"

    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=os.getcwd()).decode().strip()
    except Exception:
        git_commit = "N/A"

    # GPU check
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else "N/A"
    except ImportError:
        gpu_available = False
        gpu_name = "N/A"

    # Optional packages
    rdkit_available = importlib.util.find_spec("rdkit") is not None
    openbabel_available = importlib.util.find_spec("openbabel") is not None

    # Create nice dynamic logo
    atoms = ['C', 'N', 'O', 'S']
    random_atoms = [random.choice(atoms) for _ in range(6)]
    dynamic_logo = f"""
════════════════════════════════════════════════════════════════════════════════
   `██████╗  █████╗ ███╗   ██╗██████╗  █████╗ ██████╗  ██████╗  ██████╗██╗  ██╗
    ██╔══██╗██╔══██╗████╗  ██║██╔══██╗██╔══██╗██╔══██╗██╔═══██╗██╔════╝██║ ██╔╝
    ██████╔╝███████║██╔██╗ ██║██║  ██║███████║██║  ██║██║   ██║██║     █████╔╝ 
    ██╔═══╝ ██╔══██║██║╚██╗██║██║  ██║██╔══██║██║  ██║██║   ██║██║     ██╔═██╗ 
    ██║     ██║  ██║██║ ╚████║██████╔╝██║  ██║██████╔╝╚██████╔╝╚██████╗██║  ██╗
    ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝ ╚═╝  ╚═╝╚═════╝  ╚═════╝  ╚═════╝╚═╝  ╚═╝
                                                                                                                                                                                                                                  
               PandaDock - Python Molecular Docking Tool                             
               https://github.com/pritampanda15/PandaDock   
              (📦 PandaDock Version: {pandadock_version}\n")                
════════════════════════════════════════════════════════════════════════════════
    """
    
    # Write README
    readme_path = output_dir / "parameters.txt"
    with open(readme_path, 'w') as f:
        f.write(dynamic_logo)
        f.write("\n")

        f.write("📅 Run Started: {}\n".format(timestamp))
        f.write("🐍 Python Version: {} on {}\n".format(python_version, os_info))
        f.write(f"🖥️ CPU: {cpu_info} ({cpu_cores} cores)\n")
        f.write(f"🚀 GPU Available: {'Yes' if gpu_available else 'No'}\n")
        if gpu_available:
            f.write(f"   GPU Device: {gpu_name}\n")
        f.write(f"🧪 RDKit Installed: {'Yes' if rdkit_available else 'No'}\n")
        f.write(f"🧪 OpenBabel Installed: {'Yes' if openbabel_available else 'No'}\n")
        f.write(f"📦 PandaDock Version: {pandadock_version}\n")
        f.write(f"🔖 Git Commit: {git_commit}\n")
        f.write("\n")

        f.write("════════════════════════════════════════════════════════════════════════════════\n")
        f.write("🧬 INPUTS\n")
        f.write("───────────────\n")
        f.write(f"• Protein : {args.protein}\n")
        f.write(f"• Ligand  : {args.ligand}\n")
        if hasattr(args, 'grid_center'):
            f.write(f"• Grid Center: {args.grid_center}\n")
        if hasattr(args, 'grid_spacing'):
            f.write(f"• Grid Spacing: {args.grid_spacing} Å\n")
        if hasattr(args, 'grid_radius'):
            f.write(f"• Grid Radius: {args.grid_radius} Å\n")
        if hasattr(args, 'spherical_sampling') and args.spherical_sampling:
            f.write(f"Spherical Sampling Enabled (Radius: {getattr(args, 'sampling_radius', 'default')})\n")
        f.write("\n")

        f.write("⚙️ DOCKING CONFIGURATION\n")
        f.write("───────────────────────\n")
        f.write(f"• Algorithm         : {args.algorithm}\n")
        f.write(f"• Iterations        : {getattr(args, 'iterations', 'N/A')}\n")
        f.write(f"• Population Size   : {getattr(args, 'population_size', 'N/A')}\n")
        if args.algorithm == 'genetic':
            if hasattr(args, 'population_size'):
                {args.population_size}
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

        f.write("\n")
        f.write("🎯 SCORING\n")
        f.write("───────────────────────\n")
        if getattr(args, 'physics_based', False):
            f.write("• Scoring Function: Physics-based\n")
        elif getattr(args, 'enhanced_scoring', False):
            f.write("• Scoring Function: Enhanced\n")
        else:
            f.write("• Scoring Function: Standard\n")

        f.write("\n")
        f.write("📂 OUTPUT STRUCTURE\n")
        f.write("───────────────────────\n")
        f.write("- poses/\n")
        f.write("- plots/\n")
        f.write("- docking_report.txt\n")
        f.write("- docking_report.html\n")
        f.write("- energy_breakdown.csv\n")
        f.write("- status.json\n")

        f.write("\n════════════════════════════════════════════════════════════════════════════════\n")
        f.write("🚀 Happy Docking with PandaDock! Dock Smarter. Discover Faster\n")
        f.write("════════════════════════════════════════════════════════════════════════════════\n")

    logger.info(f"📝 Enhanced initial README created: {readme_path}")


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
        algo_name = "MC"
    elif args.genetic_algorithm:
        algo_name = "GA"
    elif args.pandadock:
        algo_name = "PD"
    elif args.random:
        algo_name = "RAND"
    elif args.default:
        algo_name = "DEFAULT"
    elif args.enhanced_scoring:
        algo_name = "ES"
    elif args.physics_based:
        algo_name = "PHYSICS"
    elif args.standard_scoring:
        algo_name = "STANDARD"
    elif args.docking:
        algo_name = "docking"
    else:
        algo_name = "DEFAULT"
    # Check if protein and ligand are provided
    if args.protein is None or args.ligand is None:
        raise ValueError("Protein and ligand files must be provided")
    # Check if protein and ligand are valid files
    if not os.path.isfile(args.protein):
        raise ValueError(f"Invalid protein file: {args.protein}")
    if not os.path.isfile(args.ligand):
        raise ValueError(f"Invalid ligand file: {args.ligand}")
    # Check if algorithm is valid
    if algo_name not in ["DEFAULT", "GA", "MC", "PD", "RAND", "ES", "PHYSICS", "STANDARD", "docking"]:
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
    for i, (pose, score) in enumerate(results[:20]):  # Save top 10 poses
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
def random_point_in_sphere(center, radius):
    """
    Generate a random point uniformly inside a sphere.
    """
    phi = np.random.uniform(0, 2 * np.pi)
    costheta = np.random.uniform(-1, 1)
    u = np.random.uniform(0, 1)

    theta = np.arccos(costheta)
    r = radius * (u ** (1/3))

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return center + np.array([x, y, z])

# def is_inside_sphere(pose, center, radius):
#     """
#     Check if the ligand pose's centroid is within the active site sphere.
#     """
#     centroid = np.mean(pose.xyz, axis=0)
#     distance = np.linalg.norm(centroid - center)
#     return distance <= radius

def is_inside_sphere(ligand, center, radius):
    for atom in ligand.atoms:
        if np.linalg.norm(atom['coords'] - center) > radius:
            return False
    return True

def enforce_sphere_boundary(pose, center, radius):
    """
    Ensure all atoms of a ligand stay within the defined spherical boundary.
    
    Parameters:
        pose: Ligand object
        center: np.array of shape (3,)
        radius: float

    Returns:
        pose: Adjusted Ligand object
    """
    max_dist = 0.0
    for atom in pose.atoms:
        dist = np.linalg.norm(atom['coords'] - center)
        max_dist = max(max_dist, dist)

    if max_dist > radius:
        # Calculate the scaling factor to bring all atoms inside
        scale = (radius * 0.95) / max_dist
        centroid = np.mean(pose.xyz, axis=0)
        for i, atom in enumerate(pose.atoms):
            direction = atom['coords'] - centroid
            new_coords = centroid + direction * scale
            pose.atoms[i]['coords'] = new_coords
            pose.xyz[i] = new_coords

    return pose
def is_fully_inside_sphere(ligand, center, radius, buffer=0.0):
        """
        Check if all atoms of a ligand are inside the sphere.
        
        Parameters:
            ligand: Ligand object with atoms or xyz coordinates
            center: Center coordinates of the sphere
            radius: Radius of the sphere
            buffer: Optional buffer distance (Å) to keep atoms from boundary
            
        Returns:
            bool: True if all atoms are within the sphere boundary
        """
        if hasattr(ligand, 'xyz') and ligand.xyz is not None:
            atom_coords = ligand.xyz
        elif hasattr(ligand, 'atoms'):
            atom_coords = np.array([atom['coords'] for atom in ligand.atoms])
        else:
            raise ValueError("Ligand must have xyz or atoms attribute")
        
        # Calculate distance of each atom to center
        distances = np.linalg.norm(atom_coords - center, axis=1)
        
        # Check if all atoms are within radius (with optional buffer)
        return np.all(distances <= (radius - buffer))

def reposition_inside_sphere(ligand, center, radius, max_attempts=50):
        """
        Reposition a ligand to ensure all atoms are inside the sphere.
        
        Parameters:
            ligand: Ligand object to reposition
            center: Center coordinates of the sphere
            radius: Radius of the sphere
            max_attempts: Maximum repositioning attempts
            
        Returns:
            bool: True if successfully repositioned, False if failed
        """
        import copy
        
        # Calculate effective molecular radius of ligand
        if hasattr(ligand, 'xyz'):
            centroid = np.mean(ligand.xyz, axis=0)
            atom_coords = ligand.xyz
        else:
            atom_coords = np.array([atom['coords'] for atom in ligand.atoms])
            centroid = np.mean(atom_coords, axis=0)
        
        # Calculate molecular radius - distance from centroid to furthest atom
        molecular_radius = np.max(np.linalg.norm(atom_coords - centroid, axis=1))
        
        # If molecular radius is too large for sphere, scale ligand
        if molecular_radius >= radius * 0.9:
            print(f"Warning: Ligand radius ({molecular_radius:.2f}Å) approaching sphere radius ({radius:.2f}Å)")
        
        # Get current centroid-to-center vector and distance
        vector_to_center = center - centroid
        dist_to_center = np.linalg.norm(vector_to_center)
        
        # Try to reposition while gradually moving inward
        for attempt in range(max_attempts):
            # Scale factor starts at 1.0 and increases to ensure inward movement
            scale_factor = 1.0 + (attempt / max_attempts) * 0.5
            
            # Calculate safe distance to avoid boundary violations
            safe_distance = radius - molecular_radius - 0.5  # 0.5Å buffer
            
            # If we're too far from center, move inward
            if dist_to_center > safe_distance:
                # Create normalized direction vector
                if dist_to_center > 0:
                    direction = vector_to_center / dist_to_center
                else:
                    # Random direction if at center
                    direction = np.random.randn(3)
                    direction = direction / np.linalg.norm(direction)
                
                # Calculate adjustment vector
                adjustment = direction * (dist_to_center - safe_distance/scale_factor)
                
                # Apply translation
                ligand.translate(adjustment)
                
                # Check if now fully inside
                if is_fully_inside_sphere(ligand, center, radius, buffer=0.2):
                    return True
            else:
                # Apply small random rotation which may help
                angle = np.random.uniform(-0.1, 0.1)  # Small rotation angle
                axis = np.random.randn(3)
                axis = axis / np.linalg.norm(axis)
                
                from scipy.spatial.transform import Rotation
                rotation = Rotation.from_rotvec(axis * angle)
                
                # Apply rotation around centroid
                centroid = np.mean(ligand.xyz, axis=0)
                ligand.translate(-centroid)
                ligand.rotate(rotation.as_matrix())
                ligand.translate(centroid)
                
                # Check if now fully inside
                if is_fully_inside_sphere(ligand, center, radius, buffer=0.2):
                    return True
        
        # Final attempt: Move directly toward center
        centroid = np.mean(ligand.xyz, axis=0)
        vector_to_center = center - centroid
        dist_to_center = np.linalg.norm(vector_to_center)
        
        if dist_to_center > 0:
            # Move to a position where molecules should fit
            target_dist = radius * 0.5  # Move to middle of sphere
            movement = (vector_to_center / dist_to_center) * (dist_to_center - target_dist)
            ligand.translate(movement)
        
        return is_fully_inside_sphere(ligand, center, radius, buffer=0.1)


def save_sphere_to_pdb(self, grid_points, output_path):
        """
        Save spherical grid points as PDB file for visualization.
        """
        with open(output_path, 'w') as f:
            for i, point in enumerate(grid_points):
                f.write(
                    f"HETATM{i+1:5d}  C   SPH A   1    "
                    f"{point[0]:8.3f}{point[1]:8.3f}{point[2]:8.3f}  1.00  0.00           C\n"
                )
            f.write("END\n")
        self.logger.info(f"Saved spherical grid to {output_path}")
        self.logger.info("Spherical grid points saved for visualization.")
        self.logger.info(f"Grid points saved to {output_path}")

#for virtaul screening
from pathlib import Path
import os
import json
import platform
import subprocess
import random
import pkg_resources
from datetime import datetime
import logging
import importlib
def get_ligand_files(directory, extensions=(".sdf", ".mol", ".mol2")):
    return [str(p) for p in Path(directory).glob("*") if p.suffix.lower() in extensions]
def get_protein_files(directory, extensions=(".pdb", ".pdbqt")):
    return [str(p) for p in Path(directory).glob("*") if p.suffix.lower() in extensions]

from rdkit import Chem
from rdkit.Chem import AllChem

def write_pose_as_pdb(mol, output_file):
    if mol is None:
        raise ValueError("Pose molecule is None")
    Chem.MolToPDBFile(mol, output_file)
    print(f"Pose saved to {output_file}")
def write_pose_as_sdf(mol, output_file):
    if mol is None:
        raise ValueError("Pose molecule is None")
    Chem.MolToSDF(mol, output_file)
    print(f"Pose saved to {output_file}")
def write_pose_as_mol2(mol, output_file):
    if mol is None:
        raise ValueError("Pose molecule is None")
    AllChem.MolToMol2File(mol, output_file)
    print(f"Pose saved to {output_file}")

def save_sphere_pdb(self, center, radius, filename="sphere.pdb", 
                   transparency=0.3, density=100, style="surface", 
                   element="He", color_by_distance=False):
    """
    Save a PDB file with a transparent sphere of dummy atoms for visualization.
    
    Parameters:
    -----------
    center : np.ndarray
        3D center of the sphere
    radius : float
        Radius of the sphere
    filename : str
        Output PDB filename
    transparency : float (0.0-1.0)
        Transparency level (0.0 = fully transparent, 1.0 = opaque)
    density : int
        Number of points to sample on the sphere
    style : str
        "surface" = points on sphere surface, "wireframe" = grid lines, "volume" = filled sphere
    element : str
        Element symbol for the dummy atoms (affects default visualization)
    color_by_distance : bool
        Whether to color-code atoms by distance from center
    """
    
    # Create output path
    if hasattr(self, 'output_dir'):
        filepath = Path(self.output_dir) / filename
    else:
        filepath = Path(filename)
    
    # Choose element for transparency effects
    # He (Helium) - often rendered as very small/transparent
    # Ne (Neon) - noble gas, often transparent
    # H (Hydrogen) - small, less obstructive
    # C (Carbon) - standard but customizable
    transparency_elements = {
        "helium": "He",
        "neon": "Ne", 
        "hydrogen": "H",
        "carbon": "C",
        "dummy": "X"
    }
    
    if element.lower() in transparency_elements:
        atom_element = transparency_elements[element.lower()]
    else:
        atom_element = element.upper()[:2]  # Take first 2 characters
    
    with open(filepath, 'w') as f:
        # Write PDB header with transparency information
        f.write("REMARK   1 SPHERE VISUALIZATION FOR DOCKING SEARCH SPACE\n")
        f.write(f"REMARK   2 CENTER: {center[0]:.3f} {center[1]:.3f} {center[2]:.3f}\n")
        f.write(f"REMARK   3 RADIUS: {radius:.3f} ANGSTROMS\n")
        f.write(f"REMARK   4 TRANSPARENCY: {transparency:.2f}\n")
        f.write(f"REMARK   5 DENSITY: {density} POINTS\n")
        f.write(f"REMARK   6 STYLE: {style.upper()}\n")
        f.write("REMARK   7 \n")
        f.write("REMARK   8 FOR PYMOL TRANSPARENCY: select sphere, elem He; set transparency, 0.7, sphere\n")
        f.write("REMARK   9 FOR CHIMERA: select :SPH; transparency 70\n")
        f.write("REMARK  10 \n")
        
        atom_count = 0
        
        if style.lower() == "surface":
            # Generate points on sphere surface
            points = generate_sphere_surface_points(center, radius, density)
            
        elif style.lower() == "wireframe":
            # Generate wireframe grid
            points = generate_sphere_wireframe(center, radius, density)
            
        elif style.lower() == "volume":
            # Generate points throughout sphere volume
            points = generate_sphere_volume_points(center, radius, density)
            
        else:  # Default to surface
            points = generate_sphere_surface_points(center, radius, density)
        
        # Write atoms
        for i, point in enumerate(points):
            atom_count += 1
            x, y, z = point
            
            # Calculate distance from center for color coding
            if color_by_distance:
                dist = np.linalg.norm(point - center)
                # Use B-factor to encode distance (0-100 scale)
                b_factor = (dist / radius) * 100.0
            else:
                b_factor = 50.0  # Neutral B-factor
            
            # Set occupancy based on transparency (inverted: high transparency = low occupancy)
            occupancy = max(0.01, min(1.00, transparency))
            
            # Write PDB ATOM record
            f.write(f"ATOM  {atom_count:5d}  {atom_element:<2s}  SPH A   1    "
                   f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{b_factor:6.2f}           {atom_element}\n")
        
        f.write("END\n")
    
    print(f"💫 Transparent sphere saved: {filepath}")
    print(f"   Center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
    print(f"   Radius: {radius:.2f} Å")
    print(f"   Points: {atom_count}")
    print(f"   Transparency: {transparency:.2f}")
    print(f"   Element: {atom_element}")
    
    # Print visualization commands
    print(f"\n🎨 Visualization Commands:")
    print(f"   PyMOL: load {filepath.name}; set transparency, {1-transparency:.1f}, elem {atom_element}")
    print(f"   ChimeraX: open {filepath.name}; transparency :{atom_count-99}-{atom_count} {int((1-transparency)*100)}")
    
    return filepath

def generate_sphere_surface_points(center, radius, n_points):
    """Generate evenly distributed points on sphere surface using Fibonacci spiral."""
    points = []
    
    # Use Fibonacci spiral for more even distribution
    golden_ratio = (1 + 5**0.5) / 2
    
    for i in range(n_points):
        # Fibonacci spiral parameters
        theta = 2 * np.pi * i / golden_ratio
        phi = np.arccos(1 - 2 * i / n_points)
        
        # Convert to Cartesian coordinates
        x = center[0] + radius * np.sin(phi) * np.cos(theta)
        y = center[1] + radius * np.sin(phi) * np.sin(theta)
        z = center[2] + radius * np.cos(phi)
        
        points.append(np.array([x, y, z]))
    
    return points

def generate_sphere_wireframe(center, radius, n_lines=20):
    """Generate wireframe representation with latitude/longitude lines."""
    points = []
    
    # Latitude lines (horizontal circles)
    for lat in range(-n_lines//2, n_lines//2 + 1):
        phi = np.pi * lat / n_lines  # -π/2 to π/2
        r_lat = radius * np.cos(phi)  # Radius at this latitude
        z = center[2] + radius * np.sin(phi)
        
        # Points along this latitude circle
        n_points_lat = max(6, int(20 * np.cos(phi)))  # Fewer points near poles
        for lon in range(n_points_lat):
            theta = 2 * np.pi * lon / n_points_lat
            x = center[0] + r_lat * np.cos(theta)
            y = center[1] + r_lat * np.sin(theta)
            points.append(np.array([x, y, z]))
    
    # Longitude lines (vertical semicircles)
    for lon in range(0, n_lines):
        theta = 2 * np.pi * lon / n_lines
        for i in range(n_lines):
            phi = np.pi * (i - n_lines/2) / n_lines
            x = center[0] + radius * np.sin(phi) * np.cos(theta)
            y = center[1] + radius * np.sin(phi) * np.sin(theta)
            z = center[2] + radius * np.cos(phi)
            points.append(np.array([x, y, z]))
    
    return points

def generate_sphere_volume_points(center, radius, n_points):
    """Generate points distributed throughout sphere volume."""
    points = []
    
    for i in range(n_points):
        # Generate random point in unit sphere
        while True:
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            z = np.random.uniform(-1, 1)
            
            if x*x + y*y + z*z <= 1.0:
                break
        
        # Scale to desired radius and translate to center
        point = center + radius * np.array([x, y, z])
        points.append(point)
    
    return points

def save_transparent_sphere_advanced(self, center, radius, filename="sphere.pdb",
                                   layers=3, transparency_gradient=True, 
                                   highlight_poles=False, grid_spacing=1.0):
    """
    Advanced transparent sphere with multiple layers and gradient effects.
    
    Parameters:
    -----------
    center : np.ndarray
        3D center of the sphere
    radius : float
        Radius of the sphere
    filename : str
        Output PDB filename
    layers : int
        Number of concentric sphere layers
    transparency_gradient : bool
        Whether to make outer layers more transparent
    highlight_poles : bool
        Whether to highlight sphere poles (binding site directions)
    grid_spacing : float
        Spacing for grid points (Angstroms)
    """
    
    filepath = Path(self.output_dir) / filename if hasattr(self, 'output_dir') else Path(filename)
    
    with open(filepath, 'w') as f:
        # Enhanced header
        f.write("REMARK   1 ADVANCED TRANSPARENT SPHERE VISUALIZATION\n")
        f.write(f"REMARK   2 CENTER: {center[0]:.3f} {center[1]:.3f} {center[2]:.3f}\n")
        f.write(f"REMARK   3 RADIUS: {radius:.3f} ANGSTROMS\n")
        f.write(f"REMARK   4 LAYERS: {layers}\n")
        f.write(f"REMARK   5 GRADIENT: {'YES' if transparency_gradient else 'NO'}\n")
        f.write("REMARK   6 \n")
        f.write("REMARK   7 CHAIN A: OUTER LAYER (MOST TRANSPARENT)\n")
        f.write("REMARK   8 CHAIN B: MIDDLE LAYER\n") 
        f.write("REMARK   9 CHAIN C: INNER LAYER (LEAST TRANSPARENT)\n")
        f.write("REMARK  10 \n")
        f.write("REMARK  11 PYMOL SETUP:\n")
        f.write("REMARK  12   set transparency, 0.8, chain A\n")
        f.write("REMARK  13   set transparency, 0.6, chain B\n")
        f.write("REMARK  14   set transparency, 0.4, chain C\n")
        f.write("REMARK  15 \n")
        
        atom_count = 0
        chain_letters = ['A', 'B', 'C', 'D', 'E']
        
        # Generate multiple layers
        for layer in range(layers):
            layer_radius = radius * (layer + 1) / layers
            layer_transparency = 0.9 - (layer * 0.3) if transparency_gradient else 0.5
            chain = chain_letters[layer % len(chain_letters)]
            
            # Generate points for this layer
            n_points = max(50, int(100 * (layer + 1) / layers))  # More points in outer layers
            layer_points = generate_sphere_surface_points(center, layer_radius, n_points)
            
            for point in layer_points:
                atom_count += 1
                x, y, z = point
                
                # Use different elements for different layers
                elements = ['He', 'Ne', 'Ar', 'Kr', 'Xe']
                element = elements[layer % len(elements)]
                
                f.write(f"ATOM  {atom_count:5d}  {element:<2s}  SPH {chain}   1    "
                       f"{x:8.3f}{y:8.3f}{z:8.3f}{layer_transparency:6.2f}{50.0:6.2f}           {element}\n")
        
        # Add pole markers if requested
        if highlight_poles:
            poles = [
                center + np.array([0, 0, radius]),    # North pole
                center + np.array([0, 0, -radius]),   # South pole
                center + np.array([radius, 0, 0]),    # East pole
                center + np.array([-radius, 0, 0]),   # West pole
                center + np.array([0, radius, 0]),    # Front pole
                center + np.array([0, -radius, 0])    # Back pole
            ]
            
            for i, pole in enumerate(poles):
                atom_count += 1
                x, y, z = pole
                f.write(f"ATOM  {atom_count:5d}  P   POL P   1    "
                       f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 99.00            P\n")
        
        f.write("END\n")
    
    print(f"✨ Advanced transparent sphere saved: {filepath}")
    print(f"   Layers: {layers}")
    print(f"   Total atoms: {atom_count}")
    print(f"   Gradient transparency: {transparency_gradient}")
    
    return filepath

def save_minimal_sphere_pdb(self, center, radius, filename="minimal_sphere.pdb", points=20):
    """
    Minimal transparent sphere with just key points for less visual clutter.
    """
    filepath = Path(self.output_dir) / filename if hasattr(self, 'output_dir') else Path(filename)
    
    with open(filepath, 'w') as f:
        f.write("REMARK   1 MINIMAL SPHERE OUTLINE\n")
        f.write(f"REMARK   2 CENTER: {center[0]:.3f} {center[1]:.3f} {center[2]:.3f}\n")
        f.write(f"REMARK   3 RADIUS: {radius:.3f} ANGSTROMS\n")
        f.write("REMARK   4 MINIMAL VISUALIZATION - ONLY KEY POINTS\n")
        f.write("REMARK   5 \n")
        
        # Generate only key directional points
        directions = [
            [1, 0, 0], [-1, 0, 0],     # X-axis
            [0, 1, 0], [0, -1, 0],     # Y-axis  
            [0, 0, 1], [0, 0, -1],     # Z-axis
            [1, 1, 0], [-1, -1, 0],    # XY diagonal
            [1, 0, 1], [-1, 0, -1],    # XZ diagonal
            [0, 1, 1], [0, -1, -1],    # YZ diagonal
        ]
        
        # Add some intermediate points for better sphere outline
        for i in range(points - len(directions)):
            theta = 2 * np.pi * i / (points - len(directions))
            phi = np.pi * np.random.random()
            direction = [np.sin(phi) * np.cos(theta), 
                        np.sin(phi) * np.sin(theta), 
                        np.cos(phi)]
            directions.append(direction)
        
        for i, direction in enumerate(directions):
            # Normalize direction
            direction = np.array(direction)
            direction = direction / np.linalg.norm(direction)
            
            # Place point on sphere surface
            point = center + radius * direction
            x, y, z = point
            
            f.write(f"ATOM  {i+1:5d}  He  SPH A   1    "
                   f"{x:8.3f}{y:8.3f}{z:8.3f}  0.20 20.00           He\n")
        
        f.write("END\n")
    
    print(f"🔘 Minimal sphere saved: {filepath} ({len(directions)} points)")
    return filepath

def save_transparent_sphere_pdb(center, radius, output_dir, filename="sphere.pdb", 
                               transparency=0.3, density=100, style="surface", 
                               element="He", color_by_distance=False):
    """
    Save a transparent sphere PDB file for docking search space visualization.
    
    Parameters:
    -----------
    center : np.ndarray or list
        3D center of the sphere [x, y, z]
    radius : float
        Radius of the sphere in Angstroms
    output_dir : Path or str
        Output directory
    filename : str
        Output PDB filename
    transparency : float (0.0-1.0)
        Transparency level (0.0 = fully transparent, 1.0 = opaque)
    density : int
        Number of points to sample on the sphere
    style : str
        "surface", "wireframe", or "minimal"
    element : str
        Element symbol for the dummy atoms
    color_by_distance : bool
        Whether to color-code atoms by distance from center
    """
    
    # Ensure center is numpy array
    center = np.array(center)
    
    # Create output path
    filepath = Path(output_dir) / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Choose element for transparency effects
    transparency_elements = {
        "helium": "He", "neon": "Ne", "hydrogen": "H", 
        "carbon": "C", "dummy": "X"
    }
    
    if element.lower() in transparency_elements:
        atom_element = transparency_elements[element.lower()]
    else:
        atom_element = element.upper()[:2]
    
    with open(filepath, 'w') as f:
        # Write PDB header with transparency information
        f.write("REMARK   1 TRANSPARENT SPHERE FOR DOCKING SEARCH SPACE\n")
        f.write(f"REMARK   2 CENTER: {center[0]:.3f} {center[1]:.3f} {center[2]:.3f}\n")
        f.write(f"REMARK   3 RADIUS: {radius:.3f} ANGSTROMS\n")
        f.write(f"REMARK   4 TRANSPARENCY: {transparency:.2f} (0=transparent, 1=opaque)\n")
        f.write(f"REMARK   5 DENSITY: {density} POINTS\n")
        f.write(f"REMARK   6 STYLE: {style.upper()}\n")
        f.write("REMARK   7 \n")
        f.write("REMARK   8 PYMOL COMMANDS:\n")
        f.write(f"REMARK   9   load {filename}\n")
        f.write(f"REMARK  10   set transparency, {1-transparency:.1f}, elem {atom_element}\n")
        f.write(f"REMARK  11   set sphere_scale, 0.3, elem {atom_element}\n")
        f.write("REMARK  12   color gray80, elem He\n")
        f.write("REMARK  13 \n")
        
        # Generate points based on style
        if style.lower() == "surface":
            points = generate_fibonacci_sphere_points(center, radius, density)
        elif style.lower() == "wireframe":
            points = generate_wireframe_points(center, radius, density)
        elif style.lower() == "minimal":
            points = generate_minimal_points(center, radius, min(density, 24))
        else:  # Default to surface
            points = generate_fibonacci_sphere_points(center, radius, density)
        
        # Write atoms
        for i, point in enumerate(points):
            x, y, z = point
            
            # Calculate distance from center for color coding
            if color_by_distance:
                dist = np.linalg.norm(point - center)
                b_factor = (dist / radius) * 100.0  # 0-100 scale
            else:
                b_factor = 50.0
            
            # Set occupancy based on transparency
            occupancy = max(0.01, min(1.00, transparency))
            
            # Write PDB ATOM record
            f.write(f"HETATM{i+1:5d}  {atom_element:<2s}  SPH A   1    "
                   f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{b_factor:6.2f}           {atom_element}\n")
        
        f.write("END\n")
    
    print(f"💫 Transparent sphere saved: {filepath}")
    print(f"   Center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
    print(f"   Radius: {radius:.2f} Å")
    print(f"   Points: {len(points)}")
    print(f"   Style: {style}")
    print(f"   Transparency: {transparency:.2f}")
    
    return filepath

def generate_fibonacci_sphere_points(center, radius, n_points):
    """Generate evenly distributed points on sphere surface using Fibonacci spiral."""
    points = []
    golden_ratio = (1 + 5**0.5) / 2
    
    for i in range(n_points):
        theta = 2 * np.pi * i / golden_ratio
        phi = np.arccos(1 - 2 * i / n_points)
        
        x = center[0] + radius * np.sin(phi) * np.cos(theta)
        y = center[1] + radius * np.sin(phi) * np.sin(theta)
        z = center[2] + radius * np.cos(phi)
        
        points.append(np.array([x, y, z]))
    
    return points

def generate_wireframe_points(center, radius, n_lines=20):
    """Generate wireframe grid points."""
    points = []
    
    # Latitude circles
    for lat in range(-n_lines//4, n_lines//4 + 1, 2):  # Fewer lines for cleaner look
        phi = np.pi * lat / n_lines
        r_lat = radius * np.cos(phi)
        z = center[2] + radius * np.sin(phi)
        
        n_points_lat = max(8, int(16 * np.cos(phi)))
        for lon in range(n_points_lat):
            theta = 2 * np.pi * lon / n_points_lat
            x = center[0] + r_lat * np.cos(theta)
            y = center[1] + r_lat * np.sin(theta)
            points.append(np.array([x, y, z]))
    
    # Longitude semicircles
    for lon in range(0, n_lines//2, 2):  # Fewer lines
        theta = 2 * np.pi * lon / n_lines
        for i in range(0, n_lines, 2):  # Skip points for sparser grid
            phi = np.pi * (i - n_lines/2) / n_lines
            x = center[0] + radius * np.sin(phi) * np.cos(theta)
            y = center[1] + radius * np.sin(phi) * np.sin(theta)
            z = center[2] + radius * np.cos(phi)
            points.append(np.array([x, y, z]))
    
    return points

def generate_minimal_points(center, radius, n_points=24):
    """Generate minimal set of key directional points."""
    points = []
    
    # Key directions (6 primary + 8 corners + 12 edges of cube)
    directions = [
        # Primary axes
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
        # Cube corners (normalized)
        [1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1],
        [-1, -1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, -1],
        # Edge midpoints
        [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
        [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
        [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1]
    ]
    
    # Take first n_points and normalize
    for i, direction in enumerate(directions[:n_points]):
        direction = np.array(direction, dtype=float)
        direction = direction / np.linalg.norm(direction)
        point = center + radius * direction
        points.append(point)
    
    return points

def save_metal_docking_results(results, metal_centers, output_dir, top_n=10):
    """
    Save metal docking results with coordination analysis.
    
    Parameters:
    -----------
    results : list
        List of (pose, score) tuples
    metal_centers : list
        List of metal centers
    output_dir : str or Path
        Output directory
    top_n : int
        Number of top poses to analyze in detail
    """
    from .metal_docking import MetalDockingPreparation
    import copy
    
    output_path = Path(output_dir)
    
    # Save detailed metal coordination analysis for top poses
    for i, (pose, score) in enumerate(results[:top_n]):
        pose_dir = output_path / f"pose_{i+1}"
        pose_dir.mkdir(exist_ok=True)
        
        # Save pose
        pose_file = pose_dir / f"pose_{i+1}_score_{score:.2f}.pdb"
        pose.write_pdb(pose_file)
        
        # Analyze metal coordination for this pose
        coordination_analysis = {}
        
        for j, metal_center in enumerate(metal_centers):
            # Find coordinating atoms in this pose
            coordinating_atoms = []
            for atom in pose.atoms:
                element = atom.get('element', atom.get('name', ''))[:1]
                if element in ['N', 'O', 'S', 'P', 'C']:
                    distance = np.linalg.norm(
                        np.array(atom['coords']) - metal_center.coords
                    )
                    if distance <= 3.5:  # Coordination distance
                        coordinating_atoms.append({
                            'atom_name': atom.get('name', ''),
                            'element': element,
                            'distance': float(distance),
                            'coords': atom['coords'].tolist()
                        })
            
            # Create temporary metal center for scoring
            temp_center = copy.deepcopy(metal_center)
            for atom_info in coordinating_atoms:
                temp_center.add_coordinating_atom({
                    'coords': np.array(atom_info['coords']),
                    'element': atom_info['element'],
                    'name': atom_info['atom_name']
                })
            
            coordination_score = temp_center.get_coordination_score()
            
            coordination_analysis[f"metal_{j+1}"] = {
                'element': metal_center.element,
                'coordinates': metal_center.coords.tolist(),
                'coordinating_atoms': coordinating_atoms,
                'coordination_number': len(coordinating_atoms),
                'coordination_score': float(coordination_score),
                'preferred_geometry': metal_center.preferred_geometry
            }
        
        # Save coordination analysis
        analysis_file = pose_dir / "coordination_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(coordination_analysis, f, indent=2)
    
    # Create summary report
    summary_file = output_path / "metal_docking_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("METAL-AWARE DOCKING SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Number of metal centers: {len(metal_centers)}\n\n")
        
        for i, metal_center in enumerate(metal_centers):
            f.write(f"Metal Center {i+1}:\n")
            f.write(f"  Element: {metal_center.element}\n")
            f.write(f"  Coordinates: ({metal_center.coords[0]:.2f}, {metal_center.coords[1]:.2f}, {metal_center.coords[2]:.2f})\n")
            f.write(f"  Preferred Geometry: {metal_center.preferred_geometry}\n")
            f.write(f"  Existing Coordination: {metal_center.coordination_number} atoms\n\n")
        
        f.write("TOP POSES COORDINATION ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        
        for i, (pose, score) in enumerate(results[:top_n]):
            f.write(f"\nPose {i+1} (Score: {score:.2f}):\n")
            
            for j, metal_center in enumerate(metal_centers):
                # Quick coordination analysis
                coordinating_count = 0
                for atom in pose.atoms:
                    element = atom.get('element', atom.get('name', ''))[:1]
                    if element in ['N', 'O', 'S', 'P']:
                        distance = np.linalg.norm(
                            np.array(atom['coords']) - metal_center.coords
                        )
                        if distance <= 3.0:
                            coordinating_count += 1
                
                f.write(f"  Metal {j+1} ({metal_center.element}): {coordinating_count} coordinating atoms\n")


def write_metal_complex_pdb(protein, ligand, metal_centers, output_file):
    """
    Write a PDB file showing the complete metal complex.
    
    Parameters:
    -----------
    protein : Protein
        Protein object
    ligand : Ligand
        Ligand object
    metal_centers : list
        List of metal centers
    output_file : str or Path
        Output PDB file path
    """
    with open(output_file, 'w') as f:
        # Write header
        f.write("REMARK   Metal-Ligand Complex from PandaDock\n")
        f.write("REMARK   Metal Centers and Coordination Environment\n")
        
        atom_counter = 1
        
        # Write protein atoms (only those near metals)
        written_protein_atoms = set()
        for metal_center in metal_centers:
            for protein_atom in protein.atoms:
                distance = np.linalg.norm(
                    np.array(protein_atom['coords']) - metal_center.coords
                )
                
                if distance <= 8.0 and id(protein_atom) not in written_protein_atoms:
                    # Write protein atom
                    element = protein_atom.get('element', protein_atom.get('name', ''))[:2]
                    f.write(f"ATOM  {atom_counter:5d} {protein_atom.get('name', 'X'):>4s} "
                           f"{protein_atom.get('residue_name', 'UNK'):>3s} A"
                           f"{protein_atom.get('residue_number', 1):4d}    "
                           f"{protein_atom['coords'][0]:8.3f}"
                           f"{protein_atom['coords'][1]:8.3f}"
                           f"{protein_atom['coords'][2]:8.3f}"
                           f"  1.00  0.00          {element:>2s}\n")
                    
                    written_protein_atoms.add(id(protein_atom))
                    atom_counter += 1
        
        # Write metal atoms
        for i, metal_center in enumerate(metal_centers):
            f.write(f"HETATM{atom_counter:5d} {metal_center.element:>4s} MET A{i+900:4d}    "
                   f"{metal_center.coords[0]:8.3f}"
                   f"{metal_center.coords[1]:8.3f}"
                   f"{metal_center.coords[2]:8.3f}"
                   f"  1.00  0.00          {metal_center.element:>2s}\n")
            atom_counter += 1
        
        # Write ligand atoms
        for atom in ligand.atoms:
            element = atom.get('element', atom.get('name', ''))[:2]
            f.write(f"HETATM{atom_counter:5d} {atom.get('name', 'X'):>4s} LIG A{999:4d}    "
                   f"{atom['coords'][0]:8.3f}"
                   f"{atom['coords'][1]:8.3f}"
                   f"{atom['coords'][2]:8.3f}"
                   f"  1.00  0.00          {element:>2s}\n")
            atom_counter += 1
        
        # Write coordination bonds as CONECT records
        metal_atom_numbers = []
        ligand_start_number = atom_counter - len(ligand.atoms)
        
        for i, metal_center in enumerate(metal_centers):
            metal_atom_number = ligand_start_number - len(metal_centers) + i
            metal_atom_numbers.append(metal_atom_number)
            
            # Find coordinating ligand atoms
            coordinating_ligand_atoms = []
            for j, atom in enumerate(ligand.atoms):
                element = atom.get('element', atom.get('name', ''))[:1]
                if element in ['N', 'O', 'S', 'P']:
                    distance = np.linalg.norm(
                        np.array(atom['coords']) - metal_center.coords
                    )
                    if distance <= 3.0:
                        coordinating_ligand_atoms.append(ligand_start_number + j)
            
            # Write CONECT records for metal-ligand bonds
            if coordinating_ligand_atoms:
                conect_line = f"CONECT{metal_atom_number:5d}"
                for coord_atom in coordinating_ligand_atoms[:4]:  # Max 4 per CONECT line
                    conect_line += f"{coord_atom:5d}"
                f.write(conect_line + "\n")
        
        f.write("END\n")


