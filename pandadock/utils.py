# utils.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import logging


def setup_logging(output_dir=None):
    """Configure logging system for PandaDock."""
    logger = logging.getLogger("pandadock")
    
    # Only configure if not already configured
    if not logger.handlers:
        # Basic configuration
        logger.setLevel(logging.INFO)
        
        # Add console handler
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        logger.addHandler(console)
        
        # Add file handler if output directory is provided
        if output_dir:
            logs_dir = Path(output_dir) / "logs"
            os.makedirs(logs_dir, exist_ok=True)
            
            file_handler = logging.FileHandler(logs_dir / "pandadock.log")
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger.addHandler(file_handler)
    
    return logger

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
