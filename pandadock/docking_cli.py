#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PandaDock Enhanced Molecular Docking CLI

High-Accuracy Algorithms:
- enhanced_hierarchical: 3-stage hierarchical search (RMSD < 2Å)
- monte_carlo: Basic Monte Carlo sampling (fast)
- genetic_algorithm: Genetic algorithm with ensemble refinement

Advanced Features:
- Comprehensive precision scoring with interaction energy decomposition
- MM-GBSA rescoring for accurate binding free energies
- Detailed interaction analysis and residue contributions
- Professional outputs (complexes, interaction reports, plots)
"""

import click
import logging
import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
from rdkit import Chem
from rdkit.Chem import AllChem
import time
import sys
import os

# Import enhanced PandaDock modules
from .docking.algorithms import (
    MonteCarloDocker, EnhancedHierarchicalDocker,
    GPU_AVAILABLE
)
from .docking.scoring.physics_based import PhysicsBasedScoring
from .docking.scoring.precision_score import PrecisionScoring
from .docking.scoring.empirical import EmpiricalScoring
from .docking.scoring.hybrid import HybridScoring
from .docking.scoring.ensemble import BoltzmannEnsemble
from .docking.refinement.mmgbsa import MMGBSARescoring
from .docking.analysis.interaction_analysis import InteractionAnalyzer
from .docking.core import DockingResult, Pose, DockingEngine
from .docking.algorithms.monte_carlo_cpu import MonteCarloDocker as MCDocker
from .docking.algorithms.genetic_algorithm_cpu import GeneticAlgorithmDocker
from .docking.algorithms.hierarchical_cpu import HierarchicalDocker
from .docking.algorithms.crystal_guided_cpu import CrystalGuidedDocker
from .docking.algorithm_selector import AlgorithmSelector, detect_batch_mode

# Import visualization components
try:
    from .docking.visualization.visualizer import DockingVisualizer
    from .docking.visualization.plotter import AffinityPlotter
except ImportError:
    class DockingVisualizer:
        pass
    class AffinityPlotter:
        def create_binding_affinity_plot(self, *args, **kwargs):
            pass
        def create_interaction_energy_plot(self, *args, **kwargs):
            pass


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

# Initialize docking engine
engine = DockingEngine()

# Register enhanced CPU algorithms
engine.register_algorithm(MCDocker())
engine.register_algorithm(GeneticAlgorithmDocker())
engine.register_algorithm(HierarchicalDocker())
engine.register_algorithm(EnhancedHierarchicalDocker())
engine.register_algorithm(CrystalGuidedDocker())

# Register GPU algorithms if available
GPU_ALGORITHMS_REGISTERED = False
if GPU_AVAILABLE:
    try:
        from .docking.algorithms.monte_carlo_gpu import CUDAMonteCarloDocker
        from .docking.algorithms.genetic_algorithm_gpu import CUDAGeneticAlgorithmDocker
        from .docking.algorithms.enhanced_hierarchical_gpu import EnhancedHierarchicalGPUDocker
        engine.register_algorithm(CUDAMonteCarloDocker())
        engine.register_algorithm(CUDAGeneticAlgorithmDocker())
        engine.register_algorithm(EnhancedHierarchicalGPUDocker())
        GPU_ALGORITHMS_REGISTERED = True
    except (ImportError, RuntimeError) as e:
        # CUDA libraries not available or GPU initialization failed
        import sys
        print(f"Warning: GPU algorithms could not be registered: {e}", file=sys.stderr)
        print(f"  GPU docking algorithms (cuda_monte_carlo, cuda_genetic_algorithm, enhanced_hierarchical_gpu) will not be available.", file=sys.stderr)
        print(f"  To enable: pip install cupy-cuda11x (or cupy-cuda12x for CUDA 12)", file=sys.stderr)

# Register comprehensive scoring functions
engine.register_scoring_function('physics_based', PhysicsBasedScoring())
engine.register_scoring_function('empirical', EmpiricalScoring())
engine.register_scoring_function('precision_score', PrecisionScoring())
engine.register_scoring_function('hybrid', HybridScoring())  # True hybrid scoring

# Register GPU scoring functions if available
try:
    from .docking.scoring.gpu_scoring import GPUPrecisionScoring, GPUMMGBSAScoring, GPUScoringManager
    engine.register_scoring_function('gpu_precision', GPUPrecisionScoring())
    engine.register_scoring_function('gpu_mmgbsa', GPUMMGBSAScoring())
    GPU_SCORING_AVAILABLE = True
except ImportError:
    GPU_SCORING_AVAILABLE = False


def show_professional_help():
    """Display professional help banner similar to AutoDock Vina"""
    help_text = """
██████╗  █████╗ ███╗   ██╗██████╗  █████╗ ██████╗  ██████╗  ██████╗██╗  ██╗
██╔══██╗██╔══██╗████╗  ██║██╔══██╗██╔══██╗██╔══██╗██╔═══██╗██╔════╝██║ ██╔╝
██████╔╝███████║██╔██╗ ██║██║  ██║███████║██║  ██║██║   ██║██║     █████╔╝
██╔═══╝ ██╔══██║██║╚██╗██║██║  ██║██╔══██║██║  ██║██║   ██║██║     ██╔═██╗
██║     ██║  ██║██║ ╚████║██████╔╝██║  ██║██████╔╝╚██████╔╝╚██████╗██║  ██╗
╚═╝     ╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝ ╚═╝  ╚═╝╚═════╝  ╚═════╝  ╚═════╝╚═╝  ╚═╝

                 Next-Generation Molecular Docking Suite
               https://github.com/pritampanda15/PandaDock

Author: Pritam Kumar Panda @ Stanford University
Version: 1.0.0
License: MIT

PandaDock is a high-performance molecular docking software with
GPU acceleration and advanced scoring functions for drug discovery.

USAGE:
    pandadock COMMAND [OPTIONS]

MAIN COMMANDS:
    dock                Perform molecular docking with advanced algorithms
    list-algorithms     List available docking algorithms and scoring functions

OTHER TOOLS:
    pandadock-prepare   Prepare ligands for docking (add hydrogens, 3D coordinates)
    pandadock-gridbox   Generate grid box configurations
    pandadock-flex      Perform flexible/induced-fit docking
    pandadock-report    Generate publication-ready analysis plots
    pandadock-metal     Specialized metal docking & metalloprotein docking suite
    pandadock-tethered  Essential for reproducing crystallographic poses and validation

EXAMPLES:
    # Fast, optimized docking (recommended)
    pandadock dock -r protein.pdb -l ligand.sdf --center 10 20 30 --box 20 20 20

    # High-accuracy docking (slower but more thorough)
    pandadock dock -r protein.pdb -l ligand.sdf --center 10 20 30 --box 20 20 20 \\
                   --algorithm enhanced_hierarchical_cpu --scoring physics_based

    # Fast mode for quick testing
    pandadock dock -r protein.pdb -l ligand.sdf --center 10 20 30 --box 20 20 20 \\
                   --algorithm enhanced_hierarchical_cpu_optimized --fast

    # List available algorithms
    pandadock list-algorithms

    # Generate publication plots
    pandadock-report -i results_directory

FEATURES:
    ✓ CPU and GPU-accelerated algorithms
    ✓ Multiple scoring functions (Physics-based, Empirical, Hybrid)
    ✓ Flexible/Induced-fit docking
    ✓ MM-GBSA rescoring
    ✓ Publication-ready visualizations
    ✓ Comprehensive interaction analysis

For detailed help on specific commands, use:
    pandadock COMMAND --help

#################################################################
"""
    click.echo(help_text)

@click.group(invoke_without_command=True, add_help_option=False)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--version', is_flag=True, help='Show version information')
@click.option('--help', '-h', is_flag=True, help='Show this help message')
@click.pass_context
def main(ctx, verbose, version, help):
    """PandaDock - Next-Generation Molecular Docking with Enhanced Accuracy"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if version:
        click.echo("PandaDock v1.0.0 - Pritam Kumar Panda @ Stanford University")
        return

    if help:
        show_professional_help()
        return

    # If no command is provided, show professional help
    if ctx.invoked_subcommand is None:
        show_professional_help()


@main.command()
@click.option('--receptor', '-r', required=True, type=click.Path(exists=True),
              help='Receptor PDB file')
@click.option('--ligand', '-l', required=True, type=click.Path(exists=True),
              help='Ligand file (SDF/MOL2/PDB)')
@click.option('--grid-config', '-g', type=click.Path(exists=True),
              help='Grid box configuration JSON file')
@click.option('--center', nargs=3, type=float, metavar='X Y Z',
              help='Grid center coordinates (Å)')
@click.option('--box', nargs=3, type=float, metavar='X Y Z',
              help='Grid box dimensions (Å)')
@click.option('--algorithm', '-a', type=click.Choice([
    'auto', 'monte_carlo_cpu', 'genetic_algorithm_cpu', 'hierarchical_cpu',
    'enhanced_hierarchical_cpu', 'crystal_guided_cpu', 'cuda_monte_carlo', 'cuda_genetic_algorithm',
    'enhanced_hierarchical_gpu'
]), default='auto', help='Docking algorithm to use (auto = intelligent selection)')
@click.option('--gpu', is_flag=True, default=False,
              help='Use GPU acceleration (requires CUDA)')
@click.option('--gpu-batch-size', type=int, default=1000,
              help='Batch size for GPU processing')
@click.option('--gpu-memory-limit', type=float, default=4.0,
              help='GPU memory limit in GB')
@click.option('--scoring', '-s', type=click.Choice([
    'physics_based', 'empirical', 'precision_score', 'hybrid', 'gpu_precision', 'gpu_mmgbsa'
]), default='physics_based', help='Scoring function to use')
@click.option('--output-dir', '-o', type=click.Path(), default='docking_output',
              help='Output directory')
@click.option('--num-poses', '-n', type=int, default=20,
              help='Number of poses to generate')
@click.option('--ensemble', is_flag=True, default=True,
              help='Use Boltzmann ensemble averaging')
@click.option('--rescoring', type=click.Choice(['none', 'mmgbsa']), default='none',
              help='Rescoring method for top poses')
@click.option('--visualize', is_flag=True, default=True,
              help='Generate visualization plots')
@click.option('--fast', is_flag=True, default=False,
              help='Use fast mode with reduced sampling for quick testing')
@click.option('--cpuworkers', type=int, default=None,
              help='Number of CPU worker threads/processes for parallel execution')
@click.option('--gpuid', type=int, default=0,
              help='GPU device ID to use (default: 0)')
def dock(receptor, ligand, grid_config, center, box, algorithm, scoring,
         output_dir, num_poses, ensemble, rescoring, visualize, fast,
         gpu, gpu_batch_size, gpu_memory_limit, cpuworkers, gpuid):
    """Perform molecular docking"""

    click.echo("PandaDock Enhanced Molecular Docking")
    click.echo(f"Receptor: {receptor}")
    click.echo(f"Ligand: {ligand}")

    # Detect batch mode and auto-select algorithm if needed
    num_ligands, batch_mode = detect_batch_mode(ligand)
    original_algorithm = algorithm

    if algorithm == 'auto':
        # Intelligent algorithm selection
        algorithm = AlgorithmSelector.auto_select(
            num_ligands=num_ligands,
            conformers_per_ligand=10,  # Default, can be detected from ligand
            gpu_available=GPU_ALGORITHMS_REGISTERED,
            user_preference='balanced',
            batch_mode=batch_mode
        )
        click.echo(f"\n{AlgorithmSelector.explain_selection(algorithm, num_ligands, 10, GPU_ALGORITHMS_REGISTERED)}")
        click.echo(f"{AlgorithmSelector.get_recommendation(algorithm, num_ligands, GPU_ALGORITHMS_REGISTERED)}\n")
    else:
        # Show warning for problematic algorithms
        warning = AlgorithmSelector.get_algorithm_warning(algorithm)
        if warning:
            click.echo(f"\n{warning}\n")

    click.echo(f"Algorithm: {algorithm}")
    click.echo(f"Scoring: {scoring}")

    # Validate grid specification
    if not grid_config and not (center and box):
        click.echo("Error: Must specify either --grid-config or --center/--box")
        sys.exit(1)

    # Load grid configuration
    if grid_config:
        with open(grid_config, 'r') as f:
            grid_data = json.load(f)
        grid_center = np.array(grid_data['center'])
        grid_dimensions = np.array(grid_data['dimensions'])
    else:
        grid_center = np.array(center)
        grid_dimensions = np.array(box)

    # GPU validation
    if gpu or algorithm.startswith('cuda') or scoring.startswith('gpu_'):
        if not GPU_AVAILABLE:
            click.echo(f"Error: GPU acceleration requested but CUDA not available")
            click.echo("Please install CuPy and ensure CUDA drivers are properly configured")
            sys.exit(1)
        else:
            click.echo(f"GPU acceleration enabled: batch_size={gpu_batch_size}, memory_limit={gpu_memory_limit}GB")

    # Force GPU algorithm if --gpu flag is used
    if gpu and not algorithm.endswith('_gpu') and not algorithm.startswith('cuda'):
        if algorithm == 'monte_carlo_cpu':
            algorithm = 'cuda_monte_carlo'
        elif algorithm == 'genetic_algorithm_cpu':
            algorithm = 'cuda_genetic_algorithm'
        elif algorithm == 'enhanced_hierarchical_cpu':
            algorithm = 'enhanced_hierarchical_gpu'
        elif algorithm == 'induced_fit_cpu':
            algorithm = 'induced_fit_gpu'
        click.echo(f"GPU flag detected - using {algorithm}")

    # Check algorithm availability
    available_algorithms = engine.list_algorithms()
    if algorithm not in available_algorithms:
        if algorithm.startswith('cuda') and not GPU_AVAILABLE:
            click.echo(f"Error: {algorithm} requires CUDA but GPU support not available")
            click.echo("Available CPU algorithms: monte_carlo_cpu, genetic_algorithm_cpu, hierarchical_cpu, enhanced_hierarchical_cpu, crystal_guided_cpu")
        else:
            click.echo(f"Error: Unknown algorithm: {algorithm}")
            click.echo(f"Available algorithms: {', '.join(available_algorithms)}")
        sys.exit(1)

    # Load ligand
    try:
        ligand_mol = load_ligand(ligand)
        if ligand_mol is None:
            click.echo(f"Error: Could not load ligand from {ligand}")
            sys.exit(1)
    except Exception as e:
        click.echo(f"Error loading ligand: {e}")
        sys.exit(1)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set ligand name
    ligand_name = Path(ligand).stem
    if not ligand_mol.HasProp('_Name'):
        ligand_mol.SetProp('_Name', ligand_name)

    click.echo(f"Grid center: ({grid_center[0]:.1f}, {grid_center[1]:.1f}, {grid_center[2]:.1f}) A")
    click.echo(f"Grid dimensions: ({grid_dimensions[0]:.1f}, {grid_dimensions[1]:.1f}, {grid_dimensions[2]:.1f}) A")
    click.echo(f"Output directory: {output_path}")

    # Set up docking parameters
    dock_params = {}
    if fast:
        click.echo("Fast mode enabled - reduced sampling for quick testing")
        dock_params.update({
            'fast': True,  # Pass fast flag to algorithms
            'num_conformers': 3,
            'num_poses_per_conformer': 10,  # Increased for better pose diversity
            'max_attempts': 100,  # More attempts to find good poses
            'energy_threshold': 20.0,  # Relaxed threshold to allow initial poses (will improve them later)
            'temperature_schedule': [50, 25],  # Gentle refinement with lower temperatures
            'refinement_steps': 10  # Minimal refinement to resolve clashes while preserving positions
        })

    # Add CPU/GPU resource control parameters
    if cpuworkers is not None:
        dock_params['cpuworkers'] = cpuworkers
        click.echo(f"CPU workers: {cpuworkers}")

    if gpu or algorithm.startswith('cuda') or scoring.startswith('gpu_'):
        dock_params.update({
            'batch_size': gpu_batch_size,
            'memory_limit_gb': gpu_memory_limit,
            'gpu_acceleration': True,
            'gpuid': gpuid
        })
        click.echo(f"GPU device ID: {gpuid}")
    elif gpuid != 0:
        # GPU ID specified but not using GPU algorithms
        dock_params['gpuid'] = gpuid
        click.echo(f"GPU device ID: {gpuid} (for potential GPU scoring functions)")

    # Perform docking
    click.echo("Starting docking...")
    start_time = time.time()

    try:
        result = engine.dock_ligand(
            receptor_file=receptor,
            ligand_mol=ligand_mol,
            grid_center=grid_center,
            grid_dimensions=grid_dimensions,
            algorithm=algorithm,
            scoring_function=scoring,
            num_poses=num_poses,
            **dock_params
        )

        runtime = time.time() - start_time
        click.echo(f"Docking completed in {runtime:.2f} seconds")

    except Exception as e:
        click.echo(f"Docking failed: {e}")
        sys.exit(1)

    if not result.poses:
        click.echo("No poses generated")
        sys.exit(1)

    click.echo(f"Generated {len(result.poses)} poses")

    # Ensemble averaging
    if ensemble:
        click.echo("Calculating Boltzmann ensemble averages...")
        boltzmann = BoltzmannEnsemble()
        boltzmann.update_docking_result_with_ensemble(result)
        click.echo(f"Ensemble binding energy: {result.ensemble_binding_energy:.3f} kcal/mol")
        click.echo(f"Ensemble confidence: {result.ensemble_confidence:.3f}")

    # Save results
    click.echo("Saving results...")
    result.save_results(output_path)

    # Save PDB outputs
    save_complex_pdbs(result, receptor, output_path, ligand_mol)
    save_pose_pdbs(result, output_path, ligand_mol)

    # MM-GBSA Rescoring for top poses
    if rescoring == 'mmgbsa':
        click.echo("Performing MM-GBSA rescoring...")
        try:
            mmgbsa = MMGBSARescoring()
            # Load receptor structure for MM-GBSA
            from Bio.PDB import PDBParser
            parser = PDBParser(QUIET=True)
            receptor_structure = parser.get_structure("receptor", receptor)

            top_poses = result.get_top_poses(5)
            for i, pose in enumerate(top_poses):
                rescore = mmgbsa.calculate_binding_free_energy(
                    pose, ligand_mol, receptor_structure
                )
                pose.mmgbsa_score = rescore
                click.echo(f"  Pose {i+1}: MM-GBSA = {rescore:.3f} kcal/mol")
        except Exception as e:
            click.echo(f"MM-GBSA rescoring failed: {e}")

    # Interaction Analysis
    click.echo("Performing interaction analysis...")
    try:
        # Load receptor structure for interaction analysis
        from Bio.PDB import PDBParser
        parser = PDBParser(QUIET=True)
        receptor_structure = parser.get_structure("receptor", receptor)

        analyzer = InteractionAnalyzer()
        top_pose = result.get_top_poses(1)[0]
        interactions = analyzer.analyze_pose_interactions(
            top_pose.coordinates, receptor_structure=receptor_structure, ligand_mol=ligand_mol
        )

        # Save interaction analysis
        interaction_file = output_path / "interaction_analysis.json"
        with open(interaction_file, 'w') as f:
            json.dump(interactions, f, indent=2)
        click.echo(f"Interaction analysis saved to: {interaction_file}")
    except Exception as e:
        click.echo(f"Interaction analysis failed: {e}")

    # Generate visualizations
    if visualize:
        click.echo("Generating visualizations...")
        try:
            plotter = AffinityPlotter()
            plotter.create_binding_affinity_plot(
                result, str(output_path / "binding_affinities.png")
            )
            click.echo("Binding affinity plot saved")

            # Create interaction energy plot if available
            if hasattr(result, 'interaction_energies') and result.interaction_energies:
                plotter.create_interaction_energy_plot(
                    result, str(output_path / "interaction_energies.png")
                )
                click.echo("Interaction energy plot saved")
        except Exception as e:
            click.echo(f"Visualization failed: {e}")

    # Show top results
    top_poses = result.get_top_poses(5)
    click.echo("\nTop 5 poses:")
    for i, pose in enumerate(top_poses, 1):
        click.echo(f"  {i}. Energy: {pose.energy:.3f} kcal/mol, "
                  f"Confidence: {pose.confidence:.3f}")

    click.echo(f"\nDocking complete! Results saved to: {output_path}")


@main.command()
def list_algorithms():
    """List available docking algorithms"""
    click.echo("PandaDock Available Docking Algorithms")
    click.echo("\nCPU Algorithms:")
    cpu_algorithms = [alg for alg in engine.list_algorithms() if not alg.startswith('cuda')]
    for alg in cpu_algorithms:
        click.echo(f"  - {alg}")

    if GPU_ALGORITHMS_REGISTERED:
        click.echo("\nGPU Algorithms:")
        gpu_algorithms = engine.list_algorithms(gpu_only=True)
        for alg in gpu_algorithms:
            click.echo(f"  - {alg}")
    else:
        if GPU_AVAILABLE:
            click.echo("\nGPU Algorithms: Not available (CUDA libraries not installed or GPU initialization failed)")
        else:
            click.echo("\nGPU Algorithms: Not available (GPU support not compiled)")

    click.echo("\nScoring Functions:")
    for scoring in engine.list_scoring_functions():
        click.echo(f"  - {scoring}")


def load_ligand(ligand_file: str) -> Optional[Chem.Mol]:
    """Load ligand molecule from file"""
    ligand_path = Path(ligand_file)

    try:
        if ligand_path.suffix.lower() == '.sdf':
            mol = Chem.SDMolSupplier(str(ligand_path), removeHs=False)[0]
        elif ligand_path.suffix.lower() == '.mol2':
            mol = Chem.MolFromMol2File(str(ligand_path), removeHs=False)
        elif ligand_path.suffix.lower() == '.pdb':
            mol = Chem.MolFromPDBFile(str(ligand_path), removeHs=False)
        else:
            # Try to guess format
            mol = Chem.MolFromMolFile(str(ligand_path), removeHs=False)

        if mol is None:
            return None

        # Add hydrogens with 3D coordinates if not present
        if mol.GetNumAtoms() < 20:  # Rough check if H are missing
            mol = Chem.AddHs(mol, addCoords=True)

        return mol

    except Exception as e:
        logging.error(f"Error loading ligand {ligand_file}: {e}")
        return None


def save_complex_pdbs(result: DockingResult, receptor_file: str, output_dir: Path, ligand_mol=None):
    """Save protein-ligand complex PDB files"""
    try:
        visualizer = DockingVisualizer()

        top_poses = result.get_top_poses(10)  # Save top 10 complexes

        for i, pose in enumerate(top_poses, 1):
            complex_file = output_dir / f"complex{i}.pdb"
            visualizer.save_complex_pdb(
                receptor_file, pose, complex_file, ligand_mol
            )

        logging.info(f"Saved {len(top_poses)} complex PDB files")

    except Exception as e:
        logging.error(f"Error saving complex PDBs: {e}")


def save_pose_pdbs(result: DockingResult, output_dir: Path, ligand_mol=None):
    """Save individual ligand pose PDB files"""
    try:
        visualizer = DockingVisualizer()

        top_poses = result.get_top_poses(10)  # Save top 10 poses

        for i, pose in enumerate(top_poses, 1):
            pose_file = output_dir / f"pose{i}.pdb"
            visualizer.save_pose_pdb(pose, pose_file, ligand_mol)

        logging.info(f"Saved {len(top_poses)} pose PDB files")

    except Exception as e:
        logging.error(f"Error saving pose PDBs: {e}")


if __name__ == '__main__':
    main()