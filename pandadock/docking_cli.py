#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PandaDock Enhanced Molecular Docking CLI

Main Commands:
- dock: Traditional hierarchical docking with Vina-style scoring
- hybrid: Dock + GNN rescoring for best accuracy (recommended)

GNN Commands:
- gnn train: Train SE(3)-equivariant GNN scoring model
- gnn predict: Predict binding affinity with trained model
- gnn benchmark: Benchmark GNN model performance
- gnn compare: Compare against baseline methods
- gnn rescore: Universal rescorer for poses from any docking tool
- gnn download-model: Download pre-trained model from GitHub

Other Tools (separate CLIs):
- pandadock-prepare: Prepare ligands for docking
- pandadock-gridbox: Generate grid box configurations
- pandadock-flex: Flexible/induced-fit docking
- pandadock-report: Generate publication-ready analysis plots
- pandadock-metal: Specialized metal docking suite
- pandadock-tethered: Tethered docking for pose validation
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

# Import PandaDock modules
from .docking.algorithms import HierarchicalDocker
from .docking.scoring.vina_scoring import VinaScoring
from .docking.scoring.physics_based import PhysicsBasedScoring
from .docking.scoring.ensemble import BoltzmannEnsemble
from .docking.refinement.mmgbsa import MMGBSARescoring
from .docking.analysis.interaction_analysis import InteractionAnalyzer
from .docking.core import DockingResult, Pose, DockingEngine

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

# Initialize docking engine with simplified algorithms
engine = DockingEngine()
engine.register_algorithm(HierarchicalDocker())

# Register scoring functions
engine.register_scoring_function('vina', VinaScoring())
engine.register_scoring_function('physics_based', PhysicsBasedScoring())


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
Version: 4.0.0
License: MIT

PandaDock is a high-performance molecular docking software featuring
an SE(3)-equivariant GNN scoring function that achieves R > 0.8 correlation
with experimental binding affinities.

USAGE:
    pandadock COMMAND [OPTIONS]

MAIN COMMANDS:
    dock                Traditional docking with Vina-style scoring
    hybrid              Dock + GNN rescoring (RECOMMENDED for best accuracy)
    list-algorithms     List available algorithms and scoring functions

GNN COMMANDS (Machine Learning):
    gnn train           Train SE(3)-equivariant GNN on protein-ligand dataset
    gnn predict         Predict binding affinity using trained GNN model
    gnn benchmark       Benchmark GNN model performance
    gnn compare         Compare GNN vs baseline scoring methods
    gnn rescore         Universal rescorer for poses from ANY docking tool
    gnn download-model  Download pre-trained model (~82 MB)

OTHER TOOLS:
    pandadock-prepare   Prepare ligands for docking (add hydrogens, 3D coordinates)
    pandadock-gridbox   Generate grid box configurations
    pandadock-flex      Perform flexible/induced-fit docking
    pandadock-report    Generate publication-ready analysis plots
    pandadock-metal     Specialized metal docking & metalloprotein docking suite
    pandadock-tethered  Tethered docking for reproducing crystallographic poses

EXAMPLES:
    # Traditional docking with Vina scoring
    pandadock dock -r protein.pdb -l ligand.sdf --center 10 20 30 --box 20 20 20

    # Hybrid docking with GNN rescoring (recommended)
    pandadock hybrid -r protein.pdb -l ligand.sdf --center 10 20 30 --box 20 20 20 \\
                     --model models/best_model.pt

    # Train GNN model
    pandadock gnn train --dataset ULVSH/ --output models/

    # Compare GNN vs baselines
    pandadock gnn compare --model models/best_model.pt --dataset ULVSH/ --output results/

FEATURES:
    ✓ SE(3)-Equivariant GNN scoring (R > 0.8)
    ✓ Vina-style empirical scoring
    ✓ Hierarchical multi-resolution search
    ✓ Hybrid dock + rescore workflow
    ✓ Publication-ready visualizations

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
    """PandaDock - Next-Generation Molecular Docking with SE(3)-Equivariant GNN"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if version:
        from pandadock import __version__
        click.echo(f"PandaDock v{__version__} - Pritam Kumar Panda @ Stanford University")
        click.echo("Featuring SE(3)-Equivariant GNN Scoring Function")
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
@click.option('--scoring', '-s', type=click.Choice(['vina', 'physics_based']),
              default='vina', help='Scoring function to use (default: vina)')
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
@click.option('--exhaustiveness', '-e', type=int, default=None,
              help='Independent search runs; higher is more thorough. '
                   'Default scales with ligand flexibility (8-32).')
@click.option('--seed', type=int, default=None,
              help='Random seed for reproducible runs (default: random)')
@click.option('--rigid-ligand', is_flag=True, default=False,
              help='Disable torsional search and dock the ligand rigidly')
@click.option('--grid-spacing', type=float, default=0.375,
              help='Affinity grid spacing in Angstrom (default: 0.375)')
def dock(receptor, ligand, grid_config, center, box, scoring,
         output_dir, num_poses, ensemble, rescoring, visualize, fast,
         exhaustiveness, seed, rigid_ligand, grid_spacing):
    """Perform molecular docking with Vina-style scoring"""

    click.echo("PandaDock Molecular Docking")
    click.echo(f"Receptor: {receptor}")
    click.echo(f"Ligand: {ligand}")
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

    click.echo(f"Grid center: ({grid_center[0]:.1f}, {grid_center[1]:.1f}, {grid_center[2]:.1f}) Å")
    click.echo(f"Grid dimensions: ({grid_dimensions[0]:.1f}, {grid_dimensions[1]:.1f}, {grid_dimensions[2]:.1f}) Å")
    click.echo(f"Output directory: {output_path}")

    # Set up docking parameters
    dock_params = {
        'exhaustiveness': exhaustiveness,
        'seed': seed,
        'rigid_ligand': rigid_ligand,
        'grid_spacing': grid_spacing,
    }
    if fast:
        click.echo("Fast mode enabled - reduced sampling for quick testing")
        click.echo("  WARNING: fast mode under-samples badly and its poses should "
                   "not be reported as results.")
        # Fast mode is for smoke-testing a setup, not for producing results.
        dock_params['exhaustiveness'] = 2
        dock_params['n_steps'] = 40

    shown = dock_params['exhaustiveness']
    click.echo(f"Exhaustiveness: {shown if shown is not None else 'auto (scales with flexibility)'}"
               + (f" | seed: {seed}" if seed is not None else "")
               + (" | rigid ligand" if rigid_ligand else ""))

    # Perform docking
    click.echo("Starting docking...")
    start_time = time.time()

    try:
        result = engine.dock_ligand(
            receptor_file=receptor,
            ligand_mol=ligand_mol,
            grid_center=grid_center,
            grid_dimensions=grid_dimensions,
            algorithm='pandadock',
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
    per_pose_interactions = []
    try:
        from Bio.PDB import PDBParser
        parser = PDBParser(QUIET=True)
        receptor_structure = parser.get_structure("receptor", receptor)

        analyzer = InteractionAnalyzer()

        # Analyse every returned mode, not only the top one: poses that score
        # alike but contact different residues are a real ambiguity, and that is
        # only visible across the set.
        per_pose_interactions = []
        for pose in result.get_top_poses(len(result.poses)):
            per_pose_interactions.append(
                analyzer.analyze_pose_interactions(
                    pose.coordinates,
                    receptor_structure=receptor_structure,
                    ligand_mol=ligand_mol,
                )
            )

        interactions = per_pose_interactions[0] if per_pose_interactions else {}
        interaction_file = output_path / "interaction_analysis.json"
        with open(interaction_file, 'w') as f:
            json.dump(
                {"top_pose": interactions, "all_poses": per_pose_interactions},
                f, indent=2, default=float,
            )
        click.echo(f"Interaction analysis saved to: {interaction_file}")
    except Exception as e:
        click.echo(f"Interaction analysis failed: {e}")

    # Generate visualizations
    if visualize:
        click.echo("Generating visualizations...")
        try:
            from .visualization.report import generate_report

            images = generate_report(
                result,
                output_path,
                ligand_mol=ligand_mol,
                receptor_file=receptor,
                interaction_analyses=per_pose_interactions,
                run_pandamap=True,
            )
            for name in sorted(images):
                click.echo(f"  {name}: {Path(images[name]).name}")
            report_html = output_path / "report.html"
            if report_html.exists():
                click.echo(f"  report: {report_html}")
        except Exception as e:
            click.echo(f"Report generation failed: {e}")

        try:
            plotter = AffinityPlotter()
            plotter.create_binding_affinity_plot(
                result, str(output_path / "binding_affinities.png")
            )
            click.echo("Binding affinity plot saved")
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
@click.option('--model', '-m', required=True, type=click.Path(exists=True),
              help='Path to trained GNN model checkpoint')
@click.option('--output-dir', '-o', type=click.Path(), default='hybrid_output',
              help='Output directory')
@click.option('--num-poses', '-n', type=int, default=50,
              help='Number of poses to generate for rescoring')
@click.option('--top-k', type=int, default=10,
              help='Number of top poses to keep after rescoring')
@click.option('--fast', is_flag=True, default=False,
              help='Use fast mode with reduced sampling')
def hybrid(receptor, ligand, grid_config, center, box, model, output_dir, num_poses, top_k, fast):
    """
    Hybrid docking: Traditional docking + GNN rescoring

    This is the RECOMMENDED workflow for best accuracy. It:
    1. Generates diverse poses using hierarchical docking
    2. Rescores all poses using the SE(3)-equivariant GNN
    3. Ranks poses by GNN-predicted binding affinity
    """
    try:
        from .gnn.scoring import GNNScoring
        from .gnn.data.graph_builder import HeterogeneousGraphBuilder, parse_molecule_file
    except ImportError as e:
        click.echo(f"Error: GNN module not available: {e}")
        click.echo("Install with: pip install torch torch-geometric")
        sys.exit(1)

    click.echo("=" * 60)
    click.echo("PandaDock Hybrid Docking (Dock + GNN Rescore)")
    click.echo("=" * 60)
    click.echo(f"Receptor: {receptor}")
    click.echo(f"Ligand: {ligand}")
    click.echo(f"GNN Model: {model}")

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

    click.echo(f"Grid center: ({grid_center[0]:.1f}, {grid_center[1]:.1f}, {grid_center[2]:.1f}) Å")
    click.echo(f"Grid dimensions: ({grid_dimensions[0]:.1f}, {grid_dimensions[1]:.1f}, {grid_dimensions[2]:.1f}) Å")

    # Phase 1: Traditional Docking
    click.echo("\n--- Phase 1: Generating Poses ---")
    dock_params = {'num_conformers': 5, 'num_poses_per_conformer': 10}
    if fast:
        dock_params = {'num_conformers': 3, 'num_poses_per_conformer': 8}

    start_time = time.time()

    try:
        result = engine.dock_ligand(
            receptor_file=receptor,
            ligand_mol=ligand_mol,
            grid_center=grid_center,
            grid_dimensions=grid_dimensions,
            algorithm='pandadock',
            scoring_function='vina',
            num_poses=num_poses,
            **dock_params
        )
    except Exception as e:
        click.echo(f"Docking failed: {e}")
        sys.exit(1)

    dock_time = time.time() - start_time
    click.echo(f"Generated {len(result.poses)} poses in {dock_time:.1f}s")

    if not result.poses:
        click.echo("No poses generated")
        sys.exit(1)

    # Phase 2: GNN Rescoring
    click.echo("\n--- Phase 2: GNN Rescoring ---")

    gnn_scorer = GNNScoring(model_path=model)
    graph_builder = HeterogeneousGraphBuilder()

    # Parse receptor once
    protein_parsed = parse_molecule_file(receptor)

    # Helper function to convert RDKit mol + pose to ParsedMolecule
    from .gnn.data.mol2_parser import ParsedMolecule, Atom

    def rdkit_pose_to_parsed(mol, pose_coords):
        """Convert RDKit mol with pose coordinates to ParsedMolecule."""
        atoms = []
        for i, atom in enumerate(mol.GetAtoms()):
            coord = pose_coords[i]
            # Map RDKit atom type to SYBYL-like type
            symbol = atom.GetSymbol()
            hyb = atom.GetHybridization()
            hyb_map = {
                Chem.HybridizationType.SP3: '3',
                Chem.HybridizationType.SP2: '2',
                Chem.HybridizationType.SP: '1',
            }
            hyb_str = hyb_map.get(hyb, '3')
            atom_type = f"{symbol}.{hyb_str}" if symbol not in ['H'] else symbol

            parsed_atom = Atom(
                id=i + 1,
                name=atom.GetSymbol(),
                x=float(coord[0]),
                y=float(coord[1]),
                z=float(coord[2]),
                atom_type=atom_type,
                charge=float(atom.GetFormalCharge()),
                residue_name='LIG',
                residue_id=1
            )
            atoms.append(parsed_atom)

        return ParsedMolecule(
            name='ligand',
            atoms=atoms,
            bonds=[],
            num_atoms=len(atoms)
        )

    rescored_poses = []
    rescore_start = time.time()

    for i, pose in enumerate(result.poses):
        try:
            # Convert pose to ParsedMolecule
            ligand_parsed = rdkit_pose_to_parsed(ligand_mol, pose.coordinates)

            # Build graph and predict
            graph = graph_builder.build_graph(protein_parsed, ligand_parsed)
            prediction = gnn_scorer.predict_from_graph(graph)

            rescored_poses.append({
                'pose': pose,
                'gnn_pec50': prediction['pec50'],
                'gnn_energy': prediction['energy'],
                'vina_energy': pose.energy,
                'activity_prob': prediction.get('activity_prob', None)
            })

            if (i + 1) % 10 == 0:
                click.echo(f"  Rescored {i+1}/{len(result.poses)} poses")

        except Exception as e:
            # Skip poses that fail rescoring
            if i == 0:  # Debug first failure
                click.echo(f"  Warning: Rescoring error: {e}")
            pass

    rescore_time = time.time() - rescore_start
    click.echo(f"Rescored {len(rescored_poses)} poses in {rescore_time:.1f}s")

    # Sort by GNN pEC50 (higher = better binding)
    rescored_poses.sort(key=lambda x: x['gnn_pec50'], reverse=True)

    # Keep top-k
    top_poses = rescored_poses[:top_k]

    # Phase 3: Output Results
    click.echo("\n--- Results ---")
    click.echo(f"\nTop {len(top_poses)} poses (ranked by GNN pEC50):")
    click.echo("-" * 70)
    click.echo(f"{'Rank':<6} {'GNN pEC50':<12} {'GNN Energy':<14} {'Vina Energy':<14} {'Activity'}")
    click.echo("-" * 70)

    for i, p in enumerate(top_poses, 1):
        activity = f"{p['activity_prob']:.2f}" if p['activity_prob'] else "N/A"
        click.echo(f"{i:<6} {p['gnn_pec50']:<12.3f} {p['gnn_energy']:<14.3f} {p['vina_energy']:<14.3f} {activity}")

    click.echo("-" * 70)

    # Save results
    import pandas as pd

    results_df = pd.DataFrame([{
        'rank': i + 1,
        'gnn_pec50': p['gnn_pec50'],
        'gnn_energy': p['gnn_energy'],
        'vina_energy': p['vina_energy'],
        'activity_prob': p['activity_prob']
    } for i, p in enumerate(top_poses)])

    results_df.to_csv(output_path / 'hybrid_results.csv', index=False)

    # Save top pose structures
    click.echo("\nSaving pose structures...")
    for i, p in enumerate(top_poses[:5], 1):
        pose_file = output_path / f"pose_{i}_pec50_{p['gnn_pec50']:.2f}.pdb"
        save_pose_to_pdb(p['pose'], pose_file, ligand_mol)

    # Save complexes
    for i, p in enumerate(top_poses[:3], 1):
        complex_file = output_path / f"complex_{i}.pdb"
        save_complex_to_pdb(receptor, p['pose'], complex_file, ligand_mol)

    total_time = time.time() - start_time
    click.echo(f"\nTotal time: {total_time:.1f}s")
    click.echo(f"Results saved to: {output_path}")
    click.echo("\nBest pose:")
    click.echo(f"  pEC50: {top_poses[0]['gnn_pec50']:.3f}")
    click.echo(f"  Predicted binding energy: {top_poses[0]['gnn_energy']:.3f} kcal/mol")


@main.command()
def list_algorithms():
    """List available docking algorithms and scoring functions"""
    click.echo("PandaDock Available Components")

    click.echo("\nDocking Algorithm:")
    click.echo("  - pandadock: Multi-stage hierarchical search")
    click.echo("               (Best traditional algorithm, R ~0.12)")

    click.echo("\nScoring Functions:")
    click.echo("  - vina: Vina-style empirical scoring (default)")
    click.echo("  - physics_based: Physics-based Lennard-Jones + electrostatics")

    click.echo("\nML-Based Scoring (pandadock gnn):")
    click.echo("  - pandadock_gnn: SE(3)-equivariant GNN scoring function")
    click.echo("                   Achieves R > 0.8 on ULVSH benchmark")
    click.echo("                   (RECOMMENDED for best accuracy)")
    click.echo("")
    click.echo("  Commands:")
    click.echo("    pandadock gnn train -d ULVSH/ -o models/")
    click.echo("    pandadock gnn predict -m model.pt -p protein.mol2 -l ligand.mol2")
    click.echo("    pandadock hybrid -r protein.pdb -l ligand.sdf -m model.pt --center X Y Z --box X Y Z")


# =============================================================================
# GNN Subcommand Group
# =============================================================================

@main.group()
def gnn():
    """PandaDock-GNN: SE(3)-Equivariant Neural Network Scoring Function"""
    pass


@gnn.command()
@click.option('--dataset', '-d', type=click.Path(exists=True),
              help='Path to ULVSH dataset directory')
@click.option('--pdbbind', '-p', type=click.Path(exists=True),
              help='Path to PDBbind dataset directory')
@click.option('--bindingdb', '-b', type=click.Path(exists=True),
              help='Path to BindingDB TSV file')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output directory for checkpoints and logs')
@click.option('--epochs', default=100, type=int, help='Number of training epochs')
@click.option('--batch-size', default=32, type=int, help='Batch size')
@click.option('--lr', default=1e-4, type=float, help='Learning rate')
@click.option('--hidden-dim', default=256, type=int, help='Hidden dimension')
@click.option('--num-layers', default=6, type=int, help='Number of EGNN layers')
@click.option('--dropout', default=0.1, type=float, help='Dropout rate')
@click.option('--split', type=click.Choice(['random', 'target']), default='random',
              help='Data split strategy')
@click.option('--patience', default=20, type=int, help='Early stopping patience')
@click.option('--balanced', is_flag=True, help='Use balanced sampling (oversample smaller dataset)')
@click.option('--gpu/--cpu', default=True, help='Use GPU if available')
@click.option('--seed', default=42, type=int, help='Random seed')
def train(dataset, pdbbind, bindingdb, output, epochs, batch_size, lr, hidden_dim, num_layers,
          dropout, split, patience, balanced, gpu, seed):
    """Train PandaDock-GNN on protein-ligand dataset(s).

    Can train on ULVSH, PDBbind, BindingDB, or any combination:

    \b
      pandadock gnn train -d ULVSH/ -o models/              # ULVSH only
      pandadock gnn train -p PDBbind/ -o models/            # PDBbind only
      pandadock gnn train -b bindingdb.tsv -o models/       # BindingDB only
      pandadock gnn train -b bindingdb.tsv -d ULVSH/ -o models/  # BindingDB + ULVSH
    """
    if not dataset and not pdbbind and not bindingdb:
        click.echo("Error: Must provide at least one dataset (-d ULVSH, -p PDBbind, or -b BindingDB)")
        return

    try:
        import torch
        import numpy as np
        import random
        from pathlib import Path
        import json

        from .gnn.data.dataset import ULVSHDataset, create_dataloaders
        from .gnn.data.pdbbind_dataset import PDBbindDataset, CombinedDataset
        from .gnn.data.graph_builder import collate_hetero_graphs
        # BindingDB import - handle if not available
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / 'BindingDB_training'))
            from bindingdb_dataset import BindingDBDataset, collate_hetero
        except ImportError:
            BindingDBDataset = None
        from .gnn.models.pandadock_gnn import PandaDockGNN, ModelConfig
        from .gnn.training.trainer import GNNTrainer, TrainingConfig
        from torch.utils.data import DataLoader, WeightedRandomSampler
    except ImportError as e:
        click.echo(f"Missing GNN dependencies: {e}")
        click.echo("Install with: pip install torch torch-geometric")
        return

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Create output directory
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo("=" * 60)
    click.echo("PandaDock-GNN Training")
    click.echo("=" * 60)
    if dataset:
        click.echo(f"ULVSH Dataset: {dataset}")
    if pdbbind:
        click.echo(f"PDBbind Dataset: {pdbbind}")
    if bindingdb:
        click.echo(f"BindingDB Dataset: {bindingdb}")
    click.echo(f"Output: {output}")
    click.echo(f"Epochs: {epochs}, Batch size: {batch_size}")
    click.echo(f"Hidden dim: {hidden_dim}, Layers: {num_layers}")
    click.echo(f"Device: {'cuda' if gpu and torch.cuda.is_available() else 'cpu'}")
    click.echo("=" * 60)

    # Load dataset(s) - supports any combination of ULVSH, PDBbind, BindingDB
    click.echo("\nLoading dataset(s)...")

    train_datasets = []
    val_datasets = []
    test_datasets = []
    dataset_sizes = {}

    # Load ULVSH if provided
    if dataset:
        click.echo("Loading ULVSH dataset...")
        ulvsh_train = ULVSHDataset(root=dataset, split='train')
        ulvsh_val = ULVSHDataset(root=dataset, split='val')
        ulvsh_test = ULVSHDataset(root=dataset, split='test')
        train_datasets.append(ulvsh_train)
        val_datasets.append(ulvsh_val)
        test_datasets.append(ulvsh_test)
        dataset_sizes['ULVSH'] = len(ulvsh_train)
        click.echo(f"  ULVSH: {len(ulvsh_train)} train, {len(ulvsh_val)} val, {len(ulvsh_test)} test")

    # Load PDBbind if provided
    if pdbbind:
        click.echo("Loading PDBbind dataset...")
        pdbbind_train = PDBbindDataset(root=pdbbind, split='train')
        pdbbind_val = PDBbindDataset(root=pdbbind, split='val')
        pdbbind_test = PDBbindDataset(root=pdbbind, split='test')
        train_datasets.append(pdbbind_train)
        val_datasets.append(pdbbind_val)
        test_datasets.append(pdbbind_test)
        dataset_sizes['PDBbind'] = len(pdbbind_train)
        click.echo(f"  PDBbind: {len(pdbbind_train)} train, {len(pdbbind_val)} val, {len(pdbbind_test)} test")

    # Load BindingDB if provided
    if bindingdb:
        if BindingDBDataset is None:
            click.echo("Error: BindingDB dataset class not found.")
            click.echo("Make sure BindingDB_training/bindingdb_dataset.py exists.")
            return
        click.echo("Loading BindingDB dataset...")
        bindingdb_train = BindingDBDataset(bindingdb, split='train')
        bindingdb_val = BindingDBDataset(bindingdb, split='val')
        bindingdb_test = BindingDBDataset(bindingdb, split='test')
        train_datasets.append(bindingdb_train)
        val_datasets.append(bindingdb_val)
        test_datasets.append(bindingdb_test)
        dataset_sizes['BindingDB'] = len(bindingdb_train)
        click.echo(f"  BindingDB: {len(bindingdb_train)} train, {len(bindingdb_val)} val, {len(bindingdb_test)} test")

    # Create data loaders
    if len(train_datasets) > 1:
        # Combined training on multiple datasets
        click.echo("Creating combined dataset...")
        train_combined = CombinedDataset(train_datasets, normalize=False)
        val_combined = CombinedDataset(val_datasets, normalize=False)
        test_combined = CombinedDataset(test_datasets, normalize=False)
        click.echo(f"  Combined: {len(train_combined)} train, {len(val_combined)} val, {len(test_combined)} test")

        if balanced and len(dataset_sizes) > 1:
            # Create weights for balanced sampling
            total = sum(dataset_sizes.values())
            n_datasets = len(dataset_sizes)
            weights = []
            for name, size in dataset_sizes.items():
                weight = total / (n_datasets * size)
                weights.extend([weight] * size)
                click.echo(f"  Balanced sampling: {name} weight={weight:.2f}")

            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            train_loader = DataLoader(train_combined, batch_size=batch_size, sampler=sampler, collate_fn=collate_hetero_graphs)
        else:
            train_loader = DataLoader(train_combined, batch_size=batch_size, shuffle=True, collate_fn=collate_hetero_graphs)

        val_loader = DataLoader(val_combined, batch_size=batch_size, shuffle=False, collate_fn=collate_hetero_graphs)
        test_loader = DataLoader(test_combined, batch_size=batch_size, shuffle=False, collate_fn=collate_hetero_graphs)

    elif bindingdb:
        # BindingDB only - use its own collate function
        train_loader = DataLoader(train_datasets[0], batch_size=batch_size, shuffle=True, collate_fn=collate_hetero)
        val_loader = DataLoader(val_datasets[0], batch_size=batch_size, shuffle=False, collate_fn=collate_hetero)
        test_loader = DataLoader(test_datasets[0], batch_size=batch_size, shuffle=False, collate_fn=collate_hetero)

    elif pdbbind:
        # PDBbind only
        train_loader = DataLoader(train_datasets[0], batch_size=batch_size, shuffle=True, collate_fn=collate_hetero_graphs)
        val_loader = DataLoader(val_datasets[0], batch_size=batch_size, shuffle=False, collate_fn=collate_hetero_graphs)
        test_loader = DataLoader(test_datasets[0], batch_size=batch_size, shuffle=False, collate_fn=collate_hetero_graphs)

    else:
        # ULVSH only (original behavior)
        train_loader, val_loader, test_loader = create_dataloaders(
            root=dataset,
            batch_size=batch_size,
            split_type=split,
            seed=seed
        )

    # Create model
    config = ModelConfig(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        predict_activity=True
    )
    model = PandaDockGNN(config)

    click.echo(f"\nModel parameters: {model.count_parameters():,}")

    # Create trainer
    train_config = TrainingConfig(
        learning_rate=lr,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        checkpoint_dir=str(output_dir),
        device='cuda' if gpu else 'cpu'
    )
    trainer = GNNTrainer(model, train_config)

    # Train
    click.echo("\nStarting training...")
    results = trainer.train(train_loader, val_loader, test_loader)

    # Save results
    results_file = output_dir / 'training_results.json'
    with open(results_file, 'w') as f:
        json_results = {
            'best_metrics': {k: float(v) for k, v in results['best_metrics'].items()},
            'elapsed_time': results['elapsed_time']
        }
        if results['test_metrics']:
            json_results['test_metrics'] = {k: float(v) for k, v in results['test_metrics'].items()}
        json.dump(json_results, f, indent=2)

    click.echo(f"\nResults saved to {results_file}")
    click.echo("\nTraining complete!")


@gnn.command()
@click.option('--model', '-m', required=True, type=click.Path(exists=True),
              help='Path to trained model checkpoint')
@click.option('--protein', '-p', required=True, type=click.Path(exists=True),
              help='Protein MOL2/PDB file')
@click.option('--ligand', '-l', required=True, type=click.Path(exists=True),
              help='Ligand MOL2/SDF file')
@click.option('--site', '-s', type=click.Path(exists=True),
              help='Optional binding site MOL2 file')
@click.option('--output', '-o', type=click.Path(), help='Output JSON file')
def predict(model, protein, ligand, site, output):
    """Predict binding affinity for a protein-ligand complex."""
    try:
        import json
        from .gnn.scoring import GNNScoring
    except ImportError as e:
        click.echo(f"Missing GNN dependencies: {e}")
        click.echo("Install with: pip install torch torch-geometric")
        return

    click.echo("Loading model...")
    scorer = GNNScoring(model_path=model)

    click.echo("Predicting...")
    result = scorer.predict_affinity(protein, ligand, site)

    click.echo("\nPrediction Results:")
    click.echo(f"  pEC50: {result['pec50']:.3f}")
    click.echo(f"  Energy: {result['energy']:.3f} kcal/mol")
    if 'activity_prob' in result:
        click.echo(f"  Activity probability: {result['activity_prob']:.3f}")
        click.echo(f"  Predicted active: {result['active']}")

    if output:
        import json
        with open(output, 'w') as f:
            json.dump(result, f, indent=2)
        click.echo(f"\nResults saved to {output}")


@gnn.command()
@click.option('--model', '-m', required=True, type=click.Path(exists=True),
              help='Path to trained model checkpoint')
@click.option('--dataset', '-d', required=True, type=click.Path(exists=True),
              help='Path to ULVSH dataset directory')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output directory for results')
@click.option('--split', default='test', type=click.Choice(['train', 'val', 'test']),
              help='Dataset split to evaluate')
def benchmark(model, dataset, output, split):
    """Benchmark GNN model performance on test set."""
    try:
        import numpy as np
        import json
        from pathlib import Path

        from .gnn.data.dataset import ULVSHDataset
        from .gnn.scoring import GNNScoring
        from .gnn.training.metrics import compute_affinity_metrics
    except ImportError as e:
        click.echo(f"Missing GNN dependencies: {e}")
        click.echo("Install with: pip install torch torch-geometric pandas")
        return

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo("Loading model...")
    scorer = GNNScoring(model_path=model)

    click.echo(f"Loading {split} dataset...")
    test_dataset = ULVSHDataset(root=dataset, split=split)

    click.echo(f"Evaluating {len(test_dataset)} complexes...")

    predictions = []
    targets = []
    compound_ids = []

    for i in range(len(test_dataset)):
        try:
            compound = test_dataset.get_compound_info(i)
            result = scorer.predict_affinity(
                compound.protein_mol2,
                compound.ligand_mol2,
                compound.site_mol2
            )
            predictions.append(result['pec50'])
            targets.append(compound.pec50)
            compound_ids.append(compound.compound_id)

            if (i + 1) % 100 == 0:
                click.echo(f"  Processed {i + 1}/{len(test_dataset)}")
        except Exception as e:
            click.echo(f"  Warning: Failed for {compound.compound_id}: {e}")

    predictions = np.array(predictions)
    targets = np.array(targets)

    # Compute metrics
    metrics = compute_affinity_metrics(targets, predictions)

    click.echo("\n" + "=" * 50)
    click.echo("Benchmark Results")
    click.echo("=" * 50)
    for key, value in metrics.items():
        click.echo(f"  {key}: {value:.4f}")

    # Save metrics
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

    click.echo(f"\nResults saved to {output_dir}")


@gnn.command()
@click.option('--model', '-m', required=True, type=click.Path(exists=True),
              help='Path to trained model checkpoint')
@click.option('--dataset', '-d', required=True, type=click.Path(exists=True),
              help='Path to ULVSH dataset directory')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output directory for comparison results')
@click.option('--split', default='test', type=click.Choice(['train', 'val', 'test', 'all']),
              help='Dataset split to evaluate')
def compare(model, dataset, output, split):
    """Compare GNN performance against all baseline scoring methods."""
    try:
        import numpy as np
        import pandas as pd
        import json
        from pathlib import Path
        from collections import defaultdict

        from .gnn.data.dataset import ULVSHDataset
        from .gnn.scoring import GNNScoring
        from .gnn.training.metrics import pearson_r, spearman_rho, rmse, mae
    except ImportError as e:
        click.echo(f"Missing dependencies: {e}")
        return

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo("=" * 70)
    click.echo("PandaDock-GNN vs Baseline Methods Comparison")
    click.echo("=" * 70)

    # Load model
    click.echo("\nLoading GNN model...")
    scorer = GNNScoring(model_path=model)

    # Load dataset
    click.echo(f"Loading {split} dataset...")
    if split == 'all':
        test_dataset = ULVSHDataset(root=dataset, split='train')
        all_compounds = test_dataset.compounds
    else:
        test_dataset = ULVSHDataset(root=dataset, split=split)
        all_compounds = [test_dataset.compounds[i] for i in test_dataset.indices]

    click.echo(f"Evaluating {len(all_compounds)} compounds...")

    # Baseline methods to compare
    baseline_methods = [
        'Hyde', 'DeltaVina', 'Gnina', 'MMPBSA', 'MMGBSA', 'VM2', 'GFN-FF', 'PM6'
    ]

    # Get experimental pEC50 values
    click.echo("\nLoading experimental values and baseline scores...")
    experimental_values = {}
    compound_to_target = {}

    for compound in all_compounds:
        compound_key = f"{compound.target}_{compound.compound_id}"
        experimental_values[compound_key] = compound.pec50
        compound_to_target[compound.compound_id] = compound.target

    # Load baseline scores from scores.tsv
    baseline_scores = defaultdict(dict)

    dataset_path = Path(dataset)
    for target in test_dataset.TARGETS:
        scores_file = dataset_path / target / 'raw' / 'scores.tsv'
        if not scores_file.exists():
            continue

        try:
            scores_df = pd.read_csv(scores_file, sep=r'\s+')
        except:
            continue

        for _, row in scores_df.iterrows():
            compound_id = row['ID']
            compound_key = f"{target}_{compound_id}"

            if compound_key not in experimental_values:
                continue

            for method in baseline_methods:
                if method in scores_df.columns:
                    try:
                        score = float(row[method])
                        if not np.isnan(score):
                            baseline_scores[method][compound_key] = score
                    except:
                        pass

    # Run GNN predictions
    click.echo("\nRunning GNN predictions...")
    gnn_predictions = {}
    gnn_pec50 = {}

    for i, compound in enumerate(all_compounds):
        compound_key = f"{compound.target}_{compound.compound_id}"

        try:
            result = scorer.predict_affinity(
                compound.protein_mol2,
                compound.ligand_mol2,
                compound.site_mol2
            )
            gnn_predictions[compound_key] = result['energy']
            gnn_pec50[compound_key] = result['pec50']

            if (i + 1) % 50 == 0:
                click.echo(f"  Processed {i + 1}/{len(all_compounds)}")
        except Exception as e:
            pass

    click.echo(f"\nSuccessfully predicted {len(gnn_predictions)} compounds")

    # Compute metrics
    click.echo("\nComputing comparison metrics...")

    results = []

    # GNN vs experimental pEC50
    common_ids = set(gnn_pec50.keys()) & set(experimental_values.keys())
    if len(common_ids) >= 5:
        y_true = np.array([experimental_values[k] for k in common_ids])
        y_pred = np.array([gnn_pec50[k] for k in common_ids])

        results.append({
            'Method': 'PandaDock-GNN',
            'Type': 'ML Scoring',
            'N': len(common_ids),
            'Pearson_R': pearson_r(y_true, y_pred),
            'Spearman_rho': spearman_rho(y_true, y_pred),
            'RMSE': rmse(y_true, y_pred),
            'MAE': mae(y_true, y_pred)
        })

    # Baseline methods
    for method in baseline_methods:
        if method not in baseline_scores:
            continue

        common_ids = set(baseline_scores[method].keys()) & set(experimental_values.keys())
        if len(common_ids) < 10:
            continue

        y_true = np.array([experimental_values[k] for k in common_ids])
        y_pred = np.array([-baseline_scores[method][k] for k in common_ids])

        results.append({
            'Method': method,
            'Type': 'ULVSH Baseline',
            'N': len(common_ids),
            'Pearson_R': pearson_r(y_true, y_pred),
            'Spearman_rho': spearman_rho(y_true, y_pred),
            'RMSE': np.nan,
            'MAE': np.nan
        })

    # Sort by Pearson R
    results = sorted(results, key=lambda x: x['Pearson_R'], reverse=True)

    # Display results
    click.echo("\n" + "=" * 70)
    click.echo("COMPARISON RESULTS (sorted by Pearson R)")
    click.echo("=" * 70)
    click.echo(f"\n{'Method':<20} {'Type':<15} {'N':>6} {'Pearson R':>12}")
    click.echo("-" * 60)

    for r in results:
        if r['Method'] == 'PandaDock-GNN':
            click.echo(f">>> {r['Method']:<16} {r['Type']:<15} {r['N']:>6} {r['Pearson_R']:>12.4f} <<<")
        else:
            click.echo(f"{r['Method']:<20} {r['Type']:<15} {r['N']:>6} {r['Pearson_R']:>12.4f}")

    click.echo("-" * 60)

    # Find GNN rank
    gnn_rank = next((i + 1 for i, r in enumerate(results) if r['Method'] == 'PandaDock-GNN'), None)
    if gnn_rank:
        click.echo(f"\nPandaDock-GNN Rank: {gnn_rank}/{len(results)}")
        if gnn_rank == 1:
            click.echo("*** PandaDock-GNN achieves BEST performance! ***")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'comparison_results.csv', index=False)

    with open(output_dir / 'comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)

    # Generate plot
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))

        methods = [r['Method'] for r in results]
        pearson_values = [r['Pearson_R'] for r in results]
        colors = ['#2ecc71' if m == 'PandaDock-GNN' else '#3498db' for m in methods]

        bars = ax.barh(range(len(methods)), pearson_values, color=colors)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods)
        ax.set_xlabel('Pearson Correlation (R)')
        ax.set_title('PandaDock-GNN vs Baseline Scoring Methods\n(Higher is Better)')
        ax.axvline(x=0, color='black', linewidth=0.5)

        for i, (bar, val) in enumerate(zip(bars, pearson_values)):
            ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_dir / 'comparison_plot.png', dpi=150, bbox_inches='tight')
        click.echo(f"\nComparison plot saved to {output_dir / 'comparison_plot.png'}")

    except ImportError:
        click.echo("\nNote: Install matplotlib for comparison plots")

    click.echo(f"\nResults saved to {output_dir}")


@gnn.command()
@click.option('--model', '-m', required=True, type=click.Path(exists=True),
              help='Path to trained model checkpoint')
@click.option('--receptor', '-r', required=True, type=click.Path(exists=True),
              help='Receptor file (PDB, MOL2)')
@click.option('--poses', '-p', required=True, type=click.Path(exists=True),
              help='Poses file (multi-conformer SDF from any docking tool)')
@click.option('--output', '-o', type=click.Path(), default='rescored_poses.csv',
              help='Output CSV file with ranked poses')
@click.option('--output-sdf', type=click.Path(),
              help='Optional: Output SDF file with poses ranked by GNN score')
@click.option('--site-radius', type=float, default=10.0,
              help='Radius around ligand centroid to extract binding site (Angstrom)')
def rescore(model, receptor, poses, output, output_sdf, site_radius):
    """
    Universal GNN rescorer for poses from any docking tool.

    Takes docked poses (from pandadock-flex, pandadock-metal, pandadock-tethered,
    or any other docking software like AutoDock Vina, Glide, etc.) and rescores
    them using the SE(3)-equivariant GNN model.

    Examples:
        # Rescore poses from pandadock-flex
        pandadock gnn rescore -m model.pt -r protein.pdb -p flex_poses.sdf

        # Rescore AutoDock Vina output
        pandadock gnn rescore -m model.pt -r receptor.pdb -p vina_out.sdf -o ranked.csv

        # Get ranked SDF with GNN scores as properties
        pandadock gnn rescore -m model.pt -r protein.pdb -p poses.sdf --output-sdf ranked.sdf
    """
    try:
        import numpy as np
        import pandas as pd
        from rdkit import Chem

        from .gnn.scoring import GNNScoring
        from .gnn.data.graph_builder import HeterogeneousGraphBuilder, parse_molecule_file
        from .gnn.data.mol2_parser import ParsedMolecule, Atom, Bond
    except ImportError as e:
        click.echo(f"Missing GNN dependencies: {e}")
        click.echo("Install with: pip install torch torch-geometric rdkit pandas")
        return

    click.echo("=" * 60)
    click.echo("PandaDock-GNN Universal Rescorer")
    click.echo("=" * 60)
    click.echo(f"Model: {model}")
    click.echo(f"Receptor: {receptor}")
    click.echo(f"Poses: {poses}")
    click.echo(f"Site radius: {site_radius} Å")
    click.echo("=" * 60)

    # Load model
    click.echo("\nLoading GNN model...")
    scorer = GNNScoring(model_path=model)
    graph_builder = HeterogeneousGraphBuilder()

    # Load receptor
    click.echo("Parsing receptor...")
    receptor_mol = parse_molecule_file(receptor)

    # Load poses from SDF
    click.echo("Loading poses from SDF...")
    pose_supplier = Chem.SDMolSupplier(poses, removeHs=False)

    all_poses = []
    for i, mol in enumerate(pose_supplier):
        if mol is not None:
            try:
                name = mol.GetProp('_Name') if mol.HasProp('_Name') else f'pose_{i+1}'
            except:
                name = f'pose_{i+1}'
            all_poses.append((name, mol))

    if not all_poses:
        click.echo("Error: No valid poses found in SDF file")
        return

    click.echo(f"Found {len(all_poses)} poses to rescore")

    # Helper functions
    def extract_site(receptor_mol, ligand_centroid, radius):
        """Extract protein atoms within radius of ligand centroid."""
        site_atoms = []
        for atom in receptor_mol.atoms:
            dist = np.sqrt(
                (atom.x - ligand_centroid[0])**2 +
                (atom.y - ligand_centroid[1])**2 +
                (atom.z - ligand_centroid[2])**2
            )
            if dist <= radius:
                site_atoms.append(atom)

        if not site_atoms:
            return receptor_mol

        site = ParsedMolecule(
            name=f"{receptor_mol.name}_site",
            atoms=site_atoms,
            bonds=[]
        )
        site.num_atoms = len(site_atoms)
        return site

    def rdkit_to_parsed(mol, name="ligand"):
        """Convert RDKit molecule to ParsedMolecule."""
        conf = mol.GetConformer()
        atoms = []

        hybridization_map = {
            Chem.HybridizationType.SP: '.1',
            Chem.HybridizationType.SP2: '.2',
            Chem.HybridizationType.SP3: '.3',
        }

        for rdkit_atom in mol.GetAtoms():
            idx = rdkit_atom.GetIdx()
            element = rdkit_atom.GetSymbol()
            pos = conf.GetAtomPosition(idx)

            hyb = rdkit_atom.GetHybridization()
            suffix = hybridization_map.get(hyb, '.3')
            if rdkit_atom.GetIsAromatic():
                sybyl_type = f'{element}.ar'
            else:
                sybyl_type = f'{element}{suffix}'

            atoms.append(Atom(
                id=idx + 1,
                name=f'{element}{idx + 1}',
                x=pos.x, y=pos.y, z=pos.z,
                atom_type=sybyl_type,
                residue_id=1,
                residue_name='LIG',
                charge=0.0
            ))

        bonds = []
        bond_type_map = {
            Chem.BondType.SINGLE: '1',
            Chem.BondType.DOUBLE: '2',
            Chem.BondType.TRIPLE: '3',
            Chem.BondType.AROMATIC: 'ar',
        }
        for bond in mol.GetBonds():
            bt = bond_type_map.get(bond.GetBondType(), '1')
            bonds.append(Bond(
                id=bond.GetIdx() + 1,
                atom1_id=bond.GetBeginAtomIdx() + 1,
                atom2_id=bond.GetEndAtomIdx() + 1,
                bond_type=bt
            ))

        parsed = ParsedMolecule(name=name, atoms=atoms, bonds=bonds)
        parsed.num_atoms = len(atoms)
        parsed.num_bonds = len(bonds)
        return parsed

    # Score each pose
    results = []
    scored_mols = []

    click.echo("\nScoring poses...")
    for i, (name, mol) in enumerate(all_poses):
        try:
            mol = Chem.AddHs(mol, addCoords=True)
            conf = mol.GetConformer()
            coords = np.array([conf.GetAtomPosition(j) for j in range(mol.GetNumAtoms())])
            centroid = coords.mean(axis=0)

            site_mol = extract_site(receptor_mol, centroid, site_radius)
            ligand_parsed = rdkit_to_parsed(mol, name=name)
            graph = graph_builder.build_graph(site_mol, ligand_parsed)
            result = scorer.predict_from_graph(graph)

            results.append({
                'pose_name': name,
                'pose_index': i + 1,
                'gnn_pKd': result['pec50'],
                'gnn_energy': result['energy'],
                'activity_prob': result.get('activity_prob', None),
                'predicted_active': result.get('active', None),
                'site_atoms': site_mol.num_atoms,
                'ligand_atoms': ligand_parsed.num_atoms
            })

            mol.SetProp('GNN_pKd', f"{result['pec50']:.3f}")
            mol.SetProp('GNN_Energy', f"{result['energy']:.3f}")
            if result.get('activity_prob') is not None:
                mol.SetProp('GNN_Activity', f"{result['activity_prob']:.3f}")
            scored_mols.append((result['pec50'], mol))

            if (i + 1) % 10 == 0 or (i + 1) == len(all_poses):
                click.echo(f"  Processed {i + 1}/{len(all_poses)} poses")

        except Exception as e:
            click.echo(f"  Warning: Failed to score pose {name}: {e}")
            results.append({
                'pose_name': name,
                'pose_index': i + 1,
                'gnn_pKd': None,
                'gnn_energy': None,
                'error': str(e)
            })

    # Create results DataFrame
    df = pd.DataFrame(results)
    df_valid = df[df['gnn_pKd'].notna()].copy()
    df_valid = df_valid.sort_values('gnn_pKd', ascending=False)
    df_valid['gnn_rank'] = range(1, len(df_valid) + 1)

    df_valid.to_csv(output, index=False)
    click.echo(f"\nResults saved to: {output}")

    # Save ranked SDF if requested
    if output_sdf:
        scored_mols.sort(key=lambda x: x[0], reverse=True)
        writer = Chem.SDWriter(output_sdf)
        for rank, (score, mol) in enumerate(scored_mols, 1):
            mol.SetProp('GNN_Rank', str(rank))
            writer.write(mol)
        writer.close()
        click.echo(f"Ranked SDF saved to: {output_sdf}")

    # Print summary
    click.echo("\n" + "=" * 60)
    click.echo("RESCORING SUMMARY")
    click.echo("=" * 60)
    click.echo(f"Total poses: {len(all_poses)}")
    click.echo(f"Successfully scored: {len(df_valid)}")
    if len(df_valid) > 0:
        click.echo(f"\nTop 5 poses by GNN score:")
        click.echo("-" * 50)
        for _, row in df_valid.head(5).iterrows():
            click.echo(f"  Rank {int(row['gnn_rank'])}: {row['pose_name']}")
            click.echo(f"    pKd: {row['gnn_pKd']:.3f}, Energy: {row['gnn_energy']:.3f} kcal/mol")
    click.echo("=" * 60)


@gnn.command('download-model')
@click.option('--output', '-o', type=click.Path(), default='models/',
              help='Output directory for the model (default: models/)')
@click.option('--version', '-v', type=str, default='latest',
              help='Model version to download (default: latest)')
@click.option('--force', '-f', is_flag=True, default=False,
              help='Overwrite existing model file')
def download_model(output, version, force):
    """
    Download pre-trained PandaDock-GNN model.

    Downloads the official pre-trained model from GitHub releases.
    The model was trained on ULVSH + PDBbind combined dataset
    and achieves R=0.88 on PDBbind validation.

    Examples:
        pandadock gnn download-model
        pandadock gnn download-model -o /path/to/models/
        pandadock gnn download-model --force
    """
    import urllib.request
    import hashlib

    # GitHub release URLs
    GITHUB_REPO = "pritampanda15/PandaDock"
    MODEL_FILENAME = "pandadock_gnn_v4.pt"

    # Model info
    MODEL_INFO = {
        'v4.0.0': {
            'url': f'https://github.com/{GITHUB_REPO}/releases/download/v4.0.0/{MODEL_FILENAME}',
            'sha256': None,  # Will be set when model is uploaded
            'size_mb': 82,
            'description': 'Combined ULVSH + PDBbind model (200 epochs)',
            'metrics': {
                'pdbbind_pearson_r': 0.88,
                'ulvsh_test_pearson_r': 0.82,
                'gaba_pearson_r': 0.68,
                'ulvsh_activity_auc': 0.94
            }
        }
    }

    # Create output directory
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / MODEL_FILENAME

    # Check if model already exists
    if output_path.exists() and not force:
        click.echo(f"Model already exists at: {output_path}")
        click.echo("Use --force to re-download")
        return

    click.echo("=" * 60)
    click.echo("PandaDock-GNN Model Download")
    click.echo("=" * 60)

    # Get version info
    if version == 'latest':
        version = 'v4.0.0'  # Current latest

    if version not in MODEL_INFO:
        click.echo(f"Error: Unknown version '{version}'")
        click.echo(f"Available versions: {', '.join(MODEL_INFO.keys())}")
        return

    info = MODEL_INFO[version]
    click.echo(f"Version: {version}")
    click.echo(f"Size: ~{info['size_mb']} MB")
    click.echo(f"Description: {info['description']}")
    click.echo(f"\nPerformance:")
    for metric, value in info['metrics'].items():
        click.echo(f"  {metric}: {value}")
    click.echo()

    # Download with progress
    url = info['url']
    click.echo(f"Downloading from: {url}")
    click.echo(f"Saving to: {output_path}")
    click.echo()

    try:
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                bar_length = 40
                filled = int(bar_length * percent / 100)
                bar = '=' * filled + '-' * (bar_length - filled)
                print(f"\r[{bar}] {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='', flush=True)

        urllib.request.urlretrieve(url, output_path, show_progress)
        print("\n")

        # Verify download
        if output_path.exists():
            file_size = output_path.stat().st_size / (1024 * 1024)
            click.echo(f"Download complete! ({file_size:.1f} MB)")

            # Verify SHA256 if available
            if info['sha256']:
                click.echo("Verifying checksum...")
                sha256_hash = hashlib.sha256()
                with open(output_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b''):
                        sha256_hash.update(chunk)
                if sha256_hash.hexdigest() == info['sha256']:
                    click.echo("Checksum verified!")
                else:
                    click.echo("WARNING: Checksum mismatch! File may be corrupted.")

            click.echo("\n" + "=" * 60)
            click.echo("SUCCESS! Model ready to use.")
            click.echo("=" * 60)
            click.echo(f"\nExample usage:")
            click.echo(f"  pandadock gnn predict -m {output_path} -p protein.mol2 -l ligand.mol2")
            click.echo(f"  pandadock gnn rescore -m {output_path} -r protein.pdb -p poses.sdf")
            click.echo(f"  pandadock hybrid -r protein.pdb -l ligand.sdf -m {output_path} --center X Y Z --box X Y Z")
        else:
            click.echo("Error: Download failed - file not found")

    except urllib.error.HTTPError as e:
        if e.code == 404:
            click.echo(f"\nError: Model not found at release URL.")
            click.echo(f"The model may not have been uploaded yet for version {version}.")
            click.echo(f"\nAlternative: Train your own model:")
            click.echo(f"  pandadock gnn train -d ULVSH/ -o models/ --epochs 100")
        else:
            click.echo(f"\nError downloading model: HTTP {e.code}")
            click.echo(f"Please try again later or download manually from:")
            click.echo(f"  {url}")

    except urllib.error.URLError as e:
        click.echo(f"\nError: Could not connect to GitHub.")
        click.echo(f"Please check your internet connection and try again.")
        click.echo(f"\nAlternatively, download manually from:")
        click.echo(f"  https://github.com/{GITHUB_REPO}/releases")

    except Exception as e:
        click.echo(f"\nError: {e}")
        click.echo(f"\nPlease download manually from:")
        click.echo(f"  https://github.com/{GITHUB_REPO}/releases")


# =============================================================================
# Helper Functions
# =============================================================================

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
            mol = Chem.MolFromMolFile(str(ligand_path), removeHs=False)

        if mol is None:
            return None

        if mol.GetNumAtoms() < 20:
            mol = Chem.AddHs(mol, addCoords=True)

        return mol

    except Exception as e:
        logging.error(f"Error loading ligand {ligand_file}: {e}")
        return None


def save_complex_pdbs(result: DockingResult, receptor_file: str, output_dir: Path, ligand_mol=None):
    """Save protein-ligand complex PDB files"""
    try:
        visualizer = DockingVisualizer()
        top_poses = result.get_top_poses(10)

        written = 0
        for i, pose in enumerate(top_poses, 1):
            complex_file = output_dir / f"complex{i}.pdb"
            visualizer.save_complex_pdb(
                receptor_file, pose, complex_file, ligand_mol
            )
            if complex_file.exists():
                written += 1

        # Count files that actually landed on disk. The writer catches its own
        # exceptions, so reporting len(top_poses) claimed success even when every
        # single write had failed and no file existed.
        if written < len(top_poses):
            logging.warning(
                "Saved %d of %d complex PDB files; %d failed to write",
                written, len(top_poses), len(top_poses) - written,
            )
        else:
            logging.info(f"Saved {written} complex PDB files")

    except Exception as e:
        logging.error(f"Error saving complex PDBs: {e}")


def save_pose_pdbs(result: DockingResult, output_dir: Path, ligand_mol=None):
    """Save individual ligand pose PDB files"""
    try:
        visualizer = DockingVisualizer()
        top_poses = result.get_top_poses(10)

        written = 0
        for i, pose in enumerate(top_poses, 1):
            pose_file = output_dir / f"pose{i}.pdb"
            visualizer.save_pose_pdb(pose, pose_file, ligand_mol)
            if pose_file.exists():
                written += 1

        if written < len(top_poses):
            logging.warning(
                "Saved %d of %d pose PDB files; %d failed to write",
                written, len(top_poses), len(top_poses) - written,
            )
        else:
            logging.info(f"Saved {written} pose PDB files")

        # An SDF alongside the PDBs preserves bond orders and formal charges,
        # which PDB cannot represent. Viewers and downstream tools infer bonds
        # from distance when given a ligand PDB, which routinely mis-assigns
        # aromatic rings.
        save_poses_sdf(result, output_dir, ligand_mol)

    except Exception as e:
        logging.error(f"Error saving pose PDBs: {e}")


def save_poses_sdf(result: DockingResult, output_dir: Path, ligand_mol=None):
    """Write all poses to a single SDF, annotated with rank and score."""
    if ligand_mol is None:
        return
    try:
        sdf_path = output_dir / "poses.sdf"
        writer = Chem.SDWriter(str(sdf_path))
        try:
            for rank, pose in enumerate(result.get_top_poses(len(result.poses)), 1):
                mol = Chem.Mol(ligand_mol)
                mol.RemoveAllConformers()
                conf = Chem.Conformer(mol.GetNumAtoms())
                n = min(mol.GetNumAtoms(), len(pose.coordinates))
                for i in range(n):
                    conf.SetAtomPosition(i, pose.coordinates[i].tolist())
                mol.AddConformer(conf, assignId=True)

                mol.SetProp("_Name", f"{result.ligand_name}_pose{rank}")
                mol.SetProp("rank", str(rank))
                mol.SetProp("score_kcal_per_mol", f"{pose.energy:.3f}")
                mol.SetProp("confidence", f"{pose.confidence:.3f}")
                for key, value in (pose.energy_components or {}).items():
                    mol.SetProp(f"energy_{key}", f"{value:.3f}")
                writer.write(mol)
        finally:
            writer.close()
        logging.info(f"Saved poses to {sdf_path}")
    except Exception as e:
        logging.error(f"Error saving poses SDF: {e}")


def save_pose_to_pdb(pose: Pose, filepath: Path, ligand_mol: Chem.Mol):
    """Save a single pose to PDB file"""
    try:
        with open(filepath, 'w') as f:
            f.write(f"REMARK   1 PandaDock Pose\n")
            f.write(f"REMARK   2 Energy: {pose.energy:.3f} kcal/mol\n")

            for i, (coord, atom) in enumerate(zip(pose.coordinates, ligand_mol.GetAtoms()), 1):
                element = atom.GetSymbol()
                name = f"{element}{i}"[:4].ljust(4)
                f.write(f"HETATM{i:5d} {name} LIG A   1    "
                       f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                       f"  1.00  0.00          {element:>2}\n")
            f.write("END\n")
    except Exception as e:
        logging.error(f"Error saving pose PDB: {e}")


def save_complex_to_pdb(receptor_file: str, pose: Pose, filepath: Path, ligand_mol: Chem.Mol):
    """Save protein-ligand complex to PDB file"""
    try:
        # Read receptor
        with open(receptor_file, 'r') as f:
            receptor_lines = [l for l in f if l.startswith(('ATOM', 'HETATM', 'TER'))]

        with open(filepath, 'w') as f:
            f.write(f"REMARK   1 PandaDock Complex\n")
            f.write(f"REMARK   2 Pose Energy: {pose.energy:.3f} kcal/mol\n")

            # Write receptor
            for line in receptor_lines:
                f.write(line)

            f.write("TER\n")

            # Write ligand
            atom_start = 10000
            for i, (coord, atom) in enumerate(zip(pose.coordinates, ligand_mol.GetAtoms()), atom_start):
                element = atom.GetSymbol()
                name = f"{element}{i-atom_start+1}"[:4].ljust(4)
                f.write(f"HETATM{i:5d} {name} LIG B   1    "
                       f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                       f"  1.00  0.00          {element:>2}\n")

            f.write("END\n")
    except Exception as e:
        logging.error(f"Error saving complex PDB: {e}")


if __name__ == '__main__':
    main()
