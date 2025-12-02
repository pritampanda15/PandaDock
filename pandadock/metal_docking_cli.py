#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PandaDock Metal Docking CLI

Specialized CLI for metal-containing ligands and metalloproteins following
Glide and AutoDock protocols for metal coordination.
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

# Import metal docking modules
from .metal_docking.metal_core import MetalDockingEngine, MetalCoordinationGeometry
from .metal_docking.metal_parameters import MetalParameterManager
from .metal_docking.metal_preparation import MetalLigandPreparator, MetalloproteinPreparator
from .metal_docking.metal_scoring import MetalAwareScoring

def show_metal_help():
    """Display professional help for metal docking"""
    help_text = """
██████╗  █████╗ ███╗   ██╗██████╗  █████╗     ███╗   ███╗███████╗████████╗ █████╗ ██╗
██╔══██╗██╔══██╗████╗  ██║██╔══██╗██╔══██╗    ████╗ ████║██╔════╝╚══██╔══╝██╔══██╗██║
██████╔╝███████║██╔██╗ ██║██║  ██║███████║    ██╔████╔██║█████╗     ██║   ███████║██║
██╔═══╝ ██╔══██║██║╚██╗██║██║  ██║██╔══██║    ██║╚██╔╝██║██╔══╝     ██║   ██╔══██║██║
██║     ██║  ██║██║ ╚████║██████╔╝██║  ██║    ██║ ╚═╝ ██║███████╗   ██║   ██║  ██║███████╗
╚═╝     ╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝ ╚═╝  ╚═╝    ╚═╝     ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝

           Specialized Metal Docking & Metalloprotein Docking Suite
               https://github.com/pritampanda15/PandaDock

Author: Pritam Kumar Panda @ Stanford University
Version: 1.0.0
License: MIT

PandaDock-Metal handles metal-containing ligands and metalloproteins
following Glide and AutoDock protocols for accurate metal coordination.

SUPPORTED METALS:
  Alkali: Li, Na, K
  Alkaline Earth: Mg, Ca, Ba
  Transition Metals: Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Mo, Pd, Ag, Cd, Pt, Au, Hg
  Post-transition: Al, Pb
  Lanthanides: La, Gd

USAGE:
    pandadock-metal COMMAND [OPTIONS]

MAIN COMMANDS:
    dock               Perform metal-aware molecular docking
    prepare-ligand     Prepare metal-containing ligands (zero-order bonds)
    analyze-protein    Analyze metalloprotein metal sites
    list-metals        List supported metal types and parameters
    validate           Validate metal ligand preparation

METAL DOCKING FEATURES:
    ✓ Zero-order bond treatment for metal coordination
    ✓ Geometric constraint validation
    ✓ Metal-specific scoring functions
    ✓ Coordination geometry optimization
    ✓ Metalloprotein site analysis
    ✓ Formal charge adjustment
    ✓ Distance-based coordination scoring

EXAMPLES:
    # Basic metal docking
    pandadock-metal dock -r metalloprotein.pdb -l metal_ligand.sdf \\
                         --center 10 20 30 --box 20 20 20

    # Prepare metal ligand with zero-order bonds
    pandadock-metal prepare-ligand -i ligand.sdf -o prepared_ligand.sdf

    # Analyze metalloprotein sites
    pandadock-metal analyze-protein -r metalloprotein.pdb

    # Advanced metal docking with constraints
    pandadock-metal dock -r protein.pdb -l ligand.sdf --center 10 20 30 --box 20 20 20 \\
                         --use-constraints --constraint-weight 15.0 \\
                         --scoring metal_constrained

KEY PROTOCOLS:
  • Single molecule ligands with reasonable coordination geometry
  • Zero-order bonds between metal and coordinating atoms
  • Formal charge adjustment for metals and ligating atoms
  • No post-docking minimization to preserve coordination
  • Metal binding preference based on protein net charge

For detailed help on specific commands, use:
    pandadock-metal COMMAND --help

#################################################################
"""
    click.echo(help_text)

@click.group(invoke_without_command=True, add_help_option=False)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--version', is_flag=True, help='Show version information')
@click.option('--help', '-h', is_flag=True, help='Show this help message')
@click.pass_context
def main(ctx, verbose, version, help):
    """PandaDock-Metal - Specialized Metal Docking Suite"""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if version:
        click.echo("PandaDock-Metal v1.0.0 - Pritam Kumar Panda @ Stanford University")
        return

    if help:
        show_metal_help()
        return

    # If no command is provided, show professional help
    if ctx.invoked_subcommand is None:
        show_metal_help()

@main.command()
@click.option('--receptor', '-r', required=True, type=click.Path(exists=True),
              help='Metalloprotein PDB file')
@click.option('--ligand', '-l', required=True, type=click.Path(exists=True),
              help='Metal-containing ligand file (SDF/MOL2/MOL/PDB)')
@click.option('--grid-config', type=click.Path(exists=True),
              help='JSON file with grid configuration')
@click.option('--center', nargs=3, type=float, metavar='X Y Z',
              help='Grid center coordinates (Å)')
@click.option('--box', nargs=3, type=float, metavar='X Y Z',
              help='Grid box dimensions (Å)')
@click.option('--algorithm', default='enhanced_hierarchical_cpu',
              type=click.Choice([
                  'monte_carlo_cpu', 'genetic_algorithm_cpu', 'hierarchical_cpu',
                  'enhanced_hierarchical_cpu', 'crystal_guided_cpu'
              ]),
              help='Docking algorithm [default: enhanced_hierarchical_cpu]')
@click.option('--scoring', default='metal_aware',
              type=click.Choice(['metal_aware', 'metal_constrained', 'physics_based']),
              help='Scoring function [default: metal_aware]')
@click.option('--output-dir', '-o', default='metal_docking_results',
              help='Output directory [default: metal_docking_results]')
@click.option('--num-poses', default=20, type=int,
              help='Number of poses to generate [default: 20]')
@click.option('--use-constraints', is_flag=True, default=True,
              help='Use geometric constraints for metal coordination [default: True]')
@click.option('--constraint-weight', default=10.0, type=float,
              help='Weight for constraint violations [default: 10.0]')
@click.option('--prepare-ligand', is_flag=True,
              help='Automatically prepare metal ligand (zero-order bonds)')
@click.option('--validate-geometry', is_flag=True, default=True,
              help='Validate coordination geometry [default: True]')
@click.option('--metal-params', type=click.Path(exists=True),
              help='Custom metal parameters file')
@click.option('--verbose', '-v', is_flag=True,
              help='Enable verbose logging')
def dock(receptor, ligand, grid_config, center, box, algorithm, scoring,
         output_dir, num_poses, use_constraints, constraint_weight,
         prepare_ligand, validate_geometry, metal_params, verbose):
    """
    Perform metal-aware molecular docking

    Specialized docking for metal-containing ligands and metalloproteins
    following Glide protocols with zero-order bonds and coordination geometry.

    Examples:
        # Basic metal docking
        pandadock-metal dock -r zinc_protein.pdb -l zinc_ligand.sdf --center 10 20 30 --box 20 20 20

        # With automatic ligand preparation
        pandadock-metal dock -r protein.pdb -l ligand.sdf --center 10 20 30 --box 20 20 20 --prepare-ligand

        # Constrained docking with higher penalty
        pandadock-metal dock -r protein.pdb -l ligand.sdf --center 10 20 30 --box 20 20 20 \\
                             --scoring metal_constrained --constraint-weight 20.0
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    click.echo("🧬 PandaDock Metal-Aware Molecular Docking")
    click.echo("=" * 50)
    click.echo(f"Receptor: {receptor}")
    click.echo(f"Ligand: {ligand}")
    click.echo(f"Algorithm: {algorithm}")
    click.echo(f"Scoring: {scoring}")

    # Validate grid specification
    if not grid_config and not (center and box):
        click.echo("❌ Error: Must specify either --grid-config or --center/--box")
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

    click.echo(f"Grid center: ({grid_center[0]:.1f}, {grid_center[1]:.1f}, {grid_center[2]:.1f}) Å")
    click.echo(f"Grid dimensions: ({grid_dimensions[0]:.1f}, {grid_dimensions[1]:.1f}, {grid_dimensions[2]:.1f}) Å")
    click.echo(f"Output directory: {output_dir}")
    click.echo("")

    # Initialize metal docking engine
    engine = MetalDockingEngine(metal_params)

    try:
        # Load ligand
        click.echo("📥 Loading ligand...")
        if ligand.endswith('.sdf'):
            ligand_mol = Chem.SDMolSupplier(ligand)[0]
        elif ligand.endswith('.mol2'):
            ligand_mol = Chem.MolFromMol2File(ligand)
        elif ligand.endswith('.pdb'):
            ligand_mol = Chem.MolFromPDBFile(ligand)
        else:
            ligand_mol = Chem.MolFromMolFile(ligand)

        if ligand_mol is None:
            click.echo(f"❌ Error: Could not load ligand from {ligand}")
            sys.exit(1)

        # Prepare ligand if requested
        if prepare_ligand:
            click.echo("🔧 Preparing metal-containing ligand...")
            ligand_mol = engine.prepare_metal_ligand(ligand)

        # Validate ligand preparation
        if validate_geometry:
            click.echo("✅ Validating metal ligand preparation...")
            validation = engine.validate_metal_preparation(ligand_mol)

            if not validation['is_valid']:
                click.echo("❌ Ligand validation failed:")
                for error in validation['errors']:
                    click.echo(f"   • {error}")
                sys.exit(1)

            if validation['warnings']:
                click.echo("⚠️  Validation warnings:")
                for warning in validation['warnings']:
                    click.echo(f"   • {warning}")

            if validation['metal_atoms']:
                click.echo(f"🔬 Found {len(validation['metal_atoms'])} metal atoms in ligand")

        # Perform metal-aware docking
        click.echo("🚀 Starting metal-aware docking...")
        start_time = time.time()

        result = engine.dock_with_metal_awareness(
            receptor_file=receptor,
            ligand_mol=ligand_mol,
            grid_center=grid_center,
            grid_dimensions=grid_dimensions,
            algorithm=algorithm,
            scoring_function=scoring,
            num_poses=num_poses,
            use_constraints=use_constraints,
            constraint_weight=constraint_weight
        )

        dock_time = time.time() - start_time

        if not result.poses:
            click.echo("❌ No poses generated")
            sys.exit(1)

        click.echo(f"✅ Docking completed in {dock_time:.2f} seconds")
        click.echo(f"📊 Generated {len(result.poses)} poses")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save results
        click.echo("💾 Saving results...")

        # Save metal docking summary
        metal_summary = result.get_metal_summary()
        with open(output_path / "metal_docking_summary.json", 'w') as f:
            json.dump(metal_summary, f, indent=2)

        # Save detailed results
        result.save_results(output_path)

        # Metal-specific analysis
        if hasattr(result, 'metal_interactions') and result.metal_interactions:
            click.echo("🔬 Performing metal interaction analysis...")

            metal_analysis = {
                'summary': metal_summary,
                'pose_details': []
            }

            for i, (pose, interactions) in enumerate(zip(result.poses[:5], result.metal_interactions[:5])):
                pose_detail = {
                    'pose_id': i + 1,
                    'energy': pose.energy,
                    'confidence': pose.confidence,
                    'metal_interactions': interactions,
                    'coordination_score': result.coordination_scores[i] if i < len(result.coordination_scores) else 0.0,
                    'constraint_violations': result.constraint_violations[i] if i < len(result.constraint_violations) else []
                }
                metal_analysis['pose_details'].append(pose_detail)

            with open(output_path / "metal_interaction_analysis.json", 'w') as f:
                json.dump(metal_analysis, f, indent=2, default=str)

        # Summary
        click.echo("")
        click.echo("📈 Top 5 poses:")
        for i, pose in enumerate(result.poses[:5], 1):
            coord_score = result.coordination_scores[i-1] if i-1 < len(result.coordination_scores) else 0.0
            violations = len(result.constraint_violations[i-1]) if i-1 < len(result.constraint_violations) else 0
            click.echo(f"  {i}. Energy: {pose.energy:.3f} kcal/mol, "
                      f"Confidence: {pose.confidence:.3f}, "
                      f"Metal Score: {coord_score:.3f}, "
                      f"Violations: {violations}")

        if metal_summary:
            click.echo(f"\n🎯 Metal Docking Summary:")
            click.echo(f"   Average coordination score: {metal_summary.get('avg_coordination_score', 0):.3f}")
            click.echo(f"   Average metal binding score: {metal_summary.get('avg_metal_binding_score', 0):.3f}")
            click.echo(f"   Poses with violations: {metal_summary.get('poses_with_violations', 0)}")
            click.echo(f"   Violation rate: {metal_summary.get('violation_rate', 0):.2%}")

        click.echo(f"\n🎉 Metal docking complete! Results saved to: {output_dir}")

    except Exception as e:
        click.echo(f"❌ Metal docking failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

@main.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True),
              help='Input ligand file (SDF/MOL2/MOL/PDB)')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output prepared ligand file')
@click.option('--validate', is_flag=True, default=True,
              help='Validate coordination geometry [default: True]')
@click.option('--metal-params', type=click.Path(exists=True),
              help='Custom metal parameters file')
def prepare_ligand(input, output, validate, metal_params):
    """
    Prepare metal-containing ligand following Glide protocol

    Converts covalent metal-ligand bonds to zero-order bonds and adjusts
    formal charges for proper metal coordination representation.

    Examples:
        pandadock-metal prepare-ligand -i raw_ligand.sdf -o prepared_ligand.sdf
        pandadock-metal prepare-ligand -i ligand.mol2 -o prepared.sdf --validate
    """
    click.echo("🔧 Preparing metal-containing ligand...")
    click.echo(f"Input: {input}")
    click.echo(f"Output: {output}")

    engine = MetalDockingEngine(metal_params)

    try:
        prepared_mol = engine.prepare_metal_ligand(input, output)

        if validate:
            validation = engine.validate_metal_preparation(prepared_mol)

            click.echo("\n✅ Validation Results:")
            click.echo(f"   Valid: {validation['is_valid']}")
            click.echo(f"   Metal atoms: {len(validation['metal_atoms'])}")

            if validation['warnings']:
                click.echo("   Warnings:")
                for warning in validation['warnings']:
                    click.echo(f"     • {warning}")

            if validation['errors']:
                click.echo("   Errors:")
                for error in validation['errors']:
                    click.echo(f"     • {error}")

            for coord_info in validation['coordination_info']:
                click.echo(f"   Metal {coord_info['metal_symbol']} (idx {coord_info['metal_index']}): "
                          f"{coord_info['coordination_number']} coordinating atoms")

        click.echo(f"\n🎉 Ligand preparation complete! Saved to: {output}")

    except Exception as e:
        click.echo(f"❌ Ligand preparation failed: {e}")
        sys.exit(1)

@main.command()
@click.option('--receptor', '-r', required=True, type=click.Path(exists=True),
              help='Metalloprotein PDB file')
@click.option('--output', '-o', type=click.Path(),
              help='Output analysis file (JSON)')
def analyze_protein(receptor, output):
    """
    Analyze metalloprotein metal binding sites

    Identifies metal centers and their coordinating residues in protein structures.

    Examples:
        pandadock-metal analyze-protein -r zinc_protein.pdb
        pandadock-metal analyze-protein -r protein.pdb -o metal_sites.json
    """
    click.echo("🔬 Analyzing metalloprotein structure...")
    click.echo(f"Receptor: {receptor}")

    engine = MetalDockingEngine()

    try:
        protein_prep = MetalloproteinPreparator()
        protein_info = protein_prep.prepare_metalloprotein(receptor)

        metal_sites = protein_info['metal_sites']

        if not metal_sites:
            click.echo("ℹ️  No metal sites identified in protein")
            return

        click.echo(f"\n🎯 Found {len(metal_sites)} metal sites:")

        for i, site in enumerate(metal_sites, 1):
            click.echo(f"\n  Site {i}: {site['metal_element']}")
            click.echo(f"    Coordinates: ({site['metal_coords'][0]:.2f}, "
                      f"{site['metal_coords'][1]:.2f}, {site['metal_coords'][2]:.2f})")
            click.echo(f"    Coordination number: {site['coordination_number']}")
            click.echo(f"    Net charge: {site['net_charge']:.1f}")

            if site['coordinating_residues']:
                click.echo(f"    Coordinating residues:")
                for res in site['coordinating_residues']:
                    click.echo(f"      • {res['residue_name']}{res['residue_id'][1]} "
                              f"({res['atom_name']}) - {res['distance']:.2f} Å")

        if output:
            with open(output, 'w') as f:
                json.dump(protein_info, f, indent=2, default=str)
            click.echo(f"\n💾 Analysis saved to: {output}")

    except Exception as e:
        click.echo(f"❌ Protein analysis failed: {e}")
        sys.exit(1)

@main.command()
def list_metals():
    """List supported metal types and their parameters"""
    click.echo("🧪 Supported Metal Types in PandaDock-Metal")
    click.echo("=" * 60)

    engine = MetalDockingEngine()
    metals = engine.get_supported_metals()

    # Group metals by category
    categories = {
        'Alkali Metals': ['Li', 'Na', 'K'],
        'Alkaline Earth': ['Mg', 'MG', 'Ca', 'CA', 'Ba'],
        'Transition Metals (1st row)': ['Ti', 'V', 'Cr', 'Mn', 'MN', 'Fe', 'FE', 'Co', 'Ni', 'Cu', 'Zn', 'ZN'],
        'Transition Metals (2nd row)': ['Mo', 'Pd', 'Ag', 'Cd'],
        'Transition Metals (3rd row)': ['Pt', 'Au', 'Hg'],
        'Post-transition': ['Al', 'Pb'],
        'Lanthanides': ['La', 'Gd']
    }

    for category, metal_list in categories.items():
        available_metals = [m for m in metal_list if m in metals]
        if available_metals:
            click.echo(f"\n{category}:")
            for metal in available_metals:
                info = engine.get_metal_info(metal)
                lines = info.strip().split('\n')
                radius = lines[1].split(': ')[1] if len(lines) > 1 else "N/A"
                click.echo(f"  {metal:3s} - vdW radius: {radius}")

    click.echo(f"\nTotal: {len(metals)} supported metal types")
    click.echo("\nFor detailed parameters of a specific metal, use:")
    click.echo("  python -c \"from pandadock.metal_docking import MetalParameterManager; print(MetalParameterManager().get_metal_info('Zn'))\"")

@main.command()
@click.option('--ligand', '-l', required=True, type=click.Path(exists=True),
              help='Metal-containing ligand file to validate')
@click.option('--metal-params', type=click.Path(exists=True),
              help='Custom metal parameters file')
def validate(ligand, metal_params):
    """
    Validate metal ligand preparation

    Checks coordination geometry, formal charges, and bond orders for
    metal-containing ligands prepared for docking.

    Examples:
        pandadock-metal validate -l prepared_ligand.sdf
        pandadock-metal validate -l ligand.mol2 --metal-params custom.txt
    """
    click.echo("✅ Validating metal ligand preparation...")
    click.echo(f"Ligand: {ligand}")

    engine = MetalDockingEngine(metal_params)

    try:
        # Load ligand
        if ligand.endswith('.sdf'):
            mol = Chem.SDMolSupplier(ligand)[0]
        elif ligand.endswith('.mol2'):
            mol = Chem.MolFromMol2File(ligand)
        elif ligand.endswith('.pdb'):
            mol = Chem.MolFromPDBFile(ligand)
        else:
            mol = Chem.MolFromMolFile(ligand)

        if mol is None:
            click.echo(f"❌ Error: Could not load ligand from {ligand}")
            sys.exit(1)

        # Validate
        validation = engine.validate_metal_preparation(mol)

        click.echo(f"\n📊 Validation Results:")
        click.echo(f"   Overall status: {'✅ VALID' if validation['is_valid'] else '❌ INVALID'}")
        click.echo(f"   Metal atoms found: {len(validation['metal_atoms'])}")

        if validation['metal_atoms']:
            click.echo(f"\n🔬 Metal Centers:")
            for coord_info in validation['coordination_info']:
                click.echo(f"   • {coord_info['metal_symbol']} (atom {coord_info['metal_index']})")
                click.echo(f"     Coordination number: {coord_info['coordination_number']}")
                click.echo(f"     Coordinating atoms: {coord_info['coordinating_atoms']}")

        if validation['warnings']:
            click.echo(f"\n⚠️  Warnings ({len(validation['warnings'])}):")
            for warning in validation['warnings']:
                click.echo(f"   • {warning}")

        if validation['errors']:
            click.echo(f"\n❌ Errors ({len(validation['errors'])}):")
            for error in validation['errors']:
                click.echo(f"   • {error}")

        if validation['is_valid']:
            click.echo(f"\n🎉 Ligand is ready for metal-aware docking!")
        else:
            click.echo(f"\n🔧 Please fix errors before proceeding with docking.")
            sys.exit(1)

    except Exception as e:
        click.echo(f"❌ Validation failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()