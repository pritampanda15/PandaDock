#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PandaDock Tethered/Constrained Docking CLI

Performs tethered docking to validate scoring functions by constraining ligands
to remain close to their crystallographic positions. This technique is essential
for scoring function validation and establishing reference binding affinities.

Features:
- Extract ligands from PDB complex files
- Perform constrained docking with distance restraints
- Score validation against known poses
- Binding affinity prediction for crystal structures
- RMSD analysis and pose quality assessment

Author: Pritam Kumar Panda @ Stanford University
"""

import click
import sys
import logging
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import numpy as np

def show_tethered_help():
    """Display professional help for tethered docking"""
    help_text = """
██████╗  █████╗ ███╗   ██╗██████╗  █████╗ ██████╗  ██████╗  ██████╗██╗  ██╗
██╔══██╗██╔══██╗████╗  ██║██╔══██╗██╔══██╗██╔══██╗██╔═══██╗██╔════╝██║ ██╔╝
██████╔╝███████║██╔██╗ ██║██║  ██║███████║██║  ██║██║   ██║██║     █████╔╝
██╔═══╝ ██╔══██║██║╚██╗██║██║  ██║██╔══██║██║  ██║██║   ██║██║     ██╔═██╗
██║     ██║  ██║██║ ╚████║██████╔╝██║  ██║██████╔╝╚██████╔╝╚██████╗██║  ██╗
╚═╝     ╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝ ╚═╝  ╚═╝╚═════╝  ╚═════╝  ╚═════╝╚═╝  ╚═╝

    🔗 Tethered/Constrained Docking Validation Suite
      https://github.com/pritampanda15/PandaDock

Author: Pritam Kumar Panda @ Stanford University
Version: 1.0.0

TETHERED DOCKING PURPOSE:
    Validate scoring functions by docking ligands with distance constraints
    to their crystallographic positions. Essential for:
    • Scoring function calibration and validation
    • Reference binding affinity establishment
    • Crystal structure pose reproduction
    • Algorithm benchmarking and assessment

USAGE:
    pandadock-tethered COMMAND [OPTIONS]

MAIN COMMANDS:
    extract           Extract ligand from PDB complex file
    dock             Perform tethered docking with constraints
    validate         Score and validate crystal structure poses
    analyze          Complete analysis: extract → dock → validate

EXAMPLES:
    # Extract ligand from complex PDB
    pandadock-tethered extract -i complex.pdb -l LIG --chain A

    # Perform tethered docking
    pandadock-tethered dock -r receptor.pdb -l ligand.sdf --reference-pose ref.pdb

    # Validate crystal structure scoring
    pandadock-tethered validate -i complex.pdb -l LIG --scoring physics_based

    # Complete workflow
    pandadock-tethered analyze -i complex.pdb -l LIG --tether-radius 2.0

FEATURES:
    ✓ Automatic ligand extraction from PDB complexes
    ✓ Flexible distance constraint definitions
    ✓ Multiple scoring function validation
    ✓ RMSD analysis and pose quality metrics
    ✓ Reference binding affinity prediction
    ✓ Comprehensive validation reports

For detailed help on specific commands:
    pandadock-tethered COMMAND --help

#################################################################
"""
    click.echo(help_text)

@click.group(invoke_without_command=True, add_help_option=False)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--help', '-h', is_flag=True, help='Show this help message')
@click.pass_context
def main(ctx, verbose, help):
    """PandaDock Tethered Docking - Validation & Constraint-Based Docking"""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if help:
        show_tethered_help()
        return

    # If no command is provided, show help
    if ctx.invoked_subcommand is None:
        show_tethered_help()

@main.command()
@click.option('--input-file', '-i', required=True, type=click.Path(exists=True),
              help='Input PDB complex file containing protein and ligand')
@click.option('--ligand-id', '-l', required=True, type=str,
              help='Ligand residue name or ID to extract (e.g., LIG, ATP, HEM)')
@click.option('--chain', '-c', type=str,
              help='Specific chain ID containing the ligand (optional)')
@click.option('--output-dir', '-o', type=click.Path(),
              help='Output directory [default: INPUT_DIR/tethered_analysis]')
@click.option('--output-format', type=click.Choice(['sdf', 'mol2', 'pdb']), default='sdf',
              help='Output format for extracted ligand [default: sdf]')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def extract(input_file, ligand_id, chain, output_dir, output_format, verbose):
    """
    Extract ligand from PDB complex file for tethered docking

    Extracts ligand coordinates and structure from a protein-ligand complex PDB file.
    The extracted ligand serves as the reference structure for tethered docking
    experiments and scoring function validation.

    Examples:
        # Extract ligand LIG from complex
        pandadock-tethered extract -i 1abc.pdb -l LIG

        # Extract ATP from chain A specifically
        pandadock-tethered extract -i kinase_complex.pdb -l ATP -c A

        # Extract to specific directory in MOL2 format
        pandadock-tethered extract -i complex.pdb -l LIG -o /path/output --output-format mol2
    """

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("🔗 PandaDock Tethered Docking - Ligand Extraction")
    print("=" * 60)
    print(f"Input complex: {input_file}")
    print(f"Ligand ID: {ligand_id}")
    print(f"Chain filter: {chain if chain else 'All chains'}")
    print(f"Output format: {output_format.upper()}")

    # Set output directory
    if not output_dir:
        output_dir = Path(input_file).parent / "tethered_analysis"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print("")

    try:
        from .tethered.ligand_extractor import LigandExtractor

        extractor = LigandExtractor()
        extracted_files = extractor.extract_ligand_from_complex(
            complex_file=input_file,
            ligand_id=ligand_id,
            chain_id=chain,
            output_dir=output_dir,
            output_format=output_format
        )

        print("📁 Extraction Results:")
        for file_type, file_path in extracted_files.items():
            print(f"   {file_type}: {file_path}")

        print(f"\n✅ Ligand extraction completed successfully!")
        print(f"📂 Files saved to: {output_dir}")

        # Provide next step hints
        print(f"\n💡 Next steps:")
        print(f"   • Use extracted receptor: {extracted_files.get('receptor', 'N/A')}")
        print(f"   • Use extracted ligand: {extracted_files.get('ligand', 'N/A')}")
        print(f"   • Use reference pose: {extracted_files.get('reference_pose', 'N/A')}")

    except ImportError:
        print("❌ Tethered docking module not found")
        print("   This feature requires the ligand extraction module")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Ligand extraction failed: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

@main.command()
@click.option('--receptor', '-r', required=True, type=click.Path(exists=True),
              help='Receptor PDB file (protein without ligand)')
@click.option('--ligand', '-l', required=True, type=click.Path(exists=True),
              help='Ligand structure file (SDF, MOL2, or PDB)')
@click.option('--reference-pose', '--ref', type=click.Path(exists=True),
              help='Reference ligand pose for constraints (PDB format)')
@click.option('--tether-radius', '-t', type=float, default=2.0,
              help='Maximum distance from reference pose (Å) [default: 2.0]')
@click.option('--scoring-function', '-s',
              type=click.Choice(['physics_based', 'empirical', 'hybrid']),
              default='physics_based',
              help='Scoring function to use [default: physics_based]')
@click.option('--algorithm', '-a',
              type=click.Choice(['genetic', 'hierarchical', 'monte_carlo']),
              default='genetic',
              help='Docking algorithm [default: genetic]')
@click.option('--output-dir', '-o', type=click.Path(),
              help='Output directory [default: current_dir/tethered_docking]')
@click.option('--num-poses', '-n', type=int, default=10,
              help='Number of poses to generate [default: 10]')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def dock(receptor, ligand, reference_pose, tether_radius, scoring_function,
         algorithm, output_dir, num_poses, verbose):
    """
    Perform tethered docking with distance constraints

    Docks a ligand while constraining it to remain within a specified distance
    of a reference pose. This is essential for scoring function validation and
    reproducing crystallographic binding modes.

    Examples:
        # Basic tethered docking
        pandadock-tethered dock -r receptor.pdb -l ligand.sdf --ref reference.pdb

        # Tight constraints with physics-based scoring
        pandadock-tethered dock -r protein.pdb -l drug.sdf --ref crystal_pose.pdb -t 1.5

        # Validation with different scoring functions
        pandadock-tethered dock -r rec.pdb -l lig.sdf --ref ref.pdb -s empirical -a monte_carlo
    """

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("🔗 PandaDock Tethered Docking - Constrained Docking")
    print("=" * 55)
    print(f"Receptor: {receptor}")
    print(f"Ligand: {ligand}")
    print(f"Reference pose: {reference_pose if reference_pose else 'None (free docking)'}")
    print(f"Tether radius: {tether_radius} Å")
    print(f"Scoring function: {scoring_function}")
    print(f"Algorithm: {algorithm}")
    print(f"Number of poses: {num_poses}")

    # Set output directory
    if not output_dir:
        output_dir = Path.cwd() / "tethered_docking"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print("")

    try:
        from .tethered.tethered_docker import TetheredDocker

        docker = TetheredDocker()
        results = docker.run_tethered_docking(
            receptor_file=receptor,
            ligand_file=ligand,
            reference_pose=reference_pose,
            tether_radius=tether_radius,
            scoring_function=scoring_function,
            algorithm=algorithm,
            output_dir=output_dir,
            num_poses=num_poses
        )

        print("📊 Docking Results:")
        print(f"   Best score: {results.get('best_score', 'N/A')} kcal/mol")
        print(f"   Poses generated: {results.get('num_poses', 0)}")
        print(f"   Average RMSD to reference: {results.get('avg_rmsd', 'N/A')} Å")
        print(f"   Within constraint radius: {results.get('poses_within_constraint', 'N/A')}")

        print(f"\n📁 Generated Files:")
        for file_type, file_path in results.get('output_files', {}).items():
            print(f"   {file_type}: {file_path}")

        print(f"\n✅ Tethered docking completed successfully!")
        print(f"📂 Results saved to: {output_dir}")

    except ImportError:
        print("❌ Tethered docking engine not found")
        print("   This feature requires the tethered docking module")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Tethered docking failed: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

@main.command()
@click.option('--input-file', '-i', required=True, type=click.Path(exists=True),
              help='Input PDB complex file or docking results directory')
@click.option('--ligand-id', '-l', type=str,
              help='Ligand residue name (required if input is PDB complex)')
@click.option('--scoring-function', '-s',
              type=click.Choice(['physics_based', 'empirical', 'hybrid']),
              default='physics_based',
              help='Scoring function for validation [default: physics_based]')
@click.option('--reference-data', type=click.Path(exists=True),
              help='Reference experimental data (CSV/JSON) for comparison')
@click.option('--output-dir', '-o', type=click.Path(),
              help='Output directory for validation report')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def validate(input_file, ligand_id, scoring_function, reference_data, output_dir, verbose):
    """
    Score and validate crystal structure poses

    Evaluates the binding affinity and pose quality of crystallographic
    structures using PandaDock's scoring functions. Provides validation
    of computational methods against experimental structures.

    Examples:
        # Validate crystal structure pose
        pandadock-tethered validate -i complex.pdb -l LIG

        # Compare against experimental binding data
        pandadock-tethered validate -i complex.pdb -l ATP --reference-data exp_data.csv

        # Validate with specific scoring function
        pandadock-tethered validate -i results_dir -s empirical
    """

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("🔗 PandaDock Tethered Docking - Pose Validation")
    print("=" * 50)
    print(f"Input: {input_file}")
    print(f"Ligand ID: {ligand_id if ligand_id else 'Auto-detect'}")
    print(f"Scoring function: {scoring_function}")
    print(f"Reference data: {reference_data if reference_data else 'None'}")

    # Set output directory
    if not output_dir:
        output_dir = Path(input_file).parent / "validation_results"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print("")

    try:
        from .tethered.pose_validator import PoseValidator

        validator = PoseValidator()
        validation_results = validator.validate_poses(
            input_file=input_file,
            ligand_id=ligand_id,
            scoring_function=scoring_function,
            reference_data=reference_data,
            output_dir=output_dir
        )

        print("📊 Validation Results:")
        print(f"   Crystal pose score: {validation_results.get('crystal_score', 'N/A')} kcal/mol")
        print(f"   Predicted Kd: {validation_results.get('predicted_kd', 'N/A')} M")
        print(f"   Pose quality: {validation_results.get('pose_quality', 'N/A')}")

        if reference_data:
            print(f"   Correlation with experimental: {validation_results.get('correlation', 'N/A')}")

        print(f"\n📁 Generated Files:")
        for file_type, file_path in validation_results.get('output_files', {}).items():
            print(f"   {file_type}: {file_path}")

        print(f"\n✅ Pose validation completed successfully!")
        print(f"📂 Results saved to: {output_dir}")

    except ImportError:
        print("❌ Pose validation module not found")
        print("   This feature requires the validation module")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Pose validation failed: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

@main.command()
@click.option('--input-file', '-i', required=True, type=click.Path(exists=True),
              help='Input PDB complex file')
@click.option('--ligand-id', '-l', required=True, type=str,
              help='Ligand residue name or ID')
@click.option('--chain', '-c', type=str,
              help='Specific chain ID containing the ligand')
@click.option('--tether-radius', '-t', type=float, default=2.0,
              help='Tethering constraint radius (Å) [default: 2.0]')
@click.option('--scoring-function', '-s',
              type=click.Choice(['physics_based', 'empirical', 'hybrid']),
              default='physics_based',
              help='Scoring function [default: physics_based]')
@click.option('--algorithm', '-a',
              type=click.Choice(['genetic', 'hierarchical', 'monte_carlo']),
              default='genetic',
              help='Docking algorithm [default: genetic]')
@click.option('--output-dir', '-o', type=click.Path(),
              help='Output directory [default: INPUT_DIR/complete_analysis]')
@click.option('--num-poses', '-n', type=int, default=10,
              help='Number of poses to generate [default: 10]')
@click.option('--reference-data', type=click.Path(exists=True),
              help='Experimental reference data for validation')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def analyze(input_file, ligand_id, chain, tether_radius, scoring_function,
            algorithm, output_dir, num_poses, reference_data, verbose):
    """
    Complete tethered docking analysis workflow

    Performs the complete workflow: ligand extraction → tethered docking → validation.
    Provides comprehensive analysis of crystal structure poses and scoring function
    performance in a single command.

    Examples:
        # Complete analysis of crystal structure
        pandadock-tethered analyze -i complex.pdb -l LIG

        # Detailed analysis with custom parameters
        pandadock-tethered analyze -i kinase.pdb -l ATP -c A -t 1.5 -s physics_based

        # Analysis with experimental validation
        pandadock-tethered analyze -i complex.pdb -l LIG --reference-data exp_data.csv
    """

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("🔗 PandaDock Tethered Docking - Complete Analysis")
    print("=" * 55)
    print(f"Input complex: {input_file}")
    print(f"Ligand ID: {ligand_id}")
    print(f"Chain filter: {chain if chain else 'All chains'}")
    print(f"Tether radius: {tether_radius} Å")
    print(f"Scoring function: {scoring_function}")
    print(f"Algorithm: {algorithm}")
    print(f"Poses to generate: {num_poses}")
    print(f"Reference data: {reference_data if reference_data else 'None'}")

    # Set output directory
    if not output_dir:
        output_dir = Path(input_file).parent / "complete_analysis"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print("")

    try:
        from .tethered.complete_analyzer import CompleteAnalyzer

        analyzer = CompleteAnalyzer()
        results = analyzer.run_complete_analysis(
            complex_file=input_file,
            ligand_id=ligand_id,
            chain_id=chain,
            tether_radius=tether_radius,
            scoring_function=scoring_function,
            algorithm=algorithm,
            output_dir=output_dir,
            num_poses=num_poses,
            experimental_data=reference_data
        )

        print("📊 Complete Analysis Results:")

        # Extract results from the nested structure
        workflow_results = results.get('tethered_docking_complete_analysis', {}).get('workflow_results', {})
        crystal_validation = workflow_results.get('crystal_validation', {})
        tethered_docking = workflow_results.get('tethered_docking', {})
        comprehensive_analysis = workflow_results.get('comprehensive_analysis', {})

        crystal_score = crystal_validation.get('crystal_score', 'N/A')
        best_score = tethered_docking.get('best_score', 'N/A')
        avg_rmsd = tethered_docking.get('avg_rmsd', 'N/A')
        poses_within = tethered_docking.get('poses_within_constraint', 'N/A')
        crystal_kd = crystal_validation.get('predicted_kd', 'N/A')

        print(f"   Crystal pose score: {crystal_score if crystal_score != 'N/A' else 'N/A'} {'kcal/mol' if crystal_score != 'N/A' else ''}")
        print(f"   Best tethered score: {best_score if best_score != 'N/A' else 'N/A'} {'kcal/mol' if best_score != 'N/A' else ''}")
        print(f"   Average RMSD: {avg_rmsd if avg_rmsd != 'N/A' else 'N/A'} {'Å' if avg_rmsd != 'N/A' else ''}")
        print(f"   Poses within constraint: {poses_within}/{num_poses}")

        if crystal_kd != 'N/A':
            if crystal_kd > 1e-3:
                kd_display = f"{crystal_kd:.2e} M (mM range)"
            elif crystal_kd > 1e-6:
                kd_display = f"{crystal_kd:.2e} M (μM range)"
            elif crystal_kd > 1e-9:
                kd_display = f"{crystal_kd:.2e} M (nM range)"
            else:
                kd_display = f"{crystal_kd:.2e} M"
            print(f"   Predicted binding affinity: {kd_display}")
        else:
            print(f"   Predicted binding affinity: N/A")

        # Display key findings
        key_findings = results.get('tethered_docking_complete_analysis', {}).get('key_findings', {})
        if key_findings:
            print(f"\n📈 Validation Summary:")
            print(f"   Score reproduction: {key_findings.get('score_reproduction', 'N/A')}")
            print(f"   Pose reproduction: {key_findings.get('pose_reproduction', 'N/A')}")
            print(f"   Overall assessment: {key_findings.get('overall_validation', 'N/A')}")

        print(f"\n📁 Output Directories:")
        output_dirs = results.get('tethered_docking_complete_analysis', {}).get('output_directories', {})
        for step, directory in output_dirs.items():
            print(f"   {step}: {directory}")

        print(f"\n✅ Complete tethered docking analysis finished!")
        print(f"📂 All results saved to: {output_dir}")

        # Find the report file
        report_file = output_dir / 'complete_analysis_report.json'
        print(f"📄 Summary report: {report_file}")

    except ImportError:
        print("❌ Complete analysis module not found")
        print("   This feature requires the full tethered docking suite")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Complete analysis failed: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()