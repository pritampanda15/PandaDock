#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PandaDock-Flex: Professional Induced-Fit Docking CLI

Implements sophisticated flexible docking with receptor conformational sampling,
similar to Schrödinger's Induced-Fit Docking and Discovery Studio's protocols.

Key Features:
- Multi-phase docking with soft potentials
- Receptor side-chain and loop refinement
- Mutual ligand-receptor adaptation
- GPU-accelerated conformational sampling
- Professional IFD scoring with energy penalties
"""

import click
import sys
import time
import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from rdkit import Chem

from .docking_cli import load_ligand
from .flex_docking.phases import (
    SoftRigidDocker, ReceptorRefiner, FinalRedocker, IFDScorer
)
from .flex_docking.core import FlexibleDockingResult
from .visualization.flex_visualizer import FlexDockingVisualizer


@click.command()
@click.option('--receptor', '-r', required=True, type=click.Path(exists=True),
              help='Receptor PDB file (cleaned, with hydrogens)')
@click.option('--ligand', '-l', required=True, type=click.Path(exists=True),
              help='Ligand file (SDF/MOL2/PDB)')
@click.option('--binding-site-coordinates', '--center', nargs=3, type=float, metavar='X Y Z',
              help='Center coordinates of binding site (Å)')
@click.option('--binding-site-radius', '--radius', type=float, default=12.0,
              help='Radius around binding site for docking/refinement (Å)')
@click.option('--flexible-residues', type=str, default=None,
              help='Comma-separated residue IDs for flexibility (e.g., "A:123,A:145,B:67")')
@click.option('--refine-distance', type=float, default=6.0,
              help='Distance from ligand for automatic residue refinement (Å)')
@click.option('--refine-loops', is_flag=True, default=False,
              help='Include loop refinement (more computationally intensive)')
@click.option('--refine-ligand', is_flag=True, default=True,
              help='Allow ligand conformational changes during refinement')

# Phase 1: Initial soft docking parameters
@click.option('--soft-potential-scale', type=float, default=0.7,
              help='VDW softening factor for initial docking (0.5-0.9)')
@click.option('--initial-poses-to-retain', type=int, default=20,
              help='Number of poses to retain from initial soft docking')

# Phase 2: Receptor refinement parameters
@click.option('--refinement-method', type=click.Choice(['minimization', 'short_md', 'monte_carlo']),
              default='minimization', help='Method for receptor refinement')
@click.option('--max-refinement-cycles', type=int, default=3,
              help='Maximum cycles of receptor refinement')

# Phase 3: Final redocking parameters
@click.option('--final-potential-scale', type=float, default=1.0,
              help='VDW scaling for final high-precision docking')
@click.option('--final-poses-to-retain', type=int, default=10,
              help='Number of final poses to retain')

# Phase 4: Scoring and output
@click.option('--scoring-function', type=click.Choice([
    'ifd_composite', 'glide_like', 'prime_energy', 'custom'
]), default='ifd_composite', help='Scoring function for final ranking')
@click.option('--output-prefix', '-o', type=str, default='flex_docking',
              help='Prefix for all output files')

# Computational parameters
@click.option('--use-gpu', is_flag=True, default=False,
              help='Use GPU acceleration for conformational sampling')
@click.option('--gpu-id', type=int, default=0,
              help='GPU device ID to use')
@click.option('--cpu-workers', type=int, default=None,
              help='Number of CPU workers for parallel processing')
@click.option('--max-memory-gb', type=float, default=8.0,
              help='Maximum memory usage in GB')

# Quality control
@click.option('--energy-cutoff', type=float, default=50.0,
              help='Energy cutoff for pose filtering (kcal/mol)')
@click.option('--rmsd-clustering', type=float, default=2.0,
              help='RMSD threshold for pose clustering (Å)')
@click.option('--verbose', '-v', is_flag=True, default=False,
              help='Enable verbose logging')

def main(receptor, ligand, binding_site_coordinates, binding_site_radius,
              flexible_residues, refine_distance, refine_loops, refine_ligand,
              soft_potential_scale, initial_poses_to_retain,
              refinement_method, max_refinement_cycles,
              final_potential_scale, final_poses_to_retain,
              scoring_function, output_prefix,
              use_gpu, gpu_id, cpu_workers, max_memory_gb,
              energy_cutoff, rmsd_clustering, verbose):
    """
    PandaDock-Flex: Professional Flexible/Induced-Fit Docking

    Performs 4-phase induced-fit docking with receptor conformational sampling:
    1. Initial soft rigid docking
    2. Receptor refinement around ligand poses
    3. Final redocking into refined receptors
    4. IFD scoring and ranking

    Example:
        pandadock-flex -r receptor.pdb -l ligand.sdf --center 129.249 120.21 145.249 -o etomidate_flex
    """

    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    logger = logging.getLogger("pandadock.flex")

    click.echo("🧬 PandaDock-Flex: Professional Induced-Fit Docking")
    click.echo("=" * 60)
    click.echo(f"Receptor: {receptor}")
    click.echo(f"Ligand: {ligand}")
    click.echo(f"Binding site: ({binding_site_coordinates[0]:.1f}, {binding_site_coordinates[1]:.1f}, {binding_site_coordinates[2]:.1f}) Å")
    click.echo(f"Refinement radius: {refine_distance:.1f} Å")
    click.echo(f"GPU acceleration: {'✓' if use_gpu else '✗'}")
    click.echo(f"Output prefix: {output_prefix}")
    click.echo()

    total_start_time = time.time()

    refiner = None

    try:
        # Validate inputs
        if not binding_site_coordinates:
            click.echo("Error: Must specify --binding-site-coordinates")
            sys.exit(1)

        binding_site = np.array(binding_site_coordinates)
        output_dir = Path(f"{output_prefix}_results")
        output_dir.mkdir(exist_ok=True)

        # Load ligand
        click.echo("📥 Loading ligand molecule...")
        ligand_mol = load_ligand(ligand)
        if ligand_mol is None:
            click.echo(f"Error: Could not load ligand from {ligand}")
            sys.exit(1)

        logger.info(f"Loaded ligand with {ligand_mol.GetNumAtoms()} atoms")

        # Initialize components with GPU/CPU configuration
        computational_config = {
            'use_gpu': use_gpu,
            'gpu_id': gpu_id,
            'cpu_workers': cpu_workers,
            'max_memory_gb': max_memory_gb
        }

        # =================================================================
        # PHASE 1: INITIAL SOFT RIGID DOCKING
        # =================================================================
        click.echo("🔍 Phase 1: Initial soft rigid docking...")
        phase1_start = time.time()

        soft_docker = SoftRigidDocker(**computational_config)
        soft_docker.configure_soft_potential(soft_potential_scale)

        initial_result = soft_docker.dock(
            receptor_file=receptor,
            ligand_mol=ligand_mol,
            binding_site_center=binding_site,
            binding_site_radius=binding_site_radius,
            max_poses=initial_poses_to_retain,
            energy_cutoff=energy_cutoff
        )

        phase1_time = time.time() - phase1_start
        click.echo(f"   ✓ Generated {len(initial_result.poses)} initial poses ({phase1_time:.1f}s)")

        if not initial_result.poses:
            click.echo("❌ No viable poses from initial docking. Exiting.")
            sys.exit(1)

        # =================================================================
        # PHASE 2: RECEPTOR REFINEMENT
        # =================================================================
        click.echo("🔧 Phase 2: Receptor refinement...")
        phase2_start = time.time()

        refiner = ReceptorRefiner(**computational_config)
        refiner.configure_refinement(
            method=refinement_method,
            max_cycles=max_refinement_cycles,
            refine_loops=refine_loops,
            refine_ligand=refine_ligand
        )

        # Determine flexible residues
        if flexible_residues:
            flex_residues = parse_residue_list(flexible_residues)
        else:
            # Auto-detect residues within refine_distance of any pose
            flex_residues = refiner.auto_detect_flexible_residues(
                receptor, initial_result.poses, refine_distance
            )

        logger.info(f"Refining {len(flex_residues)} flexible residues")

        refined_complexes = refiner.refine_receptor_conformations(
            receptor_file=receptor,
            initial_poses=initial_result.poses,
            flexible_residues=flex_residues,
            binding_site_center=binding_site
        )

        phase2_time = time.time() - phase2_start
        click.echo(f"   ✓ Generated {len(refined_complexes)} refined receptor conformations ({phase2_time:.1f}s)")

        # =================================================================
        # PHASE 3: FINAL REDOCKING
        # =================================================================
        click.echo("🎯 Phase 3: Final redocking into refined receptors...")
        phase3_start = time.time()

        final_docker = FinalRedocker(**computational_config)
        final_docker.configure_final_potential(final_potential_scale)

        final_results = []
        for i, refined_complex in enumerate(refined_complexes):
            logger.debug(f"Redocking into refined receptor {i+1}/{len(refined_complexes)}")

            redock_result = final_docker.redock(
                refined_receptor=refined_complex.receptor,
                ligand_mol=ligand_mol,
                binding_site_center=binding_site,
                binding_site_radius=binding_site_radius,
                max_poses=final_poses_to_retain // len(refined_complexes) + 1
            )

            # Combine with refinement energy cost
            for pose in redock_result.poses:
                pose.refinement_cost = refined_complex.energy_cost

            final_results.append(redock_result)

        phase3_time = time.time() - phase3_start
        total_final_poses = sum(len(r.poses) for r in final_results)
        click.echo(f"   ✓ Generated {total_final_poses} final poses ({phase3_time:.1f}s)")

        # =================================================================
        # PHASE 4: IFD SCORING AND RANKING
        # =================================================================
        click.echo("📊 Phase 4: IFD scoring and ranking...")
        phase4_start = time.time()

        ifd_scorer = IFDScorer(scoring_function)

        # Combine all final poses
        all_final_poses = []
        for result in final_results:
            all_final_poses.extend(result.poses)

        # Calculate IFD scores
        scored_poses = ifd_scorer.score_ifd_poses(
            poses=all_final_poses,
            receptor_file=receptor,
            ligand_mol=ligand_mol
        )

        # Cluster and rank poses
        clustered_poses = ifd_scorer.cluster_poses(scored_poses, rmsd_clustering)
        final_ranked_poses = clustered_poses[:final_poses_to_retain]

        phase4_time = time.time() - phase4_start
        click.echo(f"   ✓ Ranked {len(final_ranked_poses)} final IFD poses ({phase4_time:.1f}s)")

        # =================================================================
        # RESULTS OUTPUT
        # =================================================================
        total_time = time.time() - total_start_time

        # Create comprehensive result object
        flex_result = FlexibleDockingResult(
            initial_poses=initial_result.poses,
            refined_complexes=refined_complexes,
            final_poses=final_ranked_poses,
            flexible_residues=flex_residues,
            total_runtime=total_time,
            parameters={
                'soft_potential_scale': soft_potential_scale,
                'refine_distance': refine_distance,
                'refinement_method': refinement_method,
                'scoring_function': scoring_function
            }
        )

        # Save results
        click.echo("💾 Saving results...")
        flex_result.save_results(output_dir)

        # Generate visualizations
        visualizer = FlexDockingVisualizer()
        visualizer.save_flex_complexes(flex_result, receptor, ligand_mol, output_dir)
        visualizer.generate_ifd_report(flex_result, output_dir)

        # Temporary refined receptors are cleaned up by the refiner itself, in a
        # finally block, so an interrupted or failed run does not leave them
        # behind. This step used to glob "temp_refined_receptor_*.pdb" in the
        # working directory, which ran only on success and would also delete the
        # files of a concurrent run sharing that directory.
        click.echo("🧹 Cleaning up temporary files...")

        # Summary
        click.echo("\n" + "="*60)
        click.echo("🎉 PandaDock-Flex completed successfully!")
        click.echo(f"⏱️  Total time: {total_time:.1f} seconds")
        click.echo(f"📁 Results saved to: {output_dir}")
        click.echo("\nTop 5 IFD poses:")

        for i, pose in enumerate(final_ranked_poses[:5], 1):
            click.echo(f"  {i}. IFD Score: {pose.ifd_score:.3f} kcal/mol "
                      f"(Binding: {pose.binding_energy:.3f}, "
                      f"Refinement cost: {pose.refinement_cost:.3f})")

        click.echo(f"\n🔬 Flexible residues refined: {len(flex_residues)}")
        click.echo("🧬 Ready for molecular visualization and analysis!")

    except Exception as e:
        logger.error(f"Flexible docking failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    finally:
        # Runs on success, failure and Ctrl-C alike. Refined receptors are
        # roughly 375 KB each and there is one per pose, so a 20-pose run that
        # died part way through previously left about 7.5 MB in the working
        # directory with nothing to indicate where it came from.
        if refiner is not None:
            refiner.cleanup()


def parse_residue_list(residue_string: str) -> List[str]:
    """Parse comma-separated residue list like 'A:123,A:145,B:67'"""
    residues = []
    for res_spec in residue_string.split(','):
        res_spec = res_spec.strip()
        if ':' in res_spec:
            residues.append(res_spec)
        else:
            # Assume chain A if not specified
            residues.append(f"A:{res_spec}")
    return residues




if __name__ == '__main__':
    main()