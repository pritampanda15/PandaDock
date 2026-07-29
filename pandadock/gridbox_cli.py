import click
import sys
import logging
from pathlib import Path
from typing import List, Optional

from .gridbox.core import GridBox, ManualGridBoxGenerator, ResidueBasedGridBoxGenerator
from .gridbox.cavity_detection import CavityDetectionGridBoxGenerator
from .ai.binding_site_prediction import AIBindingSitePredictionGenerator
from .gridbox.ligand_based import LigandBasedGridBoxGenerator
from .gui.grid_selector import InteractiveGridBoxGenerator


@click.command()
@click.option('--receptor', '-r', required=True, type=click.Path(exists=True),
              help='Path to processed receptor file (PDB format)')
@click.option('--mode', '-m', type=click.Choice([
    'interactive', 'manual', 'residues', 'cavities', 'ai', 'similarity', 'pharmacophore'
]), default='interactive', help='Grid box generation mode')
@click.option('--center', '-c', type=float, nargs=3, help='Grid center coordinates (x y z)')
@click.option('--box', '-b', type=float, nargs=3, help='Grid box dimensions (x y z)')
@click.option('--spacing', '-s', type=float, default=0.375, help='Grid spacing in Angstroms')
@click.option('--residues', type=str, help='Comma-separated residue specifications (e.g., A:123,B:67)')
@click.option('--padding', type=float, default=5.0, help='Padding around selected residues/ligand')
@click.option('--reference-ligand', type=click.Path(exists=True),
              help='Reference ligand file for similarity-based grid generation')
@click.option('--pharmacophore', type=str,
              help='Comma-separated pharmacophore features (hydrophobic,donor,acceptor,aromatic,positive,negative)')
@click.option('--max-sites', type=int, default=3, help='Maximum number of binding sites to generate')
@click.option('--confidence-threshold', type=float, default=0.7, help='Minimum confidence for AI predictions')
@click.option('--min-volume', type=float, default=50.0, help='Minimum cavity volume for detection')
@click.option('--output', '-o', type=str, help='Output file for grid box configuration')
@click.option('--output-dir', type=str, default='gridbox_output', help='Output directory for results')
@click.option('--visualize', is_flag=True, help='Generate visualization of grid boxes')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def main(receptor, mode, center, box, spacing, residues, padding, reference_ligand,
         pharmacophore, max_sites, confidence_threshold, min_volume, output,
         output_dir, visualize, verbose):
    """
    PandaDock Grid Box Definition Tool

    Generate grid boxes for molecular docking using various intelligent methods.
    """

    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)

    click.echo("🐼 PandaDock Grid Box Generator")
    click.echo(f"Receptor: {receptor}")
    click.echo(f"Mode: {mode}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Auto-detect mode if reference ligand is provided
        if reference_ligand and mode == 'interactive':
            mode = 'similarity'
            click.echo(f"Auto-detected similarity mode due to --reference-ligand parameter")

        # Generate grid boxes based on mode
        grid_boxes = []

        if mode == 'interactive':
            click.echo("Launching interactive GUI...")
            generator = InteractiveGridBoxGenerator()
            grid_boxes = generator.generate(receptor)

        elif mode == 'manual':
            if not center or not box:
                raise click.BadParameter("Manual mode requires --center and --box parameters")

            click.echo(f"Generating manual grid box at center {center} with dimensions {box}")
            generator = ManualGridBoxGenerator()
            grid_boxes = generator.generate(
                receptor, center=center, dimensions=box, spacing=spacing
            )

        elif mode == 'residues':
            if not residues:
                raise click.BadParameter("Residues mode requires --residues parameter")

            residue_list = [r.strip() for r in residues.split(',')]
            click.echo(f"Generating grid box for residues: {residue_list}")
            generator = ResidueBasedGridBoxGenerator()
            grid_boxes = generator.generate(
                receptor, residues=residue_list, padding=padding, spacing=spacing
            )

        elif mode == 'cavities':
            click.echo(f"Detecting cavities (max sites: {max_sites}, min volume: {min_volume})")
            generator = CavityDetectionGridBoxGenerator()
            grid_boxes = generator.generate(
                receptor, max_cavities=max_sites, min_volume=min_volume, spacing=spacing
            )

        elif mode == 'ai':
            click.echo(f"AI-powered binding site prediction (confidence threshold: {confidence_threshold})")
            generator = AIBindingSitePredictionGenerator()
            grid_boxes = generator.generate(
                receptor, max_sites=max_sites, confidence_threshold=confidence_threshold, spacing=spacing
            )

        elif mode == 'similarity':
            if not reference_ligand:
                raise click.BadParameter("Similarity mode requires --reference-ligand parameter")

            click.echo(f"Ligand-based grid generation using reference: {reference_ligand}")
            generator = LigandBasedGridBoxGenerator()
            grid_boxes = generator.generate(
                receptor, reference_ligand=reference_ligand, padding=padding, spacing=spacing
            )

        elif mode == 'pharmacophore':
            if not pharmacophore:
                raise click.BadParameter("Pharmacophore mode requires --pharmacophore parameter")

            features = [f.strip() for f in pharmacophore.split(',')]
            click.echo(f"Pharmacophore-based grid generation with features: {features}")
            generator = LigandBasedGridBoxGenerator()
            grid_boxes = generator.generate(
                receptor, pharmacophore_features=features, spacing=spacing, min_score=0.8
            )

            # Limit to max_sites for better selectivity
            grid_boxes = grid_boxes[:max_sites]

        # Process results
        if not grid_boxes:
            click.echo("❌ No grid boxes generated. Try adjusting parameters or using a different mode.")
            sys.exit(1)

        click.echo(f"✅ Generated {len(grid_boxes)} grid box(es)")

        # Display grid box information
        for i, grid_box in enumerate(grid_boxes, 1):
            click.echo(f"\n📦 Grid Box {i}:")
            click.echo(f"   Center: ({grid_box.center[0]:.2f}, {grid_box.center[1]:.2f}, {grid_box.center[2]:.2f}) Å")
            click.echo(f"   Dimensions: ({grid_box.dimensions[0]:.2f}, {grid_box.dimensions[1]:.2f}, {grid_box.dimensions[2]:.2f}) Å")
            click.echo(f"   Volume: {grid_box.volume():.1f} Ų")
            click.echo(f"   Spacing: {grid_box.spacing} Å")
            click.echo(f"   Method: {grid_box.method_used}")
            click.echo(f"   Confidence: {grid_box.confidence_score:.3f}")

            if grid_box.binding_site_info:
                click.echo("   Additional Info:")
                for key, value in grid_box.binding_site_info.items():
                    if isinstance(value, (int, float)):
                        click.echo(f"     {key}: {value}")
                    elif isinstance(value, list) and len(value) <= 5:
                        click.echo(f"     {key}: {value}")
                    elif isinstance(value, str):
                        click.echo(f"     {key}: {value}")

        # Save grid boxes
        if output:
            output_file = Path(output)
        else:
            output_file = output_path / f"gridbox_{mode}.json"

        # Track what was actually written. The "next steps" hint below used to be
        # built from the --output-dir default rather than from the files just
        # saved, so with --output (a file prefix) it printed a --grid-config path
        # that did not exist and the copied command failed.
        saved_files = []

        if len(grid_boxes) == 1:
            grid_boxes[0].save(str(output_file))
            saved_files.append(output_file)
            click.echo(f"💾 Grid box saved to: {output_file}")
        else:
            # Save multiple grid boxes
            for i, grid_box in enumerate(grid_boxes):
                if output:
                    base_name = output_file.stem
                    ext = output_file.suffix or ".json"
                    multi_file = output_file.parent / f"{base_name}_{i+1}{ext}"
                else:
                    multi_file = output_path / f"gridbox_{mode}_{i+1}.json"

                grid_box.save(str(multi_file))
                saved_files.append(multi_file)
                click.echo(f"💾 Grid box {i+1} saved to: {multi_file}")

        # Generate visualization if requested
        if visualize:
            click.echo("🎨 Generating visualization...")
            try:
                _generate_visualization(receptor, grid_boxes, output_path)
                click.echo(f"📊 Visualization saved to: {output_path}/visualization.png")
            except Exception as e:
                click.echo(f"⚠️  Visualization failed: {e}")

        # Summary
        click.echo(f"\n🎉 Grid box generation complete!")
        locations = sorted({str(f.parent) for f in saved_files})
        click.echo(f"📁 Written to: {', '.join(locations) if locations else output_path}")

        # Command examples for next steps
        click.echo(f"\n💡 Next steps:")
        if len(grid_boxes) == 1:
            center = grid_boxes[0].center
            dims = grid_boxes[0].dimensions
            click.echo(f"   pandadock dock -r {receptor} -l ligand.sdf --grid-config {output_file}")
            click.echo(f"   # Or use coordinates directly:")
            click.echo(f"   pandadock dock -r {receptor} -l ligand.sdf --center {center[0]:.1f} {center[1]:.1f} {center[2]:.1f} --box {dims[0]:.1f} {dims[1]:.1f} {dims[2]:.1f}")
        else:
            click.echo(f"   # Multiple grid boxes generated - choose the best one for docking")
            click.echo(f"   pandadock dock -r {receptor} -l ligand.sdf --grid-config {saved_files[0]}")

    except Exception as e:
        click.echo(f"❌ Error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def _generate_visualization(receptor_file: str, grid_boxes: List[GridBox], output_dir: Path):
    """Generate visualization of grid boxes on protein structure"""

    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from Bio.PDB import PDBParser
        import numpy as np

        # Load receptor structure
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('receptor', receptor_file)
        atoms = list(structure.get_atoms())

        # Get CA atoms for backbone representation
        ca_atoms = [atom for atom in atoms if atom.get_name() == 'CA']
        ca_coords = np.array([atom.get_coord() for atom in ca_atoms])

        # Create 3D plot
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # Plot protein backbone
        ax.scatter(ca_coords[:, 0], ca_coords[:, 1], ca_coords[:, 2],
                  c='lightblue', alpha=0.6, s=20, label='Protein Backbone')

        # Plot grid boxes
        colors = ['red', 'green', 'blue', 'orange', 'purple']
        for i, grid_box in enumerate(grid_boxes):
            color = colors[i % len(colors)]
            center = grid_box.center
            dims = grid_box.dimensions

            # Plot grid box edges
            x_range = [center[0] - dims[0]/2, center[0] + dims[0]/2]
            y_range = [center[1] - dims[1]/2, center[1] + dims[1]/2]
            z_range = [center[2] - dims[2]/2, center[2] + dims[2]/2]

            # Draw box edges
            for x in x_range:
                for y in y_range:
                    ax.plot([x, x], [y, y], z_range, color=color, alpha=0.8, linewidth=2)
            for x in x_range:
                for z in z_range:
                    ax.plot([x, x], y_range, [z, z], color=color, alpha=0.8, linewidth=2)
            for y in y_range:
                for z in z_range:
                    ax.plot(x_range, [y, y], [z, z], color=color, alpha=0.8, linewidth=2)

            # Plot center point
            ax.scatter(*center, c=color, s=100, marker='x', linewidth=3,
                      label=f'Grid Box {i+1} (Score: {grid_box.confidence_score:.2f})')

        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.legend()
        ax.set_title('PandaDock Grid Box Visualization')

        # Save plot
        plt.tight_layout()
        plt.savefig(output_dir / 'visualization.png', dpi=300, bbox_inches='tight')
        plt.close()

    except ImportError:
        raise ImportError("Visualization requires matplotlib. Install with: pip install matplotlib")


if __name__ == '__main__':
    main()