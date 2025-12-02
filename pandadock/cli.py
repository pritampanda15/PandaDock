import click
import os
from pathlib import Path
from .preprocessing.receptor import ReceptorPreprocessor
from .preprocessing.ligand import LigandPreprocessor
from .utils.io import create_intermediate_folder, write_preprocessing_log


@click.command()
@click.option('--receptor', '-r', required=True, type=click.Path(exists=True),
              help='Path to receptor file (PDB/CIF format)')
@click.option('--ligand', '-l', required=True, type=click.Path(exists=True),
              help='Path to ligand file (SDF/MOL2/PDB format)')
@click.option('--output-dir', '-o', default='pandadock_output',
              help='Output directory for results')
@click.option('--ph', default=7.0, type=float,
              help='pH value for protonation state calculation')
@click.option('--add-hydrogens', is_flag=True, default=True,
              help='Add hydrogens to structures')
@click.option('--add-charges', is_flag=True, default=True,
              help='Add partial charges to structures')
@click.option('--minimize-ligand', is_flag=True, default=True,
              help='Minimize ligand structure')
@click.option('--fix-missing-residues', is_flag=True, default=False,
              help='Add missing residues and atoms to receptor (can be aggressive)')
def main(receptor, ligand, output_dir, ph, add_hydrogens, add_charges, minimize_ligand, fix_missing_residues):
    """
    PandaDock: GPU-accelerated molecular docking software

    Prepare receptor and ligand files for molecular docking.
    """

    click.echo(f"Starting PandaDock preprocessing...")
    click.echo(f"Receptor: {receptor}")
    click.echo(f"Ligand: {ligand}")

    # Create output directory structure
    output_path = Path(output_dir)
    intermediate_path = create_intermediate_folder(output_path)

    # Initialize preprocessors
    receptor_processor = ReceptorPreprocessor(ph=ph)
    ligand_processor = LigandPreprocessor()

    # Process receptor
    click.echo("Processing receptor...")
    receptor_result = receptor_processor.process(
        receptor,
        intermediate_path / "receptor_processed.pdb",
        add_hydrogens=add_hydrogens,
        add_charges=add_charges,
        fix_missing_residues=fix_missing_residues
    )

    # Process ligand
    click.echo("Processing ligand...")
    ligand_result = ligand_processor.process(
        ligand,
        intermediate_path,
        add_hydrogens=add_hydrogens,
        minimize=minimize_ligand
    )

    # Generate preprocessing log
    log_path = output_path / "preprocessing.out"
    write_preprocessing_log(log_path, receptor_result, ligand_result)

    click.echo(f"Preprocessing completed. Results saved to: {output_dir}")
    click.echo(f"Intermediate files: {intermediate_path}")
    click.echo(f"Preprocessing log: {log_path}")


if __name__ == '__main__':
    main()