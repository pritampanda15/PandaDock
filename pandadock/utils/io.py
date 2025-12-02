import os
from pathlib import Path
from datetime import datetime


def create_intermediate_folder(output_dir):
    """Create intermediate folder for preprocessed files"""
    output_path = Path(output_dir)
    intermediate_path = output_path / "intermediate"
    intermediate_path.mkdir(parents=True, exist_ok=True)
    return intermediate_path


def write_preprocessing_log(log_path, receptor_result, ligand_result):
    """Write detailed preprocessing log file"""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("PandaDock Preprocessing Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {timestamp}\n\n")

        # Receptor information
        f.write("RECEPTOR PROCESSING\n")
        f.write("-" * 30 + "\n")
        f.write(f"Input file: {receptor_result['input_file']}\n")
        f.write(f"Output file: {receptor_result['output_file']}\n")
        f.write(f"Format: {receptor_result['format']}\n")
        f.write(f"pH used: {receptor_result['ph_used']}\n")
        f.write(f"Original atoms: {receptor_result['original_atoms']}\n")
        f.write(f"Final atoms: {receptor_result['final_atoms']}\n")
        f.write(f"Atoms added: {receptor_result['final_atoms'] - receptor_result['original_atoms']}\n")

        if receptor_result['missing_residues_added']:
            f.write(f"\nMissing residues added ({len(receptor_result['missing_residues_added'])}):\n")
            for residue in receptor_result['missing_residues_added']:
                f.write(f"  - {residue}\n")
        else:
            f.write("\nNo missing residues found.\n")

        if receptor_result['missing_atoms_added']:
            f.write(f"\nMissing atoms added ({len(receptor_result['missing_atoms_added'])}):\n")
            for atom in receptor_result['missing_atoms_added'][:10]:  # Show first 10
                f.write(f"  - {atom}\n")
            if len(receptor_result['missing_atoms_added']) > 10:
                f.write(f"  ... and {len(receptor_result['missing_atoms_added']) - 10} more\n")
        else:
            f.write("\nNo missing atoms found.\n")

        f.write(f"\nHydrogens added: {receptor_result['hydrogens_added']}\n")
        f.write(f"Charges assigned: {receptor_result['charges_assigned']}\n")

        # Ligand information
        f.write(f"\n\nLIGAND PROCESSING\n")
        f.write("-" * 30 + "\n")
        f.write(f"Input file: {ligand_result['input_file']}\n")
        f.write(f"Output file: {ligand_result['output_file']}\n")
        f.write(f"Format: {ligand_result['format']}\n")
        f.write(f"Original atoms: {ligand_result['original_atoms']}\n")
        f.write(f"Final atoms: {ligand_result['final_atoms']}\n")
        f.write(f"Hydrogens added: {ligand_result['hydrogens_added']}\n")
        f.write(f"Minimized: {ligand_result['minimized']}\n")

        if ligand_result['energy_before'] is not None and ligand_result['energy_after'] is not None:
            energy_change = ligand_result['energy_after'] - ligand_result['energy_before']
            f.write(f"Energy before minimization: {ligand_result['energy_before']:.2f} kcal/mol\n")
            f.write(f"Energy after minimization: {ligand_result['energy_after']:.2f} kcal/mol\n")
            f.write(f"Energy change: {energy_change:.2f} kcal/mol\n")

        if ligand_result['conformers_generated'] > 0:
            f.write(f"3D conformers generated: {ligand_result['conformers_generated']}\n")

        # Summary
        f.write(f"\n\nSUMMARY\n")
        f.write("-" * 30 + "\n")
        f.write(f"Receptor atoms: {receptor_result['original_atoms']} -> {receptor_result['final_atoms']}\n")
        f.write(f"Ligand atoms: {ligand_result['original_atoms']} -> {ligand_result['final_atoms']}\n")
        f.write(f"Total atoms processed: {receptor_result['final_atoms'] + ligand_result['final_atoms']}\n")

        f.write(f"\nPreprocessing completed successfully.\n")
        f.write("Files are ready for docking workflow.\n")
        f.write("=" * 60 + "\n")