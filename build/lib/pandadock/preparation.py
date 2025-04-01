# Create a new file called preparation.py

import numpy as np
import os
import subprocess
from pathlib import Path
import tempfile
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    import pybel
    OPENBABEL_AVAILABLE = True
except ImportError:
    OPENBABEL_AVAILABLE = False


def prepare_protein(pdb_file, output_file=None, add_hydrogens=True, ph=7.4, fix_missing=True):
    """
    Prepare protein for docking by adding hydrogens, fixing missing atoms, etc.
    
    Parameters:
    -----------
    pdb_file : str
        Path to input PDB file
    output_file : str, optional
        Path to output PDB file. If None, a temporary file is created.
    add_hydrogens : bool
        Whether to add hydrogens
    ph : float
        pH for hydrogen addition
    fix_missing : bool
        Whether to fix missing atoms and sidechains
    
    Returns:
    --------
    str
        Path to prepared protein PDB file
    """
    if output_file is None:
        # Create temporary output file
        fd, output_file = tempfile.mkstemp(suffix='.pdb')
        os.close(fd)
    
    if OPENBABEL_AVAILABLE:
        # Use Open Babel for protein preparation
        print(f"Preparing protein using Open Babel...")
        
        # Read PDB file
        protein = next(pybel.readfile('pdb', pdb_file))
        
        if add_hydrogens:
            # Add hydrogens at specified pH
            protein.OBMol.AddHydrogens(False, True, ph)
            print(f"  Added hydrogens at pH {ph}")
        
        # Write output file
        protein.write('pdb', output_file, overwrite=True)
        print(f"Prepared protein saved to {output_file}")
        
    else:
        # Try to use PDBFixer through command line if installed
        try:
            print(f"Attempting to use PDBFixer for protein preparation...")
            
            cmd = ["pdbfixer", pdb_file, "--output", output_file]
            if add_hydrogens:
                cmd.extend(["--add-atoms", "hydrogens"])
            if fix_missing:
                cmd.extend(["--add-atoms", "heavy"])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"Successfully prepared protein using PDBFixer")
                print(f"Prepared protein saved to {output_file}")
            else:
                print("PDBFixer failed. Continuing with original PDB file.")
                import shutil
                shutil.copy(pdb_file, output_file)
                
        except FileNotFoundError:
            print("Neither Open Babel nor PDBFixer are available.")
            print("Using original PDB file without preparation.")
            import shutil
            shutil.copy(pdb_file, output_file)
    
    return output_file


def prepare_ligand(mol_file, output_file=None, add_hydrogens=True, generate_3d=True, minimize=True, n_conformers=10):
    """
    Prepare ligand for docking by adding hydrogens, generating 3D coordinates, etc.
    
    Parameters:
    -----------
    mol_file : str
        Path to input molecule file (MOL, SDF, etc.)
    output_file : str, optional
        Path to output SDF file. If None, a temporary file is created.
    add_hydrogens : bool
        Whether to add hydrogens
    generate_3d : bool
        Whether to generate 3D coordinates if not present
    minimize : bool
        Whether to minimize the molecule
    n_conformers : int
        Number of conformers to generate
    
    Returns:
    --------
    str
        Path to prepared ligand SDF file
    """
    if output_file is None:
        # Create temporary output file
        fd, output_file = tempfile.mkstemp(suffix='.sdf')
        os.close(fd)
    
    if RDKIT_AVAILABLE:
        # Use RDKit for ligand preparation
        print(f"Preparing ligand using RDKit...")
        
        # Read molecule
        mol = Chem.MolFromMolFile(mol_file, removeHs=False)
        if mol is None:
            raise ValueError(f"Failed to read molecule file: {mol_file}")
        
        if add_hydrogens:
            # Add hydrogens
            mol = Chem.AddHs(mol)
            print("  Added hydrogens")
        
        if generate_3d:
            # Check if 3D coordinates exist
            has_3d = False
            for atom in mol.GetAtoms():
                pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
                if pos.x != 0 or pos.y != 0 or pos.z != 0:
                    has_3d = True
                    break
            
            if not has_3d:
                # Generate 3D coordinates
                Chem.AllChem.EmbedMolecule(mol, Chem.AllChem.ETKDG())
                print("  Generated 3D coordinates")
        
        if minimize:
            # Minimize the molecule
            Chem.AllChem.MMFFOptimizeMolecule(mol)
            print("  Minimized molecule energy")
        
        if n_conformers > 1:
            # Generate conformers
            conformers = []
            
            # Keep the original conformer
            original_mol = Chem.Mol(mol)
            conformers.append(original_mol)
            
            # Generate additional conformers
            Chem.AllChem.EmbedMultipleConfs(
                mol, numConfs=n_conformers-1, 
                useExpTorsionAnglePrefs=True,
                useBasicKnowledge=True
            )
            
            # Minimize each conformer
            for conf_id in range(mol.GetNumConformers()):
                if minimize:
                    Chem.AllChem.MMFFOptimizeMoleculeConfs(mol)
                
                # Create a new molecule for this conformer
                conf_mol = Chem.Mol(mol)
                conformers.append(conf_mol)
            
            print(f"  Generated {n_conformers} conformers")
            
            # Write all conformers to SDF
            writer = Chem.SDWriter(output_file)
            for conf_mol in conformers:
                writer.write(conf_mol)
            writer.close()
        else:
            # Write single molecule to SDF
            writer = Chem.SDWriter(output_file)
            writer.write(mol)
            writer.close()
        
        print(f"Prepared ligand saved to {output_file}")
        
    elif OPENBABEL_AVAILABLE:
        # Use Open Babel for ligand preparation
        print(f"Preparing ligand using Open Babel...")
        
        # Read molecule
        mol = next(pybel.readfile(os.path.splitext(mol_file)[1][1:], mol_file))
        
        if add_hydrogens:
            # Add hydrogens
            mol.OBMol.AddHydrogens()
            print("  Added hydrogens")
        
        if generate_3d:
            # Generate 3D coordinates if needed
            if mol.dim < 3:
                mol.make3D()
                print("  Generated 3D coordinates")
        
        if minimize:
            # Minimize the molecule
            mol.localopt()
            print("  Minimized molecule energy")
        
        # Write output file
        mol.write('sdf', output_file, overwrite=True)
        print(f"Prepared ligand saved to {output_file}")
        
    else:
        print("Neither RDKit nor Open Babel are available.")
        print("Using original molecule file without preparation.")
        
        # Try to copy file to output format
        import shutil
        shutil.copy(mol_file, output_file)
    
    return output_file

