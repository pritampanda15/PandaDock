# tests/test_protein.py
import unittest
import os
import tempfile
from pandadock.protein import Protein

class TestProtein(unittest.TestCase):
    def setUp(self):
        # Create a minimal PDB file for testing
        self.temp_pdb = tempfile.NamedTemporaryFile(delete=False, suffix='.pdb')
        with open(self.temp_pdb.name, 'w') as f:
            f.write("ATOM      1  N   ASP A  30      31.550  -0.504  21.857  1.00 17.05           N\n")
            f.write("ATOM      2  CA  ASP A  30      30.539  -0.599  22.931  1.00 20.52           C\n")
            f.write("ATOM      3  C   ASP A  30      29.120  -0.525  22.374  1.00 21.35           C\n")
        
    def tearDown(self):
        os.unlink(self.temp_pdb.name)
    
    def test_load_pdb(self):
        protein = Protein(self.temp_pdb.name)
        self.assertEqual(len(protein.atoms), 3)
        self.assertEqual(protein.atoms[0]['residue_name'], 'ASP')
