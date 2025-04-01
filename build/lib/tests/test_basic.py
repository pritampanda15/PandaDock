# tests/test_basic.py
import unittest

class TestBasic(unittest.TestCase):
    def test_import(self):
        # Test that major modules can be imported
        import pandadock
        import pandadock.protein
        import pandadock.ligand
        import pandadock.scoring
        import pandadock.search
