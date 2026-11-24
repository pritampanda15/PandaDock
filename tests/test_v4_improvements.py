"""Regression and correctness tests for v4.1+ physics, GPU, GNN, and flex-docking fixes."""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

pytest.importorskip("rdkit", reason="RDKit required")
pytest.importorskip("Bio", reason="BioPython required")

from Bio.PDB import PDBParser  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402


def _make_mol(smiles: str, seed: int = 42) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    assert AllChem.EmbedMolecule(mol, params) == 0
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def _write_mini_receptor(tmp_path) -> str:
    """Minimal receptor with ALA, PRO, TYR, LYS, TRP, ASP, HIS, PHE residues."""
    lines = [
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N",
        "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 20.00           C",
        "ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00 20.00           C",
        "ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00 20.00           O",
        "ATOM      5  CB  ALA A   1       1.986  -0.760  -1.217  1.00 20.00           C",
        "ATOM      6  N   PRO A   2       3.320   1.520   0.000  1.00 20.00           N",
        "ATOM      7  CA  PRO A   2       4.000   2.800   0.000  1.00 20.00           C",
        "ATOM      8  CB  PRO A   2       5.450   2.400  -0.300  1.00 20.00           C",
        "ATOM      9  CG  PRO A   2       5.500   0.950   0.050  1.00 20.00           C",
        "ATOM     10  CD  PRO A   2       4.200   0.400   0.400  1.00 20.00           C",
        "ATOM     11  CB  TYR A   3       8.000   0.000   0.000  1.00 20.00           C",
        "ATOM     12  CG  TYR A   3       9.500   0.000   0.000  1.00 20.00           C",
        "ATOM     13  CZ  TYR A   3      12.300   0.000   0.000  1.00 20.00           C",
        "ATOM     14  OH  TYR A   3      13.650   0.000   0.000  1.00 20.00           O",
        "ATOM     15  CB  LYS A   4       8.000   5.000   0.000  1.00 20.00           C",
        "ATOM     16  CG  LYS A   4       9.500   5.000   0.000  1.00 20.00           C",
        "ATOM     17  CD  LYS A   4      10.200   6.300   0.000  1.00 20.00           C",
        "ATOM     18  CE  LYS A   4      11.700   6.300   0.000  1.00 20.00           C",
        "ATOM     19  NZ  LYS A   4      12.400   7.600   0.000  1.00 20.00           N",
        "ATOM     20  NE1 TRP A   5       8.000  10.000   0.000  1.00 20.00           N",
        "ATOM     21  ND1 HIS A   6       8.000  14.000   0.000  1.00 20.00           N",
        "ATOM     22  OD1 ASP A   7       8.000  18.000   0.000  1.00 20.00           O",
        "ATOM     23  CA  PHE A   8       0.000  10.000   0.000  1.00 20.00           C",
        "ATOM     24  CB  PHE A   8       1.500  10.000   0.000  1.00 20.00           C",
        "ATOM     25  CG  PHE A   8       2.250  11.300   0.000  1.00 20.00           C",
        "ATOM     26  CD1 PHE A   8       3.650  11.300   0.000  1.00 20.00           C",
        "ATOM     27  C   PHE A   8      -0.750  11.300   0.000  1.00 20.00           C",
        "ATOM     28  O   PHE A   8      -0.150  12.400   0.000  1.00 20.00           O",
        "ATOM     29  N   PHE A   8      -0.750   8.750   0.000  1.00 20.00           N",
        "END",
    ]
    pdb_path = tmp_path / "mini_rec.pdb"
    pdb_path.write_text("\n".join(lines) + "\n")
    return str(pdb_path)


# ------------------------------------------------------------------ Step 1


def test_vina_receptor_hydrophobic_and_hbond_typing(tmp_path):
    from pandadock.docking.scoring.vina_scoring import VinaScoring

    pdb_path = _write_mini_receptor(tmp_path)
    struct = PDBParser(QUIET=True).get_structure("rec", pdb_path)
    atoms = list(struct.get_atoms())
    vina = VinaScoring()

    hydro = vina._is_hydrophobic_receptor(atoms)
    donors, acceptors = vina._get_hbond_atoms_receptor(atoms)
    donor_set, acceptor_set = set(donors), set(acceptors)

    by_key = {(a.get_parent().get_resname().strip(), a.get_name().strip()): i for i, a in enumerate(atoms)}

    # Backbone C / CA are never hydrophobic; ALA CB, TYR CG, LYS CD are hydrophobic; TYR CZ and LYS CE are NOT
    assert not hydro[by_key[("ALA", "CA")]]
    assert not hydro[by_key[("ALA", "C")]]
    assert hydro[by_key[("ALA", "CB")]]
    assert hydro[by_key[("TYR", "CG")]]
    assert not hydro[by_key[("TYR", "CZ")]]
    assert hydro[by_key[("LYS", "CD")]]
    assert not hydro[by_key[("LYS", "CE")]]
    assert not hydro[by_key[("PRO", "CD")]]

    # Backbone ALA N: donor only, NOT acceptor
    assert by_key[("ALA", "N")] in donor_set
    assert by_key[("ALA", "N")] not in acceptor_set
    # PRO N: neither donor nor acceptor
    assert by_key[("PRO", "N")] not in donor_set
    assert by_key[("PRO", "N")] not in acceptor_set
    # LYS NZ and TRP NE1: donors only, NOT acceptors
    assert by_key[("LYS", "NZ")] in donor_set and by_key[("LYS", "NZ")] not in acceptor_set
    assert by_key[("TRP", "NE1")] in donor_set and by_key[("TRP", "NE1")] not in acceptor_set
    # HIS ND1 and TYR OH: both donor and acceptor; ASP OD1 / ALA O: acceptor only
    assert by_key[("HIS", "ND1")] in donor_set and by_key[("HIS", "ND1")] in acceptor_set
    assert by_key[("TYR", "OH")] in donor_set and by_key[("TYR", "OH")] in acceptor_set
    assert by_key[("ASP", "OD1")] in acceptor_set and by_key[("ASP", "OD1")] not in donor_set
    assert by_key[("ALA", "O")] in acceptor_set and by_key[("ALA", "O")] not in donor_set


def test_vina_ligand_hbond_typing_and_vinardo_preset():
    from pandadock.docking.scoring.vina_scoring import VinaScoring

    vina = VinaScoring()
    vinardo = VinaScoring.vinardo()
    assert vinardo.preset == "vinardo"
    assert vinardo.weights["gauss2"] == 0.0
    assert np.isclose(vina.normalize_energy(-10.0, 10), -10.0 / (1.0 + 0.05846 * 10))

    # Quaternary ammonium: neither donor nor acceptor
    quat = Chem.AddHs(Chem.MolFromSmiles("C[N+](C)(C)C"))
    d, a = vina._get_hbond_atoms_ligand(quat)
    assert d == [] and a == []

    # Protonated primary amine: donor only, NOT acceptor
    amine = Chem.AddHs(Chem.MolFromSmiles("CC[NH3+]"))
    d, a = vina._get_hbond_atoms_ligand(amine)
    assert len(d) == 1 and a == []

    # Acetamide: amide N is donor only (NOT acceptor), carbonyl O is acceptor only
    amide = Chem.AddHs(Chem.MolFromSmiles("CC(=O)NC"))
    d, a = vina._get_hbond_atoms_ligand(amide)
    n_idx = [at.GetIdx() for at in amide.GetAtoms() if at.GetSymbol() == "N"][0]
    o_idx = [at.GetIdx() for at in amide.GetAtoms() if at.GetSymbol() == "O"][0]
    assert n_idx in d and n_idx not in a
    assert o_idx in a and o_idx not in d

    # Pyrrole NH: donor only; Pyridine N: acceptor only
    pyrrole = Chem.AddHs(Chem.MolFromSmiles("c1cc[nH]c1"))
    d, a = vina._get_hbond_atoms_ligand(pyrrole)
    assert len(d) == 1 and a == []

    pyridine = Chem.AddHs(Chem.MolFromSmiles("c1ccncc1"))
    d, a = vina._get_hbond_atoms_ligand(pyridine)
    assert d == [] and len(a) == 1


def test_pandacore_executes_physics_based_rescoring_and_ring_sampling(tmp_path):
    from pandadock.docking.algorithms.pandacore import PandaCoreDocker
    from pandadock.docking.scoring.physics_based import PhysicsBasedScoring

    pdb_path = _write_mini_receptor(tmp_path)
    mol = _make_mol("C1CCCCC1O")  # flexible cyclohexane ring

    docker = PandaCoreDocker()
    docker.set_scoring_function(PhysicsBasedScoring())
    res = docker.dock(
        pdb_path,
        mol,
        grid_center=np.array([5.0, 5.0, 0.0]),
        grid_dimensions=np.array([14.0, 14.0, 14.0]),
        num_poses=3,
        exhaustiveness=1,
        n_steps=5,
        seed=7,
        sample_rings=True,
        max_ring_confs=2,
    )
    assert res.poses
    assert "rescored" in res.poses[0].energy_components
    assert "vina_grid" in res.poses[0].energy_components
    assert res.poses[0].energy == res.poses[0].energy_components["rescored"]


# ------------------------------------------------------------------ Step 2


def test_batched_lbfgs_energy_fn_cuts_gradient_calls_and_matches():
    torch = pytest.importorskip("torch")
    from pandadock.docking.gpu.optimize import LBFGSConfig, batched_lbfgs

    rng = np.random.default_rng(3)
    a = torch.tensor(rng.normal(0, 1, (4, 4)), dtype=torch.float64)
    a = a @ a.T + 4.0 * torch.eye(4, dtype=torch.float64)
    b = torch.tensor(rng.normal(0, 1, 4), dtype=torch.float64)
    x0 = torch.tensor(rng.normal(0, 2, (16, 4)), dtype=torch.float64)

    grad_calls = {"only_closure": 0, "with_energy_fn": 0}

    def closure_a(x):
        grad_calls["only_closure"] += 1
        return 0.5 * ((x @ a) * x).sum(-1) - x @ b, x @ a - b

    def closure_b(x):
        grad_calls["with_energy_fn"] += 1
        return 0.5 * ((x @ a) * x).sum(-1) - x @ b, x @ a - b

    def energy_only(x):
        return 0.5 * ((x @ a) * x).sum(-1) - x @ b

    xa, ea, ca = batched_lbfgs(x0, closure_a, LBFGSConfig(max_iter=40))
    xb, eb, cb = batched_lbfgs(x0, closure_b, LBFGSConfig(max_iter=40), energy_fn=energy_only)

    assert bool(ca.all()) and bool(cb.all())
    assert torch.allclose(xa, xb, atol=1e-6)
    assert torch.allclose(ea, eb, atol=1e-8)
    assert grad_calls["with_energy_fn"] <= grad_calls["only_closure"]


def test_cpu_mc_multi_worker_is_deterministic_and_matches_single_worker(tmp_path):
    from pandadock.docking.search import (
        AffinityGrids,
        DockingObjective,
        MonteCarloSearch,
        SearchConfig,
        TorsionTree,
    )

    pdb_path = _write_mini_receptor(tmp_path)
    struct = PDBParser(QUIET=True).get_structure("rec", pdb_path)
    mol = _make_mol("CCOC(=O)c1ccccc1")
    center = np.array([5.0, 5.0, 0.0])
    dims = np.array([14.0, 14.0, 14.0])
    grids = AffinityGrids.build(struct, mol, center, dims, spacing=0.5)
    tree = TorsionTree(mol)

    obj1 = DockingObjective(tree, grids)
    obj2 = DockingObjective(tree, grids)
    res1 = MonteCarloSearch(obj1, SearchConfig(exhaustiveness=4, n_steps=6, seed=17, n_workers=1)).run(
        center - dims / 2, center + dims / 2
    )
    res2 = MonteCarloSearch(obj2, SearchConfig(exhaustiveness=4, n_steps=6, seed=17, n_workers=2)).run(
        center - dims / 2, center + dims / 2
    )

    assert np.isclose(res1[0].energy, res2[0].energy, atol=1e-9)
    assert np.allclose(res1[0].coords, res2[0].coords, atol=1e-9)
    assert obj2.n_evaluations == obj1.n_evaluations


# ------------------------------------------------------------------ Step 3


def test_gnn_metadata_pose_scoring_hotspots_and_losses(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from pandadock.gnn.models.pandadock_gnn import ModelConfig, PandaDockGNN
    from pandadock.gnn.scoring import GNNScoring
    from pandadock.gnn.training.losses import pairwise_ranking_loss, pose_decoy_loss

    cfg = ModelConfig(
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        affinity_hidden=32,
        strip_hydrogens=True,
        site_radius=8.5,
        training_dataset="sair",
    )
    model = PandaDockGNN(cfg)
    ckpt = tmp_path / "sair_meta.pt"
    model.save(str(ckpt))

    scorer = GNNScoring(model_path=str(ckpt), device="cpu")
    assert scorer.strip_hydrogens is True
    assert scorer.site_radius == 8.5

    pdb_path = _write_mini_receptor(tmp_path)
    struct = PDBParser(QUIET=True).get_structure("rec", pdb_path)
    mol = _make_mol("c1ccccc1O")
    coords = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float64) + np.array([5.0, 5.0, 0.0])

    # Direct pose scoring works without raising NotImplementedError
    energy = scorer.calculate_binding_energy(coords, struct, mol)
    assert np.isfinite(energy)

    # Binding hotspots returns mapped protein residues and ligand atoms
    lig_sdf = tmp_path / "lig.sdf"
    w = Chem.SDWriter(str(lig_sdf))
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, coords[i].tolist())
    w.write(mol)
    w.close()
    hotspots = scorer.get_binding_hotspots(pdb_path, str(lig_sdf), top_k=5)
    assert hotspots["attention_available"] is True
    assert len(hotspots["top_protein_residues"]) > 0
    assert len(hotspots["top_ligand_atoms"]) > 0

    # Pairwise ranking and pose decoy losses
    preds = torch.tensor([5.0, 7.0, 4.0, 6.0], requires_grad=True)
    targets = torch.tensor([8.0, 5.0, 6.0, 4.0])
    groups = torch.tensor([0, 0, 1, 1])
    loss_rank = pairwise_ranking_loss(preds, targets, groups)
    assert loss_rank is not None and float(loss_rank) > 0.0
    loss_rank.backward()
    assert preds.grad is not None

    pose_scores = torch.tensor([6.0, 7.5, 5.0], requires_grad=True)
    rmsds = torch.tensor([1.0, 6.5, 5.2])
    loss_decoy = pose_decoy_loss(pose_scores, rmsds)
    assert loss_decoy is not None and float(loss_decoy) > 0.0


def test_target_grouped_batch_sampler_packs_same_target_ligands():
    from pandadock.gnn.data.sair_dataset import TargetGroupedBatchSampler

    class _FakeDataset:
        def __init__(self):
            # 4 targets x 8 ligands each across 2 shards
            self.index = [(i // 16, i) for i in range(32)]
            self.entry_target = {i: i % 4 for i in range(32)}

    ds = _FakeDataset()
    sampler = TargetGroupedBatchSampler(ds, batch_size=8, ligands_per_target=4, block_shards=2, seed=0)
    batches = list(sampler)
    assert sum(len(b) for b in batches) == 32
    # Every batch must have repeated targets (at least one target with >=2 ligands)
    for batch in batches:
        t_counts = {}
        for pos in batch:
            tid = ds.entry_target[ds.index[pos][1]]
            t_counts[tid] = t_counts.get(tid, 0) + 1
        assert max(t_counts.values()) >= 2


# ------------------------------------------------------------------ Step 4 & 5


def test_flex_docking_real_chi_rotation_rmsd_and_strain(tmp_path):
    from pandadock.docking.core import Pose
    from pandadock.flex_docking.core import FlexiblePose, FlexibleResidue
    from pandadock.flex_docking.phases import IFDScorer, ReceptorRefiner

    pdb_path = _write_mini_receptor(tmp_path)
    struct = PDBParser(QUIET=True).get_structure("rec", pdb_path)
    phe = struct[0]["A"][(" ", 8, " ")]
    atoms = list(phe.get_atoms())
    orig = np.array([a.get_coord() for a in atoms], dtype=np.float64)
    names = [a.get_name().strip().upper() for a in atoms]

    fr = FlexibleResidue("A", 8, "PHE", orig, [], 0.0)
    fr.atom_names = names
    pose = Pose(coordinates=np.array([[3.5, 11.0, 0.0]]), center=np.array([3.5, 11.0, 0.0]),
                rotation=np.array([0, 0, 0, 1.0]), conformer_id=0)

    refiner = ReceptorRefiner()
    rotamers = refiner._sample_rotamers(fr, pose)
    assert len(rotamers) > 1

    # Backbone N, CA, C, O and anchor CB must NOT move across any sampled rotamer
    fixed_idx = [i for i, n in enumerate(names) if n in {"N", "CA", "C", "O", "CB"}]
    moving_idx = [i for i, n in enumerate(names) if n not in {"N", "CA", "C", "O", "CB"}]
    for rot in rotamers:
        assert np.allclose(rot[fixed_idx], orig[fixed_idx], atol=1e-10)
        # Preserve CB-CG bond length
        cb_i, cg_i = names.index("CB"), names.index("CG")
        assert np.isclose(
            np.linalg.norm(rot[cb_i] - rot[cg_i]),
            np.linalg.norm(orig[cb_i] - orig[cg_i]),
            atol=1e-8,
        )
    # At least one rotamer must actually move distal side-chain atoms
    assert any(np.linalg.norm(rot[moving_idx] - orig[moving_idx]) > 0.5 for rot in rotamers)

    # Real receptor RMSD: identical PDBs give 0.0 (not hardcoded 1.5)
    assert refiner._calculate_receptor_rmsd(pdb_path, pdb_path, ["A:8"]) == 0.0

    # Real ligand strain: distorted conformer has strictly higher strain than minimized conformer
    mol = _make_mol("CCCC")
    min_coords = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float64)
    distorted = min_coords.copy()
    distorted[0] += np.array([0.8, 0.0, 0.0])  # stretch C-C bond

    ifd = IFDScorer()
    fp_min = FlexiblePose(coordinates=min_coords, center=min_coords.mean(0),
                          rotation=np.array([0, 0, 0, 1.0]), conformer_id=0)
    fp_bad = FlexiblePose(coordinates=distorted, center=distorted.mean(0),
                          rotation=np.array([0, 0, 0, 1.0]), conformer_id=0)
    assert ifd._calculate_ligand_strain(fp_bad, mol) > ifd._calculate_ligand_strain(fp_min, mol) + 1.0


def test_mmgbsa_real_3d_bond_energy_and_metal_ad4_params():
    from pandadock.docking.refinement.mmgbsa import MMGBSARescoring
    from pandadock.metal_docking.metal_parameters import MetalParameterManager

    mol = _make_mol("CC")
    eq_coords = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float64)
    stretched = eq_coords.copy()
    stretched[1] = eq_coords[0] + (eq_coords[1] - eq_coords[0]) * 1.5

    mmgbsa = MMGBSARescoring()
    e_eq = mmgbsa._calculate_bond_energy(mol, eq_coords)
    e_st = mmgbsa._calculate_bond_energy(mol, stretched)
    assert e_st > e_eq + 10.0

    mgr = MetalParameterManager()
    assert mgr.FALLBACK_METAL_PARAMS["ZN"]["rii"] == 1.48
    assert mgr.FALLBACK_METAL_PARAMS["ZN"]["epsii"] == 0.55
