"""
Command-line interface for PandaDock-GNN.

Provides commands for:
- Training models on ULVSH dataset
- Predicting binding affinity
- Benchmarking against baselines
- Downloading pre-trained models
- Visualizing attention weights
"""

import sys
import json
import os
from pathlib import Path
from typing import Optional
import urllib.request
import hashlib

from .. import __version__

try:
    import click
    CLICK_AVAILABLE = True
except ImportError:
    CLICK_AVAILABLE = False
    click = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def check_dependencies():
    """Check if required dependencies are available."""
    missing = []
    if not CLICK_AVAILABLE:
        missing.append('click')
    if not TORCH_AVAILABLE:
        missing.append('torch')

    try:
        import torch_geometric
    except ImportError:
        missing.append('torch-geometric')

    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install " + ' '.join(missing))
        sys.exit(1)


if CLICK_AVAILABLE:
    @click.group()
    @click.version_option(version=__version__)
    def main():
        """PandaDock-GNN: SE(3)-Equivariant Scoring Function for Molecular Docking"""
        pass

    @main.command()
    @click.option('--dataset', '-d', required=True, type=click.Path(exists=True),
                  help='Path to ULVSH dataset directory')
    @click.option('--output', '-o', required=True, type=click.Path(),
                  help='Output directory for checkpoints and logs')
    @click.option('--epochs', default=100, type=int, help='Number of training epochs')
    @click.option('--batch-size', default=32, type=int, help='Batch size')
    @click.option('--lr', default=1e-4, type=float, help='Learning rate')
    @click.option('--hidden-dim', default=256, type=int, help='Hidden dimension')
    @click.option('--num-layers', default=6, type=int, help='Number of EGNN layers')
    @click.option('--dropout', default=0.1, type=float, help='Dropout rate')
    @click.option('--split', type=click.Choice(['random', 'target']), default='random',
                  help='Data split strategy')
    @click.option('--patience', default=20, type=int, help='Early stopping patience')
    @click.option('--gpu/--cpu', default=True, help='Use GPU if available')
    @click.option('--seed', default=42, type=int, help='Random seed')
    def train(dataset, output, epochs, batch_size, lr, hidden_dim, num_layers,
              dropout, split, patience, gpu, seed):
        """Train PandaDock-GNN on ULVSH dataset."""
        check_dependencies()

        import torch
        import numpy as np
        import random

        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Create output directory
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print("PandaDock-GNN Training")
        print("=" * 60)
        print(f"Dataset: {dataset}")
        print(f"Output: {output}")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        print(f"Hidden dim: {hidden_dim}, Layers: {num_layers}")
        print(f"Device: {'cuda' if gpu and torch.cuda.is_available() else 'cpu'}")
        print("=" * 60)

        # Load dataset
        from .data.dataset import ULVSHDataset, create_dataloaders

        print("\nLoading dataset...")
        train_loader, val_loader, test_loader = create_dataloaders(
            root=dataset,
            batch_size=batch_size,
            split_type=split,
            seed=seed
        )

        # Create model
        from .models.pandadock_gnn import PandaDockGNN, ModelConfig

        config = ModelConfig(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            predict_activity=True
        )
        model = PandaDockGNN(config)

        print(f"\nModel parameters: {model.count_parameters():,}")

        # Create trainer
        from .training.trainer import GNNTrainer, TrainingConfig

        train_config = TrainingConfig(
            learning_rate=lr,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            checkpoint_dir=str(output_dir),
            device='cuda' if gpu else 'cpu'
        )
        trainer = GNNTrainer(model, train_config)

        # Train
        print("\nStarting training...")
        results = trainer.train(train_loader, val_loader, test_loader)

        # Save results
        results_file = output_dir / 'training_results.json'
        with open(results_file, 'w') as f:
            # Convert numpy values to float for JSON
            json_results = {
                'best_metrics': {k: float(v) for k, v in results['best_metrics'].items()},
                'elapsed_time': results['elapsed_time']
            }
            if results['test_metrics']:
                json_results['test_metrics'] = {k: float(v) for k, v in results['test_metrics'].items()}
            json.dump(json_results, f, indent=2)

        print(f"\nResults saved to {results_file}")
        print("\nTraining complete!")

    @main.command('train-sair')
    @click.option('--cache', '-c', required=True, type=click.Path(exists=True),
                  help='Shard cache directory built by benchmarking/sair_preprocess.py')
    @click.option('--output', '-o', required=True, type=click.Path(),
                  help='Output directory for checkpoints and logs')
    @click.option('--epochs', default=50, type=int, help='Number of training epochs')
    @click.option('--batch-size', default=64, type=int, help='Batch size')
    @click.option('--lr', default=1e-4, type=float, help='Learning rate')
    @click.option('--hidden-dim', default=256, type=int, help='Hidden dimension')
    @click.option('--num-layers', default=6, type=int, help='Number of EGNN layers')
    @click.option('--dropout', default=0.1, type=float, help='Dropout rate')
    @click.option('--patience', default=10, type=int, help='Early stopping patience')
    @click.option('--num-workers', default=4, type=int,
                  help='Dataloader worker processes')
    @click.option('--block-shards', default=16, type=int,
                  help='Shards shuffled together; larger mixes more but uses more memory')
    @click.option('--gpu/--cpu', default=True, help='Use GPU if available')
    @click.option('--seed', default=42, type=int, help='Random seed')
    def train_sair(cache, output, epochs, batch_size, lr, hidden_dim, num_layers,
                   dropout, patience, num_workers, block_shards, gpu, seed):
        """
        Train PandaDock-GNN on the SAIR shard cache.

        Splits are target-disjoint: SAIR holds many ligands per protein, so a
        random split would place near-identical complexes on both sides.
        """
        check_dependencies()

        import random

        import numpy as np
        import torch

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)

        device = 'cuda' if gpu and torch.cuda.is_available() else 'cpu'
        print("=" * 60)
        print("PandaDock-GNN Training (SAIR)")
        print("=" * 60)
        print(f"Cache: {cache}")
        print(f"Output: {output}")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        print(f"Hidden dim: {hidden_dim}, Layers: {num_layers}")
        print(f"Device: {device}")
        print("=" * 60)

        from .data.sair_dataset import create_sair_dataloaders

        print("\nLoading dataset (first run indexes the shards, which takes a few minutes)...")
        train_loader, val_loader, test_loader = create_sair_dataloaders(
            cache_dir=cache,
            batch_size=batch_size,
            seed=seed,
            num_workers=num_workers,
            block_shards=block_shards,
        )
        print(f"  train {len(train_loader.dataset):,}  "
              f"val {len(val_loader.dataset):,}  "
              f"test {len(test_loader.dataset):,}")

        from .models.pandadock_gnn import PandaDockGNN, ModelConfig

        config = ModelConfig(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            predict_activity=False,
        )
        model = PandaDockGNN(config)
        print(f"\nModel parameters: {model.count_parameters():,}")

        from .training.trainer import GNNTrainer, TrainingConfig

        train_config = TrainingConfig(
            learning_rate=lr,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            checkpoint_dir=str(output_dir),
            device=device,
        )
        trainer = GNNTrainer(model, train_config)

        print("\nStarting training...")
        results = trainer.train(train_loader, val_loader, test_loader)

        results_file = output_dir / 'training_results.json'
        with open(results_file, 'w') as f:
            json_results = {
                'best_metrics': {k: float(v) for k, v in results['best_metrics'].items()},
                'elapsed_time': results['elapsed_time'],
                'n_train': len(train_loader.dataset),
                'n_val': len(val_loader.dataset),
                'n_test': len(test_loader.dataset),
            }
            if results['test_metrics']:
                json_results['test_metrics'] = {
                    k: float(v) for k, v in results['test_metrics'].items()
                }
            json.dump(json_results, f, indent=2)

        print(f"\nResults saved to {results_file}")
        print("\nTraining complete!")

    @main.command()
    @click.option('--model', '-m', required=True, type=click.Path(exists=True),
                  help='Path to trained model checkpoint')
    @click.option('--protein', '-p', required=True, type=click.Path(exists=True),
                  help='Protein MOL2 file')
    @click.option('--ligand', '-l', required=True, type=click.Path(exists=True),
                  help='Ligand MOL2 file')
    @click.option('--site', '-s', type=click.Path(exists=True),
                  help='Optional binding site MOL2 file')
    @click.option('--output', '-o', type=click.Path(), help='Output JSON file')
    def predict(model, protein, ligand, site, output):
        """Predict binding affinity for a protein-ligand complex."""
        check_dependencies()

        from .scoring import GNNScoring

        print("Loading model...")
        scorer = GNNScoring(model_path=model)

        print("Predicting...")
        result = scorer.predict_affinity(protein, ligand, site)

        print("\nPrediction Results:")
        print(f"  pEC50: {result['pec50']:.3f}")
        print(f"  Energy: {result['energy']:.3f} kcal/mol")
        if 'activity_prob' in result:
            print(f"  Activity probability: {result['activity_prob']:.3f}")
            print(f"  Predicted active: {result['active']}")

        if output:
            with open(output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to {output}")

    @main.command()
    @click.option('--model', '-m', required=True, type=click.Path(exists=True),
                  help='Path to trained model checkpoint')
    @click.option('--dataset', '-d', required=True, type=click.Path(exists=True),
                  help='Path to ULVSH dataset directory')
    @click.option('--output', '-o', required=True, type=click.Path(),
                  help='Output directory for results')
    @click.option('--split', default='test', type=click.Choice(['train', 'val', 'test']),
                  help='Dataset split to evaluate')
    @click.option('--compare-baselines', is_flag=True,
                  help='Compare against baseline scores from scores.tsv')
    def benchmark(model, dataset, output, split, compare_baselines):
        """Benchmark model performance on test set."""
        check_dependencies()

        import torch
        import numpy as np
        import pandas as pd

        from .data.dataset import ULVSHDataset
        from .scoring import GNNScoring
        from .training.metrics import compute_affinity_metrics, compute_activity_metrics

        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("Loading model...")
        scorer = GNNScoring(model_path=model)

        print(f"Loading {split} dataset...")
        test_dataset = ULVSHDataset(root=dataset, split=split)

        print(f"Evaluating {len(test_dataset)} complexes...")

        predictions = []
        targets = []
        compound_ids = []

        for i in range(len(test_dataset)):
            try:
                compound = test_dataset.get_compound_info(i)
                result = scorer.predict_affinity(
                    compound.protein_mol2,
                    compound.ligand_mol2,
                    compound.site_mol2
                )
                predictions.append(result['pec50'])
                targets.append(compound.pec50)
                compound_ids.append(compound.compound_id)

                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{len(test_dataset)}")
            except Exception as e:
                print(f"  Warning: Failed for {compound.compound_id}: {e}")

        predictions = np.array(predictions)
        targets = np.array(targets)

        # Compute metrics
        metrics = compute_affinity_metrics(targets, predictions)

        print("\n" + "=" * 50)
        print("Benchmark Results")
        print("=" * 50)
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

        # Save results
        results_df = pd.DataFrame({
            'compound_id': compound_ids,
            'true_pec50': targets,
            'pred_pec50': predictions
        })
        results_df.to_csv(output_dir / 'predictions.csv', index=False)

        # Compare with baselines if requested
        if compare_baselines:
            print("\nComparing with baselines...")
            # Load baseline scores
            baseline_metrics = {}
            baseline_columns = [
                'Hyde', 'DeltaVina', 'Gnina', 'MMPBSA', 'MMGBSA',
                'VM2', 'GFN-FF', 'PM6'
            ]

            for target in test_dataset.targets:
                scores_file = Path(dataset) / target / 'raw' / 'scores.tsv'
                if scores_file.exists():
                    scores_df = pd.read_csv(scores_file, sep='\t')
                    for col in baseline_columns:
                        if col in scores_df.columns and col not in baseline_metrics:
                            baseline_metrics[col] = {'preds': [], 'targets': []}
                        if col in scores_df.columns:
                            for _, row in scores_df.iterrows():
                                if row['ID'] in compound_ids:
                                    idx = compound_ids.index(row['ID'])
                                    baseline_metrics[col]['preds'].append(-row[col])  # Negative for energy
                                    baseline_metrics[col]['targets'].append(targets[idx])

            print("\nBaseline Comparisons (Pearson R):")
            print(f"  PandaDock-GNN: {metrics['pearson_r']:.4f}")
            for name, data in baseline_metrics.items():
                if len(data['preds']) > 10:
                    r = compute_affinity_metrics(
                        np.array(data['targets']),
                        np.array(data['preds'])
                    )['pearson_r']
                    print(f"  {name}: {r:.4f}")

        # Save metrics
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

        print(f"\nResults saved to {output_dir}")

    @main.command()
    @click.option('--model', '-m', required=True, type=click.Path(exists=True),
                  help='Path to trained model checkpoint')
    @click.option('--receptor', '-r', required=True, type=click.Path(exists=True),
                  help='Receptor file (PDB, MOL2)')
    @click.option('--poses', '-p', required=True, type=click.Path(exists=True),
                  help='Poses file (multi-conformer SDF from any docking tool)')
    @click.option('--output', '-o', type=click.Path(), default='rescored_poses.csv',
                  help='Output CSV file with ranked poses')
    @click.option('--output-sdf', type=click.Path(),
                  help='Optional: Output SDF file with poses ranked by GNN score')
    @click.option('--site-radius', type=float, default=10.0,
                  help='Radius around ligand centroid to extract binding site (Å)')
    @click.option('--batch-size', type=int, default=1,
                  help='Batch size for processing (1 = safest)')
    def rescore(model, receptor, poses, output, output_sdf, site_radius, batch_size):
        """
        Universal GNN rescorer for poses from any docking tool.

        Takes docked poses (from pandadock-flex, pandadock-metal, pandadock-tethered,
        or any other docking software) and rescores them using the GNN model.

        Examples:
            # Rescore poses from pandadock-flex
            pandadock gnn rescore -m model.pt -r protein.pdb -p flex_output/poses.sdf

            # Rescore with custom output
            pandadock gnn rescore -m model.pt -r protein.pdb -p poses.sdf -o ranked.csv --output-sdf ranked.sdf

            # Rescore poses from AutoDock Vina
            pandadock gnn rescore -m model.pt -r receptor.pdb -p vina_out.sdf
        """
        check_dependencies()

        import torch
        import numpy as np
        import pandas as pd
        from pathlib import Path
        from rdkit import Chem
        from rdkit.Chem import AllChem

        from .scoring import GNNScoring
        from .data.graph_builder import HeterogeneousGraphBuilder, parse_molecule_file

        print("=" * 60)
        print("PandaDock-GNN Universal Rescorer")
        print("=" * 60)
        print(f"Model: {model}")
        print(f"Receptor: {receptor}")
        print(f"Poses: {poses}")
        print(f"Site radius: {site_radius} Å")
        print("=" * 60)

        # Load model
        print("\nLoading GNN model...")
        scorer = GNNScoring(model_path=model)
        graph_builder = HeterogeneousGraphBuilder()

        # Load receptor
        print("Parsing receptor...")
        receptor_mol = parse_molecule_file(receptor)
        receptor_coords = np.array([[a.x, a.y, a.z] for a in receptor_mol.atoms])

        # Load poses from SDF
        print("Loading poses from SDF...")
        pose_supplier = Chem.SDMolSupplier(poses, removeHs=False)

        all_poses = []
        for i, mol in enumerate(pose_supplier):
            if mol is not None:
                # Try to get pose name/ID
                try:
                    name = mol.GetProp('_Name') if mol.HasProp('_Name') else f'pose_{i+1}'
                except:
                    name = f'pose_{i+1}'
                all_poses.append((name, mol))

        if not all_poses:
            print("Error: No valid poses found in SDF file")
            sys.exit(1)

        print(f"Found {len(all_poses)} poses to rescore")

        # Function to extract binding site around ligand
        def extract_site(receptor_mol, ligand_centroid, radius):
            """Extract protein atoms within radius of ligand centroid."""
            from .data.mol2_parser import ParsedMolecule, Atom

            site_atoms = []
            for atom in receptor_mol.atoms:
                dist = np.sqrt(
                    (atom.x - ligand_centroid[0])**2 +
                    (atom.y - ligand_centroid[1])**2 +
                    (atom.z - ligand_centroid[2])**2
                )
                if dist <= radius:
                    site_atoms.append(atom)

            if not site_atoms:
                return receptor_mol  # Return full receptor if no atoms in radius

            site = ParsedMolecule(
                name=f"{receptor_mol.name}_site",
                atoms=site_atoms,
                bonds=[]
            )
            site.num_atoms = len(site_atoms)
            return site

        # Function to convert RDKit mol to ParsedMolecule
        def rdkit_to_parsed(mol, name="ligand"):
            """Convert RDKit molecule to ParsedMolecule."""
            from .data.mol2_parser import ParsedMolecule, Atom, Bond

            conf = mol.GetConformer()
            atoms = []

            # Hybridization mapping
            hybridization_map = {
                Chem.HybridizationType.SP: '.1',
                Chem.HybridizationType.SP2: '.2',
                Chem.HybridizationType.SP3: '.3',
            }

            for rdkit_atom in mol.GetAtoms():
                idx = rdkit_atom.GetIdx()
                element = rdkit_atom.GetSymbol()
                pos = conf.GetAtomPosition(idx)

                # Determine SYBYL type
                hyb = rdkit_atom.GetHybridization()
                suffix = hybridization_map.get(hyb, '.3')
                if rdkit_atom.GetIsAromatic():
                    sybyl_type = f'{element}.ar'
                else:
                    sybyl_type = f'{element}{suffix}'

                atoms.append(Atom(
                    id=idx + 1,
                    name=f'{element}{idx + 1}',
                    x=pos.x, y=pos.y, z=pos.z,
                    atom_type=sybyl_type,
                    residue_id=1,
                    residue_name='LIG',
                    charge=0.0
                ))

            # Get bonds
            bonds = []
            bond_type_map = {
                Chem.BondType.SINGLE: '1',
                Chem.BondType.DOUBLE: '2',
                Chem.BondType.TRIPLE: '3',
                Chem.BondType.AROMATIC: 'ar',
            }
            for bond in mol.GetBonds():
                bt = bond_type_map.get(bond.GetBondType(), '1')
                bonds.append(Bond(
                    id=bond.GetIdx() + 1,
                    atom1_id=bond.GetBeginAtomIdx() + 1,
                    atom2_id=bond.GetEndAtomIdx() + 1,
                    bond_type=bt
                ))

            parsed = ParsedMolecule(name=name, atoms=atoms, bonds=bonds)
            parsed.num_atoms = len(atoms)
            parsed.num_bonds = len(bonds)
            return parsed

        # Score each pose
        results = []
        scored_mols = []

        print("\nScoring poses...")
        for i, (name, mol) in enumerate(all_poses):
            try:
                # Add hydrogens if needed
                mol = Chem.AddHs(mol, addCoords=True)

                # Get ligand centroid
                conf = mol.GetConformer()
                coords = np.array([conf.GetAtomPosition(j) for j in range(mol.GetNumAtoms())])
                centroid = coords.mean(axis=0)

                # Extract binding site
                site_mol = extract_site(receptor_mol, centroid, site_radius)

                # Convert ligand to ParsedMolecule
                ligand_parsed = rdkit_to_parsed(mol, name=name)

                # Build graph
                graph = graph_builder.build_graph(site_mol, ligand_parsed)

                # Predict
                result = scorer.predict_from_graph(graph)

                # Store result
                results.append({
                    'pose_name': name,
                    'pose_index': i + 1,
                    'gnn_pKd': result['pec50'],
                    'gnn_energy': result['energy'],
                    'activity_prob': result.get('activity_prob', None),
                    'predicted_active': result.get('active', None),
                    'site_atoms': site_mol.num_atoms,
                    'ligand_atoms': ligand_parsed.num_atoms
                })

                # Store mol with score for SDF output
                mol.SetProp('GNN_pKd', f"{result['pec50']:.3f}")
                mol.SetProp('GNN_Energy', f"{result['energy']:.3f}")
                if result.get('activity_prob') is not None:
                    mol.SetProp('GNN_Activity', f"{result['activity_prob']:.3f}")
                scored_mols.append((result['pec50'], mol))

                if (i + 1) % 10 == 0 or (i + 1) == len(all_poses):
                    print(f"  Processed {i + 1}/{len(all_poses)} poses", flush=True)

            except Exception as e:
                print(f"  Warning: Failed to score pose {name}: {e}")
                results.append({
                    'pose_name': name,
                    'pose_index': i + 1,
                    'gnn_pKd': None,
                    'gnn_energy': None,
                    'activity_prob': None,
                    'predicted_active': None,
                    'error': str(e)
                })

        # Create results DataFrame and sort by GNN score
        df = pd.DataFrame(results)
        df_valid = df[df['gnn_pKd'].notna()].copy()
        df_valid = df_valid.sort_values('gnn_pKd', ascending=False)  # Higher pKd = better

        # Add rank
        df_valid['gnn_rank'] = range(1, len(df_valid) + 1)

        # Save CSV
        output_path = Path(output)
        df_valid.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

        # Save ranked SDF if requested
        if output_sdf:
            # Sort by GNN score (descending)
            scored_mols.sort(key=lambda x: x[0], reverse=True)

            writer = Chem.SDWriter(output_sdf)
            for rank, (score, mol) in enumerate(scored_mols, 1):
                mol.SetProp('GNN_Rank', str(rank))
                writer.write(mol)
            writer.close()
            print(f"Ranked SDF saved to: {output_sdf}")

        # Print summary
        print("\n" + "=" * 60)
        print("RESCORING SUMMARY")
        print("=" * 60)
        print(f"Total poses: {len(all_poses)}")
        print(f"Successfully scored: {len(df_valid)}")
        if len(df_valid) > 0:
            print(f"\nTop 5 poses by GNN score:")
            print("-" * 50)
            for _, row in df_valid.head(5).iterrows():
                print(f"  Rank {int(row['gnn_rank'])}: {row['pose_name']}")
                print(f"    pKd: {row['gnn_pKd']:.3f}, Energy: {row['gnn_energy']:.3f} kcal/mol")
                if row['activity_prob'] is not None:
                    print(f"    Activity: {row['activity_prob']:.3f} ({'Active' if row['predicted_active'] else 'Inactive'})")
        print("=" * 60)

    @main.command('download-model')
    @click.option('--output', '-o', type=click.Path(), default='models/',
                  help='Output directory for the model (default: models/)')
    @click.option('--model', '-m', type=click.Choice(['default', 'bindingdb', 'bindingdb-ulvsh']),
                  default='default', help='Model type to download')
    @click.option('--force', '-f', is_flag=True, default=False,
                  help='Overwrite existing model file')
    @click.option('--list', 'list_models', is_flag=True, default=False,
                  help='List available models')
    def download_model(output, model, force, list_models):
        """
        Download pre-trained PandaDock-GNN model.

        Downloads official pre-trained models from GitHub releases.

        Available models:

        \b
          default       - ULVSH + PDBbind (82 MB, R=0.88 on PDBbind)
          bindingdb     - BindingDB only (28 MB, R=0.81 on BindingDB)
          bindingdb-ulvsh - BindingDB + ULVSH (28 MB, R=0.79 combined)

        Examples:
            # Download default model (recommended)
            pandadock gnn download-model

            # Download BindingDB-trained model
            pandadock gnn download-model --model bindingdb

            # Download BindingDB + ULVSH combined model
            pandadock gnn download-model --model bindingdb-ulvsh

            # List available models
            pandadock gnn download-model --list
        """
        import urllib.request

        # GitHub release URLs
        GITHUB_REPO = "pritampanda15/PandaDock"

        # Available models
        MODELS = {
            'default': {
                'filename': 'pandadock_gnn_v4.pt',
                'url': f'https://github.com/{GITHUB_REPO}/releases/download/v4.0.0/pandadock_gnn_v4.pt',
                'size_mb': 82,
                'description': 'ULVSH + PDBbind combined (recommended)',
                'metrics': {
                    'PDBbind R': 0.88,
                    'ULVSH R': 0.82,
                    'Activity AUC': 0.94
                }
            },
            'bindingdb': {
                'filename': 'pandadock_gnn_bindingdb.pt',
                'url': f'https://github.com/{GITHUB_REPO}/releases/download/v4.0.0/pandadock_gnn_bindingdb.pt',
                'size_mb': 28,
                'description': 'BindingDB only (8,891 complexes)',
                'metrics': {
                    'BindingDB R': 0.81,
                    'Test RMSE': 0.96
                }
            },
            'bindingdb-ulvsh': {
                'filename': 'pandadock_gnn_bindingdb_ulvsh.pt',
                'url': f'https://github.com/{GITHUB_REPO}/releases/download/v4.0.0/pandadock_gnn_bindingdb_ulvsh.pt',
                'size_mb': 28,
                'description': 'BindingDB + ULVSH combined',
                'metrics': {
                    'BindingDB R': 0.79,
                    'Test RMSE': 0.96
                }
            }
        }

        # List models if requested
        if list_models:
            print("\nAvailable PandaDock-GNN Models:")
            print("=" * 60)
            for name, info in MODELS.items():
                print(f"\n{name}:")
                print(f"  File: {info['filename']}")
                print(f"  Size: ~{info['size_mb']} MB")
                print(f"  Description: {info['description']}")
                print(f"  Metrics:")
                for metric, value in info['metrics'].items():
                    print(f"    {metric}: {value}")
            print("\n" + "=" * 60)
            print("Download with: pandadock gnn download-model --model <name>")
            return

        # Get model info
        info = MODELS[model]
        model_filename = info['filename']

        # Create output directory
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / model_filename

        # Check if model already exists
        if output_path.exists() and not force:
            print(f"Model already exists at: {output_path}")
            print("Use --force to re-download")
            return

        print("=" * 60)
        print("PandaDock-GNN Model Download")
        print("=" * 60)

        print(f"\nModel: {model}")
        print(f"File: {model_filename}")
        print(f"Size: ~{info['size_mb']} MB")
        print(f"Description: {info['description']}")
        print(f"\nPerformance:")
        for metric, value in info['metrics'].items():
            print(f"  {metric}: {value}")
        print()

        # Download with progress
        url = info['url']
        print(f"Downloading from: {url}")
        print(f"Saving to: {output_path}")
        print()

        try:
            def show_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(100, downloaded * 100 / total_size)
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    bar_length = 40
                    filled = int(bar_length * percent / 100)
                    bar = '=' * filled + '-' * (bar_length - filled)
                    print(f"\r[{bar}] {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='', flush=True)

            urllib.request.urlretrieve(url, output_path, show_progress)
            print("\n")

            # Verify download
            if output_path.exists():
                file_size = output_path.stat().st_size / (1024 * 1024)
                print(f"Download complete! ({file_size:.1f} MB)")

                # Verify SHA256 if available
                if info.get('sha256'):
                    print("Verifying checksum...")
                    import hashlib
                    sha256_hash = hashlib.sha256()
                    with open(output_path, 'rb') as f:
                        for chunk in iter(lambda: f.read(4096), b''):
                            sha256_hash.update(chunk)
                    if sha256_hash.hexdigest() == info['sha256']:
                        print("Checksum verified!")
                    else:
                        print("WARNING: Checksum mismatch! File may be corrupted.")

                print("\n" + "=" * 60)
                print("SUCCESS! Model ready to use.")
                print("=" * 60)
                print(f"\nExample usage:")
                print(f"  pandadock gnn predict -m {output_path} -p protein.mol2 -l ligand.mol2")
                print(f"  pandadock gnn rescore -m {output_path} -r protein.pdb -p poses.sdf")
                print(f"  pandadock hybrid -r protein.pdb -l ligand.sdf -m {output_path} --center X Y Z --box X Y Z")
            else:
                print("Error: Download failed - file not found")

        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"\nError: Model '{model}' not found at release URL.")
                print(f"The model may not have been uploaded yet.")
                print(f"\nAlternative: Train your own model:")
                print(f"  pandadock gnn train -d ULVSH/ -o models/ --epochs 100")
            else:
                print(f"\nError downloading model: HTTP {e.code}")
                print(f"Please try again later or download manually from:")
                print(f"  {url}")

        except urllib.error.URLError as e:
            print(f"\nError: Could not connect to GitHub.")
            print(f"Please check your internet connection and try again.")
            print(f"\nAlternatively, download manually from:")
            print(f"  https://github.com/{GITHUB_REPO}/releases")

        except Exception as e:
            print(f"\nError: {e}")
            print(f"\nPlease download manually from:")
            print(f"  https://github.com/{GITHUB_REPO}/releases")

    @main.command('list-algorithms')
    def list_algorithms():
        """List available algorithms and their status."""
        print("\nPandaDock-GNN Algorithms")
        print("=" * 50)
        print("\nML-Based (this module):")
        print("  - pandadock_gnn: SE(3)-equivariant GNN scoring")
        print("    Status: Available")
        print("    Use: pandadock-gnn predict -m model.pt -p protein.mol2 -l ligand.mol2")

        print("\nPre-trained Model:")
        print("  - Download: pandadock gnn download-model")
        print("  - Trained on ULVSH + PDBbind (R = 0.88)")

        print("\nTraining:")
        print("  - ULVSH dataset: 943 compounds, 10 targets")
        print("  - Use: pandadock-gnn train -d ULVSH/ -o output/")

        print("\nRescoring (NEW):")
        print("  - Universal rescorer for poses from any docking tool")
        print("  - Use: pandadock gnn rescore -m model.pt -r receptor.pdb -p poses.sdf")

else:
    def main():
        print("Click not installed. Install with: pip install click")
        sys.exit(1)


if __name__ == '__main__':
    main()
