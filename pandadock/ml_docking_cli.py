#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PandaDock-ML: Machine Learning Enhanced Molecular Docking CLI

Advanced ML Pipeline Features:
- Pose generation using diffusion models
- Feature extraction from protein-ligand complexes
- Hybrid ML+Physics scoring
- Training on docking datasets
- Transfer learning capabilities
"""

import click
import logging
import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem
import time
import sys
import os

# Import ML-specific modules with graceful fallback
try:
    from .ml.feature_extraction import (
        ProteinFeatureExtractor, LigandFeatureExtractor,
        ComplexFeatureExtractor
    )
    from .ml.models.diffusion import DiffusionDockingModel
    from .ml.models.energy_prediction import EnergyPredictionModel
    from .ml.models.pose_ranking import PoseRankingModel
    from .ml.training.trainer import MLDockingTrainer
    from .ml.hybrid_pipeline import HybridMLPhysicsPipeline
    ML_AVAILABLE = True
except ImportError as e:
    # Fallback classes for when ML dependencies are not available
    class DiffusionDockingModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch not available. Install with: pip install torch")

    class EnergyPredictionModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch not available. Install with: pip install torch")

    class PoseRankingModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch not available. Install with: pip install torch")

    class HybridMLPhysicsPipeline:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch not available. Install with: pip install torch")

    class MLDockingTrainer:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch not available. Install with: pip install torch")

    class ProteinFeatureExtractor:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch not available. Install with: pip install torch")

    class LigandFeatureExtractor:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch not available. Install with: pip install torch")

    class ComplexFeatureExtractor:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch not available. Install with: pip install torch")

    ML_AVAILABLE = False

# Import existing PandaDock components
from .docking.core import DockingEngine, DockingResult, Pose
from .docking.algorithms import GPU_AVAILABLE
from .docking.scoring.physics_based import PhysicsBasedScoring
from .docking.scoring.precision_score import PrecisionScoring
from . import __version__

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

def show_ml_help():
    """Display ML-specific help banner"""
    help_text = """
██████╗  █████╗ ███╗   ██╗██████╗  █████╗ ██████╗  ██████╗  ██████╗██╗  ██╗      ███╗   ███╗██╗
██╔══██╗██╔══██╗████╗  ██║██╔══██╗██╔══██╗██╔══██╗██╔═══██╗██╔════╝██║ ██╔╝      ████╗ ████║██║
██████╔╝███████║██╔██╗ ██║██║  ██║███████║██║  ██║██║   ██║██║     █████╔╝ █████╗██╔████╔██║██║
██╔═══╝ ██╔══██║██║╚██╗██║██║  ██║██╔══██║██║  ██║██║   ██║██║     ██╔═██╗ ╚════╝██║╚██╔╝██║██║
██║     ██║  ██║██║ ╚████║██████╔╝██║  ██║██████╔╝╚██████╔╝╚██████╗██║  ██╗      ██║ ╚═╝ ██║███████╗
╚═╝     ╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝ ╚═╝  ╚═╝╚═════╝  ╚═════╝  ╚═════╝╚═╝  ╚═╝      ╚═╝     ╚═╝╚══════╝

                    Machine Learning Enhanced Molecular Docking
               https://github.com/pritampanda15/PandaDock

Author: Pritam Kumar Panda @ Stanford University
Version: {version} - ML Extension
License: MIT

PandaDock-ML combines machine learning with physics-based docking for
enhanced accuracy and novel pose prediction capabilities.

USAGE:
    pandadock-ml COMMAND [OPTIONS]

MAIN COMMANDS:
    dock                Perform ML-enhanced docking
    train               Train ML models on docking datasets
    hybrid-dock         Use hybrid ML+Physics pipeline
    extract-features    Extract features from protein-ligand complexes
    predict-poses      Generate poses using trained ML models

MODEL COMMANDS:
    list-models         List available ML models
    benchmark           Benchmark ML models against physics methods
    tune-hyperparams    Optimize model hyperparameters

EXAMPLES:
    # ML-enhanced docking
    pandadock-ml dock -r protein.pdb -l ligand.sdf --center 10 20 30 --box 20 20 20 \\
                      --model diffusion --hybrid-scoring

    # Train on custom dataset
    pandadock-ml train --dataset my_complexes.json --model-type pose_ranking \\
                       --epochs 100 --batch-size 32

    # Hybrid ML+Physics pipeline
    pandadock-ml hybrid-dock -r protein.pdb -l ligand.sdf --center 10 20 30 --box 20 20 20 \\
                            --ml-proposals 50 --physics-refinement enhanced_hierarchical

    # Extract features for training
    pandadock-ml extract-features --complexes complexes_dir/ --output features.h5

FEATURES:
    ✓ Diffusion-based pose generation
    ✓ Energy prediction models
    ✓ Pose ranking networks
    ✓ Hybrid ML+Physics pipelines
    ✓ Transfer learning capabilities
    ✓ Feature extraction utilities
    ✓ Model benchmarking tools

For detailed help on specific commands, use:
    pandadock-ml COMMAND --help

#################################################################
"""
    click.echo(help_text.replace("{version}", __version__))

@click.group(invoke_without_command=True, add_help_option=False)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--version', is_flag=True, help='Show version information')
@click.option('--help', '-h', is_flag=True, help='Show this help message')
@click.pass_context
def main(ctx, verbose, version, help):
    """PandaDock-ML - Machine Learning Enhanced Molecular Docking"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if version:
        click.echo("PandaDock-ML v1.0.0 - Pritam Kumar Panda @ Stanford University")
        return

    if help:
        show_ml_help()
        return

    # If no command is provided, show help
    if ctx.invoked_subcommand is None:
        show_ml_help()

@main.command()
@click.option('--receptor', '-r', required=True, type=click.Path(exists=True),
              help='Receptor PDB file')
@click.option('--ligand', '-l', required=True, type=click.Path(exists=True),
              help='Ligand file (SDF/MOL2/PDB)')
@click.option('--center', nargs=3, type=float, metavar='X Y Z',
              help='Grid center coordinates (Å)')
@click.option('--box', nargs=3, type=float, metavar='X Y Z',
              help='Grid box dimensions (Å)')
@click.option('--model', '-m', type=click.Choice([
    'diffusion', 'energy_prediction', 'pose_ranking', 'hybrid'
]), default='diffusion', help='ML model to use for pose generation')
@click.option('--model-path', type=click.Path(exists=True),
              help='Path to trained model weights')
@click.option('--output-dir', '-o', type=click.Path(), default='ml_docking_output',
              help='Output directory')
@click.option('--num-proposals', '-n', type=int, default=50,
              help='Number of ML-generated pose proposals')
@click.option('--hybrid-scoring', is_flag=True, default=True,
              help='Use hybrid ML+Physics scoring')
@click.option('--physics-refinement', type=click.Choice([
    'none', 'monte_carlo', 'enhanced_hierarchical'
]), default='enhanced_hierarchical', help='Physics-based refinement method')
@click.option('--temperature', type=float, default=1.0,
              help='Sampling temperature for diffusion models')
@click.option('--confidence-threshold', type=float, default=0.7,
              help='Minimum confidence threshold for poses')
def dock(receptor, ligand, center, box, model, model_path, output_dir,
         num_proposals, hybrid_scoring, physics_refinement, temperature,
         confidence_threshold):
    """Perform ML-enhanced molecular docking"""

    if not ML_AVAILABLE:
        click.echo("Error: ML features require PyTorch. Install with:")
        click.echo("  pip install torch")
        click.echo("  pip install -e .[ml]")
        sys.exit(1)

    click.echo("PandaDock-ML Enhanced Molecular Docking")
    click.echo(f"Receptor: {receptor}")
    click.echo(f"Ligand: {ligand}")
    click.echo(f"ML Model: {model}")

    # Validate inputs
    if not center or not box:
        click.echo("Error: Must specify --center and --box coordinates")
        sys.exit(1)

    grid_center = np.array(center)
    grid_dimensions = np.array(box)

    # Load ligand
    ligand_mol = load_ligand(ligand)
    if ligand_mol is None:
        click.echo(f"Error: Could not load ligand from {ligand}")
        sys.exit(1)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize ML pipeline
    try:
        if model == 'hybrid':
            pipeline = HybridMLPhysicsPipeline(
                model_path=model_path,
                physics_refinement=physics_refinement,
                hybrid_scoring=hybrid_scoring
            )
        else:
            # Load specific ML model
            if model == 'diffusion':
                ml_model = DiffusionDockingModel()
            elif model == 'energy_prediction':
                ml_model = EnergyPredictionModel()
            elif model == 'pose_ranking':
                ml_model = PoseRankingModel()

            if model_path:
                ml_model.load_weights(model_path)

            pipeline = HybridMLPhysicsPipeline(
                ml_model=ml_model,
                physics_refinement=physics_refinement,
                hybrid_scoring=hybrid_scoring
            )

    except Exception as e:
        click.echo(f"Error initializing ML pipeline: {e}")
        sys.exit(1)

    # Perform ML-enhanced docking
    click.echo("Starting ML-enhanced docking...")
    start_time = time.time()

    try:
        result = pipeline.dock(
            receptor_file=receptor,
            ligand_mol=ligand_mol,
            grid_center=grid_center,
            grid_dimensions=grid_dimensions,
            num_proposals=num_proposals,
            temperature=temperature,
            confidence_threshold=confidence_threshold
        )

        runtime = time.time() - start_time
        click.echo(f"ML docking completed in {runtime:.2f} seconds")

    except Exception as e:
        click.echo(f"ML docking failed: {e}")
        sys.exit(1)

    if not result.poses:
        click.echo("No poses generated")
        sys.exit(1)

    click.echo(f"Generated {len(result.poses)} poses")

    # Save results
    click.echo("Saving ML docking results...")
    result.save_results(output_path)

    # Save ML-specific metadata
    ml_metadata = {
        'model_type': model,
        'model_path': str(model_path) if model_path else None,
        'num_proposals': num_proposals,
        'hybrid_scoring': hybrid_scoring,
        'physics_refinement': physics_refinement,
        'temperature': temperature,
        'confidence_threshold': confidence_threshold,
        'ml_generation_time': getattr(result, 'ml_generation_time', 0.0),
        'physics_refinement_time': getattr(result, 'physics_refinement_time', 0.0)
    }

    with open(output_path / "ml_metadata.json", 'w') as f:
        json.dump(ml_metadata, f, indent=2)

    # Show top results
    top_poses = result.get_top_poses(5)
    click.echo("\nTop 5 ML-enhanced poses:")
    for i, pose in enumerate(top_poses, 1):
        ml_confidence = getattr(pose, 'ml_confidence', 0.0)
        click.echo(f"  {i}. Energy: {pose.energy:.3f} kcal/mol, "
                  f"Confidence: {pose.confidence:.3f}, "
                  f"ML Confidence: {ml_confidence:.3f}")

    click.echo(f"\nML docking complete! Results saved to: {output_path}")

@main.command()
@click.option('--dataset', '-d', required=True, type=click.Path(exists=True),
              help='Training dataset (JSON or HDF5 format)')
@click.option('--model-type', '-m', type=click.Choice([
    'diffusion', 'energy_prediction', 'pose_ranking'
]), required=True, help='Type of ML model to train')
@click.option('--output-dir', '-o', type=click.Path(), default='ml_models',
              help='Output directory for trained models')
@click.option('--epochs', type=int, default=100,
              help='Number of training epochs')
@click.option('--batch-size', type=int, default=32,
              help='Training batch size')
@click.option('--learning-rate', type=float, default=1e-4,
              help='Learning rate')
@click.option('--validation-split', type=float, default=0.2,
              help='Fraction of data for validation')
@click.option('--pretrained-weights', type=click.Path(exists=True),
              help='Path to pretrained weights for transfer learning')
@click.option('--gpu', is_flag=True, default=False,
              help='Use GPU for training')
def train(dataset, model_type, output_dir, epochs, batch_size, learning_rate,
          validation_split, pretrained_weights, gpu):
    """Train ML models on docking datasets"""

    if not ML_AVAILABLE:
        click.echo("Error: ML training requires PyTorch. Install with:")
        click.echo("  pip install torch")
        click.echo("  pip install -e .[ml]")
        sys.exit(1)

    click.echo(f"Training {model_type} model on {dataset}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize trainer
    try:
        trainer = MLDockingTrainer(
            model_type=model_type,
            gpu=gpu and GPU_AVAILABLE
        )

        # Load dataset
        trainer.load_dataset(dataset, validation_split=validation_split)

        # Load pretrained weights if provided
        if pretrained_weights:
            trainer.load_pretrained_weights(pretrained_weights)

        # Configure training
        trainer.configure_training(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            output_dir=output_path
        )

        # Start training
        click.echo("Starting model training...")
        start_time = time.time()

        history = trainer.train()

        training_time = time.time() - start_time
        click.echo(f"Training completed in {training_time:.2f} seconds")

        # Save training results
        trainer.save_model(output_path / f"{model_type}_model.pth")
        trainer.save_training_history(output_path / "training_history.json")

        # Show training summary
        click.echo("\nTraining Summary:")
        click.echo(f"Final training loss: {history['train_loss'][-1]:.6f}")
        click.echo(f"Final validation loss: {history['val_loss'][-1]:.6f}")
        if 'val_accuracy' in history:
            click.echo(f"Final validation accuracy: {history['val_accuracy'][-1]:.4f}")

    except Exception as e:
        click.echo(f"Training failed: {e}")
        sys.exit(1)

    click.echo(f"\nModel training complete! Saved to: {output_path}")

@main.command()
@click.option('--receptor', '-r', required=True, type=click.Path(exists=True),
              help='Receptor PDB file')
@click.option('--ligand', '-l', required=True, type=click.Path(exists=True),
              help='Ligand file (SDF/MOL2/PDB)')
@click.option('--center', nargs=3, type=float, metavar='X Y Z',
              help='Grid center coordinates (Å)')
@click.option('--box', nargs=3, type=float, metavar='X Y Z',
              help='Grid box dimensions (Å)')
@click.option('--ml-proposals', type=int, default=100,
              help='Number of ML-generated pose proposals')
@click.option('--physics-refinement', type=click.Choice([
    # Legacy aliases, all resolving to the same flexible-ligand search.
    'pandadock', 'monte_carlo_cpu', 'enhanced_hierarchical_cpu', 'genetic_algorithm_cpu'
]), default='pandadock', help='Physics algorithm for refinement')
@click.option('--ml-model-path', type=click.Path(exists=True),
              help='Path to trained ML model weights')
@click.option('--output-dir', '-o', type=click.Path(), default='hybrid_docking_output',
              help='Output directory')
@click.option('--top-k-refinement', type=int, default=20,
              help='Number of top ML proposals to refine with physics')
def hybrid_dock(receptor, ligand, center, box, ml_proposals, physics_refinement,
                ml_model_path, output_dir, top_k_refinement):
    """Use hybrid ML+Physics docking pipeline"""

    click.echo("PandaDock Hybrid ML+Physics Docking Pipeline")
    click.echo(f"ML Proposals: {ml_proposals}")
    click.echo(f"Physics Refinement: {physics_refinement}")
    click.echo(f"Top-K for refinement: {top_k_refinement}")

    # Validate inputs
    if not center or not box:
        click.echo("Error: Must specify --center and --box coordinates")
        sys.exit(1)

    grid_center = np.array(center)
    grid_dimensions = np.array(box)

    # Load ligand
    ligand_mol = load_ligand(ligand)
    if ligand_mol is None:
        click.echo(f"Error: Could not load ligand from {ligand}")
        sys.exit(1)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize hybrid pipeline
    try:
        pipeline = HybridMLPhysicsPipeline(
            model_path=ml_model_path,
            physics_refinement=physics_refinement,
            hybrid_scoring=True
        )

        # Perform hybrid docking
        click.echo("Starting hybrid ML+Physics docking...")
        start_time = time.time()

        result = pipeline.hybrid_dock(
            receptor_file=receptor,
            ligand_mol=ligand_mol,
            grid_center=grid_center,
            grid_dimensions=grid_dimensions,
            ml_proposals=ml_proposals,
            top_k_refinement=top_k_refinement
        )

        runtime = time.time() - start_time
        click.echo(f"Hybrid docking completed in {runtime:.2f} seconds")

        # Save results with detailed breakdown
        result.save_results(output_path)

        # Save hybrid-specific metadata
        hybrid_metadata = {
            'ml_proposals': ml_proposals,
            'physics_refinement': physics_refinement,
            'top_k_refinement': top_k_refinement,
            'ml_generation_time': getattr(result, 'ml_generation_time', 0.0),
            'physics_refinement_time': getattr(result, 'physics_refinement_time', 0.0),
            'hybrid_scoring_time': getattr(result, 'hybrid_scoring_time', 0.0)
        }

        with open(output_path / "hybrid_metadata.json", 'w') as f:
            json.dump(hybrid_metadata, f, indent=2)

        # Show performance breakdown
        click.echo(f"\nPerformance Breakdown:")
        click.echo(f"ML Generation: {getattr(result, 'ml_generation_time', 0.0):.2f}s")
        click.echo(f"Physics Refinement: {getattr(result, 'physics_refinement_time', 0.0):.2f}s")
        click.echo(f"Hybrid Scoring: {getattr(result, 'hybrid_scoring_time', 0.0):.2f}s")

        # Show top results
        top_poses = result.get_top_poses(5)
        click.echo("\nTop 5 hybrid poses:")
        for i, pose in enumerate(top_poses, 1):
            ml_score = getattr(pose, 'ml_score', 0.0)
            physics_score = getattr(pose, 'physics_score', 0.0)
            click.echo(f"  {i}. Total Energy: {pose.energy:.3f} kcal/mol")
            click.echo(f"      ML Score: {ml_score:.3f}, Physics Score: {physics_score:.3f}")

    except Exception as e:
        click.echo(f"Hybrid docking failed: {e}")
        sys.exit(1)

    click.echo(f"\nHybrid docking complete! Results saved to: {output_path}")

@main.command()
@click.option('--complexes', '-c', required=True, type=click.Path(exists=True),
              help='Directory containing protein-ligand complexes')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output file for extracted features (HDF5 format)')
@click.option('--feature-types', multiple=True, default=['protein', 'ligand', 'interaction'],
              help='Types of features to extract')
@click.option('--grid-resolution', type=float, default=1.0,
              help='Grid resolution for volumetric features (Å)')
@click.option('--pocket-radius', type=float, default=10.0,
              help='Radius around ligand for pocket extraction (Å)')
def extract_features(complexes, output, feature_types, grid_resolution, pocket_radius):
    """Extract features from protein-ligand complexes for ML training"""

    click.echo(f"Extracting features from {complexes}")
    click.echo(f"Feature types: {', '.join(feature_types)}")

    try:
        # Initialize feature extractors
        extractors = {}
        if 'protein' in feature_types:
            extractors['protein'] = ProteinFeatureExtractor(
                grid_resolution=grid_resolution,
                pocket_radius=pocket_radius
            )
        if 'ligand' in feature_types:
            extractors['ligand'] = LigandFeatureExtractor()
        if 'interaction' in feature_types:
            extractors['interaction'] = ComplexFeatureExtractor(
                grid_resolution=grid_resolution
            )

        # Process all complexes
        complex_files = list(Path(complexes).glob("*.pdb"))
        click.echo(f"Found {len(complex_files)} PDB files")

        all_features = []

        for i, complex_file in enumerate(complex_files):
            click.echo(f"Processing {complex_file.name} ({i+1}/{len(complex_files)})")

            try:
                features = {}
                for feat_type, extractor in extractors.items():
                    features[feat_type] = extractor.extract_from_complex(str(complex_file))

                features['complex_name'] = complex_file.stem
                all_features.append(features)

            except Exception as e:
                click.echo(f"  Error processing {complex_file.name}: {e}")
                continue

        # Save features to HDF5
        click.echo(f"Saving {len(all_features)} feature sets to {output}")
        save_features_to_hdf5(all_features, output)

        click.echo("Feature extraction complete!")

    except Exception as e:
        click.echo(f"Feature extraction failed: {e}")
        sys.exit(1)

@main.command()
def list_models():
    """List available ML models and their descriptions"""

    models = {
        'diffusion': 'Diffusion-based pose generation model',
        'energy_prediction': 'Deep learning energy prediction model',
        'pose_ranking': 'Neural network for pose ranking and selection',
        'hybrid': 'Combined ML+Physics hybrid pipeline'
    }

    click.echo("Available ML Models:")
    for model, description in models.items():
        click.echo(f"  {model}: {description}")

def load_ligand(ligand_file: str) -> Optional[Chem.Mol]:
    """Load ligand molecule from file"""
    ligand_path = Path(ligand_file)

    try:
        if ligand_path.suffix.lower() == '.sdf':
            mol = Chem.SDMolSupplier(str(ligand_path))[0]
        elif ligand_path.suffix.lower() == '.mol2':
            mol = Chem.MolFromMol2File(str(ligand_path))
        elif ligand_path.suffix.lower() == '.pdb':
            mol = Chem.MolFromPDBFile(str(ligand_path))
        else:
            mol = Chem.MolFromMolFile(str(ligand_path))

        if mol is None:
            return None

        # Add hydrogens if not present
        mol = Chem.AddHs(mol)
        return mol

    except Exception as e:
        logging.error(f"Error loading ligand {ligand_file}: {e}")
        return None

def save_features_to_hdf5(features_list: List[Dict], output_file: str):
    """Save extracted features to HDF5 format"""
    try:
        import h5py

        with h5py.File(output_file, 'w') as f:
            for i, features in enumerate(features_list):
                grp = f.create_group(f"complex_{i}")
                grp.attrs['name'] = features['complex_name']

                for feat_type, feat_data in features.items():
                    if feat_type != 'complex_name' and isinstance(feat_data, np.ndarray):
                        grp.create_dataset(feat_type, data=feat_data)

    except ImportError:
        click.echo("Error: h5py not installed. Install with: pip install h5py")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error saving features: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()