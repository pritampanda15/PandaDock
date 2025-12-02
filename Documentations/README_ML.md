# PandaDock-ML: Machine Learning Enhanced Molecular Docking

PandaDock-ML extends the PandaDock molecular docking suite with cutting-edge machine learning capabilities, combining the accuracy of physics-based methods with the speed and novel insights of ML approaches.

## 🚀 Features

### Core ML Capabilities
- **Diffusion-based pose generation**: Generate novel binding poses using diffusion models
- **Energy prediction models**: Deep learning-based binding energy prediction
- **Pose ranking networks**: Neural networks for intelligent pose selection
- **Hybrid ML+Physics pipeline**: Best of both worlds combining ML speed with physics accuracy

### Advanced Features
- **Transfer learning**: Pre-trained models that can be fine-tuned on custom datasets
- **Feature extraction utilities**: Comprehensive protein and ligand feature extraction
- **Model training framework**: Train custom ML models on your docking datasets
- **Benchmarking tools**: Compare ML models against traditional physics methods

## 📦 Installation

### Basic Installation
```bash
# Install PandaDock with ML support
pip install -e .[ml]
```

### With PyTorch (Required for ML)
```bash
# Install with PyTorch support
pip install torch torchvision torchaudio
pip install -e .[ml]
```

### Development Installation
```bash
# Clone and install in development mode
git clone https://github.com/pritampanda15/PandaDock.git
cd PandaDock
pip install -e .[ml]
```

## 🎯 Quick Start

### 1. Basic ML-Enhanced Docking
```bash
# Run ML-enhanced docking with diffusion model
pandadock-ml dock -r protein.pdb -l ligand.sdf \
    --center 10 20 30 --box 20 20 20 \
    --model diffusion --num-proposals 50
```

### 2. Hybrid ML+Physics Pipeline
```bash
# Use hybrid approach: ML proposals + physics refinement
pandadock-ml hybrid-dock -r protein.pdb -l ligand.sdf \
    --center 10 20 30 --box 20 20 20 \
    --ml-proposals 100 --physics-refinement enhanced_hierarchical_cpu \
    --top-k-refinement 20
```

### 3. Train Custom Models
```bash
# Train a pose ranking model on your dataset
pandadock-ml train --dataset my_docking_data.h5 \
    --model-type pose_ranking --epochs 100 \
    --batch-size 32 --output-dir trained_models/
```

### 4. Extract Features for Training
```bash
# Extract features from protein-ligand complexes
pandadock-ml extract-features --complexes complexes_dir/ \
    --output features.h5 --feature-types protein ligand interaction
```

## 🔬 ML Models

### Diffusion Model
- **Purpose**: Generate diverse, high-quality binding poses
- **Architecture**: Transformer-based diffusion network
- **Input**: Protein pocket features + ligand molecular descriptors
- **Output**: 3D ligand coordinates with confidence scores

### Energy Prediction Model
- **Purpose**: Fast binding energy estimation
- **Architecture**: Graph neural network with attention
- **Input**: Protein-ligand complex features
- **Output**: Binding energy (kcal/mol)

### Pose Ranking Model
- **Purpose**: Rank and select best poses from candidate sets
- **Architecture**: Multi-layer attention network
- **Input**: Pose coordinates + protein-ligand features
- **Output**: Ranking score (0-1)

## 🎛️ Command Reference

### Main Commands

#### `pandadock-ml dock`
ML-enhanced docking with various model options.

**Options:**
- `--model`: ML model type (`diffusion`, `energy_prediction`, `pose_ranking`, `hybrid`)
- `--model-path`: Path to trained model weights
- `--num-proposals`: Number of ML-generated pose proposals
- `--hybrid-scoring`: Use combined ML+Physics scoring
- `--temperature`: Sampling temperature for diffusion models
- `--confidence-threshold`: Minimum confidence threshold

#### `pandadock-ml hybrid-dock`
Full hybrid ML+Physics pipeline with refinement.

**Options:**
- `--ml-proposals`: Number of ML-generated proposals
- `--physics-refinement`: Physics algorithm for refinement
- `--top-k-refinement`: Number of top poses to refine

#### `pandadock-ml train`
Train ML models on custom datasets.

**Options:**
- `--model-type`: Type of model to train
- `--dataset`: Training dataset (JSON or HDF5)
- `--epochs`: Number of training epochs
- `--batch-size`: Training batch size
- `--learning-rate`: Learning rate
- `--pretrained-weights`: Pretrained weights for transfer learning

#### `pandadock-ml extract-features`
Extract features from protein-ligand complexes.

**Options:**
- `--complexes`: Directory containing PDB complexes
- `--feature-types`: Types of features to extract
- `--grid-resolution`: Grid resolution for volumetric features
- `--pocket-radius`: Radius around ligand for pocket extraction

### Utility Commands

#### `pandadock-ml list-models`
List available ML models and their descriptions.

## 📊 Performance Comparison

| Method | Speed | Accuracy | Novel Poses | Best Use Case |
|--------|-------|----------|-------------|---------------|
| Physics Only | Slow | High | Low | High accuracy needed |
| ML Only | Fast | Medium | High | Novel pose discovery |
| **Hybrid ML+Physics** | **Medium** | **Very High** | **Medium** | **Best overall performance** |

## 🔧 Configuration

### Model Configuration
Create a `ml_config.json` file to customize model behavior:

```json
{
  "diffusion_model": {
    "model_size": "medium",
    "timesteps": 500,
    "temperature": 1.0
  },
  "energy_model": {
    "model_size": "large",
    "hidden_dim": 512
  },
  "hybrid_pipeline": {
    "ml_weight": 0.3,
    "physics_weight": 0.7,
    "confidence_threshold": 0.7
  }
}
```

### Training Configuration
Customize training with `training_config.json`:

```json
{
  "optimizer": "adam",
  "learning_rate": 1e-4,
  "batch_size": 32,
  "epochs": 100,
  "early_stopping": {
    "patience": 20,
    "min_delta": 1e-5
  },
  "scheduler": {
    "type": "reduce_on_plateau",
    "factor": 0.5,
    "patience": 10
  }
}
```

## 📈 Dataset Format

### Training Data Format (HDF5)
```
dataset.h5
├── complex_1/
│   ├── ligand_features: [n_atoms, n_features]
│   ├── receptor_features: [n_pocket_features]
│   ├── coordinates: [n_atoms, 3]
│   ├── energy: scalar
│   └── confidence: scalar
├── complex_2/
│   └── ...
```

### Training Data Format (JSON)
```json
[
  {
    "complex_name": "1a0q_ligand",
    "ligand_features": [[...], [...]],
    "receptor_features": [...],
    "coordinates": [[x, y, z], ...],
    "energy": -8.5,
    "confidence": 0.85
  },
  ...
]
```

## 🔬 Advanced Usage

### Custom Feature Extraction
```python
from pandadock.ml.feature_extraction import ComplexFeatureExtractor

# Extract custom features
extractor = ComplexFeatureExtractor(grid_resolution=0.5)
features = extractor.extract_from_complex("complex.pdb")
```

### Custom Model Training
```python
from pandadock.ml.training.trainer import MLDockingTrainer

# Set up custom training
trainer = MLDockingTrainer(model_type='diffusion', gpu=True)
trainer.load_dataset("training_data.h5")
trainer.configure_training(epochs=200, batch_size=64)
history = trainer.train()
```

### Hybrid Pipeline Customization
```python
from pandadock.ml.hybrid_pipeline import HybridMLPhysicsPipeline

# Custom hybrid pipeline
pipeline = HybridMLPhysicsPipeline(
    model_path="trained_diffusion.pth",
    physics_refinement="enhanced_hierarchical_cpu",
    hybrid_scoring=True
)

result = pipeline.hybrid_dock(
    receptor_file="protein.pdb",
    ligand_mol=ligand,
    grid_center=center,
    grid_dimensions=dimensions,
    ml_proposals=100,
    top_k_refinement=20
)
```

## 🐛 Troubleshooting

### Common Issues

**Issue**: `ImportError: No module named 'torch'`
**Solution**: Install PyTorch: `pip install torch torchvision torchaudio`

**Issue**: `CUDA out of memory`
**Solution**: Reduce batch size or use CPU: `--gpu=false`

**Issue**: `No poses generated`
**Solution**: Check ligand format and grid parameters

**Issue**: Model weights not found
**Solution**: Verify model path or train a new model

### Performance Tips

1. **GPU Usage**: Enable GPU for training: `--gpu`
2. **Batch Size**: Increase batch size for faster training on GPUs
3. **Model Size**: Use smaller models for faster inference
4. **Hybrid Pipeline**: Balance ML proposals vs. physics refinement

## 📚 Citation

If you use PandaDock-ML in your research, please cite:

```bibtex
@software{pandadock_ml,
  title={PandaDock-ML: Machine Learning Enhanced Molecular Docking},
  author={Panda, Pritam Kumar},
  affiliation={Stanford University},
  year={2024},
  url={https://github.com/pritampanda15/PandaDock}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

PandaDock-ML is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/pritampanda15/PandaDock/issues)
- **Discussions**: [GitHub Discussions](https://github.com/pritampanda15/PandaDock/discussions)
- **Email**: pritam@stanford.edu

---

**PandaDock-ML**: Where machine learning meets molecular docking for the future of drug discovery. 🐼🧬🤖