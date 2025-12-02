import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import logging
from pathlib import Path
from ..gridbox.core import GridBox, GridBoxGenerator

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


if HAS_TORCH:
    class SimpleBindingSitePredictor(nn.Module):
        """Simple neural network for binding site prediction"""

        def __init__(self, input_dim: int = 20, hidden_dims: List[int] = [128, 64, 32]):
            super().__init__()

            layers = []
            prev_dim = input_dim

            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                ])
                prev_dim = hidden_dim

            layers.append(nn.Linear(prev_dim, 1))
            layers.append(nn.Sigmoid())

            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)
else:
    class SimpleBindingSitePredictor:
        """Placeholder class when torch is not available"""
        def __init__(self, *args, **kwargs):
            pass


class AIBindingSitePredictionGenerator(GridBoxGenerator):
    """AI-powered binding site prediction using geometric and chemical features"""

    def __init__(self):
        super().__init__()
        if HAS_TORCH:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = 'cpu'
        self.model = None
        self.feature_dim = 20

    def generate(self, receptor_file: str, max_sites: int = 5,
                 confidence_threshold: float = 0.7, spacing: float = 0.375, **kwargs) -> List[GridBox]:
        """Generate grid boxes using AI-powered binding site prediction"""

        structure = self._parse_receptor(receptor_file)

        # Extract features and predict binding sites
        candidates = self._generate_candidate_sites(structure)

        # If no candidates found, use fallback approach
        if not candidates:
            self.logger.warning("No AI candidates found, using cavity detection fallback")
            return self._fallback_cavity_detection(receptor_file, max_sites, spacing)

        predictions = self._predict_binding_sites(candidates)

        # Filter by confidence and rank
        high_confidence_sites = []
        for candidate, prediction in zip(candidates, predictions):
            if prediction['confidence'] >= confidence_threshold:
                high_confidence_sites.append({
                    'center': candidate['center'],
                    'dimensions': candidate['dimensions'],
                    'confidence': prediction['confidence'],
                    'features': candidate['features'],
                    'pharmacophore': prediction.get('pharmacophore', {})
                })

        # If still no high confidence sites, lower the threshold
        if not high_confidence_sites and candidates:
            self.logger.warning(f"No sites above confidence {confidence_threshold}, lowering threshold to 0.3")
            for candidate, prediction in zip(candidates, predictions):
                if prediction['confidence'] >= 0.3:
                    high_confidence_sites.append({
                        'center': candidate['center'],
                        'dimensions': candidate['dimensions'],
                        'confidence': prediction['confidence'],
                        'features': candidate['features'],
                        'pharmacophore': prediction.get('pharmacophore', {})
                    })

        # Sort by confidence
        high_confidence_sites.sort(key=lambda x: x['confidence'], reverse=True)

        # Convert to grid boxes
        grid_boxes = []
        for i, site in enumerate(high_confidence_sites[:max_sites]):
            grid_box = GridBox(
                center=tuple(site['center']),
                dimensions=tuple(site['dimensions']),
                spacing=spacing,
                receptor_file=receptor_file,
                method_used="ai_prediction",
                confidence_score=site['confidence'],
                binding_site_info={
                    'ai_features': site['features'],
                    'pharmacophore_features': site['pharmacophore'],
                    'prediction_rank': i + 1,
                    'model_version': 'simple_v1.0'
                }
            )
            grid_boxes.append(grid_box)

        return grid_boxes

    def _fallback_cavity_detection(self, receptor_file: str, max_sites: int, spacing: float) -> List[GridBox]:
        """Fallback to cavity detection when AI fails"""
        from ..gridbox.cavity_detection import CavityDetectionGridBoxGenerator

        cavity_generator = CavityDetectionGridBoxGenerator()
        cavity_boxes = cavity_generator.generate(
            receptor_file, max_cavities=max_sites, spacing=spacing
        )

        # Convert cavity boxes to AI-labeled boxes
        for box in cavity_boxes:
            box.method_used = "ai_cavity_fallback"
            box.confidence_score = min(box.confidence_score, 0.8)  # Lower confidence for fallback

        return cavity_boxes

    def _generate_candidate_sites(self, structure) -> List[Dict[str, Any]]:
        """Generate candidate binding sites using efficient surface-based approach"""

        atoms = list(structure.get_atoms())
        coords = np.array([atom.get_coord() for atom in atoms])

        # Use a much more efficient approach - sample points around the protein surface
        candidates = []
        protein_center = np.mean(coords, axis=0)

        # Pre-compute atom positions for efficiency
        ca_atoms = []
        for atom in atoms:
            try:
                if atom.get_name() == 'CA':  # Focus on CA atoms for surface detection
                    ca_atoms.append(atom.get_coord())
            except:
                continue

        if not ca_atoms:
            self.logger.warning("No CA atoms found, using all atoms")
            ca_coords = coords[::10]  # Subsample for efficiency
        else:
            ca_coords = np.array(ca_atoms)

        candidates_tested = 0
        count = 0

        # Sample points around each CA atom (surface-focused approach)
        for ca_coord in ca_coords:
            if count >= 50:  # Limit to prevent timeout
                break

            # Sample points around this CA atom at different distances
            for distance in [4.0, 6.0, 8.0]:  # Different cavity depths
                for theta in np.linspace(0, np.pi, 3):  # 3 polar angles
                    for phi in np.linspace(0, 2*np.pi, 4):  # 4 azimuthal angles
                        # Spherical to Cartesian
                        x = ca_coord[0] + distance * np.sin(theta) * np.cos(phi)
                        y = ca_coord[1] + distance * np.sin(theta) * np.sin(phi)
                        z = ca_coord[2] + distance * np.cos(theta)
                        center = np.array([x, y, z])

                        candidates_tested += 1

                        # Quick feasibility check
                        distances_to_atoms = np.linalg.norm(coords - center, axis=1)
                        min_distance = np.min(distances_to_atoms)

                        # Must be accessible but not too far
                        if min_distance < 3.0 or min_distance > 12.0:
                            continue

                        # Extract features (simplified for speed)
                        features = self._extract_local_features_fast(center, coords)

                        if features.get('feature_0', 0.0) < 0.05:  # Must have some protein nearby
                            continue

                        # Log first few candidates for debugging
                        if count < 3:
                            self.logger.info(f"Candidate {count}: center={center}, density={features.get('feature_0', 0.0):.3f}")

                        candidates.append({
                            'center': center,
                            'dimensions': np.array([20.0, 20.0, 20.0]),
                            'features': features
                        })
                        count += 1

                        if count >= 20:  # Limit total candidates
                            break
                    if count >= 20:
                        break
                if count >= 20:
                    break

        self.logger.info(f"Tested {candidates_tested} positions, found {count} candidates")
        return candidates

    def _extract_local_features_fast(self, center: np.ndarray, coords: np.ndarray, radius: float = 8.0) -> Dict[str, float]:
        """Fast feature extraction using coordinate array"""

        # Get atoms within radius
        distances = np.linalg.norm(coords - center, axis=1)
        local_mask = distances <= radius
        local_coords = coords[local_mask]

        if len(local_coords) == 0:
            return {f'feature_{i}': 0.0 for i in range(self.feature_dim)}

        # Simple features for speed
        features = {}

        # Protein density (normalized)
        volume = (4/3) * np.pi * radius**3
        protein_density = len(local_coords) / volume
        features['protein_density'] = min(protein_density / 0.1, 1.0)

        # Surface accessibility (rough estimate)
        surface_atoms = np.sum(distances[local_mask] > radius * 0.7)
        features['surface_accessibility'] = surface_atoms / len(local_coords)

        # Basic geometric features
        if len(local_coords) > 1:
            coord_var = np.var(local_coords, axis=0)
            features['shape_complexity'] = np.mean(coord_var) / 100.0
        else:
            features['shape_complexity'] = 0.0

        # Create feature vector
        feature_vector = {}
        for i in range(self.feature_dim):
            if i == 0:
                feature_vector[f'feature_{i}'] = features['protein_density']
            elif i == 1:
                feature_vector[f'feature_{i}'] = features['surface_accessibility']
            elif i == 2:
                feature_vector[f'feature_{i}'] = features['shape_complexity']
            else:
                feature_vector[f'feature_{i}'] = 0.1  # Default small value

        return feature_vector

    def _extract_local_features(self, center: np.ndarray, atoms: List, radius: float = 12.0) -> Dict[str, float]:
        """Extract local chemical and geometric features around a center point"""

        features = {}

        # Get atoms within radius
        local_atoms = []
        for atom in atoms:
            distance = np.linalg.norm(atom.get_coord() - center)
            if distance <= radius:
                local_atoms.append((atom, distance))

        if not local_atoms:
            return {f'feature_{i}': 0.0 for i in range(self.feature_dim)}

        # Feature 1: Protein density (normalize by expected protein density)
        volume = (4/3) * np.pi * radius**3
        protein_density = len(local_atoms) / volume  # atoms per cubic angstrom
        # Normalize by typical protein density (roughly 1.35 g/cm³ ≈ 0.1 atoms/Å³)
        normalized_density = protein_density / 0.1
        features['protein_density'] = min(normalized_density, 1.0)  # Cap at 1.0

        # Feature 2: Hydrophobic ratio
        hydrophobic_elements = {'C'}
        hydrophobic_count = sum(1 for atom, _ in local_atoms if atom.element in hydrophobic_elements)
        hydrophobic_ratio = hydrophobic_count / len(local_atoms)
        features['hydrophobic_ratio'] = hydrophobic_ratio

        # Feature 3: Hydrophilic ratio
        hydrophilic_elements = {'N', 'O', 'S'}
        hydrophilic_count = sum(1 for atom, _ in local_atoms if atom.element in hydrophilic_elements)
        hydrophilic_ratio = hydrophilic_count / len(local_atoms)
        features['hydrophilic_ratio'] = hydrophilic_ratio

        # Feature 4: Aromatic residue count
        aromatic_residues = {'PHE', 'TYR', 'TRP', 'HIS'}
        aromatic_count = 0
        for atom, _ in local_atoms:
            try:
                if atom.get_parent().get_resname() in aromatic_residues:
                    aromatic_count += 1
            except:
                pass
        features['aromatic_density'] = aromatic_count / len(local_atoms)

        # Feature 5: Charged residue count
        charged_residues = {'ARG', 'LYS', 'ASP', 'GLU', 'HIS'}
        charged_count = 0
        for atom, _ in local_atoms:
            try:
                if atom.get_parent().get_resname() in charged_residues:
                    charged_count += 1
            except:
                pass
        features['charged_density'] = charged_count / len(local_atoms)

        # Feature 6: Surface accessibility (simple approximation)
        surface_atoms = sum(1 for atom, dist in local_atoms if dist > radius * 0.7)
        features['surface_accessibility'] = surface_atoms / len(local_atoms)

        # Feature 7: Cavity depth (distance to nearest surface)
        min_distance = min(dist for _, dist in local_atoms)
        features['cavity_depth'] = min_distance / radius

        # Feature 8-10: Secondary structure features (simplified)
        features['alpha_helix_density'] = np.random.random() * 0.1  # Placeholder
        features['beta_sheet_density'] = np.random.random() * 0.1  # Placeholder
        features['loop_density'] = np.random.random() * 0.1  # Placeholder

        # Features 11-15: Geometric features
        coords = np.array([atom.get_coord() for atom, _ in local_atoms])
        if len(coords) > 1:
            # Calculate variance in coordinates (shape complexity)
            coord_var = np.var(coords, axis=0)
            features['shape_complexity_x'] = coord_var[0] / 100.0
            features['shape_complexity_y'] = coord_var[1] / 100.0
            features['shape_complexity_z'] = coord_var[2] / 100.0

            # Calculate compactness
            center_distances = np.linalg.norm(coords - center, axis=1)
            features['compactness'] = np.std(center_distances) / np.mean(center_distances)

            # Calculate eccentricity (shape descriptor)
            cov_matrix = np.cov(coords.T)
            eigenvals = np.linalg.eigvals(cov_matrix)
            features['eccentricity'] = (max(eigenvals) - min(eigenvals)) / max(eigenvals)
        else:
            features.update({
                'shape_complexity_x': 0.0, 'shape_complexity_y': 0.0, 'shape_complexity_z': 0.0,
                'compactness': 0.0, 'eccentricity': 0.0
            })

        # Features 16-20: Advanced chemical features
        features['metal_binding'] = np.random.random() * 0.1  # Placeholder
        features['hydrogen_bond_potential'] = features['hydrophilic_ratio'] * 0.8
        features['electrostatic_potential'] = features['charged_density'] * 0.6
        features['van_der_waals_surface'] = features['protein_density'] * 0.001
        features['druggability_index'] = (features['hydrophobic_ratio'] + features['aromatic_density']) * 0.5

        # Normalize and pad to feature_dim
        feature_vector = {}
        feature_keys = sorted(features.keys())[:self.feature_dim]
        for i, key in enumerate(feature_keys):
            feature_vector[f'feature_{i}'] = min(max(features[key], 0.0), 1.0)  # Clamp to [0,1]

        # Fill remaining features with zeros if needed
        for i in range(len(feature_keys), self.feature_dim):
            feature_vector[f'feature_{i}'] = 0.0

        return feature_vector

    def _predict_binding_sites(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Predict binding site likelihood using simple heuristics (placeholder for actual AI model)"""

        predictions = []

        for candidate in candidates:
            features = candidate['features']

            # Simple heuristic scoring (replace with actual AI model)
            score = 0.0

            # Get basic features
            protein_density = features.get('feature_0', 0.0)
            surface_accessibility = features.get('feature_1', 0.0)
            shape_complexity = features.get('feature_2', 0.0)

            # Score based on protein density (not too high, not too low)
            density_score = 1.0 - abs(protein_density - 0.3) / 0.3  # Optimal around 0.3
            score += max(density_score, 0.0) * 0.4

            # Prefer sites with reasonable surface accessibility
            score += surface_accessibility * 0.3

            # Prefer sites with some shape complexity
            score += min(shape_complexity, 0.5) * 0.2

            # Add some randomness to simulate model uncertainty
            score += (np.random.random() - 0.5) * 0.1

            # Clamp to [0,1]
            confidence = max(0.0, min(1.0, score))

            predictions.append({
                'confidence': confidence,
                'pharmacophore': {
                    'protein_density': protein_density,
                    'surface_accessibility': surface_accessibility,
                    'shape_complexity': shape_complexity,
                    'model_type': 'simplified_features'
                }
            })

        return predictions

    def load_model(self, model_path: str):
        """Load pre-trained AI model"""
        if not HAS_TORCH:
            self.logger.warning("PyTorch not available, using heuristic scoring only")
            return

        if Path(model_path).exists():
            self.model = SimpleBindingSitePredictor(self.feature_dim)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            self.logger.info(f"Loaded AI model from {model_path}")
        else:
            self.logger.warning(f"Model file {model_path} not found, using heuristic scoring")

    def save_model(self, model_path: str):
        """Save trained AI model"""
        if not HAS_TORCH:
            self.logger.warning("PyTorch not available, cannot save model")
            return

        if self.model is not None:
            torch.save(self.model.state_dict(), model_path)
            self.logger.info(f"Saved AI model to {model_path}")