# analysis.py
"""
Advanced pose clustering and analysis tools for PandaDock.
This module provides methods for clustering docking results, analyzing binding modes,
generating interaction fingerprints, and creating detailed reports.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from .utils import calculate_rmsd

class PoseClusterer:
    """Clustering of docking poses using various methods."""
    
    def __init__(self, method='hierarchical', rmsd_cutoff=2.0, 
                 min_cluster_size=3):
        """
        Initialize pose clusterer.
        
        Parameters:
        -----------
        method : str
            Clustering method ('hierarchical' or 'dbscan')
        rmsd_cutoff : float
            RMSD cutoff for clustering (Angstroms)
        min_cluster_size : int
            Minimum number of poses for a valid cluster
        """
        self.method = method
        self.rmsd_cutoff = rmsd_cutoff
        self.min_cluster_size = min_cluster_size
    
    def cluster_poses(self, poses, scores=None):
        """
        Cluster ligand poses based on RMSD.
        
        Parameters:
        -----------
        poses : list
            List of Ligand objects representing docking poses
        scores : list, optional
            Corresponding scores for each pose
            
        Returns:
        --------
        dict
            Clustering results with cluster assignments and representatives
        """
        if self.method == 'hierarchical':
            return self._hierarchical_clustering(poses, scores)
        else:
            raise ValueError(f"Clustering method '{self.method}' not implemented")
    
    def _hierarchical_clustering(self, poses, scores=None):
        """
        Perform hierarchical clustering based on RMSD.
        
        Parameters:
        -----------
        poses : list
            List of Ligand objects
        scores : list, optional
            Corresponding scores for each pose
            
        Returns:
        --------
        dict
            Clustering results
        """
        n_poses = len(poses)
        
        if scores is None:
            scores = [0.0] * n_poses
            
        print(f"Computing RMSD matrix for {n_poses} poses...")
        
        # Compute pairwise RMSD matrix
        rmsd_matrix = np.zeros((n_poses, n_poses))
        for i in range(n_poses):
            for j in range(i+1, n_poses):
                rmsd = calculate_rmsd(poses[i].xyz, poses[j].xyz)
                rmsd_matrix[i, j] = rmsd
                rmsd_matrix[j, i] = rmsd
        
        # Convert distance matrix to condensed form for scipy
        condensed_matrix = squareform(rmsd_matrix)
        
        # Perform hierarchical clustering
        print("Performing hierarchical clustering...")
        Z = linkage(condensed_matrix, method='average')
        
        # Form flat clusters based on RMSD cutoff
        cluster_indices = fcluster(Z, self.rmsd_cutoff, criterion='distance')
        
        # Process clusters
        clusters = {}
        for i, cluster_idx in enumerate(cluster_indices):
            if cluster_idx not in clusters:
                clusters[cluster_idx] = []
            clusters[cluster_idx].append({
                'pose_idx': i,
                'pose': poses[i],
                'score': scores[i]
            })
        
        # Format results
        result_clusters = []
        for cluster_idx, members in clusters.items():
            # Sort cluster members by score
            members.sort(key=lambda x: x['score'])
            
            # Skip small clusters
            if len(members) < self.min_cluster_size:
                continue
                
            # Add cluster to results
            result_clusters.append({
                'members': members,
                'size': len(members),
                'representative': members[0]['pose_idx'],
                'best_score': members[0]['score']
            })
        
        # Sort clusters by size (largest first)
        result_clusters.sort(key=lambda x: x['size'], reverse=True)
        
        return {
            'clusters': result_clusters,
            'n_clusters': len(result_clusters),
            'rmsd_cutoff': self.rmsd_cutoff,
            'rmsd_matrix': rmsd_matrix
        }
    
    def visualize_clusters(self, clustering_results, output_file=None):
        """
        Generate visualization of clustering results.
        
        Parameters:
        -----------
        clustering_results : dict
            Results from cluster_poses method
        output_file : str, optional
            Path to save the visualization
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        clusters = clustering_results['clusters']
        rmsd_matrix = clustering_results['rmsd_matrix']
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: RMSD heatmap
        im = ax1.imshow(rmsd_matrix, cmap='viridis', interpolation='nearest')
        ax1.set_title('Pairwise RMSD Matrix')
        ax1.set_xlabel('Pose Index')
        ax1.set_ylabel('Pose Index')
        fig.colorbar(im, ax=ax1, label='RMSD (Å)')
        
        # Plot 2: Cluster sizes and best scores
        cluster_sizes = [c['size'] for c in clusters]
        cluster_scores = [c['best_score'] for c in clusters]
        
        # Create bar plot for cluster sizes
        bars = ax2.bar(range(len(clusters)), cluster_sizes, alpha=0.7)
        ax2.set_title('Cluster Sizes and Best Scores')
        ax2.set_xlabel('Cluster Index')
        ax2.set_ylabel('Number of Poses')
        ax2.set_xticks(range(len(clusters)))
        
        # Add score labels
        for i, (bar, score) in enumerate(zip(bars, cluster_scores)):
            ax2.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f'{score:.2f}',
                ha='center',
                va='bottom',
                rotation=90,
                fontsize=9
            )
        
        # Add second y-axis for scores
        ax2_score = ax2.twinx()
        ax2_score.plot(range(len(clusters)), cluster_scores, 'ro-', alpha=0.7)
        ax2_score.set_ylabel('Best Score', color='r')
        ax2_score.tick_params(axis='y', labelcolor='r')
        
        plt.tight_layout()
        
        # Save figure if output file specified
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            
        return fig

class InteractionFingerprinter:
    """Generate and analyze protein-ligand interaction fingerprints."""
    
    def __init__(self, interaction_types=None, distance_cutoff=4.5, 
                 angle_cutoff=60, include_water=False):
        """
        Initialize interaction fingerprinter.
        
        Parameters:
        -----------
        interaction_types : list, optional
            Types of interactions to include in fingerprint
        distance_cutoff : float
            Maximum distance for interactions (Angstroms)
        angle_cutoff : float
            Angle cutoff for directional interactions (degrees)
        include_water : bool
            Whether to include water-mediated interactions
        """
        self.interaction_types = interaction_types or ['hbond', 'hydrophobic', 
                                                      'ionic', 'aromatic', 'halogen']
        self.distance_cutoff = distance_cutoff
        self.angle_cutoff = angle_cutoff
        self.include_water = include_water
        
        # Define atom types for different interactions
        self.hbond_donors = {'N', 'O'}
        self.hbond_acceptors = {'N', 'O', 'F'}
        self.hydrophobic_atoms = {'C'}
        self.halogen_atoms = {'F', 'Cl', 'Br', 'I'}
        self.charged_positive = {'N'}  # Simplified: N in Lys, Arg
        self.charged_negative = {'O'}  # Simplified: O in Asp, Glu
        self.aromatic_atoms = {'C'}    # Simplified: aromatic carbons
    
    def generate_fingerprint(self, protein, ligand):
        """
        Generate interaction fingerprint for a protein-ligand complex.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        ligand : Ligand
            Ligand pose
        
        Returns:
        --------
        dict
            Interaction fingerprint with counts of each interaction type
        """
        # Extract appropriate protein atoms
        if protein.active_site and 'atoms' in protein.active_site:
            protein_atoms = protein.active_site['atoms']
        else:
            protein_atoms = protein.atoms
        
        # Initialize fingerprint
        fingerprint = {
            'hbond': 0,
            'hydrophobic': 0,
            'ionic': 0,
            'aromatic': 0,
            'halogen': 0,
            'interactions': []  # Will store detailed interaction data
        }
        
        # Check each ligand-protein atom pair for interactions
        for l_atom in ligand.atoms:
            l_symbol = l_atom.get('symbol', 'C')
            l_coords = l_atom['coords']
            
            for p_atom in protein_atoms:
                p_symbol = p_atom.get('element', p_atom.get('name', 'C'))[0]
                p_coords = p_atom['coords']
                
                # Calculate distance
                distance = np.linalg.norm(l_coords - p_coords)
                
                # Skip if too far
                if distance > self.distance_cutoff:
                    continue
                
                # Check for hydrogen bonds
                if 'hbond' in self.interaction_types:
                    if ((l_symbol in self.hbond_donors and p_symbol in self.hbond_acceptors) or
                        (l_symbol in self.hbond_acceptors and p_symbol in self.hbond_donors)):
                        # In a full implementation, we would check angles here
                        if distance < 3.5:  # Typical H-bond distance
                            fingerprint['hbond'] += 1
                            fingerprint['interactions'].append({
                                'type': 'hbond',
                                'ligand_atom': l_atom,
                                'protein_atom': p_atom,
                                'distance': distance
                            })
                
                # Check for hydrophobic interactions
                if 'hydrophobic' in self.interaction_types:
                    if l_symbol in self.hydrophobic_atoms and p_symbol in self.hydrophobic_atoms:
                        if distance < 4.0:  # Typical hydrophobic interaction distance
                            fingerprint['hydrophobic'] += 1
                            fingerprint['interactions'].append({
                                'type': 'hydrophobic',
                                'ligand_atom': l_atom,
                                'protein_atom': p_atom,
                                'distance': distance
                            })
                
                # Check for ionic interactions
                if 'ionic' in self.interaction_types:
                    if ((l_symbol in self.charged_positive and p_symbol in self.charged_negative) or
                        (l_symbol in self.charged_negative and p_symbol in self.charged_positive)):
                        if distance < 4.0:  # Typical ionic interaction distance
                            fingerprint['ionic'] += 1
                            fingerprint['interactions'].append({
                                'type': 'ionic',
                                'ligand_atom': l_atom,
                                'protein_atom': p_atom,
                                'distance': distance
                            })
                
                # Check for halogen bonds
                if 'halogen' in self.interaction_types:
                    if l_symbol in self.halogen_atoms and p_symbol in self.hbond_acceptors:
                        if distance < 3.5:  # Typical halogen bond distance
                            fingerprint['halogen'] += 1
                            fingerprint['interactions'].append({
                                'type': 'halogen',
                                'ligand_atom': l_atom,
                                'protein_atom': p_atom,
                                'distance': distance
                            })
        
        # Aromatic interactions require special handling (simplified implementation)
        if 'aromatic' in self.interaction_types:
            # In a full implementation, we would identify aromatic rings
            # and check for π-π stacking and other aromatic interactions
            pass
        
        return fingerprint
    
    def compare_fingerprints(self, fp1, fp2):
        """
        Compare two interaction fingerprints and return similarity score.
        
        Parameters:
        -----------
        fp1 : dict
            First fingerprint
        fp2 : dict
            Second fingerprint
        
        Returns:
        --------
        float
            Similarity score (0.0 to 1.0)
        """
        # Calculate Tanimoto coefficient for interaction counts
        intersection = 0
        union = 0
        
        for interaction_type in self.interaction_types:
            count1 = fp1.get(interaction_type, 0)
            count2 = fp2.get(interaction_type, 0)
            
            intersection += min(count1, count2)
            union += max(count1, count2)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def analyze_key_interactions(self, protein, ligand):
        """
        Identify key interactions in a protein-ligand complex.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        ligand : Ligand
            Ligand pose
        
        Returns:
        --------
        list
            List of key interaction descriptions
        """
        # Generate fingerprint
        fingerprint = self.generate_fingerprint(protein, ligand)
        
        # Extract protein residue information
        residue_interactions = {}
        
        for interaction in fingerprint['interactions']:
            p_atom = interaction['protein_atom']
            
            # Get residue info
            res_name = p_atom.get('residue_name', 'UNK')
            chain_id = p_atom.get('chain_id', 'A')
            res_id = p_atom.get('residue_id', 0)
            
            # Create residue identifier
            res_key = f"{res_name} {chain_id}:{res_id}"
            
            if res_key not in residue_interactions:
                residue_interactions[res_key] = []
            
            residue_interactions[res_key].append(interaction)
        
        # Create descriptive list of key interactions
        key_interactions = []
        
        for res_key, interactions in residue_interactions.items():
            # Count interaction types
            interaction_counts = {}
            for interaction in interactions:
                int_type = interaction['type']
                if int_type not in interaction_counts:
                    interaction_counts[int_type] = 0
                interaction_counts[int_type] += 1
            
            # Generate description
            description = f"{res_key}: "
            interaction_strs = []
            
            for int_type, count in interaction_counts.items():
                if count == 1:
                    interaction_strs.append(f"1 {int_type}")
                else:
                    interaction_strs.append(f"{count} {int_type}s")
            
            description += ", ".join(interaction_strs)
            key_interactions.append(description)
        
        # Sort by residue ID
        key_interactions.sort()
        
        return key_interactions

class BindingModeClassifier:
    """Classify binding modes of docking poses."""
    
    def __init__(self, reference_modes=None, similarity_threshold=0.7):
        """
        Initialize binding mode classifier.
        
        Parameters:
        -----------
        reference_modes : dict, optional
            Dictionary of reference binding modes
        similarity_threshold : float
            Threshold for similarity to classify a pose into a mode
        """
        self.reference_modes = reference_modes or {}
        self.similarity_threshold = similarity_threshold
        self.fingerprinter = InteractionFingerprinter()
    
    def classify_pose(self, protein, pose):
        """
        Classify a docking pose into a binding mode category.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        pose : Ligand
            Ligand pose
        
        Returns:
        --------
        str
            Binding mode classification
        """
        # Generate fingerprint for pose
        pose_fp = self.fingerprinter.generate_fingerprint(protein, pose)
        
        # Compare with reference modes
        best_mode = "Unknown"
        best_similarity = 0.0
        
        for mode_name, mode_data in self.reference_modes.items():
            similarity = self.fingerprinter.compare_fingerprints(pose_fp, mode_data['fingerprint'])
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_mode = mode_name
        
        # Check if similarity exceeds threshold
        if best_similarity >= self.similarity_threshold:
            return best_mode
        else:
            return "Novel"
    
    def define_reference_mode(self, name, protein, reference_pose):
        """
        Define a new reference binding mode.
        
        Parameters:
        -----------
        name : str
            Name for the binding mode
        protein : Protein
            Protein object
        reference_pose : Ligand
            Reference ligand pose for this binding mode
        """
        # Generate fingerprint for reference pose
        fp = self.fingerprinter.generate_fingerprint(protein, reference_pose)
        
        # Store reference mode
        self.reference_modes[name] = {
            'fingerprint': fp,
            'pose': reference_pose
        }
        
        print(f"Defined reference binding mode '{name}' with {sum(fp[k] for k in fp if k != 'interactions')} interactions")
    
    def discover_modes(self, protein, poses, n_modes=5):
        """
        Discover binding modes from a set of poses.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        poses : list
            List of ligand poses
        n_modes : int
            Number of binding modes to discover
        
        Returns:
        --------
        list
            List of discovered binding modes
        """
        # Generate fingerprints for all poses
        fingerprints = [self.fingerprinter.generate_fingerprint(protein, pose) for pose in poses]
        
        # Calculate pairwise similarity matrix
        n_poses = len(poses)
        similarity_matrix = np.zeros((n_poses, n_poses))
        
        for i in range(n_poses):
            for j in range(i+1, n_poses):
                similarity = self.fingerprinter.compare_fingerprints(fingerprints[i], fingerprints[j])
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        
        # Cluster poses based on similarity
        try:
            from sklearn.cluster import AgglomerativeClustering
            
            # Convert similarity to distance
            distance_matrix = 1.0 - similarity_matrix
            
            # Apply hierarchical clustering
            clustering = AgglomerativeClustering(
                n_clusters=min(n_modes, n_poses),
                affinity='precomputed',
                linkage='average'
            )
            
            cluster_labels = clustering.fit_predict(distance_matrix)
            
            # Create binding mode data
            modes = []
            clusters = {}
            
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(i)
            
            # Process each cluster
            for label, indices in clusters.items():
                # Find most central pose (highest average similarity to other poses in cluster)
                avg_similarities = [np.mean([similarity_matrix[i, j] for j in indices if j != i]) for i in indices]
                central_idx = indices[np.argmax(avg_similarities)]
                
                # Create mode data
                modes.append({
                    'name': f"Mode {label+1}",
                    'representative': central_idx,
                    'members': indices,
                    'count': len(indices),
                    'fingerprint': fingerprints[central_idx],
                    'pose': poses[central_idx],
                    'best_score': min([self._get_pose_score(poses[i]) for i in indices])
                })
            
            # Sort modes by count (largest first)
            modes.sort(key=lambda x: x['count'], reverse=True)
            
            return modes
            
        except ImportError:
            print("Warning: scikit-learn not available. Using simplified mode discovery.")
            
            # Simple approach: just take the top n_modes poses as representatives
            modes = []
            for i in range(min(n_modes, n_poses)):
                modes.append({
                    'name': f"Mode {i+1}",
                    'representative': i,
                    'members': [i],
                    'count': 1,
                    'fingerprint': fingerprints[i],
                    'pose': poses[i],
                    'best_score': self._get_pose_score(poses[i])
                })
            
            return modes
    
    def _get_pose_score(self, pose):
        """Helper to extract score from pose if available."""
        if hasattr(pose, 'score'):
            return pose.score
        return 0.0

class EnergyDecomposition:
    """Decompose binding energy into component contributions."""
    
    def __init__(self, scoring_function):
        """
        Initialize energy decomposition analyzer.
        
        Parameters:
        -----------
        scoring_function : ScoringFunction
            Scoring function to use for decomposition
        """
        self.scoring_function = scoring_function
    
    def decompose_energy(self, protein, ligand):
        """
        Break down the binding energy into components.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        ligand : Ligand
            Ligand pose
        
        Returns:
        --------
        dict
            Energy components
        """
        # Check if scoring function has component-specific scoring methods
        if hasattr(self.scoring_function, '_calculate_vdw'):
            vdw_energy = self.scoring_function._calculate_vdw(protein, ligand)
        elif hasattr(self.scoring_function, '_calculate_vdw_energy'):
            vdw_energy = self.scoring_function._calculate_vdw_energy(protein, ligand)
        else:
            vdw_energy = None
        
        if hasattr(self.scoring_function, '_calculate_hbond'):
            hbond_energy = self.scoring_function._calculate_hbond(protein, ligand)
        elif hasattr(self.scoring_function, '_calculate_hbond_energy'):
            hbond_energy = self.scoring_function._calculate_hbond_energy(protein, ligand)
        else:
            hbond_energy = None
        
        if hasattr(self.scoring_function, '_calculate_electrostatics'):
            elec_energy = self.scoring_function._calculate_electrostatics(protein, ligand)
        else:
            elec_energy = None
        
        if hasattr(self.scoring_function, '_calculate_desolvation'):
            desolv_energy = self.scoring_function._calculate_desolvation(protein, ligand)
        else:
            desolv_energy = None
        
        if hasattr(self.scoring_function, '_calculate_hydrophobic'):
            hydrophobic_energy = self.scoring_function._calculate_hydrophobic(protein, ligand)
        else:
            hydrophobic_energy = None
        
        if hasattr(self.scoring_function, '_calculate_entropy'):
            entropy_energy = self.scoring_function._calculate_entropy(ligand)
        else:
            entropy_energy = None
        
        if hasattr(self.scoring_function, '_calculate_clashes'):
            clash_energy = self.scoring_function._calculate_clashes(protein, ligand)
        else:
            clash_energy = None
        
        # Calculate total energy
        total_energy = self.scoring_function.score(protein, ligand)
        
        # Create energy components dictionary
        components = {'total': total_energy}
        
        if vdw_energy is not None:
            components['vdw'] = vdw_energy
        
        if hbond_energy is not None:
            components['hbond'] = hbond_energy
        
        if elec_energy is not None:
            components['electrostatic'] = elec_energy
        
        if desolv_energy is not None:
            components['desolvation'] = desolv_energy
        
        if hydrophobic_energy is not None:
            components['hydrophobic'] = hydrophobic_energy
        
        if entropy_energy is not None:
            components['entropy'] = entropy_energy
        
        if clash_energy is not None:
            components['clash'] = clash_energy
        
        # Compute remainder if some components are missing
        known_sum = sum(v for k, v in components.items() if k != 'total')
        if abs(known_sum - total_energy) > 0.1:
            components['other'] = total_energy - known_sum
        
        return components
    
    def residue_contributions(self, protein, ligand, radius=2.0):
        """
        Calculate per-residue contributions to binding energy.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        ligand : Ligand
            Ligand pose
        radius : float
            Radius around ligand to consider residues (Angstroms)
        
        Returns:
        --------
        list
            List of (residue, energy) tuples, sorted by contribution
        """
        # Extract appropriate protein atoms
        if protein.active_site and 'atoms' in protein.active_site:
            protein_atoms = protein.active_site['atoms']
        else:
            protein_atoms = protein.atoms
        
        # Group atoms by residue
        residue_atoms = {}
        for atom in protein_atoms:
            res_name = atom.get('residue_name', 'UNK')
            chain_id = atom.get('chain_id', 'A')
            res_id = atom.get('residue_id', 0)
            
            res_key = f"{res_name} {chain_id}:{res_id}"
            
            if res_key not in residue_atoms:
                residue_atoms[res_key] = []
            
            residue_atoms[res_key].append(atom)
        
        # Calculate energy contribution for each residue
        contributions = []
        for res_key, atoms in residue_atoms.items():
            # Check if any atom is near the ligand
            near_ligand = False
            for atom in atoms:
                atom_pos = atom['coords']
                for l_atom in ligand.atoms:
                    l_pos = l_atom['coords']
                    if np.linalg.norm(atom_pos - l_pos) <= radius:
                        near_ligand = True
                        break
                if near_ligand:
                    break
            
            if not near_ligand:
                continue
            
            # Create temporary protein with only this residue
            from types import SimpleNamespace
            temp_protein = SimpleNamespace()
            temp_protein.atoms = atoms
            temp_protein.active_site = None
            
            # Calculate energy
            energy = self.scoring_function.score(temp_protein, ligand)
            
            contributions.append((res_key, energy))
        
        # Sort by energy (largest contribution first)
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return contributions
    
    def visualize_decomposition(self, energy_components):
        """
        Create visualization of energy component contributions.
        
        Parameters:
        -----------
        energy_components : dict
            Energy components from decompose_energy method
        
        Returns:
        --------
        plt.Figure
            Matplotlib figure with visualization
        """
        # Extract components (excluding total)
        components = {k: v for k, v in energy_components.items() if k != 'total'}
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort components by absolute value
        sorted_components = sorted(components.items(), key=lambda x: abs(x[1]), reverse=True)
        labels = [item[0] for item in sorted_components]
        values = [item[1] for item in sorted_components]
        
        # Set bar colors based on sign
        colors = ['green' if v < 0 else 'red' for v in values]
        
        # Create the bar chart
        bars = ax.bar(labels, values, color=colors)
        
        # Add labels and title
        ax.set_xlabel('Energy Component')
        ax.set_ylabel('Energy (kcal/mol)')
        ax.set_title('Binding Energy Decomposition')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + (0.1 if height > 0 else -0.1),
                f'{height:.2f}',
                ha='center',
                va='bottom' if height > 0 else 'top',
                fontsize=9
            )
        
        # Add total energy line
        total = energy_components.get('total', sum(values))
        ax.axhline(y=total, color='black', linestyle='-', alpha=0.7)
        ax.text(0, total, f'Total: {total:.2f}', va='bottom', ha='left', fontsize=9)
        
        # Adjust layout
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return fig

class DockingReportGenerator:
    """Generate comprehensive docking analysis reports."""
    
    def __init__(self, report_format='html', include_sections=None):
        """
        Initialize report generator.
        
        Parameters:
        -----------
        report_format : str
            Report format ('html', 'pdf', or 'txt')
        include_sections : list, optional
            Sections to include in the report
        """
        self.report_format = report_format
        self.include_sections = include_sections or ['summary', 'clusters', 
                                                    'interactions', 'energetics']
    
    def generate_report(self, protein, poses, scores, output_file, 
                        clustering_results=None, energy_decomposition=None):
        """
        Generate a comprehensive docking report.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        poses : list
            List of ligand poses
        scores : list
            Corresponding scores for each pose
        output_file : str
            Output file path
        clustering_results : dict, optional
            Clustering results from PoseClusterer
        energy_decomposition : dict, optional
            Energy decomposition from EnergyDecomposition
        
        Returns:
        --------
        str
            Path to generated report
        """
        if self.report_format == 'html':
            return self._generate_html_report(
                protein, poses, scores, output_file, 
                clustering_results, energy_decomposition
            )
        elif self.report_format == 'pdf':
            return self._generate_pdf_report(
                protein, poses, scores, output_file, 
                clustering_results, energy_decomposition
            )
        elif self.report_format == 'txt':
            return self._generate_txt_report(
                protein, poses, scores, output_file, 
                clustering_results, energy_decomposition
            )
        else:
            raise ValueError(f"Unsupported report format: {self.report_format}")
    
    def _generate_html_report(self, protein, poses, scores, output_file, 
                             clustering_results, energy_decomposition):
        """Generate HTML report."""
        # Create HTML content
        html_content = [
            '<!DOCTYPE html>',
            '<html>',
            '<head>',
            '    <title>PandaDock Docking Report</title>',
            '    <style>',
            '        body { font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; max-width: 1200px; margin: 0 auto; }',
            '        h1, h2, h3 { color: #2c3e50; }',
            '        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }',
            '        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }',
            '        th { background-color: #f2f2f2; }',
            '        tr:nth-child(even) { background-color: #f9f9f9; }',
            '        .chart-container { margin: 20px 0; }',
            '        .footnote { font-size: 0.8em; color: #7f8c8d; margin-top: 40px; }',
            '    </style>',
            '</head>',
            '<body>',
            '    <h1>PandaDock Docking Report</h1>'
        ]
        
        # Summary section
        if 'summary' in self.include_sections:
            html_content.extend([
                '    <h2>Summary</h2>',
                '    <table>',
                '        <tr><th>Parameter</th><th>Value</th></tr>',
                f'        <tr><td>Number of Poses</td><td>{len(poses)}</td></tr>',
                f'        <tr><td>Best Score</td><td>{min(scores):.4f}</td></tr>',
                f'        <tr><td>Average Score</td><td>{sum(scores)/len(scores):.4f}</td></tr>'
            ])
            
            if hasattr(protein, 'active_site') and protein.active_site:
                center = protein.active_site.get('center')
                radius = protein.active_site.get('radius')
                if center is not None and radius is not None:
                    html_content.append(f'        <tr><td>Active Site</td><td>Center: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}), Radius: {radius:.1f} Å</td></tr>')
            
            html_content.append('    </table>')
        
        # Top poses section
        html_content.extend([
            '    <h2>Top Poses</h2>',
            '    <table>',
            '        <tr><th>Rank</th><th>Score</th></tr>'
        ])
        
        for i, score in enumerate(sorted(scores)[:10]):
            html_content.append(f'        <tr><td>{i+1}</td><td>{score:.4f}</td></tr>')
        
        html_content.append('    </table>')
        
        # Clusters section
        if 'clusters' in self.include_sections and clustering_results:
            html_content.append('    <h2>Clustering Analysis</h2>')
            
            if 'clusters' in clustering_results:
                html_content.extend([
                    f'    <p>Found {len(clustering_results["clusters"])} clusters using {clustering_results.get("rmsd_cutoff", 2.0)} Å RMSD cutoff.</p>',
                    '    <table>',
                    '        <tr><th>Cluster</th><th>Size</th><th>Best Score</th></tr>'
                ])
                
                for i, cluster in enumerate(clustering_results['clusters']):
                    html_content.append(f'        <tr><td>{i+1}</td><td>{cluster["size"]}</td><td>{cluster["best_score"]:.4f}</td></tr>')
                
                html_content.append('    </table>')
                
                # Add cluster visualization if available
                html_content.append('    <div class="chart-container">')
                html_content.append('        <p>Cluster visualization would appear here in a full implementation.</p>')
                html_content.append('    </div>')
        
        # Interactions section
        if 'interactions' in self.include_sections:
            html_content.append('    <h2>Protein-Ligand Interactions</h2>')
            
            # Analyze top pose
            if poses:
                fingerprinter = InteractionFingerprinter()
                try:
                    interactions = fingerprinter.analyze_key_interactions(protein, poses[0])
                    
                    html_content.extend([
                        '    <h3>Key Interactions for Top Pose</h3>',
                        '    <ul>'
                    ])
                    
                    for interaction in interactions:
                        html_content.append(f'        <li>{interaction}</li>')
                    
                    html_content.append('    </ul>')
                except Exception as e:
                    html_content.append(f'    <p>Error analyzing interactions: {str(e)}</p>')
        
        # Energetics section
        if 'energetics' in self.include_sections and energy_decomposition:
            html_content.append('    <h2>Energy Decomposition</h2>')
            
            html_content.extend([
                '    <table>',
                '        <tr><th>Component</th><th>Energy (kcal/mol)</th></tr>'
            ])
            
            for component, energy in energy_decomposition.items():
                html_content.append(f'        <tr><td>{component}</td><td>{energy:.4f}</td></tr>')
            
            html_content.append('    </table>')
            
            # Add energy visualization placeholders
            html_content.append('    <div class="chart-container">')
            html_content.append('        <p>Energy decomposition chart would appear here in a full implementation.</p>')
            html_content.append('    </div>')
        
        # Footer
        html_content.extend([
            '    <div class="footnote">',
            '        <p>Generated by PandaDock Analysis Module</p>',
            '    </div>',
            '</body>',
            '</html>'
        ])
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(html_content))
        
        return output_file
    
    def _generate_pdf_report(self, protein, poses, scores, output_file, 
                            clustering_results, energy_decomposition):
        """Generate PDF report."""
        try:
            # First generate HTML content
            html_file = output_file.replace('.pdf', '_temp.html')
            self._generate_html_report(protein, poses, scores, html_file, 
                                      clustering_results, energy_decomposition)
            
            # Convert HTML to PDF
            try:
                from weasyprint import HTML
                HTML(html_file).write_pdf(output_file)
                
                # Clean up temporary HTML file
                import os
                os.remove(html_file)
                
                return output_file
            except ImportError:
                print("WeasyPrint not available. HTML report generated instead.")
                return html_file
        except Exception as e:
            print(f"Error generating PDF report: {e}")
            return None
    
    def _generate_txt_report(self, protein, poses, scores, output_file, 
                            clustering_results, energy_decomposition):
        """Generate plain text report."""
        # Create text content
        text_content = []
        
        # Header
        text_content.extend([
            "=============================================================",
            "                 PandaDock Docking Report                    ",
            "=============================================================",
            ""
        ])
        
        # Summary section
        if 'summary' in self.include_sections:
            text_content.extend([
                "SUMMARY",
                "-------",
                f"Number of Poses: {len(poses)}",
                f"Best Score: {min(scores):.4f}",
                f"Average Score: {sum(scores)/len(scores):.4f}",
                ""
            ])
            
            if hasattr(protein, 'active_site') and protein.active_site:
                center = protein.active_site.get('center')
                radius = protein.active_site.get('radius')
                if center is not None and radius is not None:
                    text_content.append(f"Active Site: Center ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}), Radius {radius:.1f} Å")
                    text_content.append("")
        
        # Top poses section
        text_content.extend([
            "TOP POSES",
            "---------",
            "Rank\tScore",
            "----\t-----"
        ])
        
        for i, score in enumerate(sorted(scores)[:10]):
            text_content.append(f"{i+1}\t{score:.4f}")
        
        text_content.append("")
        
        # Clusters section
        if 'clusters' in self.include_sections and clustering_results:
            text_content.extend([
                "CLUSTERING ANALYSIS",
                "-------------------"
            ])
            
            if 'clusters' in clustering_results:
                text_content.append(f"Found {len(clustering_results['clusters'])} clusters using {clustering_results.get('rmsd_cutoff', 2.0)} Å RMSD cutoff.")
                text_content.append("")
                text_content.append("Cluster\tSize\tBest Score")
                text_content.append("-------\t----\t----------")
                
                for i, cluster in enumerate(clustering_results['clusters']):
                    text_content.append(f"{i+1}\t{cluster['size']}\t{cluster['best_score']:.4f}")
                
                text_content.append("")
        
        # Interactions section
        if 'interactions' in self.include_sections:
            text_content.extend([
                "PROTEIN-LIGAND INTERACTIONS",
                "--------------------------"
            ])
            
            # Analyze top pose
            if poses:
                fingerprinter = InteractionFingerprinter()
                try:
                    interactions = fingerprinter.analyze_key_interactions(protein, poses[0])
                    
                    text_content.append("Key Interactions for Top Pose:")
                    for interaction in interactions:
                        text_content.append(f"- {interaction}")
                    
                    text_content.append("")
                except Exception as e:
                    text_content.append(f"Error analyzing interactions: {str(e)}")
                    text_content.append("")
        
        # Energetics section
        if 'energetics' in self.include_sections and energy_decomposition:
            text_content.extend([
                "ENERGY DECOMPOSITION",
                "--------------------",
                "Component\tEnergy (kcal/mol)",
                "---------\t----------------"
            ])
            
            for component, energy in energy_decomposition.items():
                text_content.append(f"{component}\t{energy:.4f}")
            
            text_content.append("")
        
        # Footer
        text_content.extend([
            "=============================================================",
            "Generated by PandaDock Analysis Module",
            "============================================================="
        ])
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(text_content))
        
        return output_file