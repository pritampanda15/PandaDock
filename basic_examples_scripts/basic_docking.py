#!/usr/bin/env python3
"""
Basic docking example using PandaDock's programmatic API.
This script demonstrates how to perform a simple docking with genetic algorithm
and visualize the results.
"""

import os
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

from pandadock.protein import Protein
from pandadock.ligand import Ligand
from pandadock.scoring import EnhancedScoringFunction
from pandadock.search import GeneticAlgorithm
from pandadock.utils import save_docking_results


def plot_scores(results, output_dir):
    """Create a simple score plot."""
    scores = [score for _, score in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(scores) + 1), scores, marker='o')
    plt.xlabel('Pose Rank')
    plt.ylabel('Docking Score')
    plt.title('Docking Results - Score Distribution')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save plot
    plt.tight_layout()
    plot_path = Path(output_dir) / "score_plot.png"
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Score plot saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description='Basic PandaDock Example')
    
    parser.add_argument('-p', '--protein', required=True, 
                        help='Path to protein PDB file')
    parser.add_argument('-l', '--ligand', required=True, 
                        help='Path to ligand MOL/SDF file')
    parser.add_argument('-o', '--output', default='example_results',
                        help='Output directory')
    parser.add_argument('-s', '--site', nargs=3, type=float, metavar=('X', 'Y', 'Z'),
                        help='Active site center coordinates')
    parser.add_argument('-r', '--radius', type=float, default=5.0,
                        help='Active site radius in Angstroms')
    parser.add_argument('-g', '--generations', type=int, default=50,
                        help='Number of generations for genetic algorithm')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load protein and ligand
    print(f"Loading protein from {args.protein}...")
    protein = Protein(args.protein)
    
    print(f"Loading ligand from {args.ligand}...")
    ligand = Ligand(args.ligand)
    
    # Define active site if coordinates provided
    if args.site:
        print(f"Using active site at {args.site} with radius {args.radius}Å")
        protein.define_active_site(args.site, args.radius)
    else:
        print("No active site specified, attempting to detect pockets...")
        pockets = protein.detect_pockets()
        if pockets:
            print(f"Found {len(pockets)} potential binding pockets")
            print(f"Using largest pocket as active site")
            protein.define_active_site(pockets[0]['center'], pockets[0]['radius'])
        else:
            print("No pockets detected, using whole protein")
    
    # Set up docking
    print("\nInitializing docking...")
    scoring_function = EnhancedScoringFunction()
    search_algorithm = GeneticAlgorithm(
        scoring_function,
        max_iterations=args.generations,
        population_size=100
    )
    
    # Run docking
    print(f"Starting genetic algorithm docking with {args.generations} generations...")
    results = search_algorithm.search(protein, ligand)
    
    # Save results
    print("\nDocking completed!")
    print(f"Best score: {results[0][1]:.2f}")
    save_docking_results(results, args.output)
    
    # Plot scores
    plot_scores(results, args.output)
    
    # Print summary of top poses
    print("\nTop 5 docking poses:")
    print("-" * 40)
    print("Rank\tScore")
    print("-" * 40)
    for i, (_, score) in enumerate(results[:5]):
        print(f"{i+1}\t{score:.4f}")
    print("\nResults saved to:", args.output)


if __name__ == "__main__":
    main()
