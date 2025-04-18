"""
Main entry script for PandaDock with GPU/CPU hardware acceleration.
"""

import argparse
import os
import time
import json
from datetime import datetime
from pathlib import Path
import numpy as np
from .protein import Protein
from .ligand import Ligand
from .scoring import CompositeScoringFunction, EnhancedScoringFunction
from .search import RandomSearch, GeneticAlgorithm
from .utils import save_docking_results
from .preparation import prepare_protein, prepare_ligand
from .reporting import DockingReporter
from .validation import validate_against_reference
from .main_integration import (
    add_hardware_options, 
    configure_hardware,
    setup_hardware_acceleration,
    create_optimized_scoring_function,
    create_optimized_search_algorithm,
    get_scoring_type_from_args,
    get_algorithm_type_from_args,
    get_algorithm_kwargs_from_args
)

# Import physics-based algorithms
try:
    from .physics import (MMFFMinimization, GeneralizedBornSolvation, 
                          MonteCarloSampling, PhysicsBasedScoring)
    PHYSICS_AVAILABLE = True
except ImportError:
    PHYSICS_AVAILABLE = False
    print("Warning: Physics-based modules not available. Some features will be disabled.")

def prepare_protein_configs(protein, args):
    """
    Prepare protein configurations for benchmarking.
    
    Parameters:
    -----------
    protein : Protein
        Protein object
    args : argparse.Namespace
        Command-line arguments
    
    Returns:
    --------
    list
        List of protein configurations for benchmarking
    """
    configs = []
    
    # Rigid configuration
    rigid_protein = protein  # No need to copy as we're creating new flex_protein if needed
    configs.append({
        'type': 'rigid',
        'protein': rigid_protein,
        'flexible_residues': []
    })
    
    # Flexible configuration if requested
    if hasattr(args, 'flex_residues') and args.flex_residues or (hasattr(args, 'auto_flex') and args.auto_flex):
        import copy
        flex_protein = copy.deepcopy(protein)
        flex_residues = []
        
        if hasattr(args, 'flex_residues') and args.flex_residues:
            # Use user-specified flexible residues
            flex_residues = args.flex_residues
            print(f"Using user-specified flexible residues: {', '.join(flex_residues)}")
        elif hasattr(args, 'auto_flex') and args.auto_flex:
            # Auto-detect flexible residues based on binding site
            if protein.active_site and 'residues' in protein.active_site:
                # Get residues in the active site
                binding_site_residues = protein.active_site['residues']
                
                # This function needs to be added to the appropriate class
                flex_residues = _detect_flexible_residues(protein, binding_site_residues, 
                                                          max_residues=args.max_flex_residues if hasattr(args, 'max_flex_residues') else 5)
                
                print(f"Auto-detected flexible residues: {', '.join(flex_residues)}")
            else:
                print("Warning: No active site defined. Cannot auto-detect flexible residues.")
        
        if flex_residues:
            flex_protein.define_flexible_residues(flex_residues, 
                                                 max_rotatable_bonds=args.max_flex_bonds if hasattr(args, 'max_flex_bonds') else 3)
            configs.append({
                'type': 'flexible',
                'protein': flex_protein,
                'flexible_residues': flex_residues
            })
        else:
            print("No flexible residues defined. Using only rigid configuration.")
    
    return configs

def _detect_flexible_residues(protein, binding_site_residues, max_residues=5):
    """
    Automatically detect flexible residues in the binding site.
    
    Parameters:
    -----------
    protein : Protein
        Protein object
    binding_site_residues : list
        List of residue IDs in the binding site
    max_residues : int
        Maximum number of flexible residues to detect
    
    Returns:
    --------
    list
        List of detected flexible residue IDs
    """
    # Define residues with flexible sidechains
    flexible_aa_types = [
        'ARG',  # Arginine - very flexible sidechain with multiple rotatable bonds
        'LYS',  # Lysine - long flexible sidechain
        'GLU',  # Glutamic acid - flexible sidechain with carboxyl group
        'GLN',  # Glutamine - flexible sidechain with amide group
        'MET',  # Methionine - flexible sidechain with sulfur
        'PHE',  # Phenylalanine - aromatic sidechain that can rotate
        'TYR',  # Tyrosine - aromatic sidechain with hydroxyl group
        'TRP',  # Tryptophan - large aromatic sidechain
        'LEU',  # Leucine - branched hydrophobic sidechain
        'ILE',  # Isoleucine - branched hydrophobic sidechain
        'ASP',  # Aspartic acid - shorter version of glutamic acid
        'ASN',  # Asparagine - shorter version of glutamine
        'HIS',  # Histidine - aromatic sidechain that can rotate
        'SER',  # Serine - small sidechain with hydroxyl group
        'THR',  # Threonine - small sidechain with hydroxyl group
        'CYS',  # Cysteine - sidechain with thiol group
        'VAL'   # Valine - smaller branched hydrophobic sidechain
    ]
    
    print(f"Searching for flexible residues among {len(binding_site_residues)} binding site residues")
    
    candidate_residues = []
    
    for res_id in binding_site_residues:
        if res_id in protein.residues:
            residue_atoms = protein.residues[res_id]
            
            # Get residue type
            if residue_atoms and 'residue_name' in residue_atoms[0]:
                res_type = residue_atoms[0]['residue_name']
                
                # Check if it's a residue type with flexible sidechain
                if res_type in flexible_aa_types:
                    # Calculate distance from binding site center
                    if protein.active_site and 'center' in protein.active_site:
                        center = protein.active_site['center']
                        
                        # Use CA atom or any atom for distance calculation
                        ca_atom = next((atom for atom in residue_atoms if atom.get('name', '') == 'CA'), None)
                        
                        if ca_atom:
                            distance = np.linalg.norm(ca_atom['coords'] - center)
                            candidate_residues.append((res_id, distance, res_type))
                            print(f"  Found candidate flexible residue: {res_id} ({res_type}) - distance: {distance:.2f}Å")
    
    # Sort by distance to center (closest first)
    candidate_residues.sort(key=lambda x: x[1])
    
    print(f"Selected {min(max_residues, len(candidate_residues))} flexible residues:")
    for i, (res_id, distance, res_type) in enumerate(candidate_residues[:max_residues]):
        print(f"  {i+1}. {res_id} ({res_type}) - distance: {distance:.2f}Å")
    
    # Return up to max_residues
    return [res_id for res_id, _, _ in candidate_residues[:max_residues]]

def write_results_to_txt(results, output_dir, elapsed_time, protein_path, ligand_path, algorithm, iterations):
    """
    Write detailed docking results to a text file.
    
    Parameters:
    -----------
    results : list
        List of (pose, score) tuples
    output_dir : str
        Output directory
    elapsed_time : float
        Total elapsed time in seconds
    protein_path : str
        Path to protein file
    ligand_path : str
        Path to ligand file
    algorithm : str
        Docking algorithm used
    iterations : int
        Number of iterations/generations
    """
    results_path = Path(output_dir) / "docking_results.txt"
    
    # Sort results by score (lowest first)
    sorted_results = sorted(results, key=lambda x: x[1])
    
    with open(results_path, 'w') as f:
        f.write("=====================================================\n")
        f.write("        PandaDock - Python Molecular Docking Results    \n")
        f.write("=====================================================\n\n")
        
        # Write run information
        f.write("RUN INFORMATION\n")
        f.write("--------------\n")
        f.write(f"Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Protein: {protein_path}\n")
        f.write(f"Ligand: {ligand_path}\n")
        f.write(f"Algorithm: {algorithm}\n")
        f.write(f"Iterations/Generations: {iterations}\n")
        f.write(f"Total Runtime: {elapsed_time:.2f} seconds\n\n")
        
        # Write summary of results
        f.write("RESULTS SUMMARY\n")
        f.write("--------------\n")
        f.write(f"Total Poses Generated: {len(results)}\n")
        f.write(f"Best Score: {sorted_results[0][1]:.4f}\n")
        f.write(f"Worst Score: {sorted_results[-1][1]:.4f}\n")
        f.write(f"Average Score: {sum([score for _, score in results])/len(results):.4f}\n\n")
        
        # Write top 10 poses
        f.write("TOP 10 POSES\n")
        f.write("--------------\n")
        f.write("Rank\tScore\tFile\n")
        for i, (pose, score) in enumerate(sorted_results[:10]):
            f.write(f"{i+1}\t{score:.4f}\tpose_{i+1}_score_{score:.1f}.pdb\n")
        
        f.write("\n\nFull results are available in the output directory.\n")
        f.write("=====================================================\n")
    
    print(f"Detailed results written to {results_path}")



from .reporting import DockingReporter

def check_for_updates():
    """Check for newer versions of PandaDock on PyPI and notify user if available."""
    try:
        import os
        import time
        import requests
        import pkg_resources
        from packaging import version

        # Check cache - only verify once per day
        cache_file = os.path.join(os.path.expanduser("~"), ".pandadock_version_check")
        current_time = time.time()
        
        # If cache exists and is less than 24 hours old, skip check
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                last_check = float(f.read().strip())
            if current_time - last_check < 86400:  # 24 hours
                return
            
        # Update cache timestamp
        with open(cache_file, "w") as f:
            f.write(str(current_time))
            
        # Get current installed version
        current_version = pkg_resources.get_distribution("pandadock").version
        
        # Query PyPI for the latest version
        response = requests.get("https://pypi.org/pypi/pandadock/json", timeout=2)
        latest_version = response.json()["info"]["version"]
        
        # Compare versions
        if version.parse(latest_version) > version.parse(current_version):
            print("\n" + "="*70)
            print(f"  New version available: PandaDock {latest_version} (you have {current_version})")
            print(f"  Update with: pip install --upgrade pandadock")
            print("  See release notes at: https://github.com/pritampanda15/PandaDock/releases")
            print("="*70 + "\n")
            
    except Exception:
        # Silently fail if version check doesn't work
        pass

def main():
    # Check for updates at startup
    check_for_updates()

    """Main entry point for the docking program."""
    parser = argparse.ArgumentParser(description='PandaDock: Python Molecular Docking Tool')
    
    # Required arguments
    parser.add_argument('-p', '--protein', required=True, help='Path to protein PDB file')
    parser.add_argument('-l', '--ligand', required=True, help='Path to ligand MOL/SDF file')
    
    # Optional arguments
    parser.add_argument('-o', '--output', default='docking_results', 
                        help='Output directory for docking results')
    parser.add_argument('-a', '--algorithm', choices=['random', 'genetic'], default='genetic',
                        help='Docking algorithm to use (default: genetic)')
    parser.add_argument('-i', '--iterations', type=int, default=1000,
                        help='Number of iterations/generations (default: 1000)')
    parser.add_argument('-s', '--site', nargs=3, type=float, metavar=('X', 'Y', 'Z'),
                        help='Active site center coordinates')
    parser.add_argument('-r', '--radius', type=float, default=10.0,
                        help='Active site radius in Angstroms (default: 10.0)')
    parser.add_argument('--detect-pockets', action='store_true',
                        help='Automatically detect binding pockets')
    
    # Quick mode options
    parser.add_argument('--fast-mode', action='store_true',
                        help='Run with minimal enhancements for quick results')
    parser.add_argument('--enhanced', action='store_true',
                        help='Use enhanced algorithms for more accurate (but slower) results')
    
    # Enhanced docking options
    parser.add_argument('--enhanced-scoring', action='store_true',
                        help='Use enhanced scoring function with electrostatics')
    parser.add_argument('--prepare-molecules', action='store_true',
                        help='Prepare protein and ligand before docking (recommended)')
    parser.add_argument('--reference', 
                        help='Reference ligand structure for validation')
    parser.add_argument('--exact-alignment', action='store_true',
                    help='Align docked pose exactly to reference structure')
    parser.add_argument('--population-size', type=int, default=100,
                        help='Population size for genetic algorithm (default: 100)')
    parser.add_argument('--exhaustiveness', type=int, default=1,
                        help='Number of independent docking runs (default: 1)')
    parser.add_argument('--local-opt', action='store_true',
                        help='Perform local optimization on top poses')
    parser.add_argument('--ph', type=float, default=7.4,
                        help='pH for protein preparation (default: 7.4)')
    parser.add_argument('--no-local-optimization', action='store_true',
                   help='Disable local optimization of poses (keep exact alignment)')
    
    # Physics-based options
    parser.add_argument('--physics-based', action='store_true',
                        help='Use full physics-based scoring (very slow but most accurate)')
    parser.add_argument('--mmff-minimization', action='store_true',
                        help='Use MMFF94 force field minimization (requires RDKit)')
    parser.add_argument('--monte-carlo', action='store_true',
                        help='Use Monte Carlo sampling instead of genetic algorithm')
    parser.add_argument('--mc-steps', type=int, default=1000,
                        help='Number of Monte Carlo steps (default: 1000)')
    parser.add_argument('--temperature', type=float, default=300.0,
                        help='Temperature for Monte Carlo simulation in Kelvin (default: 300K)')

    # Flexible residue options
    parser.add_argument('--flex-residues', nargs='+', 
                        help='Specify flexible residue IDs (e.g., A_42 B_57)')
    parser.add_argument('--max-flex-bonds', type=int, default=3,
                        help='Maximum rotatable bonds per residue (default: 3)')
    
    parser.add_argument('--auto-flex', action='store_true',
                    help='Automatically detect flexible residues in the binding site')
    parser.add_argument('--max-flex-residues', type=int, default=5,
                    help='Maximum number of flexible residues to detect in auto mode (default: 5)')
    
    # Add reporting options
    report_group = parser.add_argument_group('Reporting Options')
    report_group.add_argument('--report-format', choices=['text', 'csv', 'json', 'html', 'all'],
                            default='all', help='Report format (default: all)')
    report_group.add_argument('--report-name', type=str, default=None,
                            help='Custom name for the report files')
    report_group.add_argument('--detailed-energy', action='store_true',
                            help='Include detailed energy component breakdown in reports')
    report_group.add_argument('--skip-plots', action='store_true',
                            help='Skip generating plots for reports')
    
    # Add hardware acceleration options
    add_hardware_options(parser)
    
    args = parser.parse_args()
    
    print(f"============ PandaDock - Python Molecular Docking ============")
    start_time = time.time()
    
    # Configure hardware settings
    hw_config = configure_hardware(args)
    
    # Create temporary directory for prepared files
    temp_dir = Path('temp_pandadock')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Process mode flags - set parameters based on mode
    if args.fast_mode:
        print("\nRunning in fast mode with minimal enhancements")
        args.enhanced_scoring = False
        args.physics_based = False
        args.mmff_minimization = False
        args.monte_carlo = False
        args.local_opt = False
        args.exhaustiveness = 1
        args.prepare_molecules = False
        args.population_size = 50  # Smaller population
    
    if args.enhanced:
        print("\nRunning with enhanced algorithms (slower but more accurate)")
        args.enhanced_scoring = True
        args.local_opt = True
        if args.population_size < 100:
            args.population_size = 100
    
    # Prepare molecules if requested
    if args.prepare_molecules:
        print("\nPreparing molecules for docking...")
        
        # Prepare protein
        prepared_protein = prepare_protein(
            args.protein, 
            output_file=temp_dir / f"prepared_{Path(args.protein).name}",
            ph=args.ph
        )
        
        # Prepare ligand
        prepared_ligand = prepare_ligand(
            args.ligand,
            output_file=temp_dir / f"prepared_{Path(args.ligand).name}",
            n_conformers=5 if args.algorithm == 'genetic' else 1
        )
        
        # Update paths to prepared files
        protein_path = prepared_protein
        ligand_path = prepared_ligand
    else:
        protein_path = args.protein
        ligand_path = args.ligand
    
    # Load protein
    print(f"\nLoading protein from {protein_path}...")
    protein = Protein(protein_path)
    
    # Define active site
    if args.site:
        print(f"Using provided active site center: {args.site}")
        protein.define_active_site(args.site, args.radius)
    elif args.detect_pockets:
        print("Detecting binding pockets...")
        pockets = protein.detect_pockets()
        if pockets:
            print(f"Found {len(pockets)} potential binding pockets")
            print(f"Using largest pocket as active site")
            protein.define_active_site(pockets[0]['center'], pockets[0]['radius'])
        else:
            print("No pockets detected, using whole protein")
    else:
        print("No active site specified, using whole protein")
    
    # Check for flexible residues options
    if hasattr(args, 'auto_flex') and args.auto_flex:
        print("Auto-flex option detected. Will attempt to automatically find flexible residues.")
        
    if args.auto_flex or args.flex_residues:
        print("\nPreparing flexible protein configurations...")
        configs = prepare_protein_configs(protein, args)
        
        if len(configs) > 1:
            print(f"Using flexible protein configuration with {len(configs[1]['protein'].flexible_residues)} flexible residues")
            protein = configs[1]['protein']  # Use the flexible configuration
        else:
            print("No flexible configuration available, using rigid protein")
    
    # Load ligand
    print(f"\nLoading ligand from {ligand_path}...")
    ligand = Ligand(ligand_path)
    
    # Load reference ligand if provided
    reference_ligand = None
    if args.reference:
        print(f"Loading reference ligand from {args.reference}...")
        reference_ligand = Ligand(args.reference)
    
    # Setup hardware acceleration
    hybrid_manager = setup_hardware_acceleration(hw_config)
    
    # Apply MMFF minimization if requested
    if args.mmff_minimization and PHYSICS_AVAILABLE:
        print("\nApplying MMFF94 force field minimization to ligand")
        minimizer = MMFFMinimization()
        ligand = minimizer.minimize_ligand(ligand)
        print("Ligand minimization complete")
    elif args.mmff_minimization and not PHYSICS_AVAILABLE:
        print("\nWarning: MMFF minimization requested but physics module not available. Skipping.")
    
    # Create scoring function based on hardware and requested type
    scoring_type = get_scoring_type_from_args(args)
    
    if scoring_type == 'physics' and PHYSICS_AVAILABLE:
        print("\nUsing physics-based scoring function (MM-GBSA inspired)")
        scoring_function = PhysicsBasedScoring()
    else:
        # Use hardware-optimized scoring function
        scoring_function = create_optimized_scoring_function(hybrid_manager, scoring_type)
        
        if scoring_type == 'enhanced':
            print("\nUsing enhanced scoring function with hardware acceleration")
        else:
            print("\nUsing standard composite scoring function with hardware acceleration")
    
    # Create timestamp for output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.timestamp = timestamp  # Store for validation function
    
    # Setup output directory
    output_dir = f"{args.output}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Track start time for reporting
    start_time = time.time()
    args.start_time = start_time
    # Initialize the docking reporter
    reporter = DockingReporter(output_dir, args, timestamp=timestamp)
    
    # Get algorithm type and parameters
    algorithm_type = get_algorithm_type_from_args(args)
    algorithm_kwargs = get_algorithm_kwargs_from_args(args)
    
    # Create the search algorithm
    search_algorithm = create_optimized_search_algorithm(
        hybrid_manager,
        algorithm_type,
        scoring_function,
        **algorithm_kwargs
    )
    
    # Run the appropriate docking algorithm
    if args.reference and args.exact_alignment:
        print(f"Using exact alignment with reference structure...")
        all_results = search_algorithm.exact_reference_docking(
            protein, 
            ligand, 
            reference_ligand, 
            skip_optimization=args.no_local_optimization
        )
    elif args.reference and not args.exact_alignment:
        print(f"Using reference-guided docking...")
        all_results = search_algorithm.reference_guided_docking(
            protein, 
            ligand, 
            reference_ligand,
            skip_optimization=args.no_local_optimization
        )
    elif args.exhaustiveness > 1:
        print(f"\nRunning {args.exhaustiveness} independent docking runs...")
        all_results = hybrid_manager.run_ensemble_docking(
            protein=protein,
            ligand=ligand,
            n_runs=args.exhaustiveness,
            algorithm_type=algorithm_type,
            **algorithm_kwargs
        )
    elif algorithm_type == 'monte-carlo' and PHYSICS_AVAILABLE:
        print(f"\nUsing Monte Carlo sampling with {args.mc_steps} steps at {args.temperature}K")
        search_algorithm = MonteCarloSampling(
            scoring_function,
            temperature=args.temperature,
            n_steps=args.mc_steps,
            cooling_factor=0.95  # Enable simulated annealing
        )
        all_results = search_algorithm.run_sampling(protein, ligand)
    elif hasattr(args, 'enhanced') and args.enhanced:
        print("Using enhanced rigid docking algorithm...")
        all_results = search_algorithm.improve_rigid_docking(protein, ligand, args)
    else:
        print("\nPerforming standard docking...")
        all_results = search_algorithm.search(protein, ligand)
    
    # Apply local optimization to top poses if requested
    if args.local_opt and not args.no_local_optimization:
        print("\nPerforming local optimization on top poses...")
        optimized_results = []
        
        # Optimize top 10 poses
        for i, (pose, score) in enumerate(sorted(all_results, key=lambda x: x[1])[:10]):
            print(f"Optimizing pose {i+1}...")
            
            if args.mmff_minimization and PHYSICS_AVAILABLE:
                # Use MMFF minimization in protein environment
                print(f"  Using MMFF minimization in protein environment")
                minimizer = MMFFMinimization()
                opt_pose = minimizer.minimize_pose(protein, pose)
                opt_score = scoring_function.score(protein, opt_pose)
                optimized_results.append((opt_pose, opt_score))
            elif hasattr(search_algorithm, '_local_optimization'):
                # Use built-in local optimization
                opt_pose, opt_score = search_algorithm._local_optimization(pose, protein)
                optimized_results.append((opt_pose, opt_score))
            else:
                optimized_results.append((pose, score))
        
        # Combine with original results
        combined_results = optimized_results + [r for r in all_results if r not in all_results[:10]]
        all_results = combined_results
    elif args.no_local_optimization:
        print("Skipping local optimization as requested (--no-local-optimization)")
    
    # Sort all results by score
    all_results.sort(key=lambda x: x[1])

    # Extract energy components for reporting if possible
    try:
        print("Extracting energy components for detailed reporting...")
        energy_breakdown = reporter.extract_energy_components(scoring_function, protein, [pose for pose, _ in all_results[:20]])
        reporter.add_results(all_results, energy_breakdown)
    except Exception as e:
        print(f"Warning: Could not extract energy components: {e}")
        reporter.add_results(all_results)
    
    # Take the top results (avoid duplicates)
    unique_results = []
    seen_scores = set()
    
    for pose, score in all_results:
        # Round score to avoid floating point comparison issues
        rounded_score = round(score, 4)
        if rounded_score not in seen_scores:
            unique_results.append((pose, score))
            seen_scores.add(rounded_score)
            
            # Keep at most 20 poses
            if len(unique_results) >= 20:
                break
    
    # Save results
    print(f"\nDocking completed successfully!")
    print(f"Saving results to {output_dir}...")
    
    # Pass flexible residues to save_docking_results if they exist
    flexible_residues = None
    if hasattr(protein, 'flexible_residues') and protein.flexible_residues:
        flexible_residues = protein.flexible_residues
        print(f"Found {len(flexible_residues)} flexible residues to include in output")
    else:
        print("No flexible residues found on protein object")
    
    save_docking_results(unique_results, output_dir, flexible_residues=flexible_residues)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Write detailed results to text file
    write_results_to_txt(
        results=unique_results,
        output_dir=output_dir,
        elapsed_time=elapsed_time,
        protein_path=args.protein,
        ligand_path=args.ligand,
        algorithm=algorithm_type,
        iterations=args.iterations if algorithm_type != 'monte-carlo' else args.mc_steps
    )
    
    # Validate against reference if provided
    if hasattr(args, 'reference') and args.reference and not args.exact_alignment:
        
        validation_results = validate_against_reference(args, unique_results, output_dir)
        # Add validation results to reporter
        reporter.add_validation_results(validation_results)
    
    # Print summary
    print(f"\nDocking completed in {elapsed_time:.1f} seconds")
    print(f"Best score: {unique_results[0][1]:.2f}")
    print(f"Results saved to: {output_dir}")
    print(f"============================================================")
    
    
    # Generate comprehensive reports
    print("Generating comprehensive docking reports...")
    if hasattr(args, 'report_format') and args.report_format != 'all':
        # Generate only the requested format
        if args.report_format == 'text':
            reporter.generate_detailed_report()
        elif args.report_format == 'csv':
            reporter.generate_csv_report()
        elif args.report_format == 'json':
            reporter.generate_json_report()
        elif args.report_format == 'html':
            reporter.generate_html_report()
    else:
        # Generate all report formats
        reporter.generate_detailed_report()
        reporter.generate_csv_report()
        reporter.generate_json_report()
        
        # Generate HTML report with visualizations if plots are not skipped
        if not (hasattr(args, 'skip_plots') and args.skip_plots):
            html_report = reporter.generate_html_report()
            print(f"Comprehensive HTML report with visualizations saved to: {html_report}")

# Clean up temporary files
    if args.prepare_molecules:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Clean up hardware resources
    hybrid_manager.cleanup()


if __name__ == "__main__":
    main()