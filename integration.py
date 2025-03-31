# Create a new file called reporting_integration.py

from pathlib import Path
import os
import time
from datetime import datetime

def integrate_reporting(main_py_content):
    """
    Integrate the enhanced reporting module into the main.py file.
    
    Parameters:
    -----------
    main_py_content : str
        Content of the main.py file
    
    Returns:
    --------
    str
        Updated content of the main.py file
    """
    # Import the DockingReporter class
    import_line = "from .reporting import DockingReporter"
    
    # Check if import is already present
    if import_line not in main_py_content:
        # Find the place to add the import (after other imports)
        import_section_end = main_py_content.find("def main():")
        if import_section_end == -1:
            # Fallback: add after the last import
            last_import = main_py_content.rfind("import ")
            last_import_line_end = main_py_content.find("\n", last_import)
            import_section_end = last_import_line_end + 1
        
        # Add the import statement
        main_py_content = (
            main_py_content[:import_section_end] + 
            "\n" + import_line + "\n" + 
            main_py_content[import_section_end:]
        )
    
    # Now add the reporting code to the main function
    
    # Find the place to add the start_time tracking (beginning of main function)
    main_func_start = main_py_content.find("def main():")
    main_body_start = main_py_content.find("    ", main_func_start)
    
    # Add start_time tracking
    start_time_code = """
    # Track start time for reporting
    start_time = time.time()
    args.start_time = start_time
"""
    if "args.start_time = time.time()" not in main_py_content:
        main_py_content = (
            main_py_content[:main_body_start] + 
            start_time_code + 
            main_py_content[main_body_start:]
        )
    
    # Find the place to initialize the reporter (after setting up output directory)
    output_dir_setup = main_py_content.find("os.makedirs(output_dir, exist_ok=True)")
    if output_dir_setup != -1:
        reporter_init_code = """
    # Initialize the docking reporter
    reporter = DockingReporter(output_dir, args, timestamp=timestamp)
"""
        # Add after the output directory creation
        output_dir_line_end = main_py_content.find("\n", output_dir_setup)
        if "DockingReporter" not in main_py_content[output_dir_setup:output_dir_setup+500]:
            main_py_content = (
                main_py_content[:output_dir_line_end+1] + 
                reporter_init_code + 
                main_py_content[output_dir_line_end+1:]
            )
    
    # Find the place to add the energy component extraction
    sort_results = main_py_content.find("all_results.sort(key=lambda x: x[1])")
    if sort_results != -1:
        energy_extract_code = """
    # Extract energy components for reporting if possible
    try:
        print("Extracting energy components for detailed reporting...")
        energy_breakdown = reporter.extract_energy_components(scoring_function, protein, [pose for pose, _ in all_results[:20]])
        reporter.add_results(all_results, energy_breakdown)
    except Exception as e:
        print(f"Warning: Could not extract energy components: {e}")
        reporter.add_results(all_results)
"""
        # Add after sorting the results
        sort_line_end = main_py_content.find("\n", sort_results)
        if "extract_energy_components" not in main_py_content[sort_results:sort_results+500]:
            main_py_content = (
                main_py_content[:sort_line_end+1] + 
                energy_extract_code + 
                main_py_content[sort_line_end+1:]
            )
    
    # Find the place to add validation result processing
    validation_call = main_py_content.find("validate_against_reference(args, unique_results, output_dir)")
    if validation_call != -1:
        validation_code = """
        validation_results = validate_against_reference(args, unique_results, output_dir)
        # Add validation results to reporter
        reporter.add_validation_results(validation_results)
"""
        # Replace the existing validation call with our enhanced version
        validation_line_end = main_py_content.find("\n", validation_call)
        main_py_content = (
            main_py_content[:validation_call] + 
            validation_code + 
            main_py_content[validation_line_end+1:]
        )
    
    # Find the place to generate reports (at the end, before cleanup)
    cleanup_section = main_py_content.find("# Clean up temporary files")
    if cleanup_section != -1:
        reporting_code = """
    # Generate comprehensive reports
    print("Generating comprehensive docking reports...")
    reporter.generate_detailed_report()
    reporter.generate_csv_report()
    reporter.generate_json_report()
    
    # Generate HTML report with visualizations
    html_report = reporter.generate_html_report()
    print(f"Comprehensive HTML report with visualizations saved to: {html_report}")

"""
        if "reporter.generate_detailed_report()" not in main_py_content:
            main_py_content = (
                main_py_content[:cleanup_section] + 
                reporting_code + 
                main_py_content[cleanup_section:]
            )
    
    return main_py_content

def add_reporting_arguments(parser):
    """
    Add reporting-related command line arguments to the argument parser.
    
    Parameters:
    -----------
    parser : argparse.ArgumentParser
        Command line argument parser
    """
    # Create a reporting options group
    report_group = parser.add_argument_group('Reporting Options')
    
    # Add reporting arguments
    report_group.add_argument('--report-format', choices=['text', 'csv', 'json', 'html', 'all'],
                          default='all', help='Report format (default: all)')
    report_group.add_argument('--report-name', type=str, default=None,
                          help='Custom name for the report files')
    report_group.add_argument('--detailed-energy', action='store_true',
                          help='Include detailed energy component breakdown in reports')
    report_group.add_argument('--skip-plots', action='store_true',
                          help='Skip generating plots for reports')

def update_validation_function():
    """
    Return updated code for the validation_against_reference function
    that returns validation results.
    
    Returns:
    --------
    str
        Updated function code
    """
    return """
def validate_against_reference(args, results, output_dir):
    \"\"\"
    Validate docking results against a reference structure if provided.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    results : list
        List of (pose, score) tuples from docking
    output_dir : str
        Output directory
    
    Returns:
    --------
    dict or list
        Validation results
    \"\"\"
    if not hasattr(args, 'reference') or not args.reference:
        return None
    
    print("\\nValidating docking results against reference structure...")
    
    # Load reference ligand
    from .ligand import Ligand
    reference_ligand = Ligand(args.reference)
    
    # Extract poses from results
    poses = [pose for pose, _ in results]
    
    # Calculate RMSDs for all poses
    validation_results = calculate_ensemble_rmsd(poses, reference_ligand)
    
    # Report results
    best_rmsd = validation_results[0]['rmsd']
    best_index = validation_results[0]['pose_index']
    
    print(f"Best RMSD: {best_rmsd:.2f} Å (Pose {best_index + 1})")
    print(f"Docking accuracy: {'Successful' if best_rmsd < 2.0 else 'Unsuccessful'}")
    
    # Write detailed validation report
    validation_file = Path(output_dir) / "validation_report.txt"
    
    with open(validation_file, 'w') as f:
        f.write("==================================================\\n")
        f.write("       PandaDock Validation Against Reference        \\n")
        f.write("==================================================\\n\\n")
        
        f.write(f"Reference Ligand: {args.reference}\\n\\n")
        
        f.write("RMSD Summary:\\n")
        f.write("-------------\\n")
        f.write(f"Best RMSD: {best_rmsd:.4f} Å (Pose {best_index + 1})\\n")
        f.write(f"Docking accuracy: {'Successful' if best_rmsd < 2.0 else 'Unsuccessful'}\\n\\n")
        
        f.write("All Poses:\\n")
        f.write("----------\\n")
        f.write("Rank\\tPose Index\\tRMSD (Å)\\tStatus\\n")
        
        for i, result in enumerate(validation_results):
            status = "Success" if result['rmsd'] < 2.0 else "Failure"
            f.write(f"{i+1}\\t{result['pose_index']+1}\\t{result['rmsd']:.4f}\\t{status}\\n")
    
    print(f"Validation report written to {validation_file}")
    
    return validation_results
"""

def modify_write_results_function():
    """
    Return updated code for the write_results_to_txt function
    that uses the DockingReporter class.
    
    Returns:
    --------
    str
        Updated function code
    """
    return """
def write_results_to_txt(results, output_dir, elapsed_time, protein_path, ligand_path, algorithm, iterations):
    \"\"\"
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
    \"\"\"
    # Create a reporter instance with minimal arguments
    from pathlib import Path
    import argparse
    
    # Create a minimal args namespace for compatibility
    minimal_args = argparse.Namespace()
    minimal_args.protein = protein_path
    minimal_args.ligand = ligand_path
    minimal_args.algorithm = algorithm
    minimal_args.iterations = iterations
    
    # Initialize reporter
    from .reporting import DockingReporter
    reporter = DockingReporter(output_dir, minimal_args)
    
    # Add results
    reporter.add_results(results)
    
    # Generate basic report
    report_path = reporter.generate_basic_report()
    
    print(f"Detailed results written to {report_path}")
    return report_path
"""

def apply_modifications():
    """
    Apply all modifications to integrate the enhanced reporting into PandaDock.
    
    This function:
    1. Updates the main.py file to use the DockingReporter
    2. Updates the validation_against_reference function to return results
    3. Updates the write_results_to_txt function to use the DockingReporter
    
    Returns:
    --------
    tuple
        (success, message)
    """
    try:
        # Get the path to the PandaDock main.py file
        from pathlib import Path
        import sys
        import os
        
        # Try to find the location of the PandaDock package
        pandadock_dir = None
        for path in sys.path:
            potential_path = Path(path) / 'pandadock'
            if potential_path.exists() and potential_path.is_dir():
                pandadock_dir = potential_path
                break
        
        if pandadock_dir is None:
            # If not found in sys.path, check current directory
            current_dir = Path(os.getcwd())
            potential_path = current_dir / 'pandadock'
            if potential_path.exists() and potential_path.is_dir():
                pandadock_dir = potential_path
        
        if pandadock_dir is None:
            return False, "Could not locate the pandadock package directory"
        
        # Path to main.py
        main_py_path = pandadock_dir / 'main.py'
        if not main_py_path.exists():
            return False, f"Could not find main.py at {main_py_path}"
        
        # Create a backup
        import shutil
        backup_path = main_py_path.with_suffix('.py.bak')
        shutil.copy2(main_py_path, backup_path)
        
        # Read the content
        with open(main_py_path, 'r') as f:
            main_py_content = f.read()
        
        # Apply modifications
        updated_content = integrate_reporting(main_py_content)
        
        # Write the updated content
        with open(main_py_path, 'w') as f:
            f.write(updated_content)
        
        # Create the reporting.py file
        reporting_py_path = pandadock_dir / 'reporting.py'
        
        # Get the content from the artifact
        # (This would need to be handled appropriately)
        with open('reporting_module.py', 'r') as f:
            reporting_py_content = f.read()
        
        # Write the reporting module
        with open(reporting_py_path, 'w') as f:
            f.write(reporting_py_content)
        
        return True, f"Successfully integrated enhanced reporting into PandaDock at {pandadock_dir}"
    
    except Exception as e:
        return False, f"Error integrating reporting: {str(e)}"

def install_dependencies():
    """
    Install dependencies required by the enhanced reporting module.
    
    Returns:
    --------
    tuple
        (success, message)
    """
    try:
        import subprocess
        import sys
        
        # Required packages
        packages = ['matplotlib', 'pandas', 'tabulate', 'numpy']
        
        # Check which packages are already installed
        missing_packages = []
        for package in packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        # Install missing packages
        if missing_packages:
            print(f"Installing missing dependencies: {', '.join(missing_packages)}")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            return True, f"Successfully installed dependencies: {', '.join(missing_packages)}"
        else:
            return True, "All dependencies already installed"
    
    except Exception as e:
        return False, f"Error installing dependencies: {str(e)}"

# Example usage
if __name__ == "__main__":
    print("Attempting to integrate enhanced reporting into PandaDock...")
    
    # First, install dependencies
    dep_success, dep_message = install_dependencies()
    print(dep_message)
    
    if dep_success:
        # Then apply modifications
        mod_success, mod_message = apply_modifications()
        print(mod_message)
        
        if mod_success:
            print("Enhanced reporting successfully integrated!")
            print("Use PandaDock as normal, and you'll get detailed reports in the output directory.")
        else:
            print("Integration failed. You can still use the reporting module manually.")
    else:
        print("Failed to install dependencies. Integration aborted.")