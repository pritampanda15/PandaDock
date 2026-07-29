#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PandaDock Publication Report Generator CLI

Creates publication-ready plots and analysis reports from PandaDock results
"""

from .visualization.publication_plots import create_publication_plots_from_directory
from .visualization.simple_publication_plots import create_simple_plots_from_directory
from .visualization.pandamap.pandamap_integration import PandaMapIntegration
import click
import sys
import logging
from pathlib import Path

def show_report_help():
    """Display professional help for report generation"""
    help_text = """
██████╗ ██████╗  █████╗ ██╗   ██╗██████╗ ██████╗ ██████╗
██╔══██╗██╔══██╗██╔══██╗██║   ██║██╔══██╗██╔══██╗██╔══██╗
██████╔╝██║  ██║███████║██║   ██║██║  ██║██║  ██║██████╔╝
██╔══██╗██║  ██║██╔══██║██║   ██║██║  ██║██║  ██║██╔══██╗
██║  ██║██████╔╝██║  ██║╚██████╔╝██████╔╝██████╔╝██║  ██║
╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝

    📊 Publication-Ready Analysis & Visualization Suite
      https://github.com/pritampanda15/PandaDock

Author: Pritam Kumar Panda @ Stanford University
Version: 1.0.0

USAGE:
    pandadock-report COMMAND [OPTIONS]

MAIN COMMANDS:
    plots              Generate publication-ready statistical plots
    pandamap          Create 2D/3D protein-ligand interaction maps
    scoring           Generate comprehensive scoring analysis with formulas
    combined          Generate both plots and PandaMap visualizations

EXAMPLES:
    # Generate publication plots
    pandadock-report plots -i results_directory

    # Create PandaMap interaction maps
    pandadock-report pandamap -i results_directory --top-poses 3

    # Generate scoring analysis with formulas
    pandadock-report scoring -i results_directory --detailed

    # Generate combined analysis
    pandadock-report combined -i results_directory --simple --with-3d

FEATURES:
    ✓ Professional publication-ready plots (PNG/PDF)
    ✓ 2D protein-ligand interaction maps (PandaMap)
    ✓ 3D interactive HTML visualizations
    ✓ Comprehensive scoring analysis with formulas
    ✓ Energy component breakdowns and calculations
    ✓ Binding affinity estimations (Kd, IC50)
    ✓ Simple and advanced plot layouts
    ✓ Discovery Studio-style visualizations

For detailed help on specific commands:
    pandadock-report COMMAND --help

#################################################################
"""
    click.echo(help_text)

@click.group(invoke_without_command=True, add_help_option=False)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--help', '-h', is_flag=True, help='Show this help message')
@click.pass_context
def main(ctx, verbose, help):
    """PandaDock Report Generator - Publication-Ready Analysis & Visualization"""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if help:
        show_report_help()
        return

    # If no command is provided, show help
    if ctx.invoked_subcommand is None:
        show_report_help()

@main.command()
@click.option('--input-dir', '-i', required=True, type=click.Path(exists=True),
              help='Directory containing PandaDock algorithm results')
@click.option('--output-dir', '-o', type=click.Path(),
              help='Output directory for publication plots [default: INPUT_DIR/publication_plots]')
@click.option('--title', '-t', default="PandaDock Molecular Docking Analysis",
              help='Title for the analysis report')
@click.option('--simple', is_flag=True,
              help='Generate simple 4-panel layout instead of comprehensive 8-panel')
def plots(input_dir, output_dir, title, simple):
    """
    Generate publication-ready plots from PandaDock results

    DEFAULT MODE: Creates comprehensive 8-panel visualization including:
    - Algorithm performance heatmap, energy distributions
    - Runtime vs accuracy, pose statistics
    - Ensemble analysis, multi-criteria ranking
    - Interaction profiles, confidence correlations

    SIMPLE MODE (--simple): Creates clean 4-panel layout:
    - Performance matrix, runtime vs energy
    - Energy distributions, summary statistics

    Outputs both PNG and PDF versions suitable for scientific publications.

    Examples:
        pandadock-report plots -i algorithm_comparison_results
        pandadock-report plots -i results --simple --title "My Analysis"
    """

    print("📊 PandaDock Publication Plot Generator")
    print("=" * 50)
    print(f"Input directory: {input_dir}")

    if output_dir:
        print(f"Output directory: {output_dir}")
    else:
        print(f"Output directory: {input_dir}/publication_plots")

    print(f"Report title: {title}")
    print("")

    from pathlib import Path

    from .visualization.report import (
        generate_report,
        load_interaction_analyses,
        load_result_from_dir,
    )

    out = Path(output_dir) if output_dir else Path(input_dir) / "publication_plots"

    try:
        # A standard `pandadock dock` output directory is handled directly. This
        # command previously accepted only an algorithm-comparison layout, so on
        # ordinary docking output it printed "No valid results found" and then a
        # success banner, having written nothing.
        result = load_result_from_dir(Path(input_dir))

        if result is not None:
            images = generate_report(
                result,
                out,
                ligand_mol=None,
                interaction_analyses=load_interaction_analyses(Path(input_dir)),
                run_pandamap=True,
            )
            if not images:
                print("\n❌ No plots could be generated from this run.")
                sys.exit(1)
            for name in sorted(images):
                print(f"   {name}: {Path(images[name]).name}")
            report = out / "report.html"
            if report.exists():
                print(f"   report: {report}")
            print(f"\n✅ Report written to {out}")
            return

        # Otherwise fall back to the algorithm-comparison plotters.
        if simple:
            create_simple_plots_from_directory(input_dir, output_dir)
        else:
            create_publication_plots_from_directory(input_dir, output_dir)

        produced = list(out.glob("*.png")) + list(out.glob("*.pdf")) if out.exists() else []
        if not produced:
            print("\n❌ No results found in this directory.")
            print("   Expected either a `pandadock dock` output directory "
                  "(containing *_summary.json and *_poses.json) or an "
                  "algorithm-comparison directory.")
            sys.exit(1)

        print(f"\n✅ Wrote {len(produced)} files to {out}")

    except SystemExit:
        raise
    except Exception as e:
        print(f"\n❌ Error generating plots: {str(e)}")
        sys.exit(1)

@main.command()
@click.option('--input-dir', '-i', required=True, type=click.Path(exists=True),
              help='Directory containing PandaDock results with complex PDB files')
@click.option('--output-dir', '-o', type=click.Path(),
              help='Output directory for PandaMap visualizations [default: INPUT_DIR/pandamap_output]')
@click.option('--top-poses', '-n', default=3, type=int,
              help='Number of top poses to analyze [default: 3]')
@click.option('--generate-3d', is_flag=True, default=True,
              help='Generate 3D interactive HTML visualizations [default: True]')
@click.option('--skip-3d', is_flag=True,
              help='Skip 3D visualization generation (faster)')
@click.option('--verbose', '-v', is_flag=True,
              help='Enable verbose logging')
def pandamap(input_dir, output_dir, top_poses, generate_3d, skip_3d, verbose):
    """
    Generate 2D/3D protein-ligand interaction maps using PandaMap

    Creates professional interaction visualizations for docked poses:
    - 2D interaction maps (Discovery Studio-style)
    - 3D interactive HTML visualizations
    - Detailed interaction reports

    Requires complex PDB files from docking results (complex1.pdb, complex2.pdb, etc.)

    Examples:
        # Basic PandaMap analysis
        pandadock-report pandamap -i docking_results --top-poses 3

        # Skip 3D generation for faster processing
        pandadock-report pandamap -i results --skip-3d --top-poses 5

        # Custom output directory
        pandadock-report pandamap -i results -o custom_pandamap_output
    """

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Override generate_3d if skip_3d is specified
    if skip_3d:
        generate_3d = False

    print("🔬 PandaMap Protein-Ligand Interaction Analyzer")
    print("=" * 55)
    print(f"Input directory: {input_dir}")

    # Set output directory
    if not output_dir:
        output_dir = Path(input_dir) / "pandamap_output"
    else:
        output_dir = Path(output_dir)

    print(f"Output directory: {output_dir}")
    print(f"Analyzing top {top_poses} poses")
    print(f"3D visualizations: {'✅ Enabled' if generate_3d else '❌ Disabled'}")
    print("")

    try:
        # Initialize PandaMap integration
        pandamap_integration = PandaMapIntegration(str(output_dir))

        # Check PandaMap availability
        if not pandamap_integration.pandamap_available:
            print("⚠️  PandaMap not detected - generating fallback visualizations")
            print("   For full functionality, install PandaMap or ensure it's available")
            print("")

        # Check for complex PDB files first
        input_path = Path(input_dir)
        complex_files = list(input_path.glob("complex*.pdb"))
        if complex_files:
            print(f"🔍 Found {len(complex_files)} complex PDB files:")
            for i, f in enumerate(complex_files[:top_poses], 1):
                print(f"   {i}. {f.name}")
            print("")
        else:
            print("⚠️  No complex PDB files (complex*.pdb) found in input directory")
            print(f"   Searched in: {input_dir}")
            print("   PandaMap requires protein-ligand complex PDB files for analysis")
            print("")

        # Analyze complex PDB files in the results directory
        generated_files = pandamap_integration.analyze_complex_pdbs(
            results_dir=input_dir,
            top_n=top_poses,
            generate_3d=generate_3d
        )

        # Report results
        print("📊 PandaMap Analysis Results:")
        print(f"   2D interaction maps: {len(generated_files['2d_maps'])}")
        print(f"   3D visualizations: {len(generated_files['3d_maps'])}")
        print(f"   Interaction reports: {len(generated_files['reports'])}")

        if generated_files['2d_maps']:
            print("\n📁 Generated Files:")
            for i, map_file in enumerate(generated_files['2d_maps'], 1):
                print(f"   {i}. 2D Map: {Path(map_file).name}")

            for i, viz_file in enumerate(generated_files['3d_maps'], 1):
                print(f"   {i}. 3D HTML: {Path(viz_file).name}")

            for i, report_file in enumerate(generated_files['reports'], 1):
                print(f"   {i}. Report: {Path(report_file).name}")

        print(f"\n✅ PandaMap analysis completed!")
        print(f"📂 Results saved to: {output_dir}")

        # Provide usage hints
        if generated_files['3d_maps']:
            print(f"\n💡 View 3D visualizations by opening HTML files in a web browser:")
            for html_file in generated_files['3d_maps']:
                print(f"   file://{Path(html_file).absolute()}")

    except Exception as e:
        print(f"\n❌ PandaMap analysis failed: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

@main.command()
@click.option('--input-dir', '-i', required=True, type=click.Path(exists=True),
              help='Directory containing PandaDock results with scoring data')
@click.option('--output-file', '-o', type=click.Path(),
              help='Output file for scoring report [default: INPUT_DIR/scoring_analysis.txt]')
@click.option('--detailed', is_flag=True,
              help='Include detailed mathematical formulas and derivations')
@click.option('--include-references', is_flag=True, default=True,
              help='Include scientific references and citations [default: True]')
@click.option('--top-poses', '-n', default=10, type=int,
              help='Number of top poses to analyze in detail [default: 10]')
@click.option('--verbose', '-v', is_flag=True,
              help='Enable verbose logging')
def scoring(input_dir, output_file, detailed, include_references, top_poses, verbose):
    """
    Generate comprehensive scoring analysis with mathematical formulas and energy breakdowns

    Creates detailed scoring reports including:
    - Energy component breakdowns (VdW, electrostatic, hydrogen bonds, etc.)
    - Mathematical formulas and calculations
    - Binding affinity estimations (ΔG, Kd, IC50)
    - Algorithm performance analysis
    - Scientific references and methodology

    Provides transparency into how PandaDock calculates binding scores and
    estimates thermodynamic properties for drug discovery applications.

    Examples:
        # Basic scoring analysis
        pandadock-report scoring -i docking_results

        # Detailed analysis with formulas
        pandadock-report scoring -i results --detailed --top-poses 5

        # Custom output file
        pandadock-report scoring -i results -o my_scoring_report.txt
    """

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("📊 PandaDock Comprehensive Scoring Analysis")
    print("=" * 50)
    print(f"Input directory: {input_dir}")

    # Set output file
    if not output_file:
        output_file = Path(input_dir) / "scoring_analysis.txt"
    else:
        output_file = Path(output_file)

    print(f"Output file: {output_file}")
    print(f"Top poses analyzed: {top_poses}")
    print(f"Detailed formulas: {'✅ Enabled' if detailed else '❌ Disabled'}")
    print(f"References included: {'✅ Enabled' if include_references else '❌ Disabled'}")
    print("")

    try:
        from .analysis.scoring_analysis import generate_comprehensive_scoring_report

        # Generate the comprehensive scoring report
        success = generate_comprehensive_scoring_report(
            input_dir=input_dir,
            output_file=output_file,
            top_poses=top_poses,
            include_detailed_formulas=detailed,
            include_references=include_references
        )

        if success:
            print("✅ Comprehensive scoring analysis completed!")
            print(f"📄 Report saved to: {output_file}")
            print("")
            print("📚 Report includes:")
            print("   • Energy component breakdowns")
            print("   • Mathematical formulas and calculations")
            print("   • Binding affinity estimations")
            print("   • Algorithm performance analysis")
            if include_references:
                print("   • Scientific references and methodology")
            if detailed:
                print("   • Detailed mathematical derivations")
            print("")
            print("🔬 Perfect for understanding PandaDock's scoring methodology!")
        else:
            print("❌ Failed to generate scoring analysis")
            sys.exit(1)

    except ImportError:
        print("❌ Scoring analysis module not found")
        print("   This feature requires the scoring analysis module to be available")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Scoring analysis failed: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

@main.command()
@click.option('--input-dir', '-i', required=True, type=click.Path(exists=True),
              help='Directory containing PandaDock results')
@click.option('--output-dir', '-o', type=click.Path(),
              help='Output directory [default: INPUT_DIR/combined_analysis]')
@click.option('--title', '-t', default="PandaDock Complete Analysis",
              help='Title for the analysis report')
@click.option('--simple', is_flag=True,
              help='Use simple plot layout instead of comprehensive')
@click.option('--top-poses', '-n', default=3, type=int,
              help='Number of top poses for PandaMap analysis [default: 3]')
@click.option('--with-3d', is_flag=True, default=True,
              help='Include 3D visualizations [default: True]')
@click.option('--skip-3d', is_flag=True,
              help='Skip 3D visualization generation')
@click.option('--verbose', '-v', is_flag=True,
              help='Enable verbose logging')
def combined(input_dir, output_dir, title, simple, top_poses, with_3d, skip_3d, verbose):
    """
    Generate complete analysis with both publication plots and PandaMap visualizations

    Creates comprehensive analysis including:
    - Publication-ready statistical plots (PNG/PDF)
    - 2D protein-ligand interaction maps
    - 3D interactive HTML visualizations
    - Detailed interaction reports

    Examples:
        # Complete analysis with defaults
        pandadock-report combined -i results

        # Simple plots with PandaMap
        pandadock-report combined -i results --simple --top-poses 5

        # Skip 3D for faster processing
        pandadock-report combined -i results --skip-3d
    """

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Override with_3d if skip_3d is specified
    if skip_3d:
        with_3d = False

    # Set output directory
    if not output_dir:
        output_dir = Path(input_dir) / "combined_analysis"
    else:
        output_dir = Path(output_dir)

    print("🚀 PandaDock Complete Analysis Generator")
    print("=" * 45)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Analysis title: {title}")
    print(f"Plot layout: {'Simple (4-panel)' if simple else 'Comprehensive (8-panel)'}")
    print(f"Top poses for PandaMap: {top_poses}")
    print(f"3D visualizations: {'✅ Enabled' if with_3d else '❌ Disabled'}")
    print("")

    try:
        # Create subdirectories
        plots_dir = output_dir / "publication_plots"
        pandamap_dir = output_dir / "pandamap_analysis"

        # 1. Generate publication plots
        print("📊 Step 1: Generating publication plots...")
        if simple:
            create_simple_plots_from_directory(input_dir, str(plots_dir))
        else:
            create_publication_plots_from_directory(input_dir, str(plots_dir))
        print("✅ Publication plots completed")

        # 2. Generate PandaMap visualizations
        print("\n🔬 Step 2: Generating PandaMap visualizations...")
        pandamap_integration = PandaMapIntegration(str(pandamap_dir))

        if not pandamap_integration.pandamap_available:
            print("⚠️  PandaMap not detected - generating fallback visualizations")

        generated_files = pandamap_integration.analyze_complex_pdbs(
            results_dir=input_dir,
            top_n=top_poses,
            generate_3d=with_3d
        )
        print("✅ PandaMap analysis completed")

        # 3. Generate summary report
        print("\n📝 Step 3: Creating summary report...")
        summary_file = output_dir / "analysis_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"{title}\n")
            f.write("=" * len(title) + "\n\n")
            f.write(f"Analysis completed on: {Path().cwd()}\n")
            f.write(f"Input directory: {input_dir}\n")
            f.write(f"Output directory: {output_dir}\n\n")

            f.write("PUBLICATION PLOTS:\n")
            f.write(f"  Layout: {'Simple (4-panel)' if simple else 'Comprehensive (8-panel)'}\n")
            f.write(f"  Location: {plots_dir}\n\n")

            f.write("PANDAMAP ANALYSIS:\n")
            f.write(f"  2D interaction maps: {len(generated_files['2d_maps'])}\n")
            f.write(f"  3D visualizations: {len(generated_files['3d_maps'])}\n")
            f.write(f"  Interaction reports: {len(generated_files['reports'])}\n")
            f.write(f"  Location: {pandamap_dir}\n\n")

            if generated_files['3d_maps']:
                f.write("3D VISUALIZATIONS:\n")
                for html_file in generated_files['3d_maps']:
                    f.write(f"  {Path(html_file).name}\n")

        print("✅ Summary report created")

        # Final summary
        print(f"\n🎉 Complete analysis finished!")
        print(f"📂 All results saved to: {output_dir}")
        print(f"📄 Summary: {summary_file}")

        if generated_files['3d_maps']:
            print(f"\n💡 View 3D visualizations:")
            for html_file in generated_files['3d_maps']:
                print(f"   file://{Path(html_file).absolute()}")

    except Exception as e:
        print(f"\n❌ Combined analysis failed: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()