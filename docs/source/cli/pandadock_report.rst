pandadock-report - Docking Analysis and Reporting
==================================================

The ``pandadock-report`` command generates comprehensive analysis reports, visualizations, and summaries from PandaDock docking results. Essential for interpreting large docking studies and communicating results.

Synopsis
--------

.. code-block:: bash

   pandadock-report [OPTIONS]

Description
-----------

Analyzes docking results and generates:

* **Summary statistics** - Success rates, RMSD distributions, runtime
* **Interactive visualizations** - Binding modes, energy plots, interaction maps
* **Comparison reports** - Compare multiple algorithms or targets
* **Export formats** - PDF, HTML, CSV, JSON

Supports analysis of standard docking, flexible docking, metal docking, and virtual screening results.

Required Options
----------------

``-i, --input-dir PATH``
    Input directory containing docking results

    Can be output from ``pandadock``, ``pandadock-flex``, ``pandadock-metal``, etc.

``-o, --output-dir PATH``
    Output directory for report. Default: ``report_output``

Report Type Options
-------------------

``-t, --report-type TYPE``
    Type of report to generate. Default: ``standard``

    Options:

    * ``standard`` - Standard docking analysis report
    * ``comparison`` - Compare multiple docking runs
    * ``screening`` - Virtual screening enrichment analysis
    * ``validation`` - Validation study (requires reference structures)
    * ``interactive`` - Interactive HTML dashboard
    * ``publication`` - Publication-quality figures

``--format FORMAT``
    Output format. Default: ``html``

    Options:

    * ``html`` - Interactive HTML report
    * ``pdf`` - Static PDF report
    * ``pptx`` - PowerPoint presentation
    * ``jupyter`` - Jupyter notebook
    * ``markdown`` - Markdown document

Analysis Options
----------------

``--include-structures / --no-include-structures``
    Include structure visualizations. Default: enabled

``--include-interactions / --no-include-interactions``
    Include interaction analysis. Default: enabled

``--include-energy-decomposition / --no-energy-decomposition``
    Include energy decomposition. Default: enabled (if available)

``--include-statistics / --no-statistics``
    Include statistical analysis. Default: enabled

``--top-n-poses N``
    Analyze top N poses per ligand. Default: 10

``--reference-structures PATH``
    Reference structures for validation (optional)

    Directory containing PDB files for RMSD calculation.

Visualization Options
---------------------

``--viz-style STYLE``
    Visualization style. Default: ``modern``

    Options: ``modern``, ``classic``, ``publication``, ``dark``

``--color-scheme SCHEME``
    Color scheme for plots. Default: ``viridis``

    Options: ``viridis``, ``plasma``, ``cool_warm``, ``greyscale``

``--dpi DPI``
    Figure resolution (DPI). Default: 300

``--figure-format FORMAT``
    Figure file format. Default: ``png``

    Options: ``png``, ``svg``, ``pdf``

``--interactive-plots / --no-interactive-plots``
    Generate interactive plots (plotly). Default: enabled for HTML

Comparison Options
------------------

For ``--report-type comparison``:

``--compare-algorithms``
    Compare different algorithms

``--compare-scoring``
    Compare different scoring functions

``--compare-targets``
    Compare docking to different targets

``--baseline PATH``
    Baseline result for comparison

Screening Options
-----------------

For ``--report-type screening``:

``--active-compounds PATH``
    File with known active compound IDs

``--decoy-compounds PATH``
    File with decoy compound IDs

``--calculate-enrichment / --no-enrichment``
    Calculate enrichment factors. Default: enabled

``--roc-curve / --no-roc-curve``
    Generate ROC curve. Default: enabled

Export Options
--------------

``--export-csv / --no-export-csv``
    Export results to CSV. Default: enabled

``--export-json / --no-export-json``
    Export results to JSON. Default: enabled

``--export-sdf / --no-export-sdf``
    Export top poses to SDF. Default: disabled

``--export-pdb / --no-export-pdb``
    Export complexes to PDB. Default: disabled

Examples
--------

Basic Report Generation
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock-report -i docking_output/ \\
                    -o report/ \\
                    -t standard

Generate standard HTML report with visualizations.

PDF Report
^^^^^^^^^^

.. code-block:: bash

   pandadock-report -i docking_output/ \\
                    -o report_pdf/ \\
                    --format pdf \\
                    --viz-style publication \\
                    --dpi 600

Publication-Quality Report
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock-report -i docking_output/ \\
                    -t publication \\
                    --format pdf \\
                    --viz-style publication \\
                    --color-scheme greyscale \\
                    --dpi 600 \\
                    --figure-format svg \\
                    -o publication_report/

Algorithm Comparison
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock-report -t comparison \\
                    -i results1/ results2/ results3/ \\
                    --compare-algorithms \\
                    --format html \\
                    -o algorithm_comparison/

Where ``results1/``, ``results2/``, ``results3/`` contain docking results from different algorithms.

Virtual Screening Report
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock-report -i screening_results/ \\
                    -t screening \\
                    --active-compounds actives.txt \\
                    --decoy-compounds decoys.txt \\
                    --calculate-enrichment \\
                    --roc-curve \\
                    -o screening_report/

Validation Study Report
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock-report -i validation_results/ \\
                    -t validation \\
                    --reference-structures crystal_structures/ \\
                    --format html \\
                    -o validation_report/

Interactive Dashboard
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock-report -i large_screening/ \\
                    -t interactive \\
                    --interactive-plots \\
                    --format html \\
                    -o interactive_dashboard/

Generates browsable HTML dashboard with interactive plots.

Flexible Docking Report
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock-report -i flex_docking_output/ \\
                    --include-energy-decomposition \\
                    --include-interactions \\
                    --top-n-poses 20 \\
                    -o flex_report/

Export Results to CSV
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock-report -i docking_output/ \\
                    --export-csv \\
                    --export-json \\
                    --export-sdf \\
                    -o exported_results/

Multi-Target Comparison
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock-report -t comparison \\
                    -i target1_results/ target2_results/ target3_results/ \\
                    --compare-targets \\
                    -o multi_target_report/

Output Files
------------

Standard Report
^^^^^^^^^^^^^^^

* ``report.html`` or ``report.pdf`` - Main report
* ``summary_statistics.csv`` - Summary table
* ``all_results.csv`` - Detailed results
* ``figures/`` - Directory with all figures

  * ``binding_affinity_distribution.png``
  * ``rmsd_distribution.png``
  * ``top_poses_overlay.png``
  * ``interaction_heatmap.png``
  * ``energy_decomposition.png``

Comparison Report
^^^^^^^^^^^^^^^^^

* ``comparison_report.html``
* ``algorithm_performance.csv``
* ``head_to_head_comparison.png``
* ``success_rate_comparison.png``
* ``runtime_comparison.png``

Screening Report
^^^^^^^^^^^^^^^^

* ``screening_report.html``
* ``enrichment_factors.csv``
* ``roc_curve.png``
* ``enrichment_plot.png``
* ``active_recovery.png``
* ``top_hits.csv``

Validation Report
^^^^^^^^^^^^^^^^^

* ``validation_report.html``
* ``rmsd_to_crystal.csv``
* ``success_rate_by_target.png``
* ``pose_reproduction_quality.png``

Interactive Dashboard
^^^^^^^^^^^^^^^^^^^^^

* ``index.html`` - Dashboard home page
* ``js/`` - Interactive JavaScript components
* ``data/`` - JSON data for plots
* ``poses/`` - 3D structure viewers

Report Contents
---------------

Standard Report Sections
^^^^^^^^^^^^^^^^^^^^^^^^

1. **Executive Summary**

   * Number of ligands docked
   * Success rate
   * Mean binding affinity
   * Runtime statistics

2. **Binding Affinity Analysis**

   * Distribution of binding scores
   * Top scoring compounds
   * Score histogram

3. **Pose Quality Analysis**

   * RMSD distribution (if reference provided)
   * Pose clustering analysis
   * Conformational diversity

4. **Interaction Analysis**

   * Protein-ligand interaction frequency
   * Hydrogen bonding patterns
   * Hydrophobic contacts
   * Key binding residues

5. **Energy Decomposition**

   * Van der Waals contribution
   * Electrostatic energy
   * Desolvation energy
   * Component breakdown

6. **Top Compounds**

   * Ranked list with structures
   * Binding mode visualizations
   * 2D interaction diagrams

Comparison Report Sections
^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Performance Comparison**

   * Accuracy comparison
   * Speed comparison
   * Success rate comparison

2. **Head-to-Head Analysis**

   * Pose RMSD comparison
   * Binding score correlation
   * Ranking agreement

3. **Consensus Analysis**

   * Compounds ranked highly by all methods
   * Discordant predictions

Screening Report Sections
^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Enrichment Analysis**

   * Enrichment factors at 1%, 5%, 10%
   * ROC AUC
   * Active recovery rate

2. **Hit List**

   * Top scoring compounds
   * Actives in top N
   * Novel scaffolds

3. **Library Statistics**

   * Screening size
   * Chemical diversity
   * Physicochemical properties

Validation Report Sections
^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Pose Reproduction**

   * RMSD to crystal structure
   * Success rate (RMSD < 2Å)
   * Per-target breakdown

2. **Scoring Performance**

   * Correlation with experimental data
   * Ranking accuracy
   * Native pose identification

3. **Recommendations**

   * Best performing algorithm
   * Optimal parameters
   * Confidence assessment

Visualizations
--------------

Automatically Generated Plots
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Distribution Plots:**

* Binding affinity histogram
* RMSD distribution
* Runtime distribution

**Comparison Plots:**

* Algorithm performance radar chart
* Head-to-head scatter plots
* Success rate bar charts

**Interaction Plots:**

* Interaction heatmap (residue × ligand)
* 2D interaction diagrams
* 3D binding mode visualization

**Energy Plots:**

* Energy component breakdown (stacked bar)
* Per-residue contribution (heatmap)
* Energy vs RMSD scatter

**Screening Plots:**

* ROC curve
* Enrichment plot
* Active recovery curve

Best Practices
--------------

Report Organization
^^^^^^^^^^^^^^^^^^^

**For single docking study:**

.. code-block:: bash

   pandadock-report -i results/ \\
                    -t standard \\
                    --include-interactions \\
                    --include-energy-decomposition

**For large screening:**

.. code-block:: bash

   pandadock-report -i screening/ \\
                    -t screening \\
                    --top-n-poses 3 \\
                    --export-csv

**For publication:**

.. code-block:: bash

   pandadock-report -i results/ \\
                    -t publication \\
                    --format pdf \\
                    --dpi 600 \\
                    --viz-style publication \\
                    --figure-format svg

Customization
^^^^^^^^^^^^^

Create custom report template:

.. code-block:: bash

   pandadock-report -i results/ \\
                    --template custom_template.html \\
                    --custom-css style.css \\
                    -o custom_report/

Performance Optimization
^^^^^^^^^^^^^^^^^^^^^^^^

**For large datasets:**

.. code-block:: bash

   # Analyze only top poses
   pandadock-report -i large_screening/ \\
                    --top-n-poses 1 \\
                    --no-interactive-plots \\
                    --format pdf

Troubleshooting
---------------

Missing Visualizations
^^^^^^^^^^^^^^^^^^^^^^

**Problem:** Figures not generated

**Solutions:**

1. Install matplotlib: ``pip install matplotlib``
2. Install plotly for interactive: ``pip install plotly``
3. Check ``--no-include-structures`` not set

Large Report Size
^^^^^^^^^^^^^^^^^

**Problem:** HTML report very large (>100 MB)

**Solutions:**

1. Reduce pose number: ``--top-n-poses 5``
2. Disable interactive plots: ``--no-interactive-plots``
3. Use PDF format: ``--format pdf``
4. Don't embed structures: ``--no-include-structures``

Slow Report Generation
^^^^^^^^^^^^^^^^^^^^^^^

**Problem:** Report generation takes very long

**Solutions:**

1. Reduce analyzed poses: ``--top-n-poses 3``
2. Disable expensive analysis: ``--no-energy-decomposition``
3. Use simpler format: ``--format markdown``

Integration with Other Tools
-----------------------------

Export for PyMOL Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock-report -i results/ \\
                    --export-pdb \\
                    --top-n-poses 5 \\
                    -o pymol_export/

   # Load in PyMOL
   pymol pymol_export/complex*.pdb

Export for Further Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock-report -i results/ \\
                    --export-csv \\
                    --export-json \\
                    -o data_export/

   # Load in pandas
   import pandas as pd
   df = pd.read_csv('data_export/all_results.csv')

Generate Jupyter Notebook
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock-report -i results/ \\
                    --format jupyter \\
                    -o analysis_notebook/

   # Open and run
   jupyter notebook analysis_notebook/analysis.ipynb

Exit Status
-----------

Returns 0 on success, non-zero on error.

See Also
--------

* :doc:`pandadock` - Main docking command
* :doc:`pandadock_flex` - Flexible docking
* :doc:`pandadock_metal` - Metal docking
* :doc:`../algorithms/overview` - Docking algorithms
* :doc:`../scoring/overview` - Scoring functions
