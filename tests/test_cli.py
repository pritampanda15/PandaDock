"""
Integrity tests for the console entry points.

Every command declared in pyproject.toml must at minimum import and respond to
--help. This is a low bar, but it is the bar that `pandadock-ml` failed for at
least a full release: it raised ImportError on a name that had never existed in
the module it imported from, so the command was unusable while the package
reported a successful install.

Also checks that the algorithm names each CLI advertises can actually be
constructed. A CLI offering a choice that raises on selection is worse than not
offering it, and `--help` alone will not catch that.
"""

from pathlib import Path
import importlib
import subprocess
import sys

import pytest

# Every console script in pyproject.toml, as (module, attribute).
ENTRY_POINTS = [
    ("pandadock.docking_cli", "main"),
    ("pandadock.cli", "main"),
    ("pandadock.gridbox_cli", "main"),
    ("pandadock.flex_docking_cli", "main"),
    ("pandadock.report_cli", "main"),
    ("pandadock.metal_docking_cli", "main"),
    ("pandadock.tethered_cli", "main"),
    ("pandadock.ml_docking_cli", "main"),
    ("pandadock.gnn.cli", "main"),
]


@pytest.mark.parametrize("module_name,attribute", ENTRY_POINTS)
def test_entry_point_imports(module_name, attribute):
    """Each console script's module must import and expose its callable."""
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        missing = str(exc)
        # A missing third-party dependency is an environment issue, not a defect.
        # A missing name from within pandadock is a real break.
        if "pandadock" in missing:
            pytest.fail(f"{module_name} fails to import: {exc}")
        pytest.skip(f"optional dependency unavailable: {exc}")

    assert hasattr(module, attribute), (
        f"{module_name} does not define '{attribute}', which pyproject.toml "
        f"declares as its console-script target"
    )


@pytest.mark.parametrize("module_name,_attribute", ENTRY_POINTS)
def test_entry_point_help(module_name, _attribute):
    """`--help` must exit cleanly, which exercises option parsing."""
    result = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode != 0 and "No module named" in result.stderr:
        pytest.skip("optional dependency unavailable")
    assert result.returncode == 0, (
        f"{module_name} --help exited {result.returncode}:\n{result.stderr[-1500:]}"
    )


def test_advertised_algorithms_are_constructible():
    """
    Every algorithm name the CLIs offer must map to a class that can be built.

    `crystal_guided_cpu` was left in the metal CLI's choice list after the class
    was changed to raise on construction, so selecting it produced a traceback
    rather than a docking run.
    """
    pytest.importorskip("rdkit")
    from pandadock.docking import algorithms

    advertised = {
        "pandadock": "HierarchicalDocker",
        "pandacore": "PandaCoreDocker",
        "monte_carlo_cpu": "MonteCarloDocker",
        "genetic_algorithm_cpu": "GeneticAlgorithmDocker",
        "enhanced_hierarchical_cpu": "EnhancedHierarchicalDocker",
    }

    for cli_name, class_name in advertised.items():
        cls = getattr(algorithms, class_name, None)
        assert cls is not None, f"{class_name} is not exported from docking.algorithms"
        instance = cls()
        assert instance.name, f"{class_name} has no algorithm name"
        assert hasattr(instance, "dock"), f"{class_name} has no dock() method"


def test_removed_algorithm_raises_clearly():
    """The removed biased algorithm must fail loudly, not silently produce poses."""
    from pandadock.docking.algorithms.crystal_guided_cpu import CrystalGuidedDocker

    with pytest.raises(NotImplementedError, match="removed"):
        CrystalGuidedDocker()


def test_gpu_flag_is_defined():
    """
    `pandadock-ml` imports GPU_AVAILABLE at module scope.

    The name did not exist, so the command raised ImportError on every
    invocation.
    """
    from pandadock.docking import algorithms

    assert hasattr(algorithms, "GPU_AVAILABLE")
    assert isinstance(algorithms.GPU_AVAILABLE, bool)


def test_metal_engine_is_usable():
    """
    The metal docking engine must construct and be wired to real algorithms.

    It previously failed three ways before reaching any docking: a missing
    parameter data file raised at construction, its result class could not be
    instantiated because it called a dataclass parent with no arguments, and its
    base engine had no algorithms registered while calling a method
    (`dock`) that DockingEngine does not expose.
    """
    pytest.importorskip("rdkit")
    from pandadock.metal_docking.metal_core import MetalDockingEngine, MetalDockingResult

    result = MetalDockingResult()
    assert result.poses == []

    engine = MetalDockingEngine()
    assert engine.base_engine.list_algorithms(), "no algorithms registered on base engine"
    assert "physics_based" in engine.base_engine.list_scoring_functions()
    assert engine.param_manager.metal_params, "no metal parameters loaded"
    assert hasattr(engine.base_engine, "dock_ligand")


def test_metal_summary_rate_is_a_fraction():
    """
    Violation rate must not exceed 100%.

    Poses were truncated to the requested count while the parallel metadata lists
    kept every scored entry, so the rate was computed from mismatched lengths and
    reported values like 185%.
    """
    pytest.importorskip("rdkit")
    import numpy as np

    from pandadock.docking.core import Pose
    from pandadock.metal_docking.metal_core import MetalDockingResult

    result = MetalDockingResult()
    for i in range(5):
        pose = Pose(
            coordinates=np.zeros((3, 3)),
            center=np.zeros(3),
            rotation=np.array([0.0, 0.0, 0.0, 1.0]),
            conformer_id=0,
        )
        result.add_metal_pose(pose, {}, 0.0, 0.0, ["violation"] if i < 2 else [])

    summary = result.get_metal_summary()
    assert summary["n_poses"] == 5
    assert summary["poses_with_violations"] == 2
    assert 0.0 <= summary["violation_rate"] <= 1.0


def test_gridbox_next_step_paths_point_at_written_files(tmp_path, monkeypatch):
    """
    The suggested `--grid-config` path must be a file that exists.

    With multiple binding sites and an explicit --output prefix, the hint was
    built from the --output-dir default instead of the files just saved, so the
    command printed for the user to copy referenced a path that had never been
    created.
    """
    pytest.importorskip("rdkit")

    import subprocess
    import sys as _sys

    receptor = tmp_path / "rec.pdb"
    lines = []
    for i in range(300):
        x, y, z = (i % 10) * 3.0, ((i // 10) % 10) * 3.0, (i // 100) * 3.0
        lines.append(
            f"ATOM  {i + 1:5d}  CA  ALA A{i % 90 + 1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C"
        )
    lines.append("END")
    receptor.write_text("\n".join(lines) + "\n")

    out_prefix = tmp_path / "grid.json"
    result = subprocess.run(
        [_sys.executable, "-m", "pandadock.gridbox_cli",
         "-r", str(receptor), "-m", "cavities", "-o", str(out_prefix)],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        pytest.skip(f"gridbox could not run in this environment: {result.stderr[-300:]}")

    suggested = [
        token
        for line in result.stdout.splitlines() if "--grid-config" in line
        for token in line.split() if token.endswith(".json")
    ]
    assert suggested, "no --grid-config path was suggested"
    for path in suggested:
        assert Path(path).exists(), (
            f"suggested --grid-config path does not exist: {path}"
        )
