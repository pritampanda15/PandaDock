"""
Test basic package imports
"""
import importlib
import re

import pytest


def test_package_import():
    """Test that the pandadock package can be imported"""
    try:
        import pandadock
        assert True
    except ImportError:
        pytest.skip("PandaDock package not installed")


def test_version():
    """
    The package version must be a well-formed release version.

    Deliberately not pinned to a literal: a hardcoded assertion has to be edited
    on every release, which makes it a chore that gets updated mechanically
    rather than a check that catches anything.
    """
    try:
        import pandadock
    except ImportError:
        pytest.skip("PandaDock package not installed")

    if not hasattr(pandadock, '__version__'):
        pytest.fail("pandadock.__version__ is not defined")

    assert re.fullmatch(
        r"\d+\.\d+\.\d+(?:[.-]?(?:a|b|rc|dev)\d*)?", pandadock.__version__
    ), f"malformed version: {pandadock.__version__!r}"


def test_version_is_single_sourced():
    """
    Sub-packages must not declare their own version.

    pandadock.docking previously exported __version__ = "3.0.0" while the
    distribution shipped 4.0.2, so anything reading the sub-package version got a
    number three major releases out of date.
    """
    import pandadock

    for name in ("pandadock.docking", "pandadock.gnn"):
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        version = getattr(module, "__version__", None)
        if version is None:
            continue
        assert version == pandadock.__version__, (
            f"{name}.__version__ is {version!r} but the package is "
            f"{pandadock.__version__!r}; sub-packages should re-export the "
            f"package version rather than declaring their own"
        )


def test_cli_modules_exist():
    """Test that CLI modules exist"""
    try:
        from pandadock import docking_cli
        from pandadock import flex_docking_cli
        from pandadock import metal_docking_cli
        assert True
    except ImportError as e:
        pytest.skip(f"CLI modules not available: {e}")


def test_gnn_module_exists():
    """Test that GNN module exists"""
    try:
        from pandadock.gnn import GNNScoring
        from pandadock.gnn.models.pandadock_gnn import PandaDockGNN, ModelConfig
        from pandadock.gnn.data.mol2_parser import MOL2Parser
        from pandadock.gnn.data.graph_builder import HeterogeneousGraphBuilder
        assert True
    except ImportError as e:
        pytest.skip(f"GNN module not available (requires torch-geometric): {e}")


def test_basic_functionality():
    """Basic sanity test"""
    assert 1 + 1 == 2


def test_cli_banners_do_not_hardcode_a_version():
    """
    No CLI help banner may carry a literal version string.

    v4.1.0 shipped with `pandadock --help` reporting "Version: 4.0.0": the
    banner in docking_cli.py held a hardcoded literal that the release bump did
    not touch. The existing consistency check compared __version__ across
    sub-packages and so passed, because the stale value was prose in a help
    string rather than a module attribute.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent / "pandadock"
    offenders = []
    pattern = re.compile(r"Version:\s*\d+\.\d+")

    for path in root.rglob("*.py"):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{path.relative_to(root)}:{number}: {line.strip()}")

    assert not offenders, (
        "hardcoded version strings in CLI output:\n  " + "\n  ".join(offenders)
    )
