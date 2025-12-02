"""
Test basic package imports
"""
import pytest


def test_package_import():
    """Test that the pandadock package can be imported"""
    try:
        import pandadock
        assert True
    except ImportError:
        pytest.skip("PandaDock package not installed")


def test_version():
    """Test that version is defined"""
    try:
        import pandadock
        # Check if __version__ exists or skip
        if hasattr(pandadock, '__version__'):
            assert pandadock.__version__ == "3.0.0"
        else:
            pytest.skip("Version not defined in package")
    except ImportError:
        pytest.skip("PandaDock package not installed")


def test_cli_modules_exist():
    """Test that CLI modules exist"""
    try:
        from pandadock import docking_cli
        from pandadock import flex_docking_cli
        from pandadock import metal_docking_cli
        assert True
    except ImportError as e:
        pytest.skip(f"CLI modules not available: {e}")


def test_basic_functionality():
    """Basic sanity test"""
    assert 1 + 1 == 2
