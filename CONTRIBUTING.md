# Contributing to PandaDock

Thank you for your interest in contributing to PandaDock! This document provides guidelines and instructions for contributing.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [How to Contribute](#how-to-contribute)
4. [Development Setup](#development-setup)
5. [Coding Standards](#coding-standards)
6. [Testing](#testing)
7. [Documentation](#documentation)
8. [Pull Request Process](#pull-request-process)

---

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors. We expect all participants to:

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

### Unacceptable Behavior

- Harassment, discrimination, or trolling
- Personal attacks or insults
- Publishing others' private information
- Other conduct that could be considered inappropriate

---

## Getting Started

### Prerequisites

- Python 3.8+
- Git
- Familiarity with molecular docking concepts
- (Optional) CUDA for GPU development

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork:
```bash
git clone https://github.com/YOUR_USERNAME/PandaDock.git
cd PandaDock
```

3. Add upstream remote:
```bash
git remote add upstream https://github.com/pritampanda15/PandaDock.git
```

---

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

#### 1. Bug Reports
- Use GitHub Issues
- Include detailed description
- Provide steps to reproduce
- Include system information

#### 2. Feature Requests
- Open an issue first to discuss
- Explain use case and benefits
- Consider implementation complexity

#### 3. Code Contributions
- Bug fixes
- New algorithms
- Performance improvements
- Documentation improvements

#### 4. Documentation
- Fix typos or unclear explanations
- Add examples
- Improve API documentation
- Write tutorials

#### 5. Testing
- Add test cases
- Improve test coverage
- Report edge cases

---

## Development Setup

### 1. Create Development Environment

```bash
# Create virtual environment
conda create -n pandadock-dev python=3.9
conda activate pandadock-dev

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

### 2. Install Development Dependencies

```bash
pip install pytest pytest-cov black flake8 mypy
```

### 3. Verify Setup

```bash
# Run tests
pytest tests/

# Check code style
black --check pandadock/
flake8 pandadock/
```

---

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line Length**: 100 characters (not 79)
- **Formatting**: Use `black` for automatic formatting
- **Imports**: Use `isort` for organizing imports
- **Type Hints**: Required for public functions

### Code Formatting

```bash
# Format code automatically
black pandadock/

# Sort imports
isort pandadock/

# Check with flake8
flake8 pandadock/
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `MonteCarloDocker`)
- **Functions**: `snake_case` (e.g., `calculate_energy`)
- **Constants**: `UPPER_CASE` (e.g., `MAX_ITERATIONS`)
- **Private methods**: `_leading_underscore` (e.g., `_internal_method`)

### Documentation Style

Use Google-style docstrings:

```python
def dock_ligand(receptor_file: str, ligand_mol: Chem.Mol,
                grid_center: np.ndarray) -> DockingResult:
    """Performs molecular docking of ligand to receptor.

    Args:
        receptor_file: Path to receptor PDB file
        ligand_mol: RDKit molecule object for ligand
        grid_center: 3D coordinates of grid box center

    Returns:
        DockingResult object containing poses and energies

    Raises:
        ValueError: If receptor file not found
        RuntimeError: If docking fails

    Example:
        >>> result = dock_ligand("protein.pdb", ligand_mol, np.array([0,0,0]))
        >>> print(f"Best energy: {result.poses[0].energy}")
    """
    pass
```

---

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_algorithms.py

# Run with coverage
pytest --cov=pandadock tests/

# Run only fast tests (skip slow GPU tests)
pytest -m "not slow" tests/
```

### Writing Tests

Create test files in `tests/` directory:

```python
# tests/test_new_feature.py
import pytest
from pandadock.docking.algorithms import NewAlgorithm

def test_new_algorithm_basic():
    """Test basic functionality of new algorithm."""
    algorithm = NewAlgorithm()
    result = algorithm.run(...)
    assert result is not None
    assert len(result.poses) > 0

def test_new_algorithm_edge_case():
    """Test edge case handling."""
    algorithm = NewAlgorithm()
    with pytest.raises(ValueError):
        algorithm.run(invalid_input)
```

### Test Coverage

- Aim for >80% code coverage
- Test both success and failure cases
- Include edge cases and boundary conditions
- Mock external dependencies when appropriate

---

## Documentation

### Code Documentation

- Document all public classes, functions, and methods
- Use type hints
- Include examples in docstrings
- Explain complex algorithms

### README and Guides

- Update README.md if adding features
- Update ALGORITHMS.md for new algorithms
- Add examples to `examples/` directory
- Update INSTALL.md for new dependencies

### API Documentation

API docs are generated automatically from docstrings:

```bash
cd docs
make html
```

---

## Pull Request Process

### Before Submitting

1. **Update your fork**:
```bash
git fetch upstream
git rebase upstream/main
```

2. **Create feature branch**:
```bash
git checkout -b feature/my-new-feature
```

3. **Make changes**:
- Write clean, well-documented code
- Add tests for new functionality
- Update documentation

4. **Run tests and checks**:
```bash
pytest tests/
black pandadock/
flake8 pandadock/
```

5. **Commit changes**:
```bash
git add .
git commit -m "Add: Description of changes"
```

Use conventional commit messages:
- `Add:` for new features
- `Fix:` for bug fixes
- `Update:` for updates to existing features
- `Docs:` for documentation changes
- `Test:` for test additions/changes
- `Refactor:` for code refactoring

### Submitting Pull Request

1. **Push to your fork**:
```bash
git push origin feature/my-new-feature
```

2. **Open Pull Request** on GitHub:
- Clear title describing the change
- Detailed description of what and why
- Reference any related issues (e.g., "Fixes #123")
- Include test results
- Add screenshots if relevant

3. **Pull Request Template**:
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests added/updated
- [ ] All tests passing
- [ ] Code coverage maintained/improved

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings
```

### Review Process

1. Maintainers will review your PR
2. Address any requested changes
3. Once approved, PR will be merged
4. Your contribution will be acknowledged

### After Merge

1. Update your fork:
```bash
git checkout main
git pull upstream main
git push origin main
```

2. Delete feature branch:
```bash
git branch -d feature/my-new-feature
git push origin --delete feature/my-new-feature
```

---

## Development Guidelines

### Adding New Algorithms

1. Create algorithm file in `pandadock/docking/algorithms/`:
```python
from .base import DockingAlgorithm

class MyNewAlgorithm(DockingAlgorithm):
    def __init__(self):
        super().__init__(name="my_new_algorithm")

    def dock(self, receptor, ligand, **kwargs):
        # Implementation
        pass
```

2. Register in `docking_cli.py`:
```python
engine.register_algorithm(MyNewAlgorithm())
```

3. Add tests in `tests/test_algorithms.py`

4. Document in `ALGORITHMS.md`

### Adding New Scoring Functions

1. Create scoring file in `pandadock/docking/scoring/`:
```python
from .base import ScoringFunction

class MyNewScoring(ScoringFunction):
    def score(self, pose, receptor):
        # Implementation
        return score
```

2. Register in `docking_cli.py`

3. Add tests

4. Document

### Performance Optimization

- Profile code before optimizing
- Document performance improvements
- Include benchmarks
- Consider CPU and GPU implementations

---

## Bug Reports

### Good Bug Report Includes:

1. **Clear title**: Descriptive one-line summary
2. **Environment**:
   - OS and version
   - Python version
   - PandaDock version
   - CUDA version (if GPU-related)
3. **Steps to reproduce**: Minimal example
4. **Expected behavior**: What should happen
5. **Actual behavior**: What actually happens
6. **Error messages**: Full traceback
7. **Additional context**: Any other relevant information

### Bug Report Template:

```markdown
**Environment**
- OS: Ubuntu 20.04
- Python: 3.9.5
- PandaDock: 1.0.0
- CUDA: 11.8 (if applicable)

**Description**
Clear description of the bug

**To Reproduce**
```bash
pandadock dock -r protein.pdb -l ligand.sdf ...
```

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Error Message**
```
Full error traceback
```

**Additional Context**
Any other relevant information
```

---

## Feature Requests

### Good Feature Request Includes:

1. **Clear use case**: Why is this needed?
2. **Proposed solution**: How would it work?
3. **Alternatives considered**: Other approaches?
4. **Examples**: Similar features in other tools?

---

## Community

### Getting Help

- **Documentation**: https://pandadock.readthedocs.io
- **GitHub Discussions**: For questions and discussions
- **GitHub Issues**: For bugs and feature requests
- **Email**: pritampanda@stanford.edu

### Communication Channels

- **GitHub**: Primary platform for development
- **Issues**: Bug reports and feature requests
- **Discussions**: General questions and ideas
- **Pull Requests**: Code contributions

---

## Recognition

### Contributors

All contributors will be:
- Listed in CONTRIBUTORS.md
- Acknowledged in release notes
- Credited in publications (for significant contributions)

### Types of Recognition

- **Code contributors**: Implementation and bug fixes
- **Documentation contributors**: Docs and examples
- **Testing contributors**: Test cases and QA
- **Community contributors**: Support and discussions

---

## License

By contributing to PandaDock, you agree that your contributions will be licensed under the MIT License.

---

## Questions?

If you have questions about contributing:
- Open a GitHub Discussion
- Email: pritampanda@stanford.edu
- Check existing issues and documentation

---

## Thank You!

Your contributions make PandaDock better for everyone. We appreciate your time and effort!

---

**Happy Contributing!**

*The PandaDock Team*
