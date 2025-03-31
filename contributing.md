# Contributing to PandaDock

Thank you for considering contributing to PandaDock! This document outlines the process for contributing to this project.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## How Can I Contribute?

### Reporting Bugs

- Ensure the bug was not already reported by searching through [Issues](https://github.com/your-username/PandaDock/issues)
- If you cannot find an open issue addressing the problem, [open a new one](https://github.com/your-username/PandaDock/issues/new)
- Include a clear title and description, with as much relevant information as possible
- Include code samples, error messages, and steps to reproduce the issue

### Suggesting Enhancements

- Open a new issue with the enhancement tag
- Clearly describe the enhancement and the motivation for it
- Provide any references or examples that might help explain your idea

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run the tests to ensure your changes don't break existing functionality
5. Commit your changes (`git commit -m 'Add some amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Development Guidelines

### Setting Up Development Environment

```bash
# Clone your fork
git clone https://github.com/your-username/PandaDock.git
cd pandadock

# Install in development mode
pip install -e ".[dev]"
```

### Code Style

- Follow PEP 8 guidelines
- Use descriptive variable names
- Write docstrings for all functions, classes, and modules
- Organize imports properly (standard library, third-party, local)

### Testing

- Write tests for new features
- Ensure all tests pass before submitting PR
- Run tests with: `pytest`

### Documentation

- Update documentation for any changed functionality
- Provide examples for new features
- Keep the README.md up to date

## Review Process

- All submissions require review
- Reviewers may suggest changes or improvements
- Once approved, maintainers will merge your PR

Thank you for your contributions!
