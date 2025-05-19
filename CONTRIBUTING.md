# Contributing to OpenTokenizer

Thank you for your interest in contributing to OpenTokenizer! Below are the guidelines for contributing to the project.

## Development Workflow
1. **Fork the Repository**: Fork the repository to your GitHub account.
2. **Clone the Fork**: Clone your fork locally.
3. **Create a Branch**: Create a feature or bugfix branch.
4. **Make Changes**: Implement your changes and ensure tests pass.
5. **Submit a Pull Request**: Open a PR against the `main` branch.

## Code Style
- Follow PEP 8 for Python code.
- Use Rustfmt for Rust code.
- Document all public APIs.

## Testing
- Ensure all tests pass before submitting a PR.
- Add tests for new features or bug fixes.

## Reporting Issues
- Use the GitHub issue tracker to report bugs or request features.
- Include steps to reproduce the issue and expected behavior.

## License
By contributing, you agree to license your work under the MIT License.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment Setup](#development-environment-setup)
- [Contribution Workflow](#contribution-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Review Process](#review-process)
- [Release Process](#release-process)

## Code of Conduct

All contributors are expected to adhere to the project's Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork to your local machine
3. Set up your development environment
4. Create a new branch for your work
5. Make your changes
6. Test your changes
7. Submit a pull request

## Development Environment Setup

### Prerequisites

- Python 3.8 or later
- Rust (for performance-critical components)
- Poetry (for Python dependency management)

### Setup Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mltokenizer.git
   cd mltokenizer
   ```

2. Install dependencies with Poetry:
   ```bash
   poetry install
   ```

3. If you're working on Rust components:
   ```bash
   poetry run maturin develop --release
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Contribution Workflow

1. **Choose an issue**: Start by finding an issue you'd like to work on, or create a new one if you've identified a bug or improvement.

2. **Create a branch**: Always create a new branch for your work.
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**: Implement your changes following our coding standards.

4. **Test your changes**: Make sure to write tests for your changes and ensure all tests pass.
   ```bash
   pytest tests/python/unit/test_your_feature.py
   ```

5. **Document your changes**: Update or add documentation as needed.

6. **Commit your changes**: Use clear and descriptive commit messages.
   ```bash
   git commit -m "Add feature X" -m "This implements feature X to solve problem Y."
   ```

7. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Create a pull request**: Submit a PR from your branch to the main repository's `main` branch.

## Coding Standards

We follow PEP 8 and use the following tools to enforce coding standards:

- **Black**: For code formatting
- **isort**: For import sorting
- **mypy**: For type checking
- **ruff**: For linting

You can run these tools with:
```bash
poetry run black src tests
poetry run isort src tests
poetry run mypy src
poetry run ruff src tests
```

Key coding principles:
- Follow the module/component structure of the project
- Write clear, self-documenting code with appropriate comments
- Include type hints for all function parameters and return values
- Keep functions focused on a single responsibility
- Write comprehensive docstrings for all public APIs

## Testing Guidelines

- All new features should include tests
- All bug fixes should include tests that demonstrate the bug is fixed
- Unit tests go in `tests/python/unit/`
- Integration tests go in `tests/python/integration/`
- Performance tests go in `tests/python/performance/`
- Multilingual tests go in `tests/python/multilingual/`
- Rust tests go in `tests/rust/`

Run all tests with:
```bash
python scripts/run_all_tests.py
```

Or specific test categories:
```bash
pytest tests/python/unit
```

## Documentation

We use Sphinx for documentation. All public APIs should have proper docstrings. To build the documentation:

```bash
cd docs
make html
```

Documentation guidelines:
- Use NumPy-style docstrings
- Document all parameters and return values
- Include example usage for complex functions
- Keep documentation up-to-date with code changes

## Submitting Changes

When you're ready to submit your changes:

1. Make sure all tests pass
2. Make sure your code follows our coding standards
3. Update documentation if necessary
4. Push your changes to your fork
5. Create a pull request with a clear description of the changes
6. Reference any relevant issues in the PR description

## Review Process

All pull requests will be reviewed by at least one maintainer. The review process includes:

1. Checking that the code follows our standards
2. Verifying that tests pass
3. Ensuring documentation is updated
4. Testing the functionality

You may be asked to make changes before your PR is accepted. Please respond to review comments and update your PR accordingly.

## Release Process

Releases are managed by the project maintainers. The general process is:

1. Maintainers select PRs to include in a release
2. A release branch is created
3. Final testing is performed
4. Documentation is updated with release notes
5. Version numbers are updated
6. The release is packaged and published to PyPI

## Questions?

If you have any questions about contributing, please open an issue or contact the maintainers directly.

Thank you for contributing to OpenTokenizer! 