# Contributing to tt-inference-server

Thank you for your interest in contributing to tt-inference-server! This document
provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Review Process](#review-process)
- [Community](#community)

## Code of Conduct

This project adheres to the Contributor Covenant Code of Conduct. By participating,
you are expected to uphold this code. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
before contributing.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/tt-inference-server.git
   cd tt-inference-server
   ```
3. **Set up your development environment** by following the [prerequisites guide](docs/prerequisites.md)
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## How to Contribute

### Reporting Bugs

Before creating a bug report:
- Check the [existing issues](https://github.com/tenstorrent/tt-inference-server/issues) to avoid duplicates
- Collect relevant information (hardware, software versions, logs, etc.)

When filing a bug report, include:
- **Clear title** describing the issue
- **Steps to reproduce** the problem
- **Expected behavior** vs. actual behavior
- **Environment details** (OS, hardware, software versions)
- **Logs and error messages** (use code blocks for formatting)
- **Screenshots or videos** if applicable

### Suggesting Enhancements

Enhancement suggestions are welcome! Please:
- Use a clear and descriptive title
- Provide a detailed description of the proposed enhancement
- Explain why this enhancement would be useful
- Include examples or mockups if applicable

### Contributing Code

We welcome code contributions! You can contribute by:
- Fixing bugs
- Adding new features
- Improving performance
- Enhancing documentation
- Adding tests
- Refactoring code

## Development Workflow

### 1. Create an Issue

Before starting work on a significant change:
- Create or comment on an issue describing what you plan to do
- Wait for feedback from maintainers
- This ensures your work aligns with project goals

### 2. Write Your Code

- Follow the [coding standards](#coding-standards)
- Write clear commit messages
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass locally

### 3. Submit a Pull Request

- Push your changes to your fork
- Create a pull request against the `dev` branch
- Fill out the pull request template completely
- Link any related issues

## Coding Standards

### Python Code

- Follow [PEP 8](https://pep8.org/) style guide
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use type hints where appropriate
- Maximum line length: 100 characters
- Use meaningful variable and function names

### C++ Code

- Follow modern C++20 standards
- Use RAII for resource management
- Prefer `std::unique_ptr` and `std::shared_ptr` over raw pointers
- Use const correctness
- Include SPDX headers in all new files:
  ```cpp
  // SPDX-License-Identifier: Apache-2.0
  // SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
  ```

### Shell Scripts

- Use `#!/bin/bash` shebang
- Use `set -e` to exit on errors for critical scripts
- Quote variables to prevent word splitting
- Use meaningful variable names

### General Guidelines

- Write self-documenting code with clear names
- Add comments for complex logic, not obvious code
- Keep functions focused and single-purpose
- Avoid magic numbers; use named constants
- Handle errors gracefully with informative messages

## Testing

### Running Tests

```bash
# Python tests
pytest tests/

# C++ tests
cd tt-media-server/cpp_server/build
ctest --output-on-failure

# Run specific test
pytest tests/test_specific.py::test_function
```

### Writing Tests

- Write tests for all new functionality
- Ensure tests are deterministic and reproducible
- Use descriptive test names that explain what is being tested
- Follow existing test patterns in the codebase
- Aim for high code coverage, especially for critical paths

### Test Requirements

- All existing tests must pass
- New code should include appropriate tests
- Tests should be fast and focused
- Integration tests for new features
- Unit tests for individual components

## Documentation

### Code Documentation

- **Python:** Use docstrings (Google or NumPy style)
- **C++:** Use doxygen-style comments for public APIs
- Document all public functions, classes, and modules
- Include parameter descriptions and return values
- Add usage examples for complex functions

### Project Documentation

Update relevant documentation when:
- Adding new features
- Changing existing behavior
- Modifying APIs or interfaces
- Adding new dependencies
- Changing configuration options

Documentation locations:
- `README.md` - Project overview and quickstart
- `docs/` - Detailed documentation
- `tt-media-server/cpp_server/README.md` - C++ server specifics
- `tt-media-server/cpp_server/CLAUDE.md` - C++ server developer guide

## Submitting Changes

### Commit Messages

Write clear, descriptive commit messages:

```
component: Brief description (50 chars or less)

More detailed explanation if needed. Wrap at 72 characters.
Explain the problem this commit solves and why this approach
was chosen.

- Bullet points are okay
- Use present tense ("Add feature" not "Added feature")
- Reference issues: "Fixes #123" or "Relates to #456"
```

### Pull Request Guidelines

1. **Title:** Clear and descriptive (e.g., "Add Llama-3.1-8B support to C++ server")
2. **Description:** 
   - Explain what changes were made and why
   - Link related issues
   - Describe how to test the changes
   - Note any breaking changes
3. **Size:** Keep PRs focused and reasonably sized
4. **Tests:** Include tests and ensure all tests pass
5. **Documentation:** Update relevant documentation
6. **Draft PRs:** Use draft PRs for work-in-progress

### Pull Request Checklist

Before submitting, ensure:
- [ ] Code follows project style guidelines
- [ ] All tests pass locally
- [ ] New code includes tests
- [ ] Documentation is updated
- [ ] SPDX headers added to new files
- [ ] Commit messages are clear
- [ ] PR description is complete
- [ ] No merge conflicts with target branch

## Review Process

### Timeline

- Initial review: Within 1 week for most PRs
- Follow-up reviews: Within 3-5 business days
- Large/complex PRs may take longer

### What to Expect

1. **Automated Checks:** CI/CD runs tests automatically
2. **Code Review:** Maintainers review code for:
   - Correctness
   - Code quality
   - Test coverage
   - Documentation
   - Security concerns
3. **Feedback:** Address reviewer comments by:
   - Answering questions
   - Making requested changes
   - Explaining design decisions
4. **Approval:** Once approved and CI passes, your PR will be merged

### After Review

- **Requested changes:** Push additional commits to your branch
- **Merge conflicts:** Rebase or merge `dev` into your branch
- **Approval:** Maintainers will merge your PR
- **Thank you!** Your contribution is appreciated

## Community

### Getting Help

- **GitHub Issues:** For bugs and feature requests
- **GitHub Discussions:** For questions and general discussion
- **Documentation:** Check [docs/README.md](docs/README.md)

### Communication Guidelines

- Be respectful and professional
- Assume good intentions
- Provide constructive feedback
- Stay on topic
- Help others when you can

### Recognition

Contributors are recognized through:
- Commit history
- GitHub contributor graphs
- Special thanks in release notes for significant contributions

## Security

If you discover a security vulnerability:
- **DO NOT** open a public issue
- Follow the instructions in [SECURITY.md](SECURITY.md)
- Report privately through GitHub Security Advisories or email

## License

By contributing to tt-inference-server, you agree that your contributions will
be licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

See [LICENSE_understanding.txt](LICENSE_understanding.txt) for additional context
about the Apache 2.0 license and how it applies to this project.

---

## Questions?

If you have questions about contributing:
- Check existing issues and discussions
- Create a new discussion topic
- Contact the maintainers via issue comments

Thank you for contributing to tt-inference-server!
