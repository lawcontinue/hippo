# Contributing to Hippo

Thanks for your interest in contributing!

---

## Quick Start

1. Fork the repository
2. Create a feature branch: `git checkout -b my-feature`
3. Make your changes
4. Run tests: `pytest tests/ -v`
5. Commit: `git commit -am 'Add my feature'`
6. Push: `git push origin my-feature`
7. Open a Pull Request

---

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/hippo.git
cd hippo

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest --cov=hippo tests/
```

---

## Code Style

We use:
- **ruff** for linting: `ruff check hippo/`
- **black** for formatting: `black hippo/`
- **pytest** for testing

Before committing, run:
```bash
ruff check hippo/
black hippo/
pytest tests/ -v
```

---

## Testing

All contributions must include tests:
- Unit tests for new functions
- Integration tests for API endpoints
- Tests must pass before merging

**Test coverage target:** >80%

---

## Commit Messages

Use clear, descriptive commit messages:
- `feat: add model quantization support`
- `fix: resolve race condition in idle unload`
- `docs: update README with TUI instructions`

Follow conventional commits format: `type: description`

---

## Pull Request Guidelines

- Describe your changes in the PR description
- Reference related issues (fixes #123)
- Ensure all tests pass
- Add documentation for new features
- Keep PRs focused on a single change

---

## Priority Areas

We're looking for help with:
- Model quantization improvements
- Multi-GPU support
- Windows compatibility
- Documentation improvements
- Bug fixes and performance optimizations

---

## Contact

- Open a GitHub issue for bugs
- Start a Discussion for questions
- Check existing issues before creating new ones
