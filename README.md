# Depman

Enhance [Gitman](https://github.com/jacebrowning/gitman) with a short CLI wrapper (`gm`) and update checker. Uses Gitman API + GitPython.

[![PyPI](https://img.shields.io/pypi/v/depman.svg)](https://pypi.org/project/depman/)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](https://github.com/yourusername/depman/actions)

## Project structure

depman/
├── setup.py              # Packaging
├── README.md             # Docs + usage
├── depman.py             # Main CLI + logic
├── depman/commands.py    # Gm subcommand wrappers (for modularity)
├── depman/checker.py     # Update checker logic
├── tests/                # Pytest suite
│   ├── test_depman.py
│   └── test_checker.py
├── .gitignore            # Standard
└── pyproject.toml        # Optional modern packaging

## Installation

1. Clone: `git clone https://github.com/yourusername/depman.git && cd depman`
2. Install: `pip install -e .` (dev) or `pip install depman` (from PyPI)
3. Requires: Gitman (`pip install gitman`), GitPython.

## Usage

In your Gitman project root:

- **Wrap Gitman**: `depman gm install mydep` (shorter than `gitman install`)
- **Check Updates**: `depman check_updates` (top-level) or `--recursive` for nested.

Example output:


Full opts: `depman --help`, `depman gm --help`.



## Development

- Lint: `black . && flake8`
- Test: `pip install pytest pytest-mock`, `pytest`
- Publish: `python setup.py sdist bdist_wheel`, `twine upload dist/*`
- CI: Add GitHub Actions YAML for pytest on push.

## Limitations

- Nested recursion scans for `.gitman.yml` but doesn't fully traverse Gitman groups recursively (extend `get_all_requirements` if needed).
- Update checks assume standard rev types; custom remotes need tweaks.
- No `config.get_dependencies()` in public API—flattened manually.

MIT License. Contributions welcome!


