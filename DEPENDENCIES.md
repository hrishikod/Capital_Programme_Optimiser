# Dependency Guide

Install the project dependencies with your preferred package manager. Examples below assume a fresh virtual environment (conda or venv).

## Core runtime

```bash
pip install --upgrade pip
pip install numpy pandas highspy xlsxwriter openpyxl pyyaml
```

## Front end & CLI extras

```bash
pip install streamlit typer rich plotly
```

## Optional developer tooling

```bash
pip install black isort mypy pytest
```

HighsPy binaries are platform specific. On Windows, install via the pre-built wheel:

```bash
pip install highspy --index-url=https://pypi.org/simple/
```

If installation fails, consult the [HiGHS documentation](https://www.highs.dev/) for platform guidance.

