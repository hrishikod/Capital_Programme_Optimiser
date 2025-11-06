# Dependency Guide

Install dependencies into a fresh virtual environment (conda or venv recommended).

## Core runtime

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Optimiser engine

`coptpy` provides the COPT bindings used by the optimiser. Ensure the COPT binary and licence are available on your machine and then install the Python wheel:

```bash
pip install coptpy
```

Refer to [Cardinal Operations](https://www.cardinaloperations.com/) for licence provisioning.

## Optional developer tooling

```bash
pip install black isort mypy pytest
```