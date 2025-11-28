# Capital Programme Optimiser

Refactored toolkit for running the NZTA capital programme optimiser, sizing annual envelopes, and producing the board-ready Excel dashboard.

## Project structure

```
capital_programme_optimiser/
  config/
    settings.yaml            # Core paths and optimisation settings
    envelope_selection.yaml  # Default include flags for envelope helper
  optimisation/
    solver_core.py           # HiGHS optimisation logic (config-driven)
  dashboard/
    builder.py               # Excel dashboard generator
  tools/
    envelope.py              # Envelope sizing utilities
    renamer.py               # Cache renaming helper
  frontend/
    app.py                   # Streamlit user interface
  cli.py                     # Command line entry point
Solver.ipynb etc.           # Legacy notebooks retained for reference
```

## Getting started

1. Create and activate a Python 3.10+ environment.
2. Install dependencies listed in [DEPENDENCIES.md](DEPENDENCIES.md).
3. Validate the configuration in `capital_programme_optimiser/config/settings.yaml` (paths default to the NZTA project directory).

## Command line

Run the optimiser, build dashboards, or rename cache files via the CLI:

```bash
python -m capital_programme_optimiser.cli run-solver --cost-types "P50 - Real,P95 - Real,P95 - Nominal,P50 - Nominal"
python -m capital_programme_optimiser.cli build-dashboard --output-dir ./output
python -m capital_programme_optimiser.cli envelope --baseline-total 115 --baseline-years 50
python -m capital_programme_optimiser.cli rename-cache --prefix Ncor_
```

## Streamlit front end

Launch the interactive UI with dropdowns, run-mode suggestions, and dashboard button:

```bash
streamlit run capital_programme_optimiser/frontend/app.py
```

The app lets you:
- Choose cost types, scenarios, and objective dimensions.
- Toggle comparison vs. standard runs and buffer sweeps.
- Trigger optimisation runs and view logs inline.
- Compute envelope settings.
- Build the Excel front end at the push of a button.

## Envelope sizing helper

Adjust the include flags in `config/envelope_selection.yaml` to match shortlisted projects. Then:

```bash
python -m capital_programme_optimiser.cli envelope
```

or use the Streamlit tab for instant results.

## Legacy notebooks

The original notebooks remain untouched in the repo root for comparison. The refactored package provides a repeatable, configurable interface without changing the underlying optimisation logic.
