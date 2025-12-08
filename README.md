# Capital Programme Optimiser

This project implements an optimization model for the NZTA capital programme, designed to size annual funding envelopes and select projects to maximize benefits under funding constraints.

## Mathematical Formulation

The detailed mathematical formulation of the optimization problem (using the COPT solver) is documented in:
*   [COPT_SOLVER_FORMULATION.md](COPT_SOLVER_FORMULATION.md)

## Project Structure

*   **`src/`**: Contains the main solver implementation in a Jupyter Notebook (`git_copt_solver.ipynb`). This is the core optimization logic.
*   **`legacy/`**: Contains a previous refactored version of the toolkit, including a CLI, Streamlit dashboard, and configuration files. This is retained for reference.

## Dependencies

To run the solver, you need a Python environment with the following dependencies:

*   **Python 3.10+**
*   **COPT Solver**: You must have the Cardinal Optimizer (COPT) installed and a valid license.
    *   Python binding: `pip install coptpy`
*   **Python Packages**:
    *   `numpy`
    *   `pandas`
    *   `openpyxl` (for Excel I/O)
    *   `matplotlib` / `seaborn` (if used for plotting)

See `legacy/requirements.txt` for a comprehensive list of dependencies used in the legacy toolkit, many of which are likely required for the notebook as well.

## Usage

### Running the Solver

You can run the optimization via the command-line interface:

```bash
python src/main.py [options]
```

**Options:**

*   `--funding-level FLOAT`: Annual funding envelope (default: 1500.0).
*   `--dimension STRING`: Dimension to optimize (default: "Total"). Supports full names or tricodes:
    *   `TOT`: Total
    *   `INC`: Inclusive Access
    *   `HSP`: Healthy and safe people
    *   `ECO`: Economic Prosperity
    *   `ENV`: Environmental Sustainability
    *   `RES`: Resilience and Security
*   `--overflow-tiers STRING`: Overflow tiers as `threshold:penalty` pairs (default: "0.12:1000,0.15:4000,0.20:12000").
*   `--generate-only`: Generate the LP file without solving.
*   `--relax`: Generate LP relaxation (continuous variables).

**Example:**

```bash
python src/main.py --funding-level 2000 --dimension "Safety" --overflow-tiers "0.1:500,0.2:2000"
```

Alternatively, you can use the Jupyter Notebook:

1.  Navigate to the `src` directory.
2.  Open `git_copt_solver.ipynb` in Jupyter Lab or Notebook.
3.  Configure the parameters at the top of the notebook.
4.  Run all cells to execute the optimization.

### Legacy Toolkit

To use the legacy CLI or dashboard, refer to the [Legacy README](legacy/README.md).
