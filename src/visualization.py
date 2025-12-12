import matplotlib

# Use non-interactive backend for environments without display servers
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging
import os
import sys

# Import DataLoader for on-the-fly benefit calculation.
# DataLoader may be unavailable if 'data_loader.py' is missing or if this file is run outside the package context.
try:
    from .data_loader import DataLoader
except ImportError:
    logging.warning("Could not import DataLoader. Benefit recalculation will be unavailable. This may occur if 'data_loader.py' is missing or if running outside the package context.")
    DataLoader = None



def _find_dimension_key(kernels_by_dim: Dict[str, Dict[str, List[float]]], dimension: str) -> Optional[str]:
    """
    Return the matching dimension key from kernels_by_dim (case-insensitive).

    kernels_by_dim is expected to be shaped like:
    {
        "Total": {"ProjectA": [benefits_by_year], ...},
        "Safety": {...}
    }
    Returns None when no dimension matches.
    """
    if dimension in kernels_by_dim:
        return dimension
    dim_lower = dimension.lower()
    lower_map = {key.lower(): key for key in kernels_by_dim.keys()}
    return lower_map.get(dim_lower)


def build_benefit_profile(
    schedule_df: pd.DataFrame,
    kernels_by_dim: Dict[str, Dict[str, List[float]]],
    start_fy: int,
    years: int,
    dimension: str,
) -> pd.DataFrame:
    """
    Build a per-year benefit profile for the selected schedule and target dimension.

    Args:
        schedule_df: DataFrame with columns ["Project", "StartYear", "Duration"].
        kernels_by_dim: mapping of dimension -> {project -> benefit flow}.
        start_fy: first financial year (e.g., 2026).
        years: planning horizon length.
        dimension: dimension name to use when looking up kernels (case-insensitive).

    Returns:
        Single-row DataFrame (index 'Total Benefit') with columns for each year.
    """
    horizon_years = [start_fy + i for i in range(years)]
    profile: List[float] = [0.0] * years

    if schedule_df is None or schedule_df.empty or not kernels_by_dim:
        return pd.DataFrame([profile], columns=horizon_years, index=["Total Benefit"])

    dim_key = _find_dimension_key(kernels_by_dim, dimension)
    dim_kernels = kernels_by_dim.get(dim_key, {}) if dim_key else {}

    for _, row in schedule_df.iterrows():
        project = row.get("Project")
        start_year = row.get("StartYear")
        if pd.isna(start_year):
            logging.warning("Skipping project %s due to missing StartYear", project)
            continue
        try:
            start_idx = int(start_year) - int(start_fy)
        except (TypeError, ValueError):
            logging.warning("Skipping project %s due to invalid StartYear %s", project, start_year)
            continue

        ker = dim_kernels.get(project, [])
        for offset, value in enumerate(ker):
            t = start_idx + offset
            if 0 <= t < years:
                profile[t] += float(value)

    return pd.DataFrame([profile], columns=horizon_years, index=["Total Benefit"])


def plot_program_schedule(schedule_df: pd.DataFrame, output_path: Path) -> None:
    """
    Plot a simple program schedule (Gantt-style) using Project, StartYear, and Duration columns.
    Saves the figure to output_path. Returns early if schedule_df is empty.
    """
    if schedule_df is None or schedule_df.empty:
        return

    schedule_sorted = schedule_df.sort_values(["StartYear", "Project"]).reset_index(drop=True)
    fig_height = max(4.0, 0.4 * len(schedule_sorted) + 2)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    for idx, row in schedule_sorted.iterrows():
        ax.broken_barh(
            [(row["StartYear"], row["Duration"])],
            (idx - 0.4, 0.8),
            facecolors="#4C78A8",
        )

    ax.set_yticks(range(len(schedule_sorted)))
    ax.set_yticklabels(schedule_sorted["Project"])
    ax.set_xlabel("Year")
    ax.set_title("Program Schedule")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_cumulative_spend_and_benefit(
    spend_profile: pd.DataFrame,
    benefit_profile: pd.DataFrame,
    start_fy: int,
    output_path: Path,
    years: Optional[List[int]] = None,
) -> None:
    """
    Plot cumulative spend and benefit over time.

    spend_profile and benefit_profile are expected to be single-row DataFrames with
    year columns (ints). If years is not provided, columns are used; otherwise the
    provided sequence determines ordering.
    """
    if spend_profile is None or spend_profile.empty:
        return

    if years is None:
        try:
            years = [int(c) for c in spend_profile.columns]
        except (TypeError, ValueError):
            years = [start_fy + i for i in range(spend_profile.shape[1])]
    if not years:
        return

    # Aggregated spend profile is expected to be a single-row DataFrame
    spend_series = spend_profile.iloc[0].reindex(years, fill_value=0.0)

    if benefit_profile is not None and not benefit_profile.empty:
        # Benefit profile is also a single-row aggregate
        benefit_series = benefit_profile.iloc[0].reindex(years, fill_value=0.0)
    else:
        benefit_series = pd.Series([0.0] * len(years), index=years)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(years, spend_series.cumsum(), label="Cumulative Spend", color="#4C78A8")
    ax.plot(years, benefit_series.cumsum(), label="Cumulative Benefit", color="#F58518")
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative")
    ax.set_title("Cumulative Spend and Benefit")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_annual_spend_net_funding(cash_flow: pd.DataFrame, output_path: Path) -> None:
    """
    Plot annual spend alongside net balance and funding envelope.

    Expects cash_flow DataFrame to contain 'Year', 'Spend', 'Net', and 'Funding' columns.
    """
    if cash_flow is None or cash_flow.empty:
        return

    years = cash_flow["Year"].tolist()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(years, cash_flow["Spend"], label="Annual Spend", color="#4C78A8", alpha=0.8)
    ax.plot(years, cash_flow["Net"], label="Closing Net Balance", color="#54A24B", linewidth=2)
    ax.plot(years, cash_flow["Funding"], label="Funding Envelope", color="#F58518", linewidth=2, linestyle="--")

    ax.set_xlabel("Year")
    ax.set_ylabel("Amount")
    ax.set_title("Annual Spend, Net Balance, and Funding Envelope")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_visualizations(
    result,
    kernels_by_dim: Dict[str, Dict[str, List[float]]],
    start_fy: int,
    years: int,
    dimension: str,
    output_dir: Path,
) -> None:
    """
    Post-processing helper to save visual outputs for an optimization result.

    Args:
        result: OptimizationResult with schedule, spend_profile, cash_flow.
        kernels_by_dim: benefits mapped by dimension -> project -> flow.
        start_fy: first financial year.
        years: horizon length.
        dimension: target dimension for benefit aggregation.
        output_dir: directory to write PNG and CSV outputs.
    """
    if result is None or result.schedule is None:
        return

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    benefit_profile = build_benefit_profile(
        result.schedule,
        kernels_by_dim,
        start_fy,
        years,
        dimension,
    )
    benefit_profile.to_csv(out_dir / "benefit_profile.csv")
    plot_program_schedule(result.schedule, out_dir / "program_schedule.png")
    plot_cumulative_spend_and_benefit(
        result.spend_profile,
        benefit_profile,
        start_fy,
        out_dir / "cumulative_spend_benefit.png",
    )
    plot_annual_spend_net_funding(result.cash_flow, out_dir / "annual_spend_net_funding.png")



def _recalculate_benefits(
    costs_path: Path,
    benefits_path: Path,
    schedule_df: pd.DataFrame,
    start_fy: int,
    years: int,
    output_dir: Path
) -> Optional[pd.DataFrame]:
    """
    Helper to recalculate benefit profile from raw inputs.
    Returns DataFrame in Millions (matching Spend unit) or None on failure.
    """
    if not DataLoader:
        logging.warning("DataLoader not available, skipping benefit recalc.")
        return None

    try:
        logging.info("Recalculating benefits from raw inputs...")
        loader = DataLoader(str(costs_path), str(benefits_path), start_fy, years)
        
        # Load data
        _, variants, _ = loader.load_costs("P50 - Real")
        benef_df, _ = loader.load_benefits()
        
        # Map kernels
        _, kernels_by_dim = loader.map_benefit_kernels(benef_df, variants)
        
        # Build profile (Total dimension default)
        benefit_profile = build_benefit_profile(
            schedule_df,
            kernels_by_dim,
            start_fy,
            years,
            dimension="Total"
        )
        
        # Save calculated profile
        benefit_profile.to_csv(output_dir / "benefit_profile_calculated.csv")
        
        # Normalize to Millions matches Spend unit
        return benefit_profile / 1_000_000.0
        
    except Exception as e:
        logging.error("Failed to recalculate benefits: %s", e)
        return None


def visualize_from_outputs(
    schedule_csv: Path,
    cash_flow_csv: Path,
    output_dir: Path,
    costs_path: Optional[Path] = None,
    benefits_path: Optional[Path] = None,
) -> None:
    """
    Post-processing entrypoint that reads model output files and produces plots.

    Args:
        schedule_csv: path to schedule.csv
        cash_flow_csv: path to cash_flow.csv
        output_dir: directory for generated PNGs (and optional benefit profile copy)
        costs_path: optional path to raw costs input (needed for benefit recalc)
        benefits_path: optional path to raw benefits input (needed for benefit recalc)
    """
    schedule_path = Path(schedule_csv)
    cash_flow_path = Path(cash_flow_csv)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not schedule_path.exists() or not cash_flow_path.exists():
        logging.warning("Visualization skipped: missing schedule (%s) or cash_flow (%s)", schedule_path, cash_flow_path)
        return

    schedule_df = pd.read_csv(schedule_path)
    cash_flow_df = pd.read_csv(cash_flow_path)

    # Build spend profile from cash flow spend column
    spend_series = cash_flow_df.get("Spend")
    year_series = cash_flow_df.get("Year")
    if spend_series is None or year_series is None or spend_series.empty or year_series.empty:
        logging.warning("Visualization skipped: cash_flow missing Spend/Year columns")
        return

    years = year_series.tolist()
    spend_profile = pd.DataFrame([spend_series.tolist()], columns=years, index=["Total Spend"])

    # Attempt benefit recalculation if inputs provided
    benefit_profile = None
    if costs_path and benefits_path and costs_path.exists() and benefits_path.exists():
        start_fy = int(min(years))
        num_years = len(years)
        benefit_profile = _recalculate_benefits(
            costs_path, 
            benefits_path, 
            schedule_df, 
            start_fy, 
            num_years, 
            out_dir
        )
    
    # Fallback to zeros if calc failed or not attempted
    if benefit_profile is None:
        benefit_profile = pd.DataFrame([[0.0] * len(years)], columns=years, index=["Total Benefit"])

    plot_program_schedule(schedule_df, out_dir / "program_schedule.png")
    plot_cumulative_spend_and_benefit(
        spend_profile,
        benefit_profile,
        start_fy=int(years[0]),
        output_path=out_dir / "cumulative_spend_benefit.png",
        years=years,
    )
    plot_annual_spend_net_funding(cash_flow_df, out_dir / "annual_spend_net_funding.png")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate visualizations from Capital Programme Optimizer outputs.")
    parser.add_argument("--schedule-csv", type=str, default="output/schedule.csv", help="Path to schedule.csv (default: output/schedule.csv)")
    parser.add_argument("--cash-flow-csv", type=str, default="output/cash_flow.csv", help="Path to cash_flow.csv (default: output/cash_flow.csv)")
    parser.add_argument("--costs-path", type=str, default=None, help="Path to raw costs input (optional, for benefit recalc)")
    parser.add_argument("--benefits-path", type=str, default=None, help="Path to raw benefits input (optional, for benefit recalc)")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save plots (default: output)")

    args = parser.parse_args()

    visualize_from_outputs(
        schedule_csv=Path(args.schedule_csv),
        cash_flow_csv=Path(args.cash_flow_csv),
        output_dir=Path(args.output_dir),
        costs_path=Path(args.costs_path) if args.costs_path else None,
        benefits_path=Path(args.benefits_path) if args.benefits_path else None
    )

if __name__ == "__main__":
    main()
