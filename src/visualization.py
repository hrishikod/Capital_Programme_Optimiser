import matplotlib

# Use non-interactive backend for environments without display servers
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging


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
