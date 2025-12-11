import matplotlib

# Use non-interactive backend for environments without display servers
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, List


def _find_dimension_key(kernels_by_dim: Dict[str, Dict[str, List[float]]], dimension: str) -> str:
    """
    Return the exact key in kernels_by_dim that matches the requested dimension (case-insensitive).
    """
    if dimension in kernels_by_dim:
        return dimension
    dim_lower = dimension.lower()
    for key in kernels_by_dim.keys():
        if key.lower() == dim_lower:
            return key
    return ""


def build_benefit_profile(
    schedule_df: pd.DataFrame,
    kernels_by_dim: Dict[str, Dict[str, List[float]]],
    start_fy: int,
    years: int,
    dimension: str,
) -> pd.DataFrame:
    """
    Build a per-year benefit profile for the selected schedule and target dimension.
    """
    horizon_years = [start_fy + i for i in range(years)]
    profile = [0.0] * years

    if schedule_df is None or schedule_df.empty or not kernels_by_dim:
        return pd.DataFrame([profile], columns=horizon_years, index=["Total Benefit"])

    dim_key = _find_dimension_key(kernels_by_dim, dimension)
    dim_kernels = kernels_by_dim.get(dim_key, {})

    for _, row in schedule_df.iterrows():
        project = row.get("Project")
        try:
            start_idx = int(row.get("StartYear", start_fy)) - int(start_fy)
        except Exception:
            start_idx = 0

        ker = dim_kernels.get(project, [])
        for offset, value in enumerate(ker):
            t = start_idx + offset
            if 0 <= t < years:
                profile[t] += float(value)

    return pd.DataFrame([profile], columns=horizon_years, index=["Total Benefit"])


def plot_programme_schedule(schedule_df: pd.DataFrame, output_path: Path) -> None:
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
    ax.set_title("Programme Schedule")
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
) -> None:
    if spend_profile is None or spend_profile.empty:
        return

    years = [start_fy + i for i in range(spend_profile.shape[1])]
    spend_series = spend_profile.iloc[0].reindex(years, fill_value=0.0)

    if benefit_profile is not None and not benefit_profile.empty:
        benefit_series = benefit_profile.iloc[0].reindex(years, fill_value=0.0)
    else:
        benefit_series = pd.Series([0.0] * len(years), index=years)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(years, spend_series.cumsum(), label="Cumulative Spend", color="#4C78A8")
    ax.plot(years, benefit_series.cumsum(), label="Cumulative Benefit", color="#F58518")
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative ($M)")
    ax.set_title("Cumulative Spend and Benefit")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_annual_spend_net_funding(cash_flow: pd.DataFrame, output_path: Path) -> None:
    if cash_flow is None or cash_flow.empty:
        return

    years = cash_flow["Year"].tolist()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(years, cash_flow["Spend"], label="Annual Spend", color="#4C78A8", alpha=0.8)
    ax.plot(years, cash_flow["Net"], label="Closing Net Balance", color="#54A24B", linewidth=2)
    ax.plot(years, cash_flow["Funding"], label="Funding Envelope", color="#F58518", linewidth=2, linestyle="--")

    ax.set_xlabel("Year")
    ax.set_ylabel("$M")
    ax.set_title("Annual Spend, Net Balance, and Funding Envelope")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
