import os
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
cwd = Path(os.getcwd())
if str(cwd) not in sys.path:
    sys.path.insert(0, str(cwd))

from src.visualization import build_benefit_profile, visualize_from_outputs  # noqa: E402


def test_build_benefit_profile_aligns_benefits_to_start_year():
    schedule_df = pd.DataFrame([{"Project": "ProjA", "StartYear": 2026, "Duration": 2}])
    kernels_by_dim = {"Total": {"ProjA": [0.0, 5.0, 10.0]}}

    profile = build_benefit_profile(
        schedule_df,
        kernels_by_dim,
        start_fy=2026,
        years=5,
        dimension="Total",
    )

    assert list(profile.columns) == [2026, 2027, 2028, 2029, 2030]
    assert profile.iloc[0].tolist() == [0.0, 5.0, 10.0, 0.0, 0.0]


def test_visualize_from_outputs_creates_pngs(tmp_path):
    schedule_csv = tmp_path / "schedule.csv"
    cash_flow_csv = tmp_path / "cash_flow.csv"

    # Write minimal schedule and cash flow
    pd.DataFrame([{"Project": "ProjA", "StartYear": 2026, "Duration": 2}]).to_csv(schedule_csv, index=False)

    pd.DataFrame(
        {
            "Year": [2026, 2027],
            "Funding": [3.0, 3.0],
            "Spend": [1.0, 2.0],
            "Dividend": [0.0, 0.0],
            "Net": [2.0, 3.0],
            "Backlog": [0.0, 0.0],
        }
    ).to_csv(cash_flow_csv, index=False)

    visualize_from_outputs(schedule_csv, cash_flow_csv, tmp_path)

    assert (tmp_path / "program_schedule.png").exists()
    assert (tmp_path / "cumulative_spend_benefit.png").exists()
    assert (tmp_path / "annual_spend_net_funding.png").exists()
