import pandas as pd

from src.visualization import build_benefit_profile


def test_build_benefit_profile_aligns_benefits_to_start_year():
    schedule_df = pd.DataFrame(
        [{"Project": "ProjA", "StartYear": 2026, "Duration": 2}]
    )
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
