import sys
import os
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import DataLoader
from src.optimizer import CapitalProgrammeOptimizer

def calculate_pv_coefficients(
    variants: dict,
    kernels_by_dim: dict,
    allowed_starts: dict,
    start_fy: int,
    years: int,
    discount_rate: float = 0.02,
    dim: str = "Total"
):
    pv_map = {}
    disc_vec = np.array([(1.0 + discount_rate) ** t for t in range(years)])
    
    for v, starts in allowed_starts.items():
        ker = kernels_by_dim.get(dim, {}).get(v, [])
        if not ker:
            continue
            
        for s in starts:
            # Calculate PV if project v starts at s
            # Kernel is aligned with project duration.
            # We need to shift it by s and discount it.
            val = 0.0
            for k, f in enumerate(ker):
                t = s + k
                if 0 <= t < years:
                    val += float(f) / float(disc_vec[t])
            
            if val != 0.0:
                pv_map[(v, s)] = val
    return pv_map

import argparse

def main():
    parser = argparse.ArgumentParser(description="Capital Programme Optimizer")
    parser.add_argument("--generate-only", action="store_true", help="Generate LP file only, do not solve.")
    args = parser.parse_args()

    # Configuration
    # Adjust paths as necessary. Assuming running from project root or src.
    # We need to find the data file.
    # The notebook used: ROOT / "Cost_benefit_streams.xlsx"
    # Let's try to find it relative to this script.
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Use CSVs from input folder
    costs_file = project_root / "input" / "costs.csv"
    benefits_file = project_root / "input" / "benefits.csv"
    
    # Fallback to dummy if not found? Or just error.
    if not costs_file.exists():
        print(f"Error: Costs file not found at {costs_file}")
        # Check for dummy
        costs_file = project_root / "input" / "dummy_costs.csv"
        if costs_file.exists():
            print(f"Using dummy costs: {costs_file}")
            
    if not benefits_file.exists():
        print(f"Error: Benefits file not found at {benefits_file}")
        # Check for dummy
        benefits_file = project_root / "input" / "dummy_benefits.csv"
        if benefits_file.exists():
            print(f"Using dummy benefits: {benefits_file}")

    print(f"Using costs file: {costs_file}")
    print(f"Using benefits file: {benefits_file}")

    start_fy = 2026
    years = 60 # 2026 to 2085 is 60 years
    
    loader = DataLoader(str(costs_file), str(benefits_file), start_fy, years)
    
    print("Loading data...")
    try:
        data = loader.load_all(
            cost_type="P50 - Real",
            benefit_sheet=None, # Not used for CSV
            rules={} # Empty rules for now
        )
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    print(f"Loaded {len(data.variants)} variants.")
    
    # Funding envelope
    # Notebook: SURPLUS_OPTIONS_M: {"s1500": 1500.0}
    funding_level = 1500.0
    funding_target_M = [funding_level] * years
    
    print("Initializing optimizer...")
    optimizer = CapitalProgrammeOptimizer(
        variants=data.variants,
        funding_target_M=funding_target_M,
        start_fy=start_fy,
        years=years,
        max_starts_per_year=100,
        solver_backend="SCIP" # Try SCIP, user needs it installed. Or CBC.
    )
    
    print("Calculating PV coefficients...")
    pv_map = calculate_pv_coefficients(
        data.variants,
        data.kernels_by_dim,
        optimizer.allowed_starts,
        start_fy,
        years
    )
    optimizer.set_pv_coefficients(pv_map)
    
    # Export LP
    lp_dir = project_root / "linear-program-files"
    lp_dir.mkdir(exist_ok=True)
    lp_file = lp_dir / "model.lp"
    print(f"Exporting model to {lp_file}...")
    optimizer.export_model(str(lp_file))
    
    if args.generate_only:
        print("Model generated. Skipping solve step.")
        return

    print("Solving...")
    result = optimizer.solve()
    
    print(f"Status: {result.status}")
    print(f"Objective: {result.objective_value}")
    print(f"Gap: {result.gap:.4%}")
    
    if result.status in ["OPTIMAL", "FEASIBLE"]:
        print("\nSchedule:")
        print(result.schedule.head())
        print(f"\nTotal Spend: {result.spend_profile.iloc[0, :].sum():,.2f}")
        
        # Save results
        out_dir = project_root / "output"
        out_dir.mkdir(exist_ok=True)
        result.schedule.to_csv(out_dir / "schedule.csv", index=False)
        result.cash_flow.to_csv(out_dir / "cash_flow.csv", index=False)
        print(f"\nResults saved to {out_dir}")
    else:
        print("No solution found.")

if __name__ == "__main__":
    main()
