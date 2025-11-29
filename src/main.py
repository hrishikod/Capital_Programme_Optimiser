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

def main():
    # Configuration
    # Adjust paths as necessary. Assuming running from project root or src.
    # We need to find the data file.
    # The notebook used: ROOT / "Cost_benefit_streams.xlsx"
    # Let's try to find it relative to this script.
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_file = project_root / "Cost_benefit_streams.xlsx"
    
    if not data_file.exists():
        print(f"Error: Data file not found at {data_file}")
        # Try absolute path from notebook if available or ask user?
        # Notebook path: C:\Users\Adrian Desilvestro\Documents\NZTA\Project_Rons_optimisation\Cost_benefit_streams.xlsx
        # That looks like a user specific path.
        # I'll rely on the user having the file in the project root as implied by "ROOT" in notebook usually being CWD or similar.
        # Wait, notebook said: ROOT = Path(r"C:\Users\Adrian Desilvestro\Documents\NZTA\Project_Rons_optimisation")
        # But the user workspace is d:\Projects\Capital_Programme_Optimiser
        # So I should look in the workspace.
        data_file = Path(r"d:\Projects\Capital_Programme_Optimiser\Cost_benefit_streams.xlsx")
        if not data_file.exists():
             print(f"Warning: Hardcoded path {data_file} not found. Checking current dir.")
             data_file = Path("Cost_benefit_streams.xlsx")

    print(f"Using data file: {data_file}")

    start_fy = 2026
    years = 70 # TFIXED from notebook (2095 - 2026 + 1)
    
    loader = DataLoader(str(data_file), start_fy, years)
    
    print("Loading data...")
    try:
        data = loader.load_all(
            cost_type="P50 - Real",
            benefit_sheet="Benefits Linear 40yrs", # From notebook: BENEFIT_SCENARIOS
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
