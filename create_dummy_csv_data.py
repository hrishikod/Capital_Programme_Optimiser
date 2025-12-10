import pandas as pd
import numpy as np
import random
import os

def create_dummy_csv_data():
    # Configuration
    num_projects = 50
    start_year = 2026
    end_year = 2035
    years = list(range(start_year, end_year + 1))
    benefit_years = 40 # t+0 to t+39 (actually t+40 based on inspection, let's do 41)
    
    # Project names
    projects = [f"Project_{i:03d}" for i in range(1, num_projects + 1)]
    
    # 1. Generate Costs
    # Schema: Project, Cost type, Activity Class, Region, GPS Request Tier, Cost, Duration, 2026...2085
    
    costs_rows = []
    for proj in projects:
        # Random duration between 3 and 10 years
        duration = random.randint(3, 10)
        # Random start year
        # Ensure max_start is at least start_year to avoid ValueError
        max_start = max(start_year, end_year - duration)
        start_y = random.randint(start_year, max_start)
        
        total_cost = random.uniform(10, 500) * 1_000_000 # 10M to 500M
        annual_cost = total_cost / duration
        
        row = {
            "Project": proj,
            "Cost type": "P50 - Real",
            "Activity Class": "Local road improvements",
            "Region": "Canterbury", # Dummy
            "GPS Request Tier": "Tier 1",
            "Cost": total_cost,
            "Duration": duration
        }
        
        # Fill years
        for y in years:
            if start_y <= y < start_y + duration:
                row[str(y)] = annual_cost
            else:
                row[str(y)] = 0.0
        
        costs_rows.append(row)
        
    df_costs = pd.DataFrame(costs_rows)
    
    # Ensure all year columns exist and are ordered
    cols = ["Project", "Cost type", "Activity Class", "Region", "GPS Request Tier", "Cost", "Duration"] + [str(y) for y in years]
    df_costs = df_costs[cols]
    
    os.makedirs("input", exist_ok=True)
    df_costs.to_csv("input/dummy_costs.csv", index=False)
    print(f"Created input/dummy_costs.csv with {len(df_costs)} projects.")

    # 2. Generate Benefits
    # Schema: Project, Activity Class, Region, GPS Request Tier, Dimension, t+0...t+40
    
    ben_rows = []
    dimensions = ["Total", "Economic", "Safety", "Access"]
    
    for proj in projects:
        # Assume benefits start accumulating and maybe decay or stay flat
        # Just random streams for now
        
        base_benefit = random.uniform(1, 50) * 1_000_000 # Annual benefit
        
        for dim in dimensions:
            row = {
                "Project": proj,
                "Activity Class": "Local road improvements",
                "Region": "Canterbury",
                "GPS Request Tier": "Tier 1",
                "Dimension": dim
            }
            
            # If Total, sum of others? Or just independent for dummy?
            # Let's make Total bigger than others if we treat them independently, 
            # but usually Total is the sum. The loader might sum them or use Total directly.
            # Loader logic: "Total" is calculated if missing, or used if present.
            # Let's generate component dimensions and then sum for Total to be consistent.
            
            if dim == "Total":
                continue # We will calculate Total later
                
            # Generate stream
            # t+0 to t+40
            stream = []
            for t in range(41):
                # Ramp up for 5 years then flat
                factor = min(1.0, (t + 1) / 5.0)
                val = base_benefit * factor * (0.3 if dim != "Economic" else 0.7) # Split roughly
                row[f"t+{t}"] = val
            
            ben_rows.append(row)

    df_ben_components = pd.DataFrame(ben_rows)
    
    # Calculate Total
    # Group by Project and sum numeric columns
    t_cols = [f"t+{t}" for t in range(41)]
    
    # We need to preserve metadata for the Total rows
    # Just take the first row's metadata for each project
    meta_cols = ["Activity Class", "Region", "GPS Request Tier"]
    
    total_rows = []
    for proj in projects:
        proj_df = df_ben_components[df_ben_components["Project"] == proj]
        if proj_df.empty:
            continue
            
        total_vals = proj_df[t_cols].sum()
        
        row = {
            "Project": proj,
            "Dimension": "Total"
        }
        for c in meta_cols:
            row[c] = proj_df.iloc[0][c]
            
        for t_col in t_cols:
            row[t_col] = total_vals[t_col]
            
        total_rows.append(row)
        
    df_ben_total = pd.DataFrame(total_rows)
    
    df_ben = pd.concat([df_ben_components, df_ben_total], ignore_index=True)
    
    # Reorder columns
    cols_ben = ["Project", "Activity Class", "Region", "GPS Request Tier", "Dimension"] + t_cols
    df_ben = df_ben[cols_ben]
    
    df_ben.to_csv("input/dummy_benefits.csv", index=False)
    print(f"Created input/dummy_benefits.csv with {len(df_ben)} rows.")

if __name__ == "__main__":
    create_dummy_csv_data()
