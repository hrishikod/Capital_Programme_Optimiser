import pandas as pd
import numpy as np

def create_dummy_data():
    years = [2026 + i for i in range(10)] # Short horizon for test
    
    # Costs
    costs_data = {
        "Project": ["ProjA", "ProjB"],
        "Cost type": ["P50 - Real", "P50 - Real"],
    }
    for y in years:
        costs_data[y] = [10.0, 20.0] # Simple costs
    
    df_costs = pd.DataFrame(costs_data)
    
    # Benefits
    # Columns: Project, Dimension, t+0, t+1, ...
    ben_data = {
        "Project": ["ProjA", "ProjB", "ProjA", "ProjB"],
        "Dimension": ["Total", "Total", "Economic", "Economic"],
    }
    for t in range(5): # 5 year benefit stream
        ben_data[f"t+{t}"] = [2.0, 4.0, 1.0, 2.0]
        
    df_ben = pd.DataFrame(ben_data)
    
    with pd.ExcelWriter("Cost_benefit_streams.xlsx") as writer:
        df_costs.to_excel(writer, sheet_name="Costs", index=False)
        df_ben.to_excel(writer, sheet_name="Benefits Linear 40yrs", index=False)
    
    print("Created Cost_benefit_streams.xlsx")

if __name__ == "__main__":
    create_dummy_data()
