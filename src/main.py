import sys
import os
import numpy as np
import logging
from datetime import datetime
from pathlib import Path

# Determine paths robustly (handles interactive/Databricks environments)
try:
    SCRIPT_PATH = Path(__file__).resolve()
    SCRIPT_DIR = SCRIPT_PATH.parent
except NameError:
    # Interactive/Notebook fallback
    cwd = Path(os.getcwd()).resolve()
    if (cwd / "src").exists() and (cwd / "src").is_dir():
        SCRIPT_DIR = cwd / "src"
    elif cwd.name == "src":
        SCRIPT_DIR = cwd
    else:
        SCRIPT_DIR = cwd

PROJECT_ROOT = SCRIPT_DIR.parent

# Add project root to path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_loader import DataLoader
from src.cp_sat_optimizer import CapitalProgrammeOptimizer as CpSatOptimizer
from src.optimizer import CapitalProgrammeOptimizer as Optimizer

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



def run_optimization(args):
    """
    Main optimization logic, separated for easier calling from notebooks/MLflow.
    Returns the result object (or None if failed/generate-only).
    """
    
    # Parse overflow tiers
    try:
        tiers_raw = args.overflow_tiers.split(",")
        piecewise_cap_tiers = []
        for t in tiers_raw:
            thresh, pen = t.split(":")
            piecewise_cap_tiers.append((float(thresh), float(pen)))
    except ValueError:
        logging.error("Error: Invalid format for --overflow-tiers. Expected format: threshold:penalty,threshold:penalty")
        return None

    # Map dimension tricodes
    dim_map = {
        "TOT": "Total",
        "INC": "Inclusive Access",
        "HSP": "Healthy and safe people",
        "ECO": "Economic Prosperity",
        "ENV": "Environmental Sustainability",
        "RES": "Resilience and Security"
    }
    
    target_dimension = args.dimension
    if target_dimension.upper() in dim_map:
        target_dimension = dim_map[target_dimension.upper()]

    # Configuration
    # Adjust paths as necessary. Assuming running from project root or src.
    # We need to find the data file.
    # The notebook used: ROOT / "Cost_benefit_streams.xlsx"
    # Let's try to find it relative to this script.
    
    script_dir = SCRIPT_DIR
    project_root = PROJECT_ROOT
    
    # Use CSVs from input folder
    costs_file = project_root / "input" / "costs.csv"
    benefits_file = project_root / "input" / "benefits.csv"
    
    # Fallback to dummy if not found? Or just error.
    if not costs_file.exists():
        logging.error(f"Error: Costs file not found at {costs_file}")
        # Check for dummy
        costs_file = project_root / "input" / "dummy_costs.csv"
        if costs_file.exists():
            logging.info(f"Using dummy costs: {costs_file}")
            
    if not benefits_file.exists():
        logging.error(f"Error: Benefits file not found at {benefits_file}")
        # Check for dummy
        benefits_file = project_root / "input" / "dummy_benefits.csv"
        if benefits_file.exists():
            logging.info(f"Using dummy benefits: {benefits_file}")

    # Setup Logging
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"run_{timestamp}.log"

    # Reconfigure logging to include file handler
    # Remove existing handlers if any to avoid duplicates in interactive modes
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Suppress noisy py4j logs in Databricks
    logging.getLogger("py4j").setLevel(logging.WARNING)

    logging.info(f"Logging initialized. Writing to {log_file}")
    logging.info(f"Using costs file: {costs_file}")
    logging.info(f"Using benefits file: {benefits_file}")

    start_fy = args.start_year
    years = args.horizon
    
    loader = DataLoader(str(costs_file), str(benefits_file), start_fy, years)
    
    logging.info("Loading data...")
    try:
        data = loader.load_all(
            cost_type="P50 - Real",
            benefit_sheet=None, # Not used for CSV
            rules={} # Empty rules for now
        )
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        return None

    logging.info(f"Loaded {len(data.variants)} variants.")
    
    # Funding envelope
    funding_level = args.funding_level
    funding_target_M = [funding_level] * years
    
    logging.info(f"Configuration:")
    logging.info(f"  Funding Level: {funding_level}")
    logging.info(f"  Dimension: {target_dimension} (Input: {args.dimension})")
    logging.info(f"  Overflow Tiers: {piecewise_cap_tiers}")
    logging.info(f"  Start Year: {start_fy}")
    logging.info(f"  Horizon: {years} years")
    logging.info(f"  Time Limit: {args.time_limit}s")

    logging.info("Initializing optimizer...")
    if args.optimizer == "cp-sat":
        logging.info(f"Using CP-SAT optimizer.")
        optimizer = CpSatOptimizer(
            variants=data.variants,
            funding_target_M=funding_target_M,
            start_fy=start_fy,
            years=years,
            max_starts_per_year=100,
            relax_integrality=args.relax,
            piecewise_cap_tiers=piecewise_cap_tiers,
            time_limit_seconds=args.time_limit
        )
    else:
        logging.info(f"Using Optimizer.")
        optimizer = Optimizer(
            variants=data.variants,
            funding_target_M=funding_target_M,
            start_fy=start_fy,
            years=years,
            max_starts_per_year=100,
            solver_backend="SCIP",
            relax_integrality=args.relax,
            piecewise_cap_tiers=piecewise_cap_tiers,
            time_limit_seconds=args.time_limit
        )
    
    logging.info("Calculating PV coefficients...")
    pv_map = calculate_pv_coefficients(
        data.variants,
        data.kernels_by_dim,
        optimizer.allowed_starts,
        start_fy,
        years,
        dim=target_dimension
    )
    optimizer.set_pv_coefficients(pv_map)
    
    # Export LP
    lp_dir = project_root / "linear-program-files"
    lp_dir.mkdir(exist_ok=True)
    lp_file = lp_dir / "model.lp"
    logging.info(f"Exporting model to {lp_file}...")
    optimizer.export_model(str(lp_file))
    
    if args.generate_only:
        logging.info("Model generated. Skipping solve step.")
        return None

    logging.info("Solving...")
    result = optimizer.solve()
    
    logging.info(f"Status: {result.status}")
    logging.info(f"Objective: {result.objective_value}")
    logging.info(f"Gap: {result.gap:.4%}")
    
    if result.status in ["OPTIMAL", "FEASIBLE"]:
        logging.info("\nSchedule (Top 20):")
        # For dataframe printing, we might want to keep print or log as string
        logging.info("\n" + result.schedule.head(20).to_string())
        logging.info(f"\nTotal Spend: {result.spend_profile.iloc[0, :].sum():,.2f}")
        
        # Save results
        out_dir = project_root / "output"
        out_dir.mkdir(exist_ok=True)
        result.schedule.to_csv(out_dir / "schedule.csv", index=False)
        result.cash_flow.to_csv(out_dir / "cash_flow.csv", index=False)
        logging.info(f"\nResults saved to {out_dir}")
        return result
    else:
        logging.info("No solution found.")
        return result

def main():
    parser = argparse.ArgumentParser(description="Capital Programme Optimizer")
    parser.add_argument("--generate-only", action="store_true", help="Generate LP file only, do not solve.")
    parser.add_argument("--relax", action="store_true", help="Generate LP relaxation (continuous variables).")
    parser.add_argument("--funding-level", type=float, default=1500.0, help="Annual funding envelope (default: 1500.0)")
    parser.add_argument("--dimension", type=str, default="Total", help="Dimension to optimize (default: Total)")
    parser.add_argument("--overflow-tiers", type=str, default="0.12:1000,0.15:4000,0.20:12000", help="Overflow tiers as threshold:penalty pairs (default: 0.12:1000,0.15:4000,0.20:12000)")
    parser.add_argument("--start-year", type=int, default=2026, help="Start financial year (default: 2026)")
    parser.add_argument("--horizon", type=int, default=60, help="Planning horizon in years (default: 60)")
    parser.add_argument("--time-limit", type=float, default=300.0, help="Solver time limit in seconds (default: 300.0)")
    parser.add_argument("--optimizer", type=str, choices=["cp-sat", "optimizer"], default="cp-sat", help="Optimizer backend to use (default: cp-sat)")
    
    # Use parse_known_args to avoid crashing on Jupyter/Databricks kernel arguments (e.g. -f connection.json)
    args, _ = parser.parse_known_args()
    
    run_optimization(args)
