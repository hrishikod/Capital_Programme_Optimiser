from ortools.sat.python import cp_model
import numpy as np
import logging
import mlflow
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import math

@dataclass
class OptimizationResult:
    status: str
    objective_value: float
    schedule: pd.DataFrame
    spend_profile: pd.DataFrame
    cash_flow: pd.DataFrame
    gap: float
    breakdown: Dict[str, float] = None
    log_file: str = None

class CapitalProgrammeOptimizer:
    def __init__(
        self,
        variants: Dict[str, dict],
        funding_target_M: List[float],
        start_fy: int,
        years: int,
        max_starts_per_year: int = 100,
        backlog_weight: float = 1.0,
        pv_weight: float = 1e-4,
        piecewise_cap_tiers: List[Tuple[float, float]] = None,
        time_limit_seconds: float = 300.0,
        gap_limit: float = 0.01, # Not directly used in CP-SAT same way, but kept for compat
        relax_integrality: bool = False, # CP-SAT is pure integer, this flag will be ignored/warned
        scaling_factor: float = 1000.0
    ):
        self.variants = variants
        self.funding_target_M = funding_target_M
        self.start_fy = start_fy
        self.years = years
        self.max_starts_per_year = max_starts_per_year
        self.backlog_weight = backlog_weight
        self.pv_weight = pv_weight
        self.piecewise_cap_tiers = piecewise_cap_tiers or [
            (0.12, 1000.0), (0.15, 4000.0), (0.20, 12000.0)
        ]
        self.relax_integrality = relax_integrality
        self.scaling_factor = scaling_factor
        self.time_limit_seconds = time_limit_seconds
        
        self.model = cp_model.CpModel()
        
        # Calculate big_M in scaled units
        total_funding = sum(funding_target_M)
        self.big_M_scaled = int(total_funding * 2.0 * self.scaling_factor)
        
        self._build_model()

    def _scale(self, value: float) -> int:
        return int(round(value * self.scaling_factor))

    def _descale(self, value: int) -> float:
        return float(value) / self.scaling_factor

    @mlflow.trace(name="_build_model", span_type="build")
    def _build_model(self):
        # 1. Pre-calculate allowed starts
        self.allowed_starts: Dict[str, List[int]] = {}
        for variant_id, meta in self.variants.items():
            duration = meta["dur"]
            latest_start_idx = self.years - duration
            if latest_start_idx >= 0:
                self.allowed_starts[variant_id] = list(range(latest_start_idx + 1))
            else:
                self.allowed_starts[variant_id] = []

        # 2. Decision Variables
        self.x: Dict[Tuple[str, int], cp_model.IntVar] = {}
        
        # x[variant_id, start_year_idx]: Binary start variable
        for variant_id, starts in self.allowed_starts.items():
            for start_year_idx in starts:
                self.x[(variant_id, start_year_idx)] = self.model.NewBoolVar(f"x_{variant_id}_{start_year_idx}")

        # y[t]: Envelope active (Binary)
        self.y: List[cp_model.IntVar] = [
            self.model.NewBoolVar(f"y_{t}") for t in range(self.years)
        ]

        # Financial variables (Scaled Integers)
        # Use a safe upper bound (Total funding * ~5) to avoid overflow
        # int64 max is ~9e18, so scaling by 1000 is safe.
        
        upper_bound = int(sum(self.funding_target_M) * 5 * self.scaling_factor) # Safe upper bound
        
        self.funding: List[cp_model.IntVar] = []
        self.spend_vars: List[cp_model.IntVar] = [] 
        self.net: List[cp_model.IntVar] = []
        self.dividend: List[cp_model.IntVar] = []
        self.backlog: List[cp_model.IntVar] = []
        self.excess_tiers: List[List[cp_model.IntVar]] = [[] for _ in range(self.years)]

        for t in range(self.years):
            ub_fund = self._scale(self.funding_target_M[t])
            # Funding is exactly y[t] * target, or 0.
            # Modeled as a variable for clarity in constraints.
            self.funding.append(self.model.NewIntVar(0, ub_fund, f"fund_{t}"))
            
            # Net, Dividend, Backlog can be large
            self.net.append(self.model.NewIntVar(0, upper_bound, f"net_{t}"))
            self.dividend.append(self.model.NewIntVar(0, upper_bound, f"div_{t}"))
            self.backlog.append(self.model.NewIntVar(0, upper_bound, f"backlog_{t}"))
            
            # Spend variable for year t
            self.spend_vars.append(self.model.NewIntVar(0, upper_bound, f"spend_{t}"))

            # Excess tiers
            env_S = self.funding_target_M[t]
            for i, (thresh_start, _) in enumerate(self.piecewise_cap_tiers):
                is_last = (i == len(self.piecewise_cap_tiers) - 1)
                if not is_last:
                    thresh_next = self.piecewise_cap_tiers[i+1][0]
                    width = self._scale(env_S * (thresh_next - thresh_start))
                    self.excess_tiers[t].append(self.model.NewIntVar(0, width, f"exc_{t}_{i}"))
                else:
                    self.excess_tiers[t].append(self.model.NewIntVar(0, upper_bound, f"exc_{t}_{i}"))

        # 3. Constraints

        # Single start per project
        for variant_id in self.variants:
            if self.allowed_starts[variant_id]:
                self.model.Add(
                    sum(self.x[(variant_id, s)] for s in self.allowed_starts[variant_id]) == 1
                )

        # Max starts per year
        for t in range(self.years):
            starts_in_t = []
            for variant_id, starts in self.allowed_starts.items():
                if t in starts:
                     starts_in_t.append(self.x[(variant_id, t)])
            if starts_in_t:
                self.model.Add(sum(starts_in_t) <= self.max_starts_per_year)

        # Spend expressions and linking to spend_vars
        for t in range(self.years):
            terms = []
            for variant_id, starts in self.allowed_starts.items():
                spend_vec = self.variants[variant_id]["spend"]
                for start_year_idx in starts:
                    if start_year_idx <= t < start_year_idx + len(spend_vec):
                        amount = spend_vec[t - start_year_idx]
                        if amount > 0:
                            scaled_amount = self._scale(amount)
                            terms.append(self.x[(variant_id, start_year_idx)] * scaled_amount)
            
            # self.spend_vars[t] == sum(terms)
            self.model.Add(self.spend_vars[t] == sum(terms))

        # Envelope logic
        for t in range(self.years):
            # funding[t] == y[t] * target
            target_scaled = self._scale(self.funding_target_M[t])
            self.model.Add(self.funding[t] == self.y[t] * target_scaled)
            
            # y[t] >= y[t+1] (Monotonicity)
            if t < self.years - 1:
                self.model.Add(self.y[t] >= self.y[t+1])
            
            # y[t] must be 1 if there is spend (BigM constraint)
            self.model.Add(self.spend_vars[t] <= self.y[t] * upper_bound)

        # Net balance flow: net[t] = net[t-1] + funding[t] - spend[t] - dividend[t]
        # Initial year:
        self.model.Add(self.funding[0] == self.net[0] + self.spend_vars[0] + self.dividend[0])
        
        for t in range(1, self.years):
            # net[t] == net[t-1] + funding[t] - spend[t] - dividend[t]
            # => net[t] + spend[t] + dividend[t] == net[t-1] + funding[t]
            self.model.Add(
                self.net[t] + self.spend_vars[t] + self.dividend[t] == self.net[t-1] + self.funding[t]
            )

        # Dividend restriction: dividend[t] <= M * (1 - y[t+1])
        # If y[t+1] is 1, dividend[t] must be 0.
        for t in range(self.years - 1):
            self.model.Add(self.dividend[t] <= self.big_M_scaled * (1 - self.y[t+1]))
        
        # Piecewise soft cap
        base_thresh = self.piecewise_cap_tiers[0][0]
        for t in range(self.years):
            base_cap = self._scale(self.funding_target_M[t] * base_thresh)
            sum_excess = sum(self.excess_tiers[t])
            
            # Link net[t] to base cap + excess tiers
            rhs_terms = [base_cap, sum_excess]
            if t < self.years - 1:
                rhs_terms.append(self.big_M_scaled * (1 - self.y[t+1]))
            else:
                rhs_terms.append(self.big_M_scaled) # No cap in final years
            
            self.model.Add(self.net[t] <= sum(rhs_terms))

        # Backlog constraints
        # backlog[t] >= net[t] - M*(1-y[t+1])
        # backlog[t] <= net[t] + M*(1-y[t+1])
        # If y[t+1]=1, backlog[t] == net[t]. Else loose.
        for t in range(self.years - 1):
            term = self.big_M_scaled * (1 - self.y[t+1])
            self.model.Add(self.backlog[t] >= self.net[t] - term)
            self.model.Add(self.backlog[t] <= self.net[t] + term)
        
        self.model.Add(self.backlog[self.years - 1] == 0)

        # 4. Objective
        # Scale objective components to integer units.
        self.obj_scale = 10000
        
        # Backlog term
        # backlog variables are already scaled by scaling_factor.
        # We want the final objective value to represent something meaningful or just be minimized.
        
        self.obj_backlog = sum(self.backlog) * int(self.backlog_weight * self.obj_scale)
        
        # Excess term
        excess_terms = []
        for t in range(self.years):
            for i, (_, weight) in enumerate(self.piecewise_cap_tiers):
                # weight is penalty per dollar.
                # excess_tiers are scaled dollars.
                # so excess * weight is scaled dollars * penalty.
                # we multiply by obj_scale.
                excess_terms.append(self.excess_tiers[t][i] * int(weight * self.obj_scale))
        self.obj_excess = sum(excess_terms)
        
        self.pv_expr = 0 # Will be updated
        
        self.model.Minimize(self.obj_backlog + self.obj_excess)

    @mlflow.trace(name="set_pv_coefficients", span_type="build")
    def set_pv_coefficients(self, pv_map: Dict[Tuple[str, int], float]):
        """
        Updates the objective to include PV rewards.
        pv_map: {(variant_id, start_year_idx): pv_value}
        """
        self.pv_map = pv_map
        pv_terms = []
        for (variant_id, start_year_idx), coeff in pv_map.items():
            if (variant_id, start_year_idx) in self.x:
                # Scale PV coefficient to match objective units (ScaledDollars * ObjScale)
                
                scaled_coeff = int(coeff * self.scaling_factor * self.obj_scale * self.pv_weight)
                pv_terms.append(self.x[(variant_id, start_year_idx)] * scaled_coeff)
        
        if pv_terms:
            self.pv_expr = sum(pv_terms)
            # Update objective to include PV term (Maximize PV => Minimize -PV)
            self.model.Minimize(self.obj_backlog + self.obj_excess - self.pv_expr)

    def export_model(self, filepath: str):
        """Exports the model to a text file (CP-SAT format)."""
        with open(filepath, "w") as f:
            f.write(str(self.model))

    @mlflow.trace(name="solve", span_type="optimization")
    def solve(self) -> OptimizationResult:
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit_seconds
        solver.parameters.log_search_progress = True
        solver.parameters.log_to_stdout = False
        solver.log_callback = lambda line: logging.info(line)
        
        status_code = solver.Solve(self.model)
        
        status_map = {
            cp_model.OPTIMAL: "OPTIMAL",
            cp_model.FEASIBLE: "FEASIBLE",
            cp_model.INFEASIBLE: "INFEASIBLE",
            cp_model.MODEL_INVALID: "MODEL_INVALID",
            cp_model.UNKNOWN: "UNKNOWN"
        }
        status = status_map.get(status_code, "UNKNOWN")
        
        if status not in ["OPTIMAL", "FEASIBLE"]:
            return OptimizationResult(status, 0.0, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0.0, None, None)

        # Extract results
        # Schedule
        schedule_rows = []
        for (variant_id, start_year_idx), var in self.x.items():
            if solver.Value(var) > 0.5:
                schedule_rows.append({
                    "Project": variant_id,
                    "StartYear": self.start_fy + start_year_idx,
                    "Duration": self.variants[variant_id]["dur"]
                })
        schedule_df = pd.DataFrame(schedule_rows)
        
        # Spend Profile
        spend_data = {}
        for t in range(self.years):
            spend_data[self.start_fy + t] = self._descale(solver.Value(self.spend_vars[t]))
        spend_df = pd.DataFrame([spend_data], index=["Total Spend"])
        
        # Cash Flow
        cash_rows = []
        for t in range(self.years):
            cash_rows.append({
                "Year": self.start_fy + t,
                "Funding": self._descale(solver.Value(self.funding[t])),
                "Spend": self._descale(solver.Value(self.spend_vars[t])),
                "Dividend": self._descale(solver.Value(self.dividend[t])),
                "Net": self._descale(solver.Value(self.net[t])),
                "Backlog": self._descale(solver.Value(self.backlog[t]))
            })
        cash_df = pd.DataFrame(cash_rows)
        
        # Gap calculation
        # CP-SAT provides best bound and objective value
        gap = 0.0
        try:
            obj_val = solver.ObjectiveValue()
            best_bound = solver.BestObjectiveBound()
            if abs(obj_val) > 1e-6:
                gap = abs(obj_val - best_bound) / abs(obj_val)
        except:
            pass
            
        # --- OBJECTIVE VALUE BREAKDOWN ---
        
        # 1. Calculate Real Backlog ($M)
        # Sum of all backlog variables / currency scaling factor
        raw_backlog_sum = sum(solver.Value(var) for var in self.backlog)
        real_backlog_total = raw_backlog_sum / self.scaling_factor

        # 2. Calculate Real Excess Penalty ($M)
        # Sum of all excess tier variables / currency scaling factor
        raw_excess_sum = 0
        for t in range(self.years):
            for tier_vars in self.excess_tiers[t]:
                raw_excess_sum += solver.Value(tier_vars)
        real_excess_total = raw_excess_sum / self.scaling_factor

        # 3. Calculate Real PV ($M)
        # Iterate active schedule vars and lookup original PV
        real_pv_total = 0.0
        if hasattr(self, 'pv_map') and self.pv_map:
             for (variant_id, start_year_idx), var in self.x.items():
                 if solver.Value(var) > 0.5:
                     real_pv_total += self.pv_map.get((variant_id, start_year_idx), 0.0)

        # 4. Global Descaling Factor
        global_divisor = self.scaling_factor * self.obj_scale
        descaled_objective = solver.ObjectiveValue() / global_divisor

        breakdown = {
            "real_backlog_M": real_backlog_total,
            "real_excess_above_soft_cap_M": real_excess_total,
            "real_pv_M": real_pv_total,
            # Useful for debugging weights:
            "weighted_backlog_contribution": (raw_backlog_sum * self.backlog_weight * self.obj_scale) / global_divisor,
        }

        return OptimizationResult(
            status=status,
            objective_value=descaled_objective,
            schedule=schedule_df,
            spend_profile=spend_df,
            cash_flow=cash_df,
            gap=gap,
            breakdown=breakdown,
            log_file=None # Will be populated by main wrapper if needed
        )
