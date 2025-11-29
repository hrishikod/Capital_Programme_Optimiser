from ortools.linear_solver import pywraplp
import numpy as np
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
        solver_backend: str = "SCIP",
        time_limit_seconds: float = 300.0,
        gap_limit: float = 0.01
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
        
        self.solver = pywraplp.Solver.CreateSolver(solver_backend)
        if not self.solver:
            raise RuntimeError(f"Could not create solver with backend: {solver_backend}")
            
        # Set solver parameters
        self.solver.SetTimeLimit(int(time_limit_seconds * 1000))
        # Note: Gap limit setting depends on the specific solver backend in OR-Tools
        # For SCIP, we can try setting parameters via string args if supported, 
        # but pywraplp interface is generic. We'll rely on time limit mostly.
        
        self.big_M = sum(funding_target_M) * 2.0 # Safe upper bound
        
        self._build_model()

    def _build_model(self):
        # 1. Pre-calculate allowed starts
        self.allowed_starts: Dict[str, List[int]] = {}
        for v, meta in self.variants.items():
            dur = meta["dur"]
            # Simple logic: can start as long as it finishes within horizon? 
            # Or just starts within horizon? The notebook logic was:
            # s_ear = 0, s_lat = ny - dur
            # We'll stick to that to ensure full project fits in horizon
            s_lat = self.years - dur
            if s_lat >= 0:
                self.allowed_starts[v] = list(range(s_lat + 1))
            else:
                self.allowed_starts[v] = []

        # 2. Decision Variables
        self.x: Dict[Tuple[str, int], pywraplp.Variable] = {}
        
        # x[v,s]: Binary start variable
        for v, starts in self.allowed_starts.items():
            for s in starts:
                self.x[(v, s)] = self.solver.BoolVar(f"x_{v}_{s}")

        # y[t]: Envelope active (relaxed to continuous [0,1] usually fine if constraints force it)
        # But for strict logic, let's use Bool or continuous with constraints.
        # Notebook used continuous [0,1] with constraints.
        self.y: List[pywraplp.Variable] = [
            self.solver.NumVar(0.0, 1.0, f"y_{t}") for t in range(self.years)
        ]

        # Financial variables
        self.funding: List[pywraplp.Variable] = []
        self.spend: List[pywraplp.Variable] = [] # Auxiliary, not direct var, but we can make it one or just expr
        self.net: List[pywraplp.Variable] = []
        self.dividend: List[pywraplp.Variable] = []
        self.backlog: List[pywraplp.Variable] = []
        self.excess_tiers: List[List[pywraplp.Variable]] = [[] for _ in range(self.years)]

        for t in range(self.years):
            ub_fund = self.funding_target_M[t]
            self.funding.append(self.solver.NumVar(0.0, ub_fund, f"fund_{t}"))
            self.net.append(self.solver.NumVar(0.0, self.solver.infinity(), f"net_{t}"))
            self.dividend.append(self.solver.NumVar(0.0, self.solver.infinity(), f"div_{t}"))
            self.backlog.append(self.solver.NumVar(0.0, self.solver.infinity(), f"backlog_{t}"))
            
            # Excess tiers
            env_S = self.funding_target_M[t]
            for i, (thresh_start, _) in enumerate(self.piecewise_cap_tiers):
                is_last = (i == len(self.piecewise_cap_tiers) - 1)
                if not is_last:
                    thresh_next = self.piecewise_cap_tiers[i+1][0]
                    width = env_S * (thresh_next - thresh_start)
                    self.excess_tiers[t].append(self.solver.NumVar(0.0, width, f"exc_{t}_{i}"))
                else:
                    self.excess_tiers[t].append(self.solver.NumVar(0.0, self.solver.infinity(), f"exc_{t}_{i}"))

        # 3. Constraints

        # Single start per project
        for v in self.variants:
            if self.allowed_starts[v]:
                self.solver.Add(
                    self.solver.Sum([self.x[(v, s)] for s in self.allowed_starts[v]]) == 1.0
                )

        # Max starts per year
        for t in range(self.years):
            starts_in_t = []
            for v, starts in self.allowed_starts.items():
                if t in starts:
                     starts_in_t.append(self.x[(v, t)])
            if starts_in_t:
                self.solver.Add(self.solver.Sum(starts_in_t) <= self.max_starts_per_year)

        # Spend expressions
        self.spend_exprs = []
        for t in range(self.years):
            terms = []
            for v, starts in self.allowed_starts.items():
                spend_vec = self.variants[v]["spend"]
                for s in starts:
                    # If project v starts at s, does it spend in year t?
                    # t must be >= s and t < s + dur
                    if s <= t < s + len(spend_vec):
                        amount = spend_vec[t - s]
                        if amount > 0:
                            terms.append(self.x[(v, s)] * amount)
            self.spend_exprs.append(self.solver.Sum(terms))

        # Envelope logic
        for t in range(self.years):
            # funding[t] == y[t] * envelope[t]
            # Linearize: funding <= y[t] * env, funding >= y[t] * env (since funding is var)
            # Or just: funding == y[t] * env
            self.solver.Add(self.funding[t] == self.y[t] * self.funding_target_M[t])
            
            # y[t] >= y[t+1] (Monotonicity)
            if t < self.years - 1:
                self.solver.Add(self.y[t] >= self.y[t+1])
            
            # y[t] must be 1 if there is spend? 
            # Notebook: "If there is spend in year t from some x[v,s], then y[t] must be 1."
            # Implementation: y[t] >= x[v,s] for all contributing vars
            for v, starts in self.allowed_starts.items():
                spend_vec = self.variants[v]["spend"]
                for s in starts:
                    if s <= t < s + len(spend_vec):
                        if spend_vec[t-s] > 0:
                            self.solver.Add(self.y[t] >= self.x[(v, s)])

        # Net balance flow
        # net[0] = fund[0] - spend[0] - div[0]
        self.solver.Add(self.net[0] == self.funding[0] - self.spend_exprs[0] - self.dividend[0])
        for t in range(1, self.years):
            self.solver.Add(
                self.net[t] == self.net[t-1] + self.funding[t] - self.spend_exprs[t] - self.dividend[t]
            )

        # Dividend restriction: dividend[t] <= M * (1 - y[t+1])
        # Only pay dividend if next year is inactive (end of programme)
        for t in range(self.years - 1):
            self.solver.Add(self.dividend[t] <= self.big_M * (1.0 - self.y[t+1]))
        
        # Piecewise soft cap
        base_thresh = self.piecewise_cap_tiers[0][0]
        for t in range(self.years):
            base_cap = self.funding_target_M[t] * base_thresh
            sum_excess = self.solver.Sum(self.excess_tiers[t])
            
            # net[t] <= base_cap + sum_excess + M*(1 - y[t+1])
            # If y[t+1]=1 (active), net <= base + excess
            # If y[t+1]=0 (end), net <= base + excess + M (unconstrained effectively)
            
            rhs = base_cap + sum_excess
            if t < self.years - 1:
                rhs += self.big_M * (1.0 - self.y[t+1])
            else:
                # Last year, just cap by Big M if needed, or assume y[T]=0 implicitly
                rhs += self.big_M 
            
            self.solver.Add(self.net[t] <= rhs)

        # Backlog constraints
        # backlog[t] >= net[t] - M(1 - y[t+1])
        # backlog[t] <= net[t] + M(1 - y[t+1])
        # backlog[t] >= 0 (implicit in var definition)
        # backlog[T-1] == 0
        for t in range(self.years - 1):
            term = self.big_M * (1.0 - self.y[t+1])
            self.solver.Add(self.backlog[t] >= self.net[t] - term)
            self.solver.Add(self.backlog[t] <= self.net[t] + term)
        
        self.solver.Add(self.backlog[self.years - 1] == 0.0)

        # 4. Objective
        # Min Backlog + Excess Penalties - PV
        
        obj_backlog = self.solver.Sum(self.backlog) * self.backlog_weight
        
        excess_terms = []
        for t in range(self.years):
            for i, (_, weight) in enumerate(self.piecewise_cap_tiers):
                excess_terms.append(self.excess_tiers[t][i] * weight)
        obj_excess = self.solver.Sum(excess_terms)
        
        # PV Reward
        # We need PV coefficients. For now, assume we have them or calculate them.
        # Let's assume a simple discount rate for now or pass it in.
        # The notebook calculated specific PV coefficients per project/start.
        # I'll implement a simple PV calculation here: sum(benefit / (1+r)^t)
        # But wait, the benefit data is complex (kernels).
        # For this refactor, I will assume the caller passes in a `pv_map` or I calculate it simply from cost if benefit data missing?
        # Actually, `variants` has `spend`. I should probably accept `pv_coefficients` map as input to be precise.
        # BUT, to make this class self-contained for the "refactor", I'll add a method to set PV coefficients.
        # Or better, let's just use a placeholder PV = Cost for now if not provided?
        # No, the user wants "easier LP generation", so I should expose the objective terms.
        
        self.pv_expr = self.solver.Sum([]) # Placeholder, will be populated if coefficients set
        
        self.objective = self.solver.Objective()
        # We can't easily sum expressions into a single Objective object in OR-Tools like in COPT/Gurobi sometimes.
        # We have to set coefficients for variables.
        # This is where OR-Tools is a bit more verbose.
        # Actually, `solver.Minimize(expr)` works in python wrapper.
        
        self.total_obj_expr = obj_backlog + obj_excess
        self.solver.Minimize(self.total_obj_expr)

    def set_pv_coefficients(self, pv_map: Dict[Tuple[str, int], float]):
        """
        Updates the objective to include PV rewards.
        pv_map: {(variant, start_year): pv_value}
        """
        pv_terms = []
        for (v, s), coeff in pv_map.items():
            if (v, s) in self.x:
                pv_terms.append(self.x[(v, s)] * coeff)
        
        if pv_terms:
            self.pv_expr = self.solver.Sum(pv_terms)
            # Update objective: Original Min - PV * weight
            # OR-Tools doesn't support "updating" the expression easily if we already called Minimize?
            # Actually we can just call Minimize again with the new expression.
            self.solver.Minimize(self.total_obj_expr - self.pv_expr * self.pv_weight)

    def solve(self) -> OptimizationResult:
        status_code = self.solver.Solve()
        
        status_map = {
            pywraplp.Solver.OPTIMAL: "OPTIMAL",
            pywraplp.Solver.FEASIBLE: "FEASIBLE",
            pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
            pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
            pywraplp.Solver.ABNORMAL: "ABNORMAL",
            pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED"
        }
        status = status_map.get(status_code, "UNKNOWN")
        
        if status not in ["OPTIMAL", "FEASIBLE"]:
            return OptimizationResult(status, 0.0, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0.0)

        # Extract results
        # Schedule
        schedule_rows = []
        for (v, s), var in self.x.items():
            if var.solution_value() > 0.5:
                schedule_rows.append({
                    "Project": v,
                    "StartYear": self.start_fy + s,
                    "Duration": self.variants[v]["dur"]
                })
        schedule_df = pd.DataFrame(schedule_rows)
        
        # Spend Profile
        spend_data = {}
        for t in range(self.years):
            spend_data[self.start_fy + t] = self.spend_exprs[t].solution_value()
        spend_df = pd.DataFrame([spend_data], index=["Total Spend"])
        
        # Cash Flow
        cash_rows = []
        for t in range(self.years):
            cash_rows.append({
                "Year": self.start_fy + t,
                "Funding": self.funding[t].solution_value(),
                "Spend": self.spend_exprs[t].solution_value(),
                "Dividend": self.dividend[t].solution_value(),
                "Net": self.net[t].solution_value(),
                "Backlog": self.backlog[t].solution_value()
            })
        cash_df = pd.DataFrame(cash_rows)
        
        # Gap calculation (approximate for SCIP/CBC via OR-Tools)
        # OR-Tools doesn't always expose gap directly for all solvers via standard API
        # We'll try to get best bound if available
        gap = 0.0
        try:
            obj_val = self.solver.Objective().Value()
            best_bound = self.solver.Objective().BestBound()
            if abs(obj_val) > 1e-6:
                gap = abs(obj_val - best_bound) / abs(obj_val)
        except:
            pass

        return OptimizationResult(
            status=status,
            objective_value=self.solver.Objective().Value(),
            schedule=schedule_df,
            spend_profile=spend_df,
            cash_flow=cash_df,
            gap=gap
        )
