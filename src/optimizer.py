from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
from ortools.linear_solver import pywraplp


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
        gap_limit: float = 0.01,
        relax_integrality: bool = False,
    ):
        self.variants = variants
        self.funding_target_M = funding_target_M
        self.start_fy = start_fy
        self.years = years
        self.max_starts_per_year = max_starts_per_year
        self.backlog_weight = backlog_weight
        self.pv_weight = pv_weight
        self.piecewise_cap_tiers = piecewise_cap_tiers or [(0.12, 1000.0), (0.15, 4000.0), (0.20, 12000.0)]
        self.relax_integrality = relax_integrality

        self.solver = pywraplp.Solver.CreateSolver(solver_backend)
        if not self.solver:
            raise RuntimeError(f"Could not create solver with backend: {solver_backend}")

        # Set solver parameters
        self.solver.SetTimeLimit(int(time_limit_seconds * 1000))

        self.big_M = sum(funding_target_M) * 2.0  # Safe upper bound

        self._build_model()

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
        self.x: Dict[Tuple[str, int], pywraplp.Variable] = {}

        # x[variant_id, start_year_idx]: Binary start variable (or continuous [0,1] if relaxed)
        for variant_id, starts in self.allowed_starts.items():
            for start_year_idx in starts:
                if self.relax_integrality:
                    self.x[(variant_id, start_year_idx)] = self.solver.NumVar(0.0, 1.0, f"x_{variant_id}_{start_year_idx}")
                else:
                    self.x[(variant_id, start_year_idx)] = self.solver.BoolVar(f"x_{variant_id}_{start_year_idx}")

        # y[t]: Envelope active
        self.y: List[pywraplp.Variable] = [self.solver.NumVar(0.0, 1.0, f"y_{t}") for t in range(self.years)]

        # Financial variables
        self.funding: List[pywraplp.Variable] = []
        self.spend: List[pywraplp.Variable] = []
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
                is_last = i == len(self.piecewise_cap_tiers) - 1
                if not is_last:
                    thresh_next = self.piecewise_cap_tiers[i + 1][0]
                    width = env_S * (thresh_next - thresh_start)
                    self.excess_tiers[t].append(self.solver.NumVar(0.0, width, f"exc_{t}_{i}"))
                else:
                    self.excess_tiers[t].append(self.solver.NumVar(0.0, self.solver.infinity(), f"exc_{t}_{i}"))

        # 3. Constraints

        # Single start per project
        for variant_id in self.variants:
            if self.allowed_starts[variant_id]:
                self.solver.Add(
                    self.solver.Sum([self.x[(variant_id, s)] for s in self.allowed_starts[variant_id]]) == 1.0,
                    name=f"SingleStart_{variant_id}",
                )

        # Max starts per year
        for t in range(self.years):
            starts_in_t = []
            for variant_id, starts in self.allowed_starts.items():
                if t in starts:
                    starts_in_t.append(self.x[(variant_id, t)])
            if starts_in_t:
                self.solver.Add(self.solver.Sum(starts_in_t) <= self.max_starts_per_year, name=f"MaxStarts_{t}")

        # Spend expressions
        self.spend_exprs = []
        for t in range(self.years):
            terms = []
            for variant_id, starts in self.allowed_starts.items():
                spend_vec = self.variants[variant_id]["spend"]
                for start_year_idx in starts:
                    if start_year_idx <= t < start_year_idx + len(spend_vec):
                        amount = spend_vec[t - start_year_idx]
                        if amount > 0:
                            terms.append(self.x[(variant_id, start_year_idx)] * amount)
            self.spend_exprs.append(self.solver.Sum(terms))

        # Envelope logic
        for t in range(self.years):
            self.solver.Add(self.funding[t] == self.y[t] * self.funding_target_M[t], name=f"FundingDef_{t}")

            # y[t] >= y[t+1] (Monotonicity)
            if t < self.years - 1:
                self.solver.Add(self.y[t] >= self.y[t + 1], name=f"Monotonicity_{t}")

            # y[t] must be 1 if there is spend
            for variant_id, starts in self.allowed_starts.items():
                spend_vec = self.variants[variant_id]["spend"]
                for start_year_idx in starts:
                    if start_year_idx <= t < start_year_idx + len(spend_vec):
                        if spend_vec[t - start_year_idx] > 0:
                            self.solver.Add(
                                self.y[t] >= self.x[(variant_id, start_year_idx)],
                                name=f"Activity_{t}_{variant_id}_{start_year_idx}",
                            )

        # Net balance flow
        self.solver.Add(self.net[0] == self.funding[0] - self.spend_exprs[0] - self.dividend[0], name="NetBalance_0")
        for t in range(1, self.years):
            self.solver.Add(
                self.net[t] == self.net[t - 1] + self.funding[t] - self.spend_exprs[t] - self.dividend[t],
                name=f"NetBalance_{t}",
            )

        # Dividend restriction: dividend[t] <= M * (1 - y[t+1])
        for t in range(self.years - 1):
            self.solver.Add(self.dividend[t] <= self.big_M * (1.0 - self.y[t + 1]), name=f"DividendRestr_{t}")

        # Piecewise soft cap
        base_thresh = self.piecewise_cap_tiers[0][0]
        for t in range(self.years):
            base_cap = self.funding_target_M[t] * base_thresh
            sum_excess = self.solver.Sum(self.excess_tiers[t])

            rhs = base_cap + sum_excess
            if t < self.years - 1:
                rhs += self.big_M * (1.0 - self.y[t + 1])
            else:
                rhs += self.big_M

            self.solver.Add(self.net[t] <= rhs, name=f"SoftCap_{t}")

        # Backlog constraints
        for t in range(self.years - 1):
            term = self.big_M * (1.0 - self.y[t + 1])
            self.solver.Add(self.backlog[t] >= self.net[t] - term, name=f"BacklogLB_{t}")
            self.solver.Add(self.backlog[t] <= self.net[t] + term, name=f"BacklogUB_{t}")

        self.solver.Add(self.backlog[self.years - 1] == 0.0, name="BacklogFinal")

        # 4. Objective
        obj_backlog = self.solver.Sum(self.backlog) * self.backlog_weight

        excess_terms = []
        for t in range(self.years):
            for i, (_, weight) in enumerate(self.piecewise_cap_tiers):
                excess_terms.append(self.excess_tiers[t][i] * weight)
        obj_excess = self.solver.Sum(excess_terms)

        self.pv_expr = self.solver.Sum([])

        self.objective = self.solver.Objective()
        self.total_obj_expr = obj_backlog + obj_excess
        self.solver.Minimize(self.total_obj_expr)

    def set_pv_coefficients(self, pv_map: Dict[Tuple[str, int], float]):
        """
        Updates the objective to include PV rewards.
        pv_map: {(variant_id, start_year_idx): pv_value}
        """
        pv_terms = []
        for (variant_id, start_year_idx), coeff in pv_map.items():
            if (variant_id, start_year_idx) in self.x:
                pv_terms.append(self.x[(variant_id, start_year_idx)] * coeff)

        if pv_terms:
            self.pv_expr = self.solver.Sum(pv_terms)
            # Update objective to include PV term
            self.solver.Minimize(self.total_obj_expr - self.pv_expr * self.pv_weight)

    def export_model(self, filepath: str):
        """Exports the model to an LP file."""
        with open(filepath, "w") as f:
            f.write(self.solver.ExportModelAsLpFormat(False))

    def solve(self) -> OptimizationResult:
        status_code = self.solver.Solve()

        status_map = {
            pywraplp.Solver.OPTIMAL: "OPTIMAL",
            pywraplp.Solver.FEASIBLE: "FEASIBLE",
            pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
            pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
            pywraplp.Solver.ABNORMAL: "ABNORMAL",
            pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
        }
        status = status_map.get(status_code, "UNKNOWN")

        if status not in ["OPTIMAL", "FEASIBLE"]:
            return OptimizationResult(status, 0.0, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0.0)

        # Extract results
        # Schedule
        schedule_rows = []
        for (variant_id, start_year_idx), var in self.x.items():
            if var.solution_value() > 0.5:
                schedule_rows.append(
                    {
                        "Project": variant_id,
                        "StartYear": self.start_fy + start_year_idx,
                        "Duration": self.variants[variant_id]["dur"],
                    }
                )
        schedule_df = pd.DataFrame(schedule_rows)

        # Spend Profile
        spend_data = {}
        for t in range(self.years):
            spend_data[self.start_fy + t] = self.spend_exprs[t].solution_value()
        spend_df = pd.DataFrame([spend_data], index=["Total Spend"])

        # Cash Flow
        cash_rows = []
        for t in range(self.years):
            cash_rows.append(
                {
                    "Year": self.start_fy + t,
                    "Funding": self.funding[t].solution_value(),
                    "Spend": self.spend_exprs[t].solution_value(),
                    "Dividend": self.dividend[t].solution_value(),
                    "Net": self.net[t].solution_value(),
                    "Backlog": self.backlog[t].solution_value(),
                }
            )
        cash_df = pd.DataFrame(cash_rows)

        # Gap calculation
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
            gap=gap,
        )
