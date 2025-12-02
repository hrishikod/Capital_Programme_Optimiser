# Assumptions and Design Decisions

This document captures assumptions and logic derived from the original notebook implementation and design decisions made during the refactoring process.

## Project Scheduling
- **Allowed Starts**: A project is allowed to start in any year `s` such that it can fully complete within the planning horizon (`years`).
  - Logic: `latest_start_idx = years - duration`.
  - Constraint: `0 <= start_year_idx <= latest_start_idx`.

## Financial Constraints
- **Funding Envelope**: 
  - The funding constraint is modeled as `funding[t] == y[t] * funding_target[t]`.
  - `y[t]` is a variable (continuous [0,1] or binary) representing if the programme is "active" in year `t`.
  - Monotonicity: `y[t] >= y[t+1]` ensures the programme doesn't restart after stopping.
  - Activity: If any project spends money in year `t`, `y[t]` must be active (>= 1).

- **Dividends**:
  - Dividends can only be paid when the programme is winding down or finished.
  - Constraint: `dividend[t] <= Big_M * (1 - y[t+1])`. This implies dividends are zero if the next year is active.

- **Piecewise Soft Cap**:
  - Excess funding usage is penalized via piecewise linear tiers.
  - Logic: `net[t] <= base_cap + sum(excess_tiers) + Big_M * (1 - y[t+1])`.
  - If the programme ends (`y[t+1]=0`), the cap is effectively removed (via `Big_M`), allowing the remaining funds to be "dumped" or handled without penalty.

- **Backlog**:
  - Backlog tracks unspent/unallocated funds carried over.
  - Constraint: `backlog[t] == net[t]` (approx) but relaxed with `Big_M` when programme ends.
  - Final year backlog is forced to 0.

## Optimization Objective
- **Components**:
  - Minimize `Backlog * backlog_weight`
  - Minimize `Excess_Spend * tier_weight`
  - Maximize `PV_Benefits * pv_weight` (implemented as minimizing negative PV).
- **Solver**:
  - Uses SCIP by default.
  - Gap limit and time limit are configurable.

## Data Loading
- **Cost Type**: Defaults to "P50 - Real".
- **Missing Data**: Missing values in CSVs are treated as 0.0.
- **Number Parsing**: Commas and spaces in number strings are removed before parsing.

## Implementation Details
- **Objective Updates**: The objective function is initially set with Backlog and Excess Spend terms. It is updated later to include PV rewards by calling `Minimize` again with the combined expression.
- **Gap Calculation**: The optimality gap is calculated as `abs(obj - bound) / abs(obj)` if the objective is non-zero. This is an approximation as OR-Tools generic API doesn't always expose the solver's internal gap directly.
