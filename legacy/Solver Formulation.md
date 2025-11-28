This document outlines the Mixed-Integer Linear Programming (MILP) formulation used in Solver.py. The problem solves for an optimal capital programme schedule using a Lexicographical (Multi-objective) approach, prioritizing benefit realization while adhering to a flexible "Prefix Capacity" funding envelope.



1. Indices and Sets



$v \in \mathcal{V}$: The set of candidate projects (variants).

$t \in \mathcal{T} = \{0, \dots, N-1\}$: The planning horizon (years).

$s \in \mathcal{S}_v \subseteq \mathcal{T}$: The set of allowed start years for project $v$.

$k \in \{0, \dots, D_v - 1\}$: Relative year index within the duration of project $v$.



2. Parameters



$c_{v,k}$: The cost (spend) of project $v$ in its $k$-th year of execution.

$b_{v,k}$: The benefit of project $v$ in its $k$-th year of execution.

$r$: The discount rate.

$LB_t, UB_t$: Lower and upper bounds for the funding envelope in year $t$.

$Cap_{total}$: The limit on the total budget across the entire horizon (if CONSERVE_TOTAL is active).

$M_{starts}$: Maximum number of new project starts allowed in a single year.

$w_t^{env}$: Penalty weights for using budget in later years (used in the tertiary objective).



3. Decision Variables



Project Starts (Binary):

$$ x_{v,s} \in \{0, 1\}$$

Where $x_{v,s} = 1$ if project $v$ starts in year $s$, otherwise $0$.

Envelope (Continuous):

$$ y_t \ge 0$$

Where $y_t$ represents the allocated budget capacity in year $t$.



4. Constraints





A. Project Selection



A project can start at most once. (If the project is strictly forced to be included, this becomes an equality constraint).

$$\sum_{s \in \mathcal{S}_v} x_{v,s} \le 1 \quad \forall v \in \mathcal{V}$$



B. Yearly Start Limits



To prevent operational bottlenecks, limit the number of projects starting in the same year.

$$\sum_{v \in \mathcal{V} : t \in \mathcal{S}_v} x_{v,t} \le M_{starts} \quad \forall t \in \mathcal{T}$$



C. Envelope Bounds



The yearly envelope variable is constrained by the scenario settings (e.g., minimum baselines and buffer levels).

$$LB_t \le y_t \le UB_t \quad \forall t \in \mathcal{T}$$



D. Total Capacity (Optional)



If configured to conserve the total baseline budget over the horizon:

$$\sum_{t \in \mathcal{T}} y_t \le Cap_{total}$$

(Note: This constraint ensures the flexible envelope does not exceed the total budget of the fixed baseline).



E. Prefix Capacity Constraint (Cumulative Spending)



This formulation allows "carry-forward" (underspend in year $t$ can be used in $t+1$) but prohibits "borrowing" (cumulative spending cannot exceed cumulative envelope up to year $t$).

Let the total spend in year $t$ be:

$$Spend_t = \sum_{v \in \mathcal{V}} \sum_{s \in \mathcal{S}_v, s \le t < s+D_v} c_{v, t-s} \cdot x_{v,s}$$

The constraint ensures cumulative spend does not exceed cumulative envelope at any point in time:

$$\sum_{\tau=0}^{t} Spend_\tau \le \sum_{\tau=0}^{t} y_\tau \quad \forall t \in \mathcal{T}$$



5. Objectives (Lexicographical)



The solver resolves objectives hierarchically. After optimizing objective $k$, the solver adds a constraint ensuring the result of objective $k$ does not degrade (within a tolerance) while optimizing objective $k+1$.



Objective 1: Maximize Primary Dimension PV



Maximize the Net Present Value (PV) of the selected primary benefit dimension (e.g., "Safety" or "Economic").

$$\text{Maximize } Z_1 = \sum_{v \in \mathcal{V}} \sum_{s \in \mathcal{S}_v} PV_{v,s}^{primary} \cdot x_{v,s}$$

Where the coefficient $PV_{v,s}$ is pre-calculated:

$$PV_{v,s} = \sum_{k=0}^{D_v-1} \frac{b_{v,k}}{(1+r)^{s+k}}$$



Objective 2: Maximize Total Benefit PV



Maximize the PV of all combined benefit dimensions.

$$\text{Maximize } Z_2 = \sum_{v \in \mathcal{V}} \sum_{s \in \mathcal{S}_v} PV_{v,s}^{total} \cdot x_{v,s}$$



Objective 3: Minimize Late Envelope (Smoothing)



Minimize the envelope usage, weighted to penalize later years more heavily. This preferences front-loading and early allocation of the budget.

$$\text{Minimize } Z_3 = \sum_{t \in \mathcal{T}} w_t^{env} \cdot y_t$$
