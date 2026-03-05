# Capital Project Optimiser Formulation (Current CP-SAT Implementation)

## Fixed funding with piecewise soft cap

This document describes the model currently implemented in `src/cp_sat_optimizer.py`.
The optimisation is a pure CP-SAT integer model using scaled integer financial variables.

## 1. Sets and Indices

* $v \in V$: Set of projects (variants).
* $t \in \{0, \dots, T-1\}$: Time periods (years), where $T$ is the planning horizon.
* $s$: Start year index for a project.
* $A_v = \{0, \dots, T-D_v\}$ if $D_v \le T$, else $A_v = \emptyset$: Allowed starts for project $v$.
* $i$: Index for piecewise soft-cap tiers.

## 2. Parameters

* $E_t$: Funding target in year $t$ (`funding_target_M`).
* $S_{v,k}$: Spend of project $v$ in its $k$-th year of duration.
* $D_v$: Duration of project $v$.
* $Cap_{starts}$: Maximum number of project starts allowed per year (`max_starts_per_year`).
* $M$: Big-M for gating constraints (`big_M_scaled = 2 * sum(E_t) * scale`).
* $U$: Shared upper bound for financial integer vars (`upper_bound = 5 * sum(E_t) * scale`).
* $Tier_{i,\text{thresh}}$: Soft-cap threshold fraction for tier $i$ (defaults: 0.12, 0.15, 0.20).
* $Tier_{i,\text{weight}}$: Tier penalty weight (defaults: 1000, 4000, 12000).
* $\alpha_{backlog}$: Backlog weight (`backlog_weight`, default 1.0).
* $\alpha_{pv}$: PV weight (`pv_weight`, default $1e-4$).
* $W_{v,s}$: PV coefficient for project $v$ starting at $s$ (provided externally via `set_pv_coefficients`).
* `scale`: Financial scaling factor (`scaling_factor`, default 1000).
* `obj_scale`: Objective scaling factor (fixed at 10000).

## 3. Decision Variables

* $x_{v,s} \in \{0,1\}$ for $s \in A_v$: 1 if project $v$ starts in year $s$.
* $y_t \in \{0,1\}$: Envelope active indicator.
* $funding_t, spend_t, dividend_t, net_t, backlog_t \in \mathbb{Z}_{\ge 0}$ (scaled units).
* $excess_{i,t} \in \mathbb{Z}_{\ge 0}$ (scaled units).

Implementation notes:

* `relax_integrality` is currently ignored (CP-SAT remains integer).
* `gap_limit` is accepted in API but not applied to solver parameters.

## 4. Objective Function

The implementation minimizes:

$$
Z = c_{backlog}\sum_t backlog_t + \sum_t\sum_i c_i\,excess_{i,t} - \sum_v\sum_{s \in A_v} c^{pv}_{v,s}\,x_{v,s}
$$

where:

* $c_{backlog} = \lfloor \alpha_{backlog} \cdot obj\_scale \rfloor$
* $c_i = \lfloor Tier_{i,\text{weight}} \cdot obj\_scale \rfloor$
* $c^{pv}_{v,s} = \lfloor W_{v,s} \cdot scale \cdot obj\_scale \cdot \alpha_{pv} \rfloor$

If no PV map is provided, the PV term is omitted.

## 5. Constraints

### 5.1 Project Constraints

* **Single Start:**

$$
\sum_{s \in A_v} x_{v,s} = 1 \quad \forall v \text{ with } A_v \neq \emptyset
$$

Projects with $A_v = \emptyset$ (duration longer than horizon) have no start variables.

* **Starts Capacity:**

$$
\sum_{v: t \in A_v} x_{v,t} \le Cap_{starts} \quad \forall t
$$

### 5.2 Financial Dynamics

* **Spend Calculation:**

$$
spend_t = \sum_v \sum_{s \in A_v} x_{v,s} \cdot S_{v,t-s}
$$

Only valid active-year terms are included ($s \le t < s + D_v$), and the implementation only adds strictly positive spend terms.

* **Funding Draw:**

$$
funding_t = y_t \cdot E_t \quad \forall t
$$

* **Net Balance:**

$$
net_0 = funding_0 - spend_0 - dividend_0
$$

$$
net_t = net_{t-1} + funding_t - spend_t - dividend_t \quad \forall t > 0
$$

### 5.3 Envelope and Dividend Logic

* **Spend-Activation Link:**

$$
spend_t \le U \cdot y_t \quad \forall t
$$

* **Monotonic Envelope:**

$$
y_t \ge y_{t+1} \quad \forall t < T-1
$$

* **Dividend Restriction:**

$$
dividend_t \le M \cdot (1 - y_{t+1}) \quad \forall t < T-1
$$

### 5.4 Piecewise Soft Cap on Net Balance

For $t < T-1$:

$$
net_t \le E_t \cdot Tier_{0,\text{thresh}} + \sum_i excess_{i,t} + M \cdot (1 - y_{t+1})
$$

For $t = T-1$ (as implemented):

$$
net_{T-1} \le E_{T-1} \cdot Tier_{0,\text{thresh}} + \sum_i excess_{i,T-1} + M
$$

Tier bounds:

$$
0 \le excess_{i,t} \le E_t \cdot (Tier_{i+1,\text{thresh}} - Tier_{i,\text{thresh}}) \quad \text{for non-last tiers}
$$

The last tier is not threshold-capped in code; it is only bounded by the global integer upper bound $U$.

### 5.5 Backlog Constraints

$$
backlog_t \ge net_t - M \cdot (1 - y_{t+1}) \quad \forall t < T-1
$$

$$
backlog_t \le net_t + M \cdot (1 - y_{t+1}) \quad \forall t < T-1
$$

$$
backlog_{T-1} = 0
$$

This enforces $backlog_t = net_t$ while $y_{t+1}=1$, and relaxes backlog once the envelope turns off.

## 6. Scenario Permutations Around the Core Model

The wrapper `solve_with_permutations(...)` solves four default combinations:

* Cost scenarios:
  * `Base Real` (original spends)
  * `P95 Real` (spends multiplied by 1.2)
* Benefit levels:
  * `base` (original PV map)
  * `high` (PV map multiplied by 1.2)

Each combination is solved independently and then summarized.
