# Capital Project Optimiser Formulation 
### Fixed funding with piecewise soft cap
This document describes the mathematical formulation of the capital project optimiser problem. The problem is to optimise the capital project portfolio subject to fixed funding with a piecewise soft cap on the net balance.

## 1. Sets and Indices

*   $v \in V$: Set of projects (variants).
*   $t \in \{0, \dots, T-1\}$: Time periods (years), where $T$ is the planning horizon (`Tn`).
*   $s$: Start year index for a project.
*   $i$: Index for piecewise soft cap tiers.

## 2. Parameters

*   $E_t$: Funding envelope capacity (target) in year $t$ (`funding_target_S`).
*   $S_{v,k}$: Spend of project $v$ in the $k$-th year of its duration ($k=0, \dots, D_v-1$).
*   $D_v$: Duration of project $v$ in years.
*   $W_{v,s}$: PV coefficient (benefit) for starting project $v$ at year $s$.
*   $Cap_{starts}$: Maximum number of project starts allowed per year (`max_starts_per_year`).
*   $M$: A large constant used for big-M constraints (`net_bigM`), typically sum of all funding targets.
*   $Tier_{i, \text{thresh}}$: Threshold (fraction of envelope) for the $i$-th piecewise soft cap tier.
*   $Tier_{i, \text{weight}}$: Penalty weight for the $i$-th piecewise soft cap tier.
*   $\alpha_{backlog}$: Weight for backlog penalty (`BACKLOG_WEIGHT` = 1.0).
*   $\alpha_{pv}$: Weight for PV reward (`PV_WEIGHT` = 1e-4).

## 3. Decision Variables

*   $x_{v,s} \in \{0, 1\}$: Binary variable, equal to 1 if project $v$ starts at year $s$, 0 otherwise. (Relaxed to continuous $[0,1]$ if `relax_binaries` is True).
*   $y_t \in [0, 1]$: Envelope active indicator for year $t$. Monotone non-increasing ($y_t \ge y_{t+1}$).
*   $funding_t \ge 0$: Funding actually drawn in year $t$.
*   $dividend_t \ge 0$: Unused funding returned in year $t$ (restricted to end-of-life).
*   $net_t \ge 0$: Net balance at the end of year $t$.
*   $excess\_tier_{i,t} \ge 0$: Excess net balance allocated to tier $i$ in year $t$.
*   $backlog_t \ge 0$: Auxiliary variable representing the backlog penalty in year $t$.

## 4. Objective Function

Minimize the weighted sum of backlog penalties and excess tier penalties, minus the weighted PV reward:

$$
\text{Minimize } Z = \alpha_{backlog} \sum_{t=0}^{T-1} backlog_t + \sum_{t=0}^{T-1} \sum_{i} (Tier_{i, \text{weight}} \cdot excess\_tier_{i,t}) - \alpha_{pv} \sum_{v \in V} \sum_{s} (W_{v,s} \cdot x_{v,s})
$$

## 5. Constraints

### 5.1. Project Constraints

*   **Single Start:** Each project must start exactly once within the allowed start window.
    $$\sum_{s} x_{v,s} = 1 \quad \forall v \in V 
	$$

*   **Starts Capacity:** Limit the number of project starts in any given year.
    $$\sum_{v} x_{v,t} \le Cap_{starts} \quad \forall t 
	$$

### 5.2. Financial Dynamics

*   **Spend Calculation:** Total spend in year $t$ is the sum of spend from all active projects.
    $$Spend_t = \sum_{v \in V} \sum_{s} x_{v,s} \cdot S_{v, t-s} 
	$$
    *(Sum includes only valid $s$ such that $s \le t < s + D_v$)*

*   **Funding Draw:** Funding drawn is determined by the envelope capacity and the active indicator.
    $$funding_t = y_t \cdot E_t \quad \forall t 
	$$

*   **Net Balance:**
    $$net_0 = funding_0 - Spend_0 - dividend_0 
	$$

    $$net_t = net_{t-1} + funding_t - Spend_t - dividend_t \quad \forall t > 0 
	$$

### 5.3. Envelope and Dividend Logic

*   **Envelope Activation:** If any project spends money in year $t$, the envelope must be active ($y_t=1$).
    $$y_t \ge x_{v,s} \quad \forall v, s \text{ contributing to } Spend_t $$

*   **Monotonicity:** The envelope active period must be contiguous from the start (cannot turn off and on again).
    $$y_t \ge y_{t+1} \quad \forall t < T-1 $$

*   **Dividend Restriction:** Dividends can only be paid out at the end of the programme's life (when the envelope becomes inactive).
    $$ dividend_t \le M \cdot (1 - y_{t+1}) \quad \forall t < T-1 $$

### 5.4. Piecewise Soft Cap on Net Balance

The net balance is capped by a base threshold plus a series of excess tiers. If the balance exceeds the base, it spills into the excess tiers which incur penalties.

$$net_t \le E_t \cdot Tier_{0, \text{thresh}} + \sum_i excess\_tier_{i,t} + M \cdot (1 - y_{t+1}) $$

*   **Tier Capacity:** Each excess tier has a maximum capacity based on the envelope size.
    $$0 \le excess\_tier_{i,t} \le E_t \cdot (Tier_{i+1, \text{thresh}} - Tier_{i, \text{thresh}}) $$

*(Note: The term $M \cdot (1 - y_{t+1})$ allows the net balance to be arbitrarily large (up to $M$) without penalty once the programme ends, i.e., when $y_{t+1}=0$.)*

### 5.5. Backlog Constraints

The backlog variable tracks the net balance but is forced to zero after the programme ends.

$$backlog_t \ge net_t - M \cdot (1 - y_{t+1}) $$
$$backlog_t \le net_t + M \cdot (1 - y_{t+1}) $$
$$backlog_{T-1} = 0 $$

*(Effectively, $backlog_t = net_t$ while $y_{t+1}=1$, and is unconstrained (can be 0) when $y_{t+1}=0$.)*
