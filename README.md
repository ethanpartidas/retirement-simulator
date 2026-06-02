# retirement-simulator

A dynamic-programming Monte Carlo retirement portfolio optimiser.

Given your current age, net worth, savings rate, and spending target, it finds
the stock/bond allocation at every point in time that optimises one of three goals:

| Mode | What it optimises |
|---|---|
| `earliest_retirement` | Retire as early as possible at a configurable age percentile |
| `max_wealth` | Maximise nominal net worth at a fixed target retirement age at a configurable wealth percentile |
| `semi_retirement` | Minimise semi-retirement age, sustaining reduced cash flow until a fixed full retirement age |

All modes use the same fan-out distributional DP — backward induction over a
wealth grid, propagating full quantile distributions of outcomes rather than
point estimates. Forward simulation then runs Monte Carlo paths through the
resulting policy.

---

## Quick start

### 1. Clone

```bash
git clone https://github.com/ethanpartidas/retirement-simulator.git
cd retirement-simulator

```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

```

### 3. Install dependencies

```bash
pip install -r requirements.txt

```

> **Numba** is listed as optional but gives a ~10–50× speed-up on the backward
> induction step. It takes a minute to JIT-compile on the first run; subsequent
> runs use a cache and start instantly.

### 4. Edit `config.toml`

All parameters live in `config.toml`. The key knobs are:

```toml
[person]
current_age = 25
end_age     = 95
current_nw  = 150_000

[cashflow]
monthly_savings_today                  = 4_000   # pre-retirement monthly savings (today's $)
retirement_spend_today                 = 4_000   # monthly spend in full retirement (today's $)
semi_retirement_monthly_cashflow_today = 1_000   # net monthly cash flow during semi-retirement (today's $)

[kids]
kids_start_age             = 30
kids_end_age               = 50
kids_monthly_expense_today = 1_000

[market]
stock_mu_nom        = 0.10    # expected nominal stock return
stock_vol           = 0.15
bond_mu_nom         = 0.045
bond_vol            = 0.05
inflation_rate      = 0.03
retirement_bond_vol = 0.01    # post-retirement bond vol (model a bond ladder)

[simulation]
iterations           = 10_000
age_percentile       = 0.95      # e.g. 0.95 → "hit target age in 95% of scenarios"
wealth_percentile    = 0.05      # e.g. 0.05 → "protect the bottom 5% wealth outcomes"
num_quantiles        = 50        # DP resolution; 50 → 2500 branches per step
mode                 = "semi_retirement"   # "earliest_retirement", "max_wealth", or "semi_retirement"
fixed_retirement_age = 55        # used in max_wealth and semi_retirement modes

```

### 5. Run

```bash
python simulate.py                    # uses config.toml
python simulate.py --config alt.toml  # custom config

```

Output: three-panel chart saved to `results.png` and printed summary stats.

---

## How it works

### Backward induction (DP)

Starting from the terminal age and working backwards, the algorithm maintains a
distribution of future outcomes (retirement age or nominal wealth) at every
`(age, wealth)` grid point.

At each step, for every allocation in `{0%, 10%, …, 100%}` stocks:

1. **N wealth outcomes** are computed from N Gaussian quantiles of the
one-year return distribution.
2. **Fan-out**: each of those N next-year wealth outcomes is paired with each
of the N outcome-quantiles stored in the next year's DP node, giving
N² joint paths.
3. The N² outcomes are sorted and compressed back to N quantiles for storage.
4. The allocation that minimises (or maximises) the value at the target percentile
is stored as the optimal policy.

### Forward simulation

10 000 independent Monte Carlo paths are rolled forward through the policy
table. Each path uses `retirement_bond_vol` (rather than the accumulation
`bond_vol`) once full retirement is reached, modeling the reduced variance of a
bond-ladder or annuity in decumulation.

---

## Output

Three panels are produced:

* **Wealth trajectories** — 100 sample paths + median + P5/P95 bands.
* **Policy heatmap** — optimal stock weight at every `(age, wealth)` node. In `semi_retirement` mode, this displays a mixed view: accumulation policy for pre-threshold cells, and semi-retirement policy for cells that have met the semi-retirement threshold.
* **DP heatmap** — predicted percentile outcome (retirement age or nominal wealth)
across the grid.

---

## Modes in detail

### `earliest_retirement`

The DP value stored at each node is the predicted retirement age. "Retiring"
is defined as crossing a spending-floor threshold: the present value of all
future inflation-adjusted spending, multiplied by 1.1 as a buffer.

The optimiser minimises the `age_percentile` retirement age, i.e.,
*"in the worst X% of scenarios, how old will I be when I can retire?"*

### `max_wealth`

A fixed `fixed_retirement_age` is assumed. The DP value stored at each node
is predicted nominal net worth at that age. The optimiser maximises the `wealth_percentile` nominal wealth.

### `semi_retirement`

Uses a two-phase DP approach.

* **Phase 1** optimises allocations during the semi-retirement window to find the minimum net worth required at each age to successfully hit the full retirement threshold by `fixed_retirement_age`.
* **Phase 2** optimises accumulation allocations to hit those Phase 1 net worth targets as early as possible.

---

## Performance

| Setup | Typical backward-induction time |
| --- | --- |
| No Numba (pure NumPy) | ~5–20 min |
| With Numba (first run, JIT compile) | ~2–4 min |
| With Numba (cached) | **~30–90 sec** |

Parameters that most affect runtime: `num_quantiles` (scales as Q²),
`num_nw_buckets` (101, hard-coded), number of age steps.

---

## Requirements

* Python ≥ 3.10
* numpy, scipy, matplotlib
* numba *(optional but recommended)*
* tomli *(only needed on Python < 3.11)*
