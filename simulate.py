"""
simulate.py — Retirement portfolio optimiser.

Three optimisation modes (set in config.toml):

  earliest_retirement  –  find the allocation that minimises retirement age at
                           the configured percentile.  DP stores predicted
                           retirement age at each (age, wealth) node.

  max_wealth           –  fix a target retirement age and maximise nominal net
                           worth at that age.  DP stores predicted nominal wealth
                           at each (age, wealth) node.

  semi_retirement      –  fix a full retirement age and find the allocation that
                           minimises semi-retirement age at the configured
                           percentile.  From current age to semi-retirement the
                           monthly savings is monthly_savings_today (inflation-
                           adjusted).  From semi-retirement to fixed_retirement_age
                           the portfolio is 100 % bonds and the net monthly cash
                           flow is semi_retirement_monthly_cashflow_today
                           (inflation-adjusted, positive = net income).  After
                           fixed_retirement_age the portfolio is 100 % bonds and
                           the spend is retirement_spend_today (inflation-adjusted).

Usage:
    python simulate.py                     # uses config.toml in cwd
    python simulate.py --config my.toml    # custom config path
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ── Config loading ───────────────────────────────────────────────────────────

try:
    import tomllib          # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        sys.exit(
            "tomli not found.  Install it:  pip install tomli\n"
            "(or upgrade to Python 3.11+ which includes tomllib in the stdlib)"
        )


def load_config(path: str = "config.toml") -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def flatten_config(cfg: dict) -> dict:
    flat: dict = {}
    for section in cfg.values():
        flat.update(section)
    return flat


def resolve_params(flat: dict) -> dict:
    """Apply defaults and derived values."""
    flat.setdefault("retirement_bond_vol", flat["bond_vol"])
    flat.setdefault("mode", "earliest_retirement")
    flat.setdefault("fixed_retirement_age", flat["end_age"])
    flat.setdefault("semi_retirement_monthly_cashflow_today", 0.0)
    return flat


# ── Numba setup ──────────────────────────────────────────────────────────────

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print(
        "Numba not installed – running without JIT compilation.\n"
        "Install it for a large speed-up:  pip install numba"
    )

    def njit(*args, **kwargs):          # transparent no-op decorator
        def decorator(fn):
            return fn
        return decorator if args and callable(args[0]) else decorator


# ── Shared grid helpers ──────────────────────────────────────────────────────

def make_grids(num_quantiles: int):
    quantiles = np.linspace(
        1 / (2 * num_quantiles), 1 - 1 / (2 * num_quantiles), num_quantiles
    )
    z_scores = norm.ppf(quantiles)
    total = num_quantiles * num_quantiles
    outcome_percentiles = np.linspace(1 / (2 * total), 1 - 1 / (2 * total), total)
    return quantiles, z_scores, outcome_percentiles


# ── MODE 1: Earliest-retirement backward induction ───────────────────────────

@njit(cache=True)
def _backward_earliest(
    nw_grid, ages, actions, quantiles, z_scores, outcome_percentiles,
    dist_grid, optimal_policy,
    monthly_savings_today, retirement_spend_today,
    stock_mu_nom, stock_vol, bond_mu_nom, bond_vol,
    inflation_rate, target_percentile, max_nw,
):
    num_quantiles = len(quantiles)

    for a_idx in range(len(ages) - 2, -1, -1):
        age = ages[a_idx]
        years_to_end = ages[-1] - age

        inf_factor = (1 + inflation_rate) ** a_idx
        yr_savings = monthly_savings_today * 12 * inf_factor
        yr_spend   = retirement_spend_today * 12 * inf_factor

        nominal_floor = 0.0
        for k in range(years_to_end):
            nominal_floor += (
                yr_spend * (1 + inflation_rate) ** k / (1 + bond_mu_nom) ** k
            )
        target_goal = nominal_floor * 1.1

        next_dist = dist_grid[a_idx + 1]   # shape (nw_buckets, Q)

        for nw_idx in range(len(nw_grid)):
            nw = nw_grid[nw_idx]

            if nw >= target_goal:
                optimal_policy[a_idx, nw_idx] = 0.0
                for q in range(num_quantiles):
                    dist_grid[a_idx, nw_idx, q] = age
                continue

            best_action = 0.0
            best_score  = 1e18
            best_dist   = np.empty(num_quantiles)
            for q in range(num_quantiles):
                best_dist[q] = ages[-1]

            for alloc in actions:
                mu  = alloc * stock_mu_nom + (1 - alloc) * bond_mu_nom
                vol = np.sqrt((alloc * stock_vol) ** 2 + ((1 - alloc) * bond_vol) ** 2)

                next_nws = (nw + yr_savings) * (1 + mu + z_scores * vol)

                mat = np.empty((num_quantiles, num_quantiles))
                for q_idx in range(num_quantiles):
                    interp = np.interp(next_nws, nw_grid, next_dist[:, q_idx])
                    for nw_out in range(num_quantiles):
                        nw_next = next_nws[nw_out]
                        if nw_next >= target_goal and nw < target_goal:
                            denom    = nw_next - nw
                            fraction = (target_goal - nw) / denom if denom > 0 else 0.0
                            fraction = min(max(fraction, 0.0), 1.0)
                            mat[nw_out, q_idx] = age + fraction
                        else:
                            mat[nw_out, q_idx] = interp[nw_out]

                all_out = mat.flatten()
                all_out.sort()

                metric = np.interp(target_percentile, outcome_percentiles, all_out)

                # tie-breaker: protect lower-tail wealth
                lo_idx = min(int((1 - target_percentile) * num_quantiles), num_quantiles - 1)
                safe   = np.sort(next_nws)[lo_idx]
                score  = metric - 0.0001 * (safe / max_nw)

                if score < best_score:
                    best_score  = score
                    best_action = alloc
                    best_dist   = np.interp(quantiles, outcome_percentiles, all_out)

            optimal_policy[a_idx, nw_idx] = best_action
            for q in range(num_quantiles):
                dist_grid[a_idx, nw_idx, q] = best_dist[q]

    return dist_grid, optimal_policy


# ── MODE 2: Max-wealth backward induction ────────────────────────────────────

@njit(cache=True)
def _backward_max_wealth(
    nw_grid, ages, actions, quantiles, z_scores, outcome_percentiles,
    dist_grid, optimal_policy, mult_grid,
    monthly_savings_today,
    stock_mu_nom, stock_vol, bond_mu_nom, bond_vol,
    inflation_rate, target_percentile, max_nw,
    fixed_retirement_age,
):
    """
    Unified DP for max_wealth mode.

    dist_grid stores absolute predicted terminal NW for nodes within the grid.
    mult_grid  stores wealth-independent growth multipliers (age-only, no NW dim).

    During fan-out, next_nw values that exceed max_nw are evaluated using the
    multiplier from mult_grid rather than extrapolating from the grid edge,
    eliminating the cap artifact without any pre-set NW threshold.

    Both grids run the same backward pass; mult_grid is always available for
    any out-of-bounds next_nw regardless of which NW bucket we're in.
    """
    num_quantiles = len(quantiles)

    ret_a_idx = -1
    for i in range(len(ages)):
        if ages[i] == fixed_retirement_age:
            ret_a_idx = i
            break

    # Terminal conditions
    for nw_idx in range(len(nw_grid)):
        for q in range(num_quantiles):
            dist_grid[ret_a_idx, nw_idx, q] = nw_grid[nw_idx]
    for q in range(num_quantiles):
        mult_grid[ret_a_idx, q] = 1.0

    for a_idx in range(ret_a_idx - 1, -1, -1):
        inf_factor = (1 + inflation_rate) ** a_idx
        yr_savings = monthly_savings_today * 12 * inf_factor

        next_dist = dist_grid[a_idx + 1]   # shape (nw_buckets, Q)
        next_mult = mult_grid[a_idx + 1]   # shape (Q,)

        # ── High-NW multiplier DP (age-only, no savings) ──────────────────
        best_hi_action = 0.0
        best_hi_metric  = -1e18
        best_hi_dist   = np.zeros(num_quantiles)

        for alloc in actions:
            mu  = alloc * stock_mu_nom + (1 - alloc) * bond_mu_nom
            vol = np.sqrt((alloc * stock_vol) ** 2 + ((1 - alloc) * bond_vol) ** 2)
            one_yr_mults = 1.0 + mu + z_scores * vol

            mat = np.empty((num_quantiles, num_quantiles))
            for q_idx in range(num_quantiles):
                for j in range(num_quantiles):
                    mat[j, q_idx] = one_yr_mults[j] * next_mult[q_idx]

            all_out = mat.flatten()
            all_out.sort()

            metric = np.interp(target_percentile, outcome_percentiles, all_out)\

            if metric > best_hi_metric:
                best_hi_metric  = metric
                best_hi_action = alloc
                best_hi_dist   = np.interp(quantiles, outcome_percentiles, all_out)

        for q in range(num_quantiles):
            mult_grid[a_idx, q] = best_hi_dist[q]

        # ── Full grid DP (all NW buckets, with savings) ───────────────────
        for nw_idx in range(len(nw_grid)):
            nw = nw_grid[nw_idx]

            best_action = 0.0
            best_metric  = -1e18
            best_dist   = np.zeros(num_quantiles)

            for alloc in actions:
                mu  = alloc * stock_mu_nom + (1 - alloc) * bond_mu_nom
                vol = np.sqrt((alloc * stock_vol) ** 2 + ((1 - alloc) * bond_vol) ** 2)

                next_nws = (nw + yr_savings) * (1 + mu + z_scores * vol)

                mat = np.empty((num_quantiles, num_quantiles))
                for q_idx in range(num_quantiles):
                    for j in range(num_quantiles):
                        nw_next = next_nws[j]
                        if nw_next <= 0.0:
                            mat[j, q_idx] = 0.0
                        elif nw_next > max_nw:
                            # Use multiplier DP: next_nw * future_multiplier
                            mat[j, q_idx] = nw_next * next_mult[q_idx]
                        else:
                            mat[j, q_idx] = np.interp(nw_next, nw_grid, next_dist[:, q_idx])

                all_out = mat.flatten()
                all_out.sort()

                metric = np.interp(target_percentile, outcome_percentiles, all_out)

                if metric > best_metric:
                    best_metric  = metric
                    best_action = alloc
                    best_dist   = np.interp(quantiles, outcome_percentiles, all_out)

            optimal_policy[a_idx, nw_idx] = best_action
            for q in range(num_quantiles):
                dist_grid[a_idx, nw_idx, q] = best_dist[q]

    return dist_grid, optimal_policy, mult_grid


# ── MODE 3: Semi-retirement helpers ──────────────────────────────────────────

def precompute_w_semi(
    ages,
    current_age,
    fixed_retirement_age,
    end_age,
    retirement_spend_today,
    semi_retirement_monthly_cashflow_today,
    bond_mu_nom,
    inflation_rate,
):
    """
    Precompute the semi-retirement wealth threshold for every age in `ages`.

    For each candidate semi-retirement age s (s_idx years from current_age),
    w_semi[s] is the NW needed AT age s to fund all future cashflows:

      k = 0 .. (fixed_retirement_age - s - 1)  : semi-retirement years
          annual_cf = semi_retirement_monthly_cashflow_today * 12 * (1+inf)^(s_idx+k)
          (positive = net income, negative = net expense)

      k = (fixed_retirement_age - s) .. (end_age - s - 1)  : full retirement years
          annual_cf = -retirement_spend_today * 12 * (1+inf)^(s_idx+k)
          (always an outflow)

    All cashflows inflated from current_age (factor (1+inf)^(s_idx+k)).
    All discounted back to age s (factor (1+bond_mu_nom)^k, where k is years from s).

    w_semi[s] = -pv_total * 1.1   (flip sign: we need this NW to cover net outflows)

    Ages at or beyond fixed_retirement_age get w_semi = 0.
    """
    w_semi = np.zeros(len(ages))

    for a_idx, age in enumerate(ages):
        if age >= fixed_retirement_age:
            w_semi[a_idx] = 0.0
            continue

        s_idx         = int(age - current_age)   # years from current_age to age s
        years_to_end  = end_age - int(age)        # total years to simulate from s
        years_semi    = fixed_retirement_age - int(age)

        pv_total = 0.0
        for k in range(years_to_end):
            yr_inflation = (1 + inflation_rate) ** (s_idx + k)
            yr_discount  = (1 + bond_mu_nom)    ** k

            if k < years_semi:
                annual_cf = semi_retirement_monthly_cashflow_today * 12 * yr_inflation
            else:
                annual_cf = -retirement_spend_today * 12 * yr_inflation

            pv_total += annual_cf / yr_discount

        w_semi[a_idx] = -pv_total * 1.1

    return w_semi


# ── MODE 3: Semi-retirement backward induction ────────────────────────────────

@njit(cache=True)
def _backward_semi(
    nw_grid, ages, actions, quantiles, z_scores, outcome_percentiles,
    dist_grid, optimal_policy,
    monthly_savings_today,
    stock_mu_nom, stock_vol, bond_mu_nom, bond_vol,
    inflation_rate, target_percentile, max_nw,
    w_semi,
):
    """
    Backward induction for semi_retirement mode.

    Identical in structure to _backward_earliest, but the per-age goal threshold
    comes from the precomputed w_semi array rather than being computed inline
    from retirement_spend_today.

    When a cell's nw >= w_semi[a_idx], every quantile for that cell is set to
    the current age (semi-retirement achieved), exactly mirroring earliest_retirement.
    """
    num_quantiles = len(quantiles)

    for a_idx in range(len(ages) - 2, -1, -1):
        age = ages[a_idx]

        inf_factor = (1 + inflation_rate) ** a_idx
        yr_savings = monthly_savings_today * 12 * inf_factor

        target_goal = w_semi[a_idx]

        next_dist = dist_grid[a_idx + 1]   # shape (nw_buckets, Q)

        for nw_idx in range(len(nw_grid)):
            nw = nw_grid[nw_idx]

            if nw >= target_goal:
                optimal_policy[a_idx, nw_idx] = 0.0
                for q in range(num_quantiles):
                    dist_grid[a_idx, nw_idx, q] = age
                continue

            best_action = 0.0
            best_score  = 1e18
            best_dist   = np.empty(num_quantiles)
            for q in range(num_quantiles):
                best_dist[q] = ages[-1]

            for alloc in actions:
                mu  = alloc * stock_mu_nom + (1 - alloc) * bond_mu_nom
                vol = np.sqrt((alloc * stock_vol) ** 2 + ((1 - alloc) * bond_vol) ** 2)

                next_nws = (nw + yr_savings) * (1 + mu + z_scores * vol)

                # next_dist at a_idx+1: if a_idx+1 is at or past fixed_retirement_age,
                # target_goal for that node is 0, so those nodes are already marked
                # with their age. The interp will correctly read those values.
                mat = np.empty((num_quantiles, num_quantiles))
                for q_idx in range(num_quantiles):
                    interp = np.interp(next_nws, nw_grid, next_dist[:, q_idx])
                    for nw_out in range(num_quantiles):
                        nw_next = next_nws[nw_out]
                        if nw_next >= target_goal and nw < target_goal:
                            denom    = nw_next - nw
                            fraction = (target_goal - nw) / denom if denom > 0 else 0.0
                            fraction = min(max(fraction, 0.0), 1.0)
                            mat[nw_out, q_idx] = age + fraction
                        else:
                            mat[nw_out, q_idx] = interp[nw_out]

                all_out = mat.flatten()
                all_out.sort()

                metric = np.interp(target_percentile, outcome_percentiles, all_out)

                # tie-breaker: protect lower-tail wealth
                lo_idx = min(int((1 - target_percentile) * num_quantiles), num_quantiles - 1)
                safe   = np.sort(next_nws)[lo_idx]
                score  = metric - 0.0001 * (safe / max_nw)

                if score < best_score:
                    best_score  = score
                    best_action = alloc
                    best_dist   = np.interp(quantiles, outcome_percentiles, all_out)

            optimal_policy[a_idx, nw_idx] = best_action
            for q in range(num_quantiles):
                dist_grid[a_idx, nw_idx, q] = best_dist[q]

    return dist_grid, optimal_policy


# ── Top-level simulation ──────────────────────────────────────────────────────

def run_simulation(p: dict):
    max_nw        = 10_000_000
    num_nw_buckets = 101
    nw_grid       = np.linspace(0, max_nw, num_nw_buckets)
    ages          = np.arange(p["current_age"], p["end_age"] + 1)
    total_years   = p["end_age"] - p["current_age"]

    actions               = np.linspace(0.0, 1.0, 11)
    num_quantiles         = p["num_quantiles"]
    target_pct            = p["target_percentile"]
    mode                  = p["mode"]
    fixed_ret_age         = int(p["fixed_retirement_age"])
    ret_bond_vol          = p["retirement_bond_vol"]

    quantiles, z_scores, outcome_percentiles = make_grids(num_quantiles)
    optimal_policy = np.zeros((len(ages), num_nw_buckets))

    # ── Backward induction ───────────────────────────────────────────────────

    if mode == "earliest_retirement":
        dist_grid = np.zeros((len(ages), num_nw_buckets, num_quantiles))
        dist_grid[-1, :, :] = p["end_age"]   # terminal: retire by end_age

        print(
            f"[Mode: earliest_retirement] "
            f"{num_quantiles} quantiles, P{target_pct*100:.0f} target"
        )
        t0 = time.perf_counter()
        dist_grid, optimal_policy = _backward_earliest(
            nw_grid, ages, actions, quantiles, z_scores, outcome_percentiles,
            dist_grid, optimal_policy,
            p["monthly_savings_today"], p["retirement_spend_today"],
            p["stock_mu_nom"], p["stock_vol"], p["bond_mu_nom"], p["bond_vol"],
            p["inflation_rate"], target_pct, max_nw,
        )

    elif mode == "max_wealth":
        if fixed_ret_age not in ages:
            sys.exit(
                f"fixed_retirement_age={fixed_ret_age} is outside "
                f"[{ages[0]}, {ages[-1]}]"
            )

        dist_grid = np.zeros((len(ages), num_nw_buckets, num_quantiles))
        mult_grid = np.zeros((len(ages), num_quantiles))

        print(
            f"[Mode: max_wealth] target retirement age {fixed_ret_age}, "
            f"{num_quantiles} quantiles, P{target_pct*100:.0f} target"
        )
        t0 = time.perf_counter()
        dist_grid, optimal_policy, mult_grid = _backward_max_wealth(
            nw_grid, ages, actions, quantiles, z_scores, outcome_percentiles,
            dist_grid, optimal_policy, mult_grid,
            p["monthly_savings_today"],
            p["stock_mu_nom"], p["stock_vol"], p["bond_mu_nom"], p["bond_vol"],
            p["inflation_rate"], target_pct, max_nw,
            fixed_ret_age,
        )

    elif mode == "semi_retirement":
        if fixed_ret_age not in ages:
            sys.exit(
                f"fixed_retirement_age={fixed_ret_age} is outside "
                f"[{ages[0]}, {ages[-1]}]"
            )

        w_semi = precompute_w_semi(
            ages,
            p["current_age"],
            fixed_ret_age,
            p["end_age"],
            p["retirement_spend_today"],
            p["semi_retirement_monthly_cashflow_today"],
            p["bond_mu_nom"],
            p["inflation_rate"],
        )

        print(
            f"[Mode: semi_retirement] full retirement age {fixed_ret_age}, "
            f"semi_ret_threshold_today=${w_semi[0]:,.0f}, "
            f"{num_quantiles} quantiles, P{target_pct*100:.0f} target"
        )

        # Terminal condition: at fixed_retirement_age every node is already
        # semi-retired (semi_ret happened at or before that age).
        # We fill from current_age up to fixed_retirement_age; nodes at
        # fixed_retirement_age get the age value directly (can always semi-retire
        # by full retirement at the latest).
        dist_grid = np.full((len(ages), num_nw_buckets, num_quantiles), float(fixed_ret_age))

        t0 = time.perf_counter()
        dist_grid, optimal_policy = _backward_semi(
            nw_grid, ages, actions, quantiles, z_scores, outcome_percentiles,
            dist_grid, optimal_policy,
            p["monthly_savings_today"],
            p["stock_mu_nom"], p["stock_vol"], p["bond_mu_nom"], p["bond_vol"],
            p["inflation_rate"], target_pct, max_nw,
            w_semi,
        )

    else:
        sys.exit(
            f"Unknown mode: {mode!r}. "
            "Use 'earliest_retirement', 'max_wealth', or 'semi_retirement'."
        )

    print(f"Backward induction done in {time.perf_counter() - t0:.1f}s")

    # ── Forward simulation ────────────────────────────────────────────────────

    iterations    = p["iterations"]
    all_paths     = np.zeros((iterations, total_years + 1))
    ret_ages_list = []
    ret_nws_list  = []
    rng           = np.random.default_rng()

    # Precompute w_semi array for forward sim (only used in semi_retirement mode)
    if mode == "semi_retirement":
        w_semi_fwd = precompute_w_semi(
            ages,
            p["current_age"],
            fixed_ret_age,
            p["end_age"],
            p["retirement_spend_today"],
            p["semi_retirement_monthly_cashflow_today"],
            p["bond_mu_nom"],
            p["inflation_rate"],
        )
    else:
        w_semi_fwd = None

    for i in range(iterations):
        nw      = float(p["current_nw"])
        all_paths[i, 0] = nw
        ret_age      = None   # semi-retirement age (modes: semi) or full ret age (others)
        full_ret_age_actual = None  # only used in semi_retirement mode
        ret_nw  = None

        for year_idx in range(1, total_years + 1):
            a_idx        = year_idx - 1
            age          = p["current_age"] + a_idx
            years_to_end = p["end_age"] - age

            inf_factor = (1 + p["inflation_rate"]) ** a_idx
            yr_savings = p["monthly_savings_today"] * 12 * inf_factor
            yr_spend   = p["retirement_spend_today"] * 12 * inf_factor

            if mode == "earliest_retirement":
                nominal_floor = sum(
                    yr_spend
                    * (1 + p["inflation_rate"]) ** k
                    / (1 + p["bond_mu_nom"]) ** k
                    for k in range(years_to_end)
                )
                target_goal = nominal_floor * 1.1

                if ret_age is None and nw >= target_goal:
                    ret_age = age
                    ret_nw  = nw

            elif mode == "max_wealth":
                target_goal = None
                if ret_age is None and age >= fixed_ret_age:
                    ret_age = age
                    ret_nw  = nw

            elif mode == "semi_retirement":
                # Check semi-retirement trigger
                if ret_age is None and nw >= w_semi_fwd[a_idx]:
                    ret_age = age
                    ret_nw  = nw
                # Check full retirement trigger
                if ret_age is not None and full_ret_age_actual is None and age >= fixed_ret_age:
                    full_ret_age_actual = age

            # ── Accumulation / semi-retirement / full-retirement phases ──────

            if mode in ("earliest_retirement", "max_wealth"):
                if ret_age is None:
                    # Accumulation: follow optimal policy
                    nw_bucket = int(np.argmin(np.abs(nw_grid - nw)))
                    sw  = optimal_policy[a_idx, nw_bucket]
                    bw  = 1.0 - sw
                    mu  = sw * p["stock_mu_nom"] + bw * p["bond_mu_nom"]
                    vol = np.sqrt((sw * p["stock_vol"]) ** 2 + (bw * p["bond_vol"]) ** 2)
                    ret = rng.normal(mu, vol)
                    nw  = (nw + yr_savings) * (1 + ret)
                else:
                    # Decumulation: 100% bonds
                    ret = rng.normal(p["bond_mu_nom"], ret_bond_vol)
                    nw  = (nw - yr_spend) * (1 + ret)

            elif mode == "semi_retirement":
                if ret_age is None:
                    # Accumulation: follow optimal policy
                    nw_bucket = int(np.argmin(np.abs(nw_grid - nw)))
                    sw  = optimal_policy[a_idx, nw_bucket]
                    bw  = 1.0 - sw
                    mu  = sw * p["stock_mu_nom"] + bw * p["bond_mu_nom"]
                    vol = np.sqrt((sw * p["stock_vol"]) ** 2 + (bw * p["bond_vol"]) ** 2)
                    ret = rng.normal(mu, vol)
                    nw  = (nw + yr_savings) * (1 + ret)

                elif full_ret_age_actual is None:
                    # Semi-retirement: 100% bonds, semi-retirement cashflow
                    semi_cf_annual = p["semi_retirement_monthly_cashflow_today"] * 12 * inf_factor
                    ret = rng.normal(p["bond_mu_nom"], ret_bond_vol)
                    # positive cashflow = net income (adds to nw before growth)
                    nw  = (nw + semi_cf_annual) * (1 + ret)

                else:
                    # Full retirement: 100% bonds, spend retirement_spend_today
                    ret = rng.normal(p["bond_mu_nom"], ret_bond_vol)
                    nw  = (nw - yr_spend) * (1 + ret)

            all_paths[i, year_idx] = max(0.0, nw)

        # Record retirement ages/nw
        if mode == "semi_retirement":
            ret_ages_list.append(ret_age if ret_age is not None else fixed_ret_age)
        else:
            ret_ages_list.append(ret_age if ret_age is not None else p["end_age"])
        ret_nws_list.append(ret_nw if ret_nw is not None else 0.0)

    extra = {}
    if mode == "semi_retirement":
        extra["w_semi"] = w_semi_fwd
        extra["fixed_ret_age"] = fixed_ret_age

    return (
        all_paths,
        np.array(ret_ages_list),
        np.array(ret_nws_list),
        ages,
        optimal_policy,
        dist_grid,
        mode,
        nw_grid,
        extra,
    )


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(paths, ret_ages, ret_nws, timeline, policy, dist_matrix, mode, p, nw_grid, extra=None):
    target_pct = p["target_percentile"]
    fig, axes  = plt.subplots(1, 3, figsize=(20, 6))
    ax1, ax2, ax3 = axes

    # Panel 1: wealth trajectories
    for idx in np.random.choice(len(paths), size=min(100, len(paths)), replace=False):
        ax1.plot(timeline, paths[idx], color="tab:blue", alpha=0.1)
    ax1.plot(timeline, np.median(paths, axis=0), color="black", lw=2, label="Median")
    ax1.plot(
        timeline,
        np.percentile(paths, 5,  axis=0), color="red",   lw=1, ls="--", label="P5/P95"
    )
    ax1.plot(
        timeline,
        np.percentile(paths, 95, axis=0), color="green", lw=1, ls="--"
    )
    ax1.set_title("Wealth Trajectories")
    ax1.set_xlabel("Age")
    ax1.set_ylabel("Net Worth ($)")
    ax1.legend(fontsize=8)

    # Panel 2: policy heatmap
    im2 = ax2.imshow(
        policy.T, aspect="auto", origin="lower",
        extent=[timeline[0], timeline[-1], 0, 10_000_000], cmap="RdYlGn",
    )
    ax2.set_title("Optimal Stock Allocation")
    ax2.set_xlabel("Age")
    fig.colorbar(im2, ax=ax2, label="Stock weight")

    # Panel 3: DP distribution heatmap
    q_idx = min(int(target_pct * dist_matrix.shape[2]), dist_matrix.shape[2] - 1)

    if mode == "earliest_retirement":
        ret_lower  = np.percentile(ret_ages, 5)
        ret_upper  = np.percentile(ret_ages, 95)
        median_age = np.median(ret_ages)
        ax1.axvspan(
            ret_lower, ret_upper, color="orange", alpha=0.2,
            label=f"90% CI ({int(ret_lower)}–{int(ret_upper)})"
        )
        ax1.legend(fontsize=8)

        im3 = ax3.imshow(
            dist_matrix[:, :, q_idx].T, aspect="auto", origin="lower",
            extent=[timeline[0], timeline[-1], 0, 10_000_000], cmap="RdYlGn_r",
        )
        ax3.set_title(f"Predicted P{target_pct*100:.0f} Retirement Age")
        ax3.set_xlabel("Age")
        fig.colorbar(im3, ax=ax3, label="Retirement age")

        summary = (
            f"Median retirement age: {median_age:.1f}   "
            f"P{target_pct*100:.0f} worst-case: {ret_upper:.1f}"
        )

    elif mode == "max_wealth":
        med_nw = np.median(ret_nws[ret_nws > 0])
        p_lo   = np.percentile(ret_nws[ret_nws > 0], (1 - target_pct) * 100)
        p_hi   = np.percentile(ret_nws[ret_nws > 0], target_pct * 100)

        ax1.axvline(p["fixed_retirement_age"], color="orange", lw=2, ls="--",
                    label=f"Target retirement age {p['fixed_retirement_age']}")
        ax1.legend(fontsize=8)

        # Build display layer: pre-retirement = DP quantile, post-retirement = nw_grid value
        ret_a_idx   = int(p["fixed_retirement_age"]) - int(timeline[0])
        display_layer = dist_matrix[:, :, q_idx].copy()
        for a_idx in range(ret_a_idx, len(timeline)):
            for nw_idx in range(display_layer.shape[1]):
                display_layer[a_idx, nw_idx] = nw_grid[nw_idx]

        im3 = ax3.imshow(
            display_layer.T, aspect="auto", origin="lower",
            extent=[timeline[0], timeline[-1], 0, 10_000_000], cmap="RdYlGn",
            vmin=0, vmax=10_000_000,
        )
        ax3.axvline(p["fixed_retirement_age"], color="orange", lw=1, ls="--")
        ax3.set_title(f"Predicted P{target_pct*100:.0f} Nominal Wealth at Retirement")
        ax3.set_xlabel("Age")
        fig.colorbar(im3, ax=ax3, label="Nominal net worth ($)")

        summary = (
            f"Nominal NW at retirement — Median: ${med_nw:,.0f}   "
            f"P{(1-target_pct)*100:.0f}: ${p_lo:,.0f}   "
            f"P{target_pct*100:.0f}: ${p_hi:,.0f}"
        )

    elif mode == "semi_retirement":
        ret_lower  = np.percentile(ret_ages, 5)
        ret_upper  = np.percentile(ret_ages, 95)
        median_age = np.median(ret_ages)
        fixed_ret_age = extra["fixed_ret_age"] if extra else p["fixed_retirement_age"]

        ax1.axvspan(
            ret_lower, ret_upper, color="orange", alpha=0.2,
            label=f"Semi-ret 90% CI ({int(ret_lower)}–{int(ret_upper)})"
        )
        ax1.axvline(fixed_ret_age, color="purple", lw=2, ls="--",
                    label=f"Full retirement age {fixed_ret_age}")
        ax1.legend(fontsize=8)

        im3 = ax3.imshow(
            dist_matrix[:, :, q_idx].T, aspect="auto", origin="lower",
            extent=[timeline[0], timeline[-1], 0, 10_000_000], cmap="RdYlGn_r",
        )
        ax3.axvline(fixed_ret_age, color="purple", lw=1, ls="--")
        ax3.set_title(f"Predicted P{target_pct*100:.0f} Semi-Retirement Age")
        ax3.set_xlabel("Age")
        fig.colorbar(im3, ax=ax3, label="Semi-retirement age")

        summary = (
            f"Median semi-retirement age: {median_age:.1f}   "
            f"P{target_pct*100:.0f} worst-case: {ret_upper:.1f}   "
            f"Full retirement at: {fixed_ret_age}"
        )

    plt.suptitle(f"Mode: {mode}  |  {summary}", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = "results.png"
    plt.savefig(out_path, dpi=150)
    print(f"\n{summary}")
    print(f"Plot saved to {out_path}")
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Retirement portfolio optimiser (DP + Monte Carlo)"
    )
    parser.add_argument(
        "--config", default="config.toml",
        help="Path to TOML config file (default: config.toml)"
    )
    args   = parser.parse_args()

    cfg    = load_config(args.config)
    params = resolve_params(flatten_config(cfg))

    (paths, ret_ages, ret_nws,
     timeline, policy, dist_matrix, mode, nw_grid, extra) = run_simulation(params)

    plot_results(
        paths, ret_ages, ret_nws,
        timeline, policy, dist_matrix, mode, params, nw_grid, extra,
    )


if __name__ == "__main__":
    main()