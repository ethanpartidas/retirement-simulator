"""
simulate.py — Retirement portfolio optimiser.

Two optimisation modes (set in config.toml):

  earliest_retirement  –  find the allocation that minimises retirement age at
                           the configured percentile.  DP stores predicted
                           retirement age at each (age, wealth) node.

  max_wealth           –  fix a target retirement age and maximise nominal net
                           worth at that age.  DP stores predicted nominal wealth
                           at each (age, wealth) node.

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

    else:
        sys.exit(f"Unknown mode: {mode!r}. Use 'earliest_retirement' or 'max_wealth'.")

    print(f"Backward induction done in {time.perf_counter() - t0:.1f}s")

    # ── Forward simulation ────────────────────────────────────────────────────

    iterations    = p["iterations"]
    all_paths     = np.zeros((iterations, total_years + 1))
    ret_ages_list = []
    ret_nws_list  = []
    rng           = np.random.default_rng()

    for i in range(iterations):
        nw      = float(p["current_nw"])
        all_paths[i, 0] = nw
        ret_age = None
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
                    ret_nw  = nw   # nominal NW at retirement

            elif mode == "max_wealth":
                target_goal = None
                if ret_age is None and age >= fixed_ret_age:
                    ret_age = age
                    ret_nw  = nw   # nominal NW at retirement

            # Accumulation phase
            if ret_age is None:
                nw_bucket = int(np.argmin(np.abs(nw_grid - nw)))
                sw  = optimal_policy[a_idx, nw_bucket]
                bw  = 1.0 - sw
                mu  = sw * p["stock_mu_nom"] + bw * p["bond_mu_nom"]
                vol = np.sqrt((sw * p["stock_vol"]) ** 2 + (bw * p["bond_vol"]) ** 2)
                ret = rng.normal(mu, vol)
                nw  = (nw + yr_savings) * (1 + ret)

            # Decumulation phase — use retirement_bond_vol
            else:
                ret = rng.normal(p["bond_mu_nom"], ret_bond_vol)
                nw  = (nw - yr_spend) * (1 + ret)

            all_paths[i, year_idx] = max(0.0, nw)

        ret_ages_list.append(ret_age if ret_age is not None else p["end_age"])
        ret_nws_list.append(ret_nw  if ret_nw  is not None else 0.0)

    return (
        all_paths,
        np.array(ret_ages_list),
        np.array(ret_nws_list),
        ages,
        optimal_policy,
        dist_grid,
        mode,
        nw_grid,
    )


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(paths, ret_ages, ret_nws, timeline, policy, dist_matrix, mode, p, nw_grid):
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

    else:  # max_wealth
        med_nw = np.median(ret_nws[ret_nws > 0])
        p_lo   = np.percentile(ret_nws[ret_nws > 0], (1 - target_pct) * 100)
        p_hi   = np.percentile(ret_nws[ret_nws > 0], target_pct * 100)

        ax1.axvline(p["fixed_retirement_age"], color="orange", lw=2, ls="--",
                    label=f"Target retirement age {p['fixed_retirement_age']}")
        ax1.legend(fontsize=8)

        # Build display layer: pre-retirement = DP quantile, post-retirement = nw_grid value
        ret_a_idx   = int(p["fixed_retirement_age"]) - int(timeline[0])
        display_layer = dist_matrix[:, :, q_idx].copy()   # shape (ages, nw_buckets)
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
     timeline, policy, dist_matrix, mode, nw_grid) = run_simulation(params)

    plot_results(
        paths, ret_ages, ret_nws,
        timeline, policy, dist_matrix, mode, params, nw_grid
    )


if __name__ == "__main__":
    main()