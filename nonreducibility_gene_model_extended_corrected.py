#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extended analysis script for:
"Non-reducibility of structural effects in gene regulatory dynamics"

Corrected version:
1. Structural sweeps now use an adaptive search domain so high-alpha fixed points
   are not spuriously missed outside [0, 3] x [0, 3].
2. Zero-total structural conditions are automatically rechecked on a larger domain.
3. Additional CSV and figure outputs are produced for these rechecks.

All numerical outputs and figures are saved to the Desktop.
"""

from __future__ import annotations

import os
import math
import json
import time
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.integrate import solve_ivp

# =========================
# Output settings
# =========================

OUTDIR = os.path.expanduser("~/Desktop/nonreducibility_results_extended_corrected")
LOGFILE = os.path.join(OUTDIR, "analysis.log")

# =========================
# Core methods settings
# =========================

SEARCH_X_MIN = 0.0
SEARCH_X_MAX = 3.0
SEARCH_Y_MIN = 0.0
SEARCH_Y_MAX = 3.0

DEFAULT_GRID_N = 50
ROBUST_GRID_LIST = [50, 70]
ETA = 1.0e-8
EPSILON_DEFAULT = 0.2
ROBUST_EPSILON_LIST = [0.15, 0.20, 0.25]

ALPHA_RANGE = np.linspace(0.5, 5.0, 20)
N_RANGE = np.linspace(1.0, 4.0, 20)
STRUCTURAL_ALPHA = 2.0
STRUCTURAL_N = 2.0
RHO_LIST = [0.05, 0.10, 0.15]

# Added analyses
STRUCTURAL_RHO_FINE = np.linspace(0.0, 0.30, 61)  # step 0.005
STRUCTURAL_ALPHA_SWEEP = np.linspace(0.5, 5.0, 25)
STRUCTURAL_N_SWEEP = np.linspace(1.0, 4.0, 25)
STRUCTURAL_RHO_SWEEP = np.linspace(0.0, 0.30, 31)

BASIN_GRID_N = 60
BASIN_RHO_LIST = [0.00, 0.05, 0.10, 0.15]
BASIN_TMAX = 80.0
BASIN_TOL = 0.15

ROOT_METHOD = "hybr"
ROOT_TOL = 1.0e-12
DOMAIN_CHECK_MAX = 5.0
DOMAIN_CHECK_GRID = 60
ZERO_RECHECK_GRID = 80
ZERO_RECHECK_MARGIN = 1.5
RANDOM_SEED = 42


@dataclass
class FixedPointRecord:
    system: str
    alpha: float
    n: float
    rho: float
    x: float
    y: float
    stable: bool
    classification: str
    eig1_real: float
    eig1_imag: float
    eig2_real: float
    eig2_imag: float


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOGFILE, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def make_grid(grid_n: int,
              xmin: float = SEARCH_X_MIN,
              xmax: float = SEARCH_X_MAX,
              ymin: float = SEARCH_Y_MIN,
              ymax: float = SEARCH_Y_MAX) -> np.ndarray:
    xs = np.linspace(xmin, xmax, grid_n)
    ys = np.linspace(ymin, ymax, grid_n)
    return np.array([(x, y) for x in xs for y in ys], dtype=float)


def in_domain(x: float, y: float,
              xmin: float = SEARCH_X_MIN,
              xmax: float = SEARCH_X_MAX,
              ymin: float = SEARCH_Y_MIN,
              ymax: float = SEARCH_Y_MAX) -> bool:
    return (xmin <= x <= xmax) and (ymin <= y <= ymax)


def euclidean_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sqrt(np.sum((p - q) ** 2)))


def unique_points(points: List[np.ndarray], epsilon: float) -> List[np.ndarray]:
    unique: List[np.ndarray] = []
    for p in points:
        is_new = True
        for q in unique:
            if euclidean_distance(p, q) < epsilon:
                is_new = False
                break
        if is_new:
            unique.append(p)
    return unique


def adaptive_structural_domain(alpha: float, rho: float, n: float) -> float:
    """
    Heuristic upper bound for structural fixed-point search.
    High-alpha regimes can place stable fixed points outside [0, 3] x [0, 3].
    This function expands the search box with alpha and modestly with rho.
    """
    base = max(3.0, alpha + ZERO_RECHECK_MARGIN)
    rho_term = 2.0 * rho
    n_term = 0.25 * max(0.0, n - 2.0)
    return float(max(3.0, base + rho_term + n_term))


# =========================
# Model definitions
# =========================


def baseline_residual(z: np.ndarray, alpha: float, n: float) -> np.ndarray:
    x, y = z
    return np.array([
        alpha / (1.0 + y ** n) - x,
        alpha / (1.0 + x ** n) - y,
    ], dtype=float)


def structural_residual(z: np.ndarray, alpha: float, n: float, rho: float) -> np.ndarray:
    x, y = z
    return np.array([
        alpha / (1.0 + y ** n) + rho * (x ** 2 / (1.0 + x ** 2)) - x,
        alpha / (1.0 + x ** n) - y,
    ], dtype=float)


def baseline_rhs(t: float, z: np.ndarray, alpha: float, n: float) -> np.ndarray:
    return baseline_residual(z, alpha, n)


def structural_rhs(t: float, z: np.ndarray, alpha: float, n: float, rho: float) -> np.ndarray:
    return structural_residual(z, alpha, n, rho)


def baseline_jacobian(z: np.ndarray, alpha: float, n: float) -> np.ndarray:
    x, y = z
    j11 = -1.0
    j12 = -alpha * n * (y ** (n - 1.0)) / (1.0 + y ** n) ** 2
    j21 = -alpha * n * (x ** (n - 1.0)) / (1.0 + x ** n) ** 2
    j22 = -1.0
    return np.array([[j11, j12], [j21, j22]], dtype=float)


def structural_jacobian(z: np.ndarray, alpha: float, n: float, rho: float) -> np.ndarray:
    x, y = z
    j11 = rho * (2.0 * x / (1.0 + x ** 2) ** 2) - 1.0
    j12 = -alpha * n * (y ** (n - 1.0)) / (1.0 + y ** n) ** 2
    j21 = -alpha * n * (x ** (n - 1.0)) / (1.0 + x ** n) ** 2
    j22 = -1.0
    return np.array([[j11, j12], [j21, j22]], dtype=float)


# =========================
# Fixed-point analysis
# =========================


def solve_multistart(system: str,
                     alpha: float,
                     n: float,
                     rho: float = 0.0,
                     grid_n: int = DEFAULT_GRID_N,
                     epsilon: float = EPSILON_DEFAULT,
                     domain_max: float = 3.0) -> List[np.ndarray]:
    pts = make_grid(grid_n, 0.0, domain_max, 0.0, domain_max)
    solutions: List[np.ndarray] = []

    if system == "baseline":
        fun = lambda z: baseline_residual(z, alpha, n)
    elif system == "structural":
        fun = lambda z: structural_residual(z, alpha, n, rho)
    else:
        raise ValueError(f"Unknown system: {system}")

    for guess in pts:
        try:
            res = root(fun, x0=guess, method=ROOT_METHOD, tol=ROOT_TOL)
        except Exception:
            continue

        if not res.success:
            continue

        z = np.asarray(res.x, dtype=float)
        if not np.all(np.isfinite(z)):
            continue

        x, y = float(z[0]), float(z[1])
        if not in_domain(x, y, 0.0, domain_max, 0.0, domain_max):
            continue

        residual = fun(z)
        residual_norm = float(np.sqrt(np.sum(residual ** 2)))
        if residual_norm >= ETA:
            continue

        solutions.append(np.array([x, y], dtype=float))

    return unique_points(solutions, epsilon=epsilon)


def eigenvalues_and_stability(system: str,
                              z: np.ndarray,
                              alpha: float,
                              n: float,
                              rho: float = 0.0) -> Tuple[np.ndarray, bool]:
    if system == "baseline":
        J = baseline_jacobian(z, alpha, n)
    elif system == "structural":
        J = structural_jacobian(z, alpha, n, rho)
    else:
        raise ValueError(f"Unknown system: {system}")

    eigvals = np.linalg.eigvals(J)
    stable = bool(np.all(np.real(eigvals) < 0.0))
    return eigvals, stable


def classify_stable_points(stable_points: List[np.ndarray], epsilon: float) -> Dict[int, str]:
    classes: Dict[int, str] = {}
    for i, p in enumerate(stable_points):
        x_i, y_i = float(p[0]), float(p[1])
        if abs(x_i - y_i) < epsilon:
            classes[i] = "diagonal"
            continue

        paired = False
        for j, q in enumerate(stable_points):
            if i == j:
                continue
            x_j, y_j = float(q[0]), float(q[1])
            mirror_dist = math.sqrt((x_j - y_i) ** 2 + (y_j - x_i) ** 2)
            if mirror_dist < epsilon:
                paired = True
                break
        classes[i] = "paired" if paired else "unpaired"
    return classes


def analyze_condition(system: str,
                      alpha: float,
                      n: float,
                      rho: float = 0.0,
                      grid_n: int = DEFAULT_GRID_N,
                      epsilon: float = EPSILON_DEFAULT,
                      domain_max: float = 3.0) -> Tuple[List[FixedPointRecord], Dict[str, int], List[np.ndarray]]:
    points = solve_multistart(system, alpha, n, rho, grid_n, epsilon, domain_max)

    stable_points: List[np.ndarray] = []
    eig_store: List[np.ndarray] = []
    for p in points:
        eigvals, stable = eigenvalues_and_stability(system, p, alpha, n, rho)
        if stable:
            stable_points.append(p)
            eig_store.append(eigvals)

    classes = classify_stable_points(stable_points, epsilon)
    records: List[FixedPointRecord] = []
    counts = {"diagonal": 0, "paired": 0, "unpaired": 0, "stable_total": 0}

    for i, p in enumerate(stable_points):
        eigvals = eig_store[i]
        cls = classes[i]
        counts[cls] += 1
        counts["stable_total"] += 1
        records.append(FixedPointRecord(
            system=system,
            alpha=float(alpha),
            n=float(n),
            rho=float(rho),
            x=float(p[0]),
            y=float(p[1]),
            stable=True,
            classification=cls,
            eig1_real=float(np.real(eigvals[0])),
            eig1_imag=float(np.imag(eigvals[0])),
            eig2_real=float(np.real(eigvals[1])),
            eig2_imag=float(np.imag(eigvals[1])),
        ))

    return records, counts, stable_points


# =========================
# Existing analyses
# =========================


def run_baseline_sweep() -> Tuple[pd.DataFrame, pd.DataFrame]:
    logging.info("Running baseline parameter sweep...")
    summary_rows, detail_rows = [], []
    total_conditions = len(ALPHA_RANGE) * len(N_RANGE)
    done = 0

    for alpha in ALPHA_RANGE:
        for n in N_RANGE:
            records, counts, _ = analyze_condition("baseline", float(alpha), float(n))
            detail_rows.extend(asdict(rec) for rec in records)
            summary_rows.append({
                "alpha": float(alpha),
                "n": float(n),
                "stable_diagonal_count": counts["diagonal"],
                "stable_paired_count": counts["paired"],
                "stable_unpaired_count": counts["unpaired"],
                "stable_total_count": counts["stable_total"],
                "domain_max_used": 3.0,
            })
            done += 1
            if done % 25 == 0 or done == total_conditions:
                logging.info("Baseline sweep progress: %d / %d", done, total_conditions)

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def run_structural_evaluation() -> Tuple[pd.DataFrame, pd.DataFrame]:
    logging.info("Running structural evaluation...")
    summary_rows, detail_rows = [], []
    for rho in RHO_LIST:
        dmax = adaptive_structural_domain(STRUCTURAL_ALPHA, rho, STRUCTURAL_N)
        records, counts, _ = analyze_condition("structural", STRUCTURAL_ALPHA, STRUCTURAL_N, float(rho), domain_max=dmax)
        detail_rows.extend(asdict(rec) for rec in records)
        summary_rows.append({
            "rho": float(rho),
            "alpha": float(STRUCTURAL_ALPHA),
            "n": float(STRUCTURAL_N),
            "stable_diagonal_count": counts["diagonal"],
            "stable_paired_count": counts["paired"],
            "stable_unpaired_count": counts["unpaired"],
            "stable_total_count": counts["stable_total"],
            "domain_max_used": dmax,
        })
    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def run_robustness_checks() -> pd.DataFrame:
    logging.info("Running robustness checks...")
    rows = []

    for eps in ROBUST_EPSILON_LIST:
        for grid_n in ROBUST_GRID_LIST:
            for rho in RHO_LIST:
                dmax = adaptive_structural_domain(STRUCTURAL_ALPHA, rho, STRUCTURAL_N)
                _, counts, _ = analyze_condition(
                    "structural", STRUCTURAL_ALPHA, STRUCTURAL_N, float(rho),
                    grid_n=grid_n, epsilon=eps, domain_max=dmax
                )
                rows.append({
                    "system": "structural",
                    "epsilon": float(eps),
                    "grid_n": int(grid_n),
                    "rho": float(rho),
                    "alpha": float(STRUCTURAL_ALPHA),
                    "n": float(STRUCTURAL_N),
                    "stable_diagonal_count": counts["diagonal"],
                    "stable_paired_count": counts["paired"],
                    "stable_unpaired_count": counts["unpaired"],
                    "stable_total_count": counts["stable_total"],
                    "domain_max_used": dmax,
                })

    for eps in ROBUST_EPSILON_LIST:
        for grid_n in ROBUST_GRID_LIST:
            _, counts, _ = analyze_condition(
                "baseline", STRUCTURAL_ALPHA, STRUCTURAL_N, 0.0, grid_n=grid_n, epsilon=eps
            )
            rows.append({
                "system": "baseline",
                "epsilon": float(eps),
                "grid_n": int(grid_n),
                "rho": 0.0,
                "alpha": float(STRUCTURAL_ALPHA),
                "n": float(STRUCTURAL_N),
                "stable_diagonal_count": counts["diagonal"],
                "stable_paired_count": counts["paired"],
                "stable_unpaired_count": counts["unpaired"],
                "stable_total_count": counts["stable_total"],
                "domain_max_used": 3.0,
            })

    return pd.DataFrame(rows)


def run_domain_check() -> pd.DataFrame:
    logging.info("Running representative larger-domain checks...")
    rows = []
    representative_baseline_conditions = [(0.5, 1.0), (2.0, 2.0), (5.0, 4.0)]

    for alpha, n in representative_baseline_conditions:
        _, cnt_small, _ = analyze_condition("baseline", alpha, n, 0.0, grid_n=DEFAULT_GRID_N, domain_max=3.0)
        _, cnt_large, _ = analyze_condition("baseline", alpha, n, 0.0, grid_n=DOMAIN_CHECK_GRID, domain_max=DOMAIN_CHECK_MAX)
        rows.append({
            "system": "baseline",
            "alpha": alpha,
            "n": n,
            "rho": 0.0,
            "domain_max_small": 3.0,
            "domain_max_large": DOMAIN_CHECK_MAX,
            "stable_total_small": cnt_small["stable_total"],
            "stable_total_large": cnt_large["stable_total"],
            "stable_unpaired_small": cnt_small["unpaired"],
            "stable_unpaired_large": cnt_large["unpaired"],
        })

    for rho in RHO_LIST:
        dsmall = adaptive_structural_domain(STRUCTURAL_ALPHA, rho, STRUCTURAL_N)
        dlarge = max(DOMAIN_CHECK_MAX, dsmall + 1.0)
        _, cnt_small, _ = analyze_condition("structural", STRUCTURAL_ALPHA, STRUCTURAL_N, rho, grid_n=DEFAULT_GRID_N, domain_max=dsmall)
        _, cnt_large, _ = analyze_condition("structural", STRUCTURAL_ALPHA, STRUCTURAL_N, rho, grid_n=DOMAIN_CHECK_GRID, domain_max=dlarge)
        rows.append({
            "system": "structural",
            "alpha": STRUCTURAL_ALPHA,
            "n": STRUCTURAL_N,
            "rho": rho,
            "domain_max_small": dsmall,
            "domain_max_large": dlarge,
            "stable_total_small": cnt_small["stable_total"],
            "stable_total_large": cnt_large["stable_total"],
            "stable_unpaired_small": cnt_small["unpaired"],
            "stable_unpaired_large": cnt_large["unpaired"],
        })

    return pd.DataFrame(rows)


# =========================
# New analyses requested
# =========================


def run_structural_alpha_rho_sweep() -> Tuple[pd.DataFrame, pd.DataFrame]:
    logging.info("Running structural alpha-rho sweep at n = %.3f ...", STRUCTURAL_N)
    summary_rows, detail_rows = [], []
    total = len(STRUCTURAL_ALPHA_SWEEP) * len(STRUCTURAL_RHO_SWEEP)
    done = 0

    for alpha in STRUCTURAL_ALPHA_SWEEP:
        for rho in STRUCTURAL_RHO_SWEEP:
            dmax = adaptive_structural_domain(alpha, rho, STRUCTURAL_N)
            records, counts, _ = analyze_condition(
                "structural", float(alpha), STRUCTURAL_N, float(rho), domain_max=dmax
            )
            detail_rows.extend(asdict(rec) for rec in records)
            summary_rows.append({
                "alpha": float(alpha),
                "n": float(STRUCTURAL_N),
                "rho": float(rho),
                "stable_diagonal_count": counts["diagonal"],
                "stable_paired_count": counts["paired"],
                "stable_unpaired_count": counts["unpaired"],
                "stable_total_count": counts["stable_total"],
                "domain_max_used": dmax,
            })
            done += 1
            if done % 50 == 0 or done == total:
                logging.info("Structural alpha-rho sweep progress: %d / %d", done, total)

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def run_structural_n_rho_sweep() -> Tuple[pd.DataFrame, pd.DataFrame]:
    logging.info("Running structural n-rho sweep at alpha = %.3f ...", STRUCTURAL_ALPHA)
    summary_rows, detail_rows = [], []
    total = len(STRUCTURAL_N_SWEEP) * len(STRUCTURAL_RHO_SWEEP)
    done = 0

    for n in STRUCTURAL_N_SWEEP:
        for rho in STRUCTURAL_RHO_SWEEP:
            dmax = adaptive_structural_domain(STRUCTURAL_ALPHA, rho, n)
            records, counts, _ = analyze_condition(
                "structural", STRUCTURAL_ALPHA, float(n), float(rho), domain_max=dmax
            )
            detail_rows.extend(asdict(rec) for rec in records)
            summary_rows.append({
                "alpha": float(STRUCTURAL_ALPHA),
                "n": float(n),
                "rho": float(rho),
                "stable_diagonal_count": counts["diagonal"],
                "stable_paired_count": counts["paired"],
                "stable_unpaired_count": counts["unpaired"],
                "stable_total_count": counts["stable_total"],
                "domain_max_used": dmax,
            })
            done += 1
            if done % 50 == 0 or done == total:
                logging.info("Structural n-rho sweep progress: %d / %d", done, total)

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def run_structural_rho_bifurcation_scan() -> pd.DataFrame:
    logging.info("Running fine rho scan for transition tracking ...")
    rows = []

    for rho in STRUCTURAL_RHO_FINE:
        dmax = adaptive_structural_domain(STRUCTURAL_ALPHA, rho, STRUCTURAL_N)
        records, counts, _ = analyze_condition("structural", STRUCTURAL_ALPHA, STRUCTURAL_N, float(rho), grid_n=70, domain_max=dmax)
        if len(records) == 0:
            rows.append({
                "rho": float(rho),
                "alpha": float(STRUCTURAL_ALPHA),
                "n": float(STRUCTURAL_N),
                "branch_index": -1,
                "x": np.nan,
                "y": np.nan,
                "classification": "none",
                "eig1_real": np.nan,
                "eig2_real": np.nan,
                "stable_total_count": counts["stable_total"],
                "domain_max_used": dmax,
            })
        else:
            records_sorted = sorted(records, key=lambda r: (r.x, r.y))
            for bi, rec in enumerate(records_sorted):
                rows.append({
                    "rho": float(rho),
                    "alpha": float(STRUCTURAL_ALPHA),
                    "n": float(STRUCTURAL_N),
                    "branch_index": int(bi),
                    "x": rec.x,
                    "y": rec.y,
                    "classification": rec.classification,
                    "eig1_real": rec.eig1_real,
                    "eig2_real": rec.eig2_real,
                    "stable_total_count": counts["stable_total"],
                    "domain_max_used": dmax,
                })

    return pd.DataFrame(rows)


def integrate_to_attractor(system: str,
                           z0: np.ndarray,
                           alpha: float,
                           n: float,
                           rho: float,
                           stable_points: List[np.ndarray]) -> Tuple[np.ndarray, str, int]:
    if system == "baseline":
        rhs = lambda t, z: baseline_rhs(t, z, alpha, n)
    else:
        rhs = lambda t, z: structural_rhs(t, z, alpha, n, rho)

    sol = solve_ivp(rhs, (0.0, BASIN_TMAX), z0, method="RK45", rtol=1e-7, atol=1e-9)
    zT = sol.y[:, -1]

    if np.any(~np.isfinite(zT)):
        return np.array([np.nan, np.nan]), "integration_failed", -1

    if len(stable_points) == 0:
        return zT, "no_stable_point", -1

    distances = [euclidean_distance(zT, p) for p in stable_points]
    best_idx = int(np.argmin(distances))
    if distances[best_idx] < BASIN_TOL:
        return zT, "assigned", best_idx
    return zT, "unassigned", -1


def run_basin_analysis() -> Tuple[pd.DataFrame, pd.DataFrame]:
    logging.info("Running basin-of-attraction analysis ...")
    basin_rows = []
    attractor_rows = []
    init_grid = make_grid(BASIN_GRID_N)

    for rho in BASIN_RHO_LIST:
        system = "baseline" if abs(rho) < 1e-12 else "structural"
        alpha = STRUCTURAL_ALPHA
        n = STRUCTURAL_N
        rho_eff = 0.0 if system == "baseline" else float(rho)
        dmax = 3.0 if system == "baseline" else adaptive_structural_domain(alpha, rho_eff, n)
        records, counts, stable_points = analyze_condition(system, alpha, n, rho_eff, grid_n=70, domain_max=dmax)
        classes = classify_stable_points(stable_points, EPSILON_DEFAULT) if len(stable_points) > 0 else {}

        for idx, p in enumerate(stable_points):
            attractor_rows.append({
                "system": system,
                "alpha": alpha,
                "n": n,
                "rho": rho_eff,
                "attractor_index": idx,
                "x": float(p[0]),
                "y": float(p[1]),
                "classification": classes.get(idx, "unknown"),
                "domain_max_used": dmax,
            })

        for z0 in init_grid:
            zT, assign_status, attractor_index = integrate_to_attractor(system, z0, alpha, n, rho_eff, stable_points)
            basin_rows.append({
                "system": system,
                "alpha": alpha,
                "n": n,
                "rho": rho_eff,
                "x0": float(z0[0]),
                "y0": float(z0[1]),
                "xT": float(zT[0]) if np.isfinite(zT[0]) else np.nan,
                "yT": float(zT[1]) if np.isfinite(zT[1]) else np.nan,
                "assignment_status": assign_status,
                "attractor_index": attractor_index,
                "stable_total_count": counts["stable_total"],
                "domain_max_used": dmax,
            })

    return pd.DataFrame(basin_rows), pd.DataFrame(attractor_rows)


def recheck_zero_total_structural_conditions(df_alpha_rho_summary: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Re-run only structural alpha-rho conditions that produced stable_total_count == 0
    using an enlarged domain and denser grid, to distinguish true absence from domain artifact.
    """
    logging.info("Rechecking structural alpha-rho conditions with stable_total_count == 0 ...")
    zero_df = df_alpha_rho_summary[df_alpha_rho_summary["stable_total_count"] == 0].copy()

    summary_rows = []
    detail_rows = []

    total = len(zero_df)
    done = 0

    for _, row in zero_df.iterrows():
        alpha = float(row["alpha"])
        rho = float(row["rho"])
        n = float(row["n"])
        old_dmax = float(row["domain_max_used"])
        new_dmax = max(old_dmax + 1.5, adaptive_structural_domain(alpha, rho, n) + 1.5)

        records, counts, _ = analyze_condition(
            "structural",
            alpha,
            n,
            rho,
            grid_n=ZERO_RECHECK_GRID,
            epsilon=EPSILON_DEFAULT,
            domain_max=new_dmax,
        )

        detail_rows.extend(asdict(rec) for rec in records)
        summary_rows.append({
            "alpha": alpha,
            "n": n,
            "rho": rho,
            "old_domain_max": old_dmax,
            "new_domain_max": new_dmax,
            "stable_diagonal_count": counts["diagonal"],
            "stable_paired_count": counts["paired"],
            "stable_unpaired_count": counts["unpaired"],
            "stable_total_count": counts["stable_total"],
        })

        done += 1
        if done % 25 == 0 or done == total:
            logging.info("Zero-total recheck progress: %d / %d", done, total)

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


# =========================
# Plotting
# =========================


def save_fig1_baseline_heatmap(df_summary: pd.DataFrame) -> None:
    pivot = df_summary.pivot(index="n", columns="alpha", values="stable_unpaired_count").sort_index(ascending=True)
    plt.figure(figsize=(8, 6))
    im = plt.imshow(pivot.values, origin="lower", aspect="auto",
                    extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()])
    plt.colorbar(im, label="Number of unpaired stable fixed points")
    plt.xlabel("alpha")
    plt.ylabel("n")
    plt.title("Figure 1. Baseline system: unpaired stable fixed points")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig1_baseline_unpaired_heatmap.png"), dpi=300)
    plt.close()


def save_fig2_robustness(df_robust: pd.DataFrame) -> None:
    structural = df_robust[df_robust["system"] == "structural"].copy()
    structural["setting"] = structural.apply(
        lambda r: f"eps={r['epsilon']:.2f}, grid={int(r['grid_n'])}x{int(r['grid_n'])}", axis=1
    )
    plt.figure(figsize=(9, 6))
    for setting in structural["setting"].unique():
        sub = structural[structural["setting"] == setting].sort_values("rho")
        plt.plot(sub["rho"].values, sub["stable_unpaired_count"].values, marker="o", label=setting)
    plt.xlabel("rho")
    plt.ylabel("Number of unpaired stable fixed points")
    plt.title("Figure 2. Structural robustness of unpaired attractor count")
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig2_robustness_unpaired_counts.png"), dpi=300, bbox_inches="tight")
    plt.close()


def save_fig3_structural_locations(df_struct_detail: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 6))
    for rho in sorted(df_struct_detail["rho"].unique()):
        sub = df_struct_detail[df_struct_detail["rho"] == rho]
        plt.scatter(sub["x"], sub["y"], label=f"rho = {rho:.2f}", s=70)
    plt.xlabel("x*")
    plt.ylabel("y*")
    plt.title("Figure 3. Stable fixed-point locations in structural system")
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig3_structural_fixed_point_locations.png"), dpi=300, bbox_inches="tight")
    plt.close()


def save_fig4_alpha_rho_heatmap(df_summary: pd.DataFrame) -> None:
    pivot = df_summary.pivot(index="alpha", columns="rho", values="stable_unpaired_count").sort_index(ascending=True)
    plt.figure(figsize=(8.5, 6))
    im = plt.imshow(pivot.values, origin="lower", aspect="auto",
                    extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()])
    plt.colorbar(im, label="Number of unpaired stable fixed points")
    plt.xlabel("rho")
    plt.ylabel("alpha")
    plt.title("Figure 4. Structural alpha-rho sweep at fixed n")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig4_structural_alpha_rho_unpaired_heatmap.png"), dpi=300)
    plt.close()


def save_fig5_n_rho_heatmap(df_summary: pd.DataFrame) -> None:
    pivot = df_summary.pivot(index="n", columns="rho", values="stable_unpaired_count").sort_index(ascending=True)
    plt.figure(figsize=(8.5, 6))
    im = plt.imshow(pivot.values, origin="lower", aspect="auto",
                    extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()])
    plt.colorbar(im, label="Number of unpaired stable fixed points")
    plt.xlabel("rho")
    plt.ylabel("n")
    plt.title("Figure 5. Structural n-rho sweep at fixed alpha")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig5_structural_n_rho_unpaired_heatmap.png"), dpi=300)
    plt.close()


def save_fig6_bifurcation(df_bif: pd.DataFrame) -> None:
    valid = df_bif[df_bif["branch_index"] >= 0].copy()
    plt.figure(figsize=(8.5, 6))
    if len(valid) > 0:
        plt.scatter(valid["rho"], valid["x"], s=18, label="x*", alpha=0.85)
        plt.scatter(valid["rho"], valid["y"], s=18, label="y*", alpha=0.85)
    plt.xlabel("rho")
    plt.ylabel("Stable fixed-point coordinate")
    plt.title("Figure 6. Fine rho scan of stable fixed-point coordinates")
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig6_structural_rho_bifurcation_scan.png"), dpi=300, bbox_inches="tight")
    plt.close()


def save_fig7_basin_maps(df_basin: pd.DataFrame, df_attractors: pd.DataFrame) -> None:
    for rho in sorted(df_basin["rho"].unique()):
        sub = df_basin[df_basin["rho"] == rho].copy()
        attrs = df_attractors[df_attractors["rho"] == rho].copy()
        if len(sub) == 0:
            continue

        plt.figure(figsize=(7, 6))
        plt.scatter(sub["x0"], sub["y0"], c=sub["attractor_index"], s=10)
        if len(attrs) > 0:
            plt.scatter(attrs["x"], attrs["y"], marker="x", s=120, linewidths=2)
        plt.xlabel("Initial x0")
        plt.ylabel("Initial y0")
        plt.title(f"Figure 7. Basin map at rho = {rho:.2f}")
        plt.tight_layout()
        fname = f"fig7_basin_map_rho_{rho:.2f}.png"
        plt.savefig(os.path.join(OUTDIR, fname), dpi=300)
        plt.close()


def save_fig8_zero_recheck(df_recheck: pd.DataFrame) -> None:
    if len(df_recheck) == 0:
        return
    pivot = df_recheck.pivot(index="alpha", columns="rho", values="stable_total_count").sort_index(ascending=True)
    plt.figure(figsize=(8.5, 6))
    im = plt.imshow(pivot.values, origin="lower", aspect="auto",
                    extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()])
    plt.colorbar(im, label="Stable fixed-point count after recheck")
    plt.xlabel("rho")
    plt.ylabel("alpha")
    plt.title("Figure 8. Rechecked zero-total structural alpha-rho conditions")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig8_zero_total_recheck_heatmap.png"), dpi=300)
    plt.close()


# =========================
# Reporting
# =========================


def make_results_summary(df_baseline_summary: pd.DataFrame,
                         df_struct_summary: pd.DataFrame,
                         df_struct_detail: pd.DataFrame,
                         df_robust: pd.DataFrame,
                         df_domain_check: pd.DataFrame,
                         df_alpha_rho: pd.DataFrame,
                         df_n_rho: pd.DataFrame,
                         df_bif: pd.DataFrame,
                         df_basin: pd.DataFrame,
                         df_attractors: pd.DataFrame,
                         df_zero_recheck: pd.DataFrame) -> pd.DataFrame:
    rows = []

    rows.append({
        "section": "3.1 baseline sweep",
        "metric": "number of baseline parameter combinations",
        "value": int(len(df_baseline_summary)),
    })
    rows.append({
        "section": "3.1 baseline sweep",
        "metric": "maximum baseline unpaired stable fixed points across parameter sweep",
        "value": int(df_baseline_summary["stable_unpaired_count"].max()),
    })
    rows.append({
        "section": "3.1 baseline sweep",
        "metric": "number of baseline parameter combinations with any unpaired stable fixed point",
        "value": int((df_baseline_summary["stable_unpaired_count"] > 0).sum()),
    })

    for rho in sorted(df_struct_summary["rho"].unique()):
        sub = df_struct_detail[df_struct_detail["rho"] == rho].copy().sort_values(["x", "y"])
        if len(sub) > 0:
            row = sub.iloc[0]
            rows.append({
                "section": "3.2 structural evaluation",
                "metric": f"stable fixed point at rho={rho:.2f}",
                "value": f"({row['x']:.6f}, {row['y']:.6f})",
            })
            rows.append({
                "section": "3.2 structural evaluation",
                "metric": f"stable unpaired count at rho={rho:.2f}",
                "value": int(df_struct_summary[df_struct_summary["rho"] == rho]["stable_unpaired_count"].iloc[0]),
            })

    rows.append({
        "section": "3.3 robustness",
        "metric": "minimum structural unpaired stable count across robustness settings",
        "value": int(df_robust[df_robust["system"] == "structural"]["stable_unpaired_count"].min()),
    })
    rows.append({
        "section": "3.3 robustness",
        "metric": "maximum structural unpaired stable count across robustness settings",
        "value": int(df_robust[df_robust["system"] == "structural"]["stable_unpaired_count"].max()),
    })
    rows.append({
        "section": "3.3 robustness",
        "metric": "maximum baseline unpaired stable count across robustness settings",
        "value": int(df_robust[df_robust["system"] == "baseline"]["stable_unpaired_count"].max()),
    })

    rows.append({
        "section": "3.4 structural alpha-rho sweep",
        "metric": "number of alpha-rho conditions with any unpaired stable fixed point",
        "value": int((df_alpha_rho["stable_unpaired_count"] > 0).sum()),
    })
    rows.append({
        "section": "3.4 structural alpha-rho sweep",
        "metric": "number of alpha-rho conditions with zero stable fixed points before recheck",
        "value": int((df_alpha_rho["stable_total_count"] == 0).sum()),
    })

    if len(df_zero_recheck) > 0:
        rows.append({
            "section": "3.4 structural alpha-rho sweep",
            "metric": "number of zero-total alpha-rho conditions with stable fixed points after recheck",
            "value": int((df_zero_recheck["stable_total_count"] > 0).sum()),
        })

    rows.append({
        "section": "3.5 structural n-rho sweep",
        "metric": "number of n-rho conditions with any unpaired stable fixed point",
        "value": int((df_n_rho["stable_unpaired_count"] > 0).sum()),
    })

    valid_bif = df_bif[df_bif["branch_index"] >= 0].copy()
    if len(valid_bif) > 0:
        rows.append({
            "section": "3.6 fine rho scan",
            "metric": "minimum rho with any stable fixed point in fine scan",
            "value": float(valid_bif["rho"].min()),
        })
        rows.append({
            "section": "3.6 fine rho scan",
            "metric": "maximum rho with any unpaired stable fixed point in fine scan",
            "value": float(valid_bif[valid_bif["classification"] == "unpaired"]["rho"].max()) if np.any(valid_bif["classification"] == "unpaired") else np.nan,
        })

    basin_summary = (
        df_basin.groupby(["system", "rho", "attractor_index"], dropna=False)
        .size()
        .reset_index(name="basin_cell_count")
    )
    for _, row in basin_summary.iterrows():
        rows.append({
            "section": "3.7 basin analysis",
            "metric": f"system={row['system']}, rho={row['rho']:.2f}, attractor_index={int(row['attractor_index'])}",
            "value": int(row["basin_cell_count"]),
        })

    for _, row in df_domain_check.iterrows():
        rows.append({
            "section": "domain check",
            "metric": f"{row['system']} alpha={row['alpha']}, n={row['n']}, rho={row['rho']}",
            "value": (
                f"stable_total_small={int(row['stable_total_small'])}, "
                f"stable_total_large={int(row['stable_total_large'])}, "
                f"unpaired_small={int(row['stable_unpaired_small'])}, "
                f"unpaired_large={int(row['stable_unpaired_large'])}"
            ),
        })

    return pd.DataFrame(rows)


def write_readme() -> None:
    text = f"""Extended corrected non-reducibility gene model analysis outputs

Folder:
{OUTDIR}

Main output files:
- baseline_sweep_summary.csv
- baseline_sweep_fixed_points.csv
- structural_summary.csv
- structural_fixed_points.csv
- robustness_results.csv
- domain_check_results.csv
- structural_alpha_rho_sweep_summary.csv
- structural_alpha_rho_sweep_fixed_points.csv
- structural_n_rho_sweep_summary.csv
- structural_n_rho_sweep_fixed_points.csv
- structural_rho_bifurcation_scan.csv
- basin_assignments.csv
- basin_attractors.csv
- structural_alpha_rho_zero_total_recheck.csv
- structural_alpha_rho_zero_total_recheck_fixed_points.csv
- results_summary_for_manuscript.csv
- fig1_baseline_unpaired_heatmap.png
- fig2_robustness_unpaired_counts.png
- fig3_structural_fixed_point_locations.png
- fig4_structural_alpha_rho_unpaired_heatmap.png
- fig5_structural_n_rho_unpaired_heatmap.png
- fig6_structural_rho_bifurcation_scan.png
- fig7_basin_map_rho_0.00.png
- fig7_basin_map_rho_0.05.png
- fig7_basin_map_rho_0.10.png
- fig7_basin_map_rho_0.15.png
- fig8_zero_total_recheck_heatmap.png
- analysis_settings.json
- analysis.log

Key correction:
Structural sweeps no longer use a fixed [0, 3] x [0, 3] search box at high alpha.
An adaptive domain is used, and all zero-total alpha-rho conditions are rechecked.
"""
    with open(os.path.join(OUTDIR, "README.txt"), "w", encoding="utf-8") as f:
        f.write(text)


def write_settings() -> None:
    settings = {
        "SEARCH_DOMAIN": {
            "x_min": SEARCH_X_MIN,
            "x_max": SEARCH_X_MAX,
            "y_min": SEARCH_Y_MIN,
            "y_max": SEARCH_Y_MAX,
        },
        "DEFAULT_GRID_N": DEFAULT_GRID_N,
        "ROBUST_GRID_LIST": ROBUST_GRID_LIST,
        "ETA": ETA,
        "EPSILON_DEFAULT": EPSILON_DEFAULT,
        "ROBUST_EPSILON_LIST": ROBUST_EPSILON_LIST,
        "ALPHA_RANGE": ALPHA_RANGE.tolist(),
        "N_RANGE": N_RANGE.tolist(),
        "STRUCTURAL_ALPHA": STRUCTURAL_ALPHA,
        "STRUCTURAL_N": STRUCTURAL_N,
        "RHO_LIST": RHO_LIST,
        "STRUCTURAL_RHO_FINE": STRUCTURAL_RHO_FINE.tolist(),
        "STRUCTURAL_ALPHA_SWEEP": STRUCTURAL_ALPHA_SWEEP.tolist(),
        "STRUCTURAL_N_SWEEP": STRUCTURAL_N_SWEEP.tolist(),
        "STRUCTURAL_RHO_SWEEP": STRUCTURAL_RHO_SWEEP.tolist(),
        "BASIN_GRID_N": BASIN_GRID_N,
        "BASIN_RHO_LIST": BASIN_RHO_LIST,
        "BASIN_TMAX": BASIN_TMAX,
        "BASIN_TOL": BASIN_TOL,
        "ROOT_METHOD": ROOT_METHOD,
        "ROOT_TOL": ROOT_TOL,
        "DOMAIN_CHECK_MAX": DOMAIN_CHECK_MAX,
        "DOMAIN_CHECK_GRID": DOMAIN_CHECK_GRID,
        "ZERO_RECHECK_GRID": ZERO_RECHECK_GRID,
        "ZERO_RECHECK_MARGIN": ZERO_RECHECK_MARGIN,
        "RANDOM_SEED": RANDOM_SEED,
    }
    with open(os.path.join(OUTDIR, "analysis_settings.json"), "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)


def main() -> None:
    np.random.seed(RANDOM_SEED)
    ensure_outdir(OUTDIR)
    setup_logging()

    start = time.time()
    logging.info("Starting extended corrected analysis...")
    logging.info("Output directory: %s", OUTDIR)

    write_settings()
    write_readme()

    df_baseline_summary, df_baseline_detail = run_baseline_sweep()
    df_baseline_summary.to_csv(os.path.join(OUTDIR, "baseline_sweep_summary.csv"), index=False)
    df_baseline_detail.to_csv(os.path.join(OUTDIR, "baseline_sweep_fixed_points.csv"), index=False)

    df_struct_summary, df_struct_detail = run_structural_evaluation()
    df_struct_summary.to_csv(os.path.join(OUTDIR, "structural_summary.csv"), index=False)
    df_struct_detail.to_csv(os.path.join(OUTDIR, "structural_fixed_points.csv"), index=False)

    df_robust = run_robustness_checks()
    df_robust.to_csv(os.path.join(OUTDIR, "robustness_results.csv"), index=False)

    df_domain_check = run_domain_check()
    df_domain_check.to_csv(os.path.join(OUTDIR, "domain_check_results.csv"), index=False)

    df_alpha_rho_summary, df_alpha_rho_detail = run_structural_alpha_rho_sweep()
    df_alpha_rho_summary.to_csv(os.path.join(OUTDIR, "structural_alpha_rho_sweep_summary.csv"), index=False)
    df_alpha_rho_detail.to_csv(os.path.join(OUTDIR, "structural_alpha_rho_sweep_fixed_points.csv"), index=False)

    df_n_rho_summary, df_n_rho_detail = run_structural_n_rho_sweep()
    df_n_rho_summary.to_csv(os.path.join(OUTDIR, "structural_n_rho_sweep_summary.csv"), index=False)
    df_n_rho_detail.to_csv(os.path.join(OUTDIR, "structural_n_rho_sweep_fixed_points.csv"), index=False)

    df_bif = run_structural_rho_bifurcation_scan()
    df_bif.to_csv(os.path.join(OUTDIR, "structural_rho_bifurcation_scan.csv"), index=False)

    df_basin, df_attractors = run_basin_analysis()
    df_basin.to_csv(os.path.join(OUTDIR, "basin_assignments.csv"), index=False)
    df_attractors.to_csv(os.path.join(OUTDIR, "basin_attractors.csv"), index=False)

    df_zero_recheck, df_zero_recheck_detail = recheck_zero_total_structural_conditions(df_alpha_rho_summary)
    df_zero_recheck.to_csv(os.path.join(OUTDIR, "structural_alpha_rho_zero_total_recheck.csv"), index=False)
    df_zero_recheck_detail.to_csv(os.path.join(OUTDIR, "structural_alpha_rho_zero_total_recheck_fixed_points.csv"), index=False)

    save_fig1_baseline_heatmap(df_baseline_summary)
    save_fig2_robustness(df_robust)
    save_fig3_structural_locations(df_struct_detail)
    save_fig4_alpha_rho_heatmap(df_alpha_rho_summary)
    save_fig5_n_rho_heatmap(df_n_rho_summary)
    save_fig6_bifurcation(df_bif)
    save_fig7_basin_maps(df_basin, df_attractors)
    save_fig8_zero_recheck(df_zero_recheck)

    df_results_summary = make_results_summary(
        df_baseline_summary,
        df_struct_summary,
        df_struct_detail,
        df_robust,
        df_domain_check,
        df_alpha_rho_summary,
        df_n_rho_summary,
        df_bif,
        df_basin,
        df_attractors,
        df_zero_recheck,
    )
    df_results_summary.to_csv(os.path.join(OUTDIR, "results_summary_for_manuscript.csv"), index=False)

    elapsed = time.time() - start
    logging.info("Extended corrected analysis finished in %.2f seconds.", elapsed)
    print("\nDone.")
    print(f"All results were saved to:\n{OUTDIR}\n")


if __name__ == "__main__":
    main()

