"""
Phase 7: Spatial Prioritization & Sensitivity Analysis for FIREGUARD ID.

Keeps vulnerability and operational priority conceptually distinct by combining:
1. Relative vulnerability score (Phase 6)
2. Multi-year persistence / chronic recurrence ratio (active_years_count / 5)
3. Spatial corridor spillover (mean vulnerability of queen-adjacent cells)
4. Geographic remoteness / peripheral accessibility proxy (haversine distance from
   Pekanbaru regional hub [0.5071N, 101.4478E], explicitly marked as proxy pending OSM roads)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.geospatial.geometry import haversine_km
from src.utils import ensure_dir, get_logger, load_yaml, resolve_path

logger = get_logger("fireguard.prioritization")

PEKANBARU_LAT = 0.5071
PEKANBARU_LON = 101.4478

PRIORITY_WEIGHT_SCENARIOS = {
    "balanced_operational": {"vuln": 0.45, "persistence": 0.25, "corridor": 0.20, "remoteness": 0.10},
    "vulnerability_dominant": {"vuln": 0.70, "persistence": 0.15, "corridor": 0.10, "remoteness": 0.05},
    "corridor_containment": {"vuln": 0.35, "persistence": 0.20, "corridor": 0.35, "remoteness": 0.10},
    "remote_access_focus": {"vuln": 0.40, "persistence": 0.20, "corridor": 0.15, "remoteness": 0.25},
}


def _compute_neighbor_mean(df: pd.DataFrame, col: str) -> np.ndarray:
    """Computes queen-neighborhood mean of a column for each grid cell."""
    pos = {(int(r.row_idx), int(r.col_idx)): idx for idx, r in enumerate(df.itertuples())}
    vals = df[col].to_numpy(dtype=float)
    out = np.zeros(len(df), dtype=float)
    for idx, r in enumerate(df.itertuples()):
        ri, ci = int(r.row_idx), int(r.col_idx)
        nbs = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nb = pos.get((ri + dr, ci + dc))
                if nb is not None:
                    nbs.append(vals[nb])
        out[idx] = float(np.mean(nbs)) if nbs else vals[idx]
    return out


def run_phase7_prioritization(repo_root: Path | None = None) -> dict[str, Any]:
    proj_cfg = load_yaml("config/project.yaml", repo_root=repo_root)
    processed_dir = resolve_path(proj_cfg["paths"]["processed_dir"], repo_root=repo_root)
    prio_dir = ensure_dir(resolve_path(proj_cfg["paths"]["reports_dir"], repo_root=repo_root) / "prioritization")

    df = pd.read_parquet(processed_dir / "riau_vulnerability_profile.parquet")

    # Component 1: Phase 6 relative vulnerability_score (already in [0, 1])
    c_vuln = df["vulnerability_score"].to_numpy(dtype=float)

    # Component 2: Chronic multi-year persistence (normalized active_years_count in [0, 1])
    c_persist = np.clip(df["active_years_count"].to_numpy(dtype=float) / 5.0, 0.0, 1.0)

    # Component 3: Spatial corridor spillover (queen-neighbor mean vulnerability)
    c_corridor = _compute_neighbor_mean(df, "vulnerability_score")

    # Component 4: Geographic remoteness proxy (haversine distance to Pekanbaru normalized to [0, 1])
    dist_hub_km = haversine_km(
        df["land_rep_lat"].to_numpy(dtype=float),
        df["land_rep_lon"].to_numpy(dtype=float),
        PEKANBARU_LAT,
        PEKANBARU_LON,
    )
    c_remote = dist_hub_km / max(float(np.max(dist_hub_km)), 1e-9)

    df["dist_to_pekanbaru_km"] = np.round(dist_hub_km, 2)
    df["factor_vulnerability"] = np.round(c_vuln, 6)
    df["factor_persistence"] = np.round(c_persist, 6)
    df["factor_spatial_corridor"] = np.round(c_corridor, 6)
    df["factor_remoteness_proxy"] = np.round(c_remote, 6)

    w_base = PRIORITY_WEIGHT_SCENARIOS["balanced_operational"]
    df["priority_score"] = np.round(
        w_base["vuln"] * c_vuln
        + w_base["persistence"] * c_persist
        + w_base["corridor"] * c_corridor
        + w_base["remoteness"] * c_remote,
        6,
    )

    # Deterministic priority ranking (1 = highest operational priority)
    df["priority_rank"] = df["priority_score"].rank(ascending=False, method="first").astype(int)

    # Categorical priority tiers by percentile rank (distinct from cluster IDs)
    n = len(df)
    conditions = [
        df["priority_rank"] <= int(round(0.10 * n)),
        df["priority_rank"] <= int(round(0.25 * n)),
        df["priority_rank"] <= int(round(0.50 * n)),
    ]
    choices = [
        "Tier 1 (Critical Priority - Top 10%)",
        "Tier 2 (High Priority - Top 10-25%)",
        "Tier 3 (Elevated Priority - Top 25-50%)",
    ]
    df["priority_class"] = np.select(conditions, choices, default="Tier 4 (Baseline Monitoring - Bottom 50%)")

    # Sensitivity analysis across alternative weight profiles
    sens_records = []
    base_top85 = set(df.nsmallest(85, "priority_rank")["grid_id"])
    for sname, w in PRIORITY_WEIGHT_SCENARIOS.items():
        alt_score = (
            w["vuln"] * c_vuln
            + w["persistence"] * c_persist
            + w["corridor"] * c_corridor
            + w["remoteness"] * c_remote
        )
        rho, _ = spearmanr(df["priority_score"], alt_score)
        alt_top85 = set(df.assign(_s=alt_score).nlargest(85, "_s")["grid_id"])
        jaccard = len(base_top85 & alt_top85) / max(len(base_top85 | alt_top85), 1)
        max_rank_shift = int(
            np.max(np.abs(df["priority_rank"].to_numpy() - pd.Series(alt_score).rank(ascending=False, method="first").to_numpy()))
        )
        sens_records.append(
            {
                "scenario": sname,
                **w,
                "spearman_rho_vs_base": round(float(rho), 6),
                "top85_jaccard_vs_base": round(float(jaccard), 4),
                "max_rank_shift_cells": max_rank_shift,
            }
        )
    sens_df = pd.DataFrame(sens_records)
    sens_df.to_csv(prio_dir / "sensitivity_analysis.csv", index=False)

    # Save locked spatial priorities for Phase 8 optimization
    out_parquet = processed_dir / "riau_spatial_priorities.parquet"
    out_csv = processed_dir / "riau_spatial_priorities.csv"
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)

    # Top 20 priority zones summary
    top20 = df.nsmallest(20, "priority_rank")[
        [
            "priority_rank",
            "grid_id",
            "center_lat",
            "center_lon",
            "hotspot_count",
            "fire_days",
            "active_years_count",
            "vulnerability_class",
            "priority_score",
            "priority_class",
            "is_hdbscan_noise",
        ]
    ]
    top20.to_csv(prio_dir / "top20_priority_zones.csv", index=False)

    summary = {
        "n_cells": int(len(df)),
        "priority_class_counts": df["priority_class"].value_counts().to_dict(),
        "baseline_weights": w_base,
        "vulnerability_vs_priority_spearman": round(
            float(spearmanr(df["vulnerability_score"], df["priority_score"])[0]), 6
        ),
        "sensitivity": sens_records,
    }
    with open(prio_dir / "prioritization_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    lines = [
        "# FIREGUARD ID — Spatial Prioritization Report (Phase 7)",
        "",
        "## 1. Framework & Factor Separation",
        "Operational priority is kept distinct from raw cluster membership and from vulnerability alone:",
        "- `factor_vulnerability` (`w=0.45`): Phase 6 empirical relative vulnerability score",
        "- `factor_persistence` (`w=0.25`): Multi-year chronic recurrence (`active_years_count / 5`)",
        "- `factor_spatial_corridor` (`w=0.20`): Mean vulnerability across queen-adjacent Riau cells",
        "- `factor_remoteness_proxy` (`w=0.10`): Geodesic distance from Pekanbaru regional hub (proxy for response travel burden; real OSM road network unacquired)",
        "",
        f"- **Spearman Correlation (Vulnerability vs Priority Score):** `{summary['vulnerability_vs_priority_spearman']}` (correlated, yet distinct due to persistence, corridor spillover, and remoteness).",
        "",
        "## 2. Priority Class Distribution",
        *(f"- **{k}:** {v} cells" for k, v in summary["priority_class_counts"].items()),
        "",
        "## 3. Weight Sensitivity Analysis",
        sens_df.to_markdown(index=False),
        "",
        "## 4. Top 10 Priority Grid Cells",
        top20.head(10).to_markdown(index=False),
    ]
    with open(prio_dir / "prioritization_report.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    logger.info("Phase 7 spatial prioritization complete -> %s", out_parquet)
    return summary


if __name__ == "__main__":
    run_phase7_prioritization()
