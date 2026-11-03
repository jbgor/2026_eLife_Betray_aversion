"""Simulate Model 2 choices on the observed trials and save the cell means.

Figures are drawn by analysis.R.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from fitting import ensure_dir
from model import (
    SIM_DIR,
    FIT_DIR,
    SEED,
    WINNER,
    choice_probs,
    model_config,
    simulate_choices,
    valid_participants,
)

N_SIM = 500
PROBS = (0.1, 0.3, 0.5, 0.7, 0.9)
STAKES = (0.2, 0.4, 0.6, 0.8, 1.0)
TASK_NAME = {0: "Risk", 1: "Cooperation"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Model 2 choice simulation.")
    p.add_argument("--n-sim", type=int, default=N_SIM)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def _theta(wide: pd.DataFrame, sid: int, names: tuple[str, ...]) -> np.ndarray:
    return wide.loc[sid, [f"param_{n}" for n in names]].to_numpy(float)


def _cell_means(choice: np.ndarray, condition: np.ndarray, factor: np.ndarray, levels: tuple[float, ...]) -> dict[tuple[int, float], float]:
    out: dict[tuple[int, float], float] = {}
    for cond in (0, 1):
        for level in levels:
            idx = (condition == cond) & np.isclose(factor, level)
            out[(cond, float(level))] = float(choice[idx].mean()) if idx.any() else np.nan
    return out


def _ci95(values: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    m = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(len(x)))
    return m, m - 1.96 * se, m + 1.96 * se


def main() -> None:
    args = parse_args()
    cfg = model_config(WINNER)
    wide = pd.read_csv(FIT_DIR / WINNER / "subject_params_wide.csv").set_index("subject_id")
    people = [p for p in valid_participants() if p.subject_id in wide.index]
    rng = np.random.default_rng(args.seed)

    packed = []
    obs_prob_rows = []
    obs_stake_rows = []
    for p in people:
        theta = _theta(wide, p.subject_id, cfg.param_names)
        pr = choice_probs(theta, p, WINNER)
        packed.append((p, pr))
        for (cond, prob), val in _cell_means(p.choice, p.condition, p.prob, PROBS).items():
            obs_prob_rows.append(
                {
                    "subject_id": p.subject_id,
                    "task": TASK_NAME[cond],
                    "condition": cond,
                    "prob": prob,
                    "p_choice": val,
                }
            )
        for (cond, stake), val in _cell_means(p.choice, p.condition, p.money, STAKES).items():
            obs_stake_rows.append(
                {
                    "subject_id": p.subject_id,
                    "task": TASK_NAME[cond],
                    "condition": cond,
                    "stake": stake,
                    "p_choice": val,
                }
            )
    obs_subj = pd.DataFrame(obs_prob_rows)
    obs_stake = pd.DataFrame(obs_stake_rows)

    sim_means = np.zeros((args.n_sim, 2, len(PROBS)))
    sim_stake = np.zeros((args.n_sim, 2, len(STAKES)))
    for s in range(args.n_sim):
        prob_cells = []
        stake_cells = []
        for p, pr in packed:
            y = simulate_choices(pr, rng)
            prob_cells.append(_cell_means(y, p.condition, p.prob, PROBS))
            stake_cells.append(_cell_means(y, p.condition, p.money, STAKES))
        for j, cond in enumerate((0, 1)):
            for k, prob in enumerate(PROBS):
                sim_means[s, j, k] = np.nanmean([c[(cond, prob)] for c in prob_cells])
            for k, stake in enumerate(STAKES):
                sim_stake[s, j, k] = np.nanmean([c[(cond, stake)] for c in stake_cells])

    summary = _summarize(obs_subj, sim_means, PROBS, "prob")
    stake_summary = _summarize(obs_stake, sim_stake, STAKES, "stake")
    out_dir = ensure_dir(SIM_DIR)
    summary.to_csv(out_dir / "simulation_by_probability.csv", index=False)
    stake_summary.to_csv(out_dir / "simulation_by_stake.csv", index=False)
    meta = {
        "model": WINNER,
        "n_subjects": len(people),
        "n_sim": args.n_sim,
        "seed": args.seed,
    }
    (out_dir / "simulation_meta.json").write_text(json.dumps(meta, indent=2))
    print(summary.to_string(index=False))
    print(stake_summary.to_string(index=False))
    print("Simulation tables written. Figures are drawn by analysis.R.")


def _summarize(obs_subj: pd.DataFrame, sim_means: np.ndarray, levels: tuple[float, ...], level_name: str) -> pd.DataFrame:
    rows = []
    for j, cond in enumerate((0, 1)):
        for k, level in enumerate(levels):
            sub = obs_subj[(obs_subj["condition"] == cond) & np.isclose(obs_subj[level_name], level)]
            obs_m, obs_lo, obs_hi = _ci95(sub["p_choice"].to_numpy())
            sim = sim_means[:, j, k]
            sim_lo, sim_hi = float(np.quantile(sim, 0.025)), float(np.quantile(sim, 0.975))
            rows.append(
                {
                    "task": TASK_NAME[cond],
                    "condition": cond,
                    level_name: float(level),
                    "observed": obs_m,
                    "observed_ci_lo": obs_lo,
                    "observed_ci_hi": obs_hi,
                    "simulated_mean": float(sim.mean()),
                    "simulated_q025": sim_lo,
                    "simulated_q975": sim_hi,
                    "simulated_min": float(sim.min()),
                    "simulated_max": float(sim.max()),
                    "in_95": bool(sim_lo <= obs_m <= sim_hi),
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
