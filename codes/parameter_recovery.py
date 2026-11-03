"""Parameter recovery for M0–M2. Saves tables; figures are drawn by analysis.R."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy import stats

from fitting import ensure_dir, fit_subject
from model import (
    MAXITER,
    MODEL_ORDER,
    N_SIM_PARAM,
    N_STARTS,
    PARAM_PLAIN,
    RECOVERY_DIR,
    SEED,
    SHORT,
    ParticipantData,
    choice_probs,
    load_data,
    model_config,
    sample_params_uniform,
)


def _one_recovery(payload: dict) -> list[dict]:
    """true_params ~ U[lb, ub]; data.choice = model probabilities; then MLE."""
    cfg = model_config(payload["model_name"])
    true = np.asarray(payload["true_params"], float)
    data = ParticipantData(
        payload["sim_id"],
        np.asarray(payload["prob"], float),
        np.asarray(payload["money"], float),
        np.zeros(len(payload["prob"])),
        np.asarray(payload["condition"], float),
    )
    sim = data.copy_with_choice(choice_probs(true, data, payload["model_name"]))
    fit = fit_subject(payload["model_name"], sim, payload["n_starts"], payload["seed"], payload["maxiter"])
    est = np.asarray([fit[f"param_{n}"] for n in cfg.param_names], float)
    rows = []
    for name, t, e in zip(cfg.param_names, true, est):
        rows.append(
            {
                "sim_id": payload["sim_id"],
                "model_name": payload["model_name"],
                "short_name": SHORT[payload["model_name"]],
                "parameter": name,
                "parameter_symbol": PARAM_PLAIN[name],
                "true_value": float(t),
                "estimated_value": float(e),
                "bias": float(e - t),
                "nll": float(fit["nll"]),
            }
        )
    return rows


def _corr_matrix(rec: pd.DataFrame, param_names: tuple[str, ...]) -> pd.DataFrame:
    wide_true = rec.pivot(index="sim_id", columns="parameter", values="true_value")
    wide_est = rec.pivot(index="sim_id", columns="parameter", values="estimated_value")
    mat = np.array([[np.corrcoef(wide_true[a], wide_est[b])[0, 1] for b in param_names] for a in param_names])
    labels = [PARAM_PLAIN[c] for c in param_names]
    return pd.DataFrame(mat, index=labels, columns=labels)


def _summarize(rec: pd.DataFrame, param_names: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for param in param_names:
        g = rec[rec["parameter"] == param]
        r, p = stats.pearsonr(g["true_value"], g["estimated_value"])
        rows.append(
            {
                "model_name": rec["model_name"].iloc[0],
                "short_name": rec["short_name"].iloc[0],
                "parameter": param,
                "parameter_symbol": PARAM_PLAIN[param],
                "pearson_r": float(r),
                "pearson_p": float(p),
                "mean_bias": float(g["bias"].mean()),
                "mae": float(g["bias"].abs().mean()),
                "rmse": float(np.sqrt((g["bias"] ** 2).mean())),
                "n": int(len(g)),
            }
        )
    return pd.DataFrame(rows)


def write_recovery_points(sub: pd.DataFrame, param_names: tuple[str, ...], path) -> None:
    points = sub.loc[:, ["sim_id", "parameter", "true_value", "estimated_value"]].rename(columns={"sim_id": "sim"})
    points["parameter"] = pd.Categorical(points["parameter"], list(param_names), ordered=True)
    points = points.sort_values(["parameter", "sim"]).reset_index(drop=True)
    points.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parameter recovery for M0–M2 (n_sim=500, first xlsx subject).")
    p.add_argument("--models", nargs="*", default=list(MODEL_ORDER), help="Default: all of M0 M1 M2.")
    p.add_argument("--n-sim", type=int, default=N_SIM_PARAM)
    p.add_argument("--n-starts", type=int, default=N_STARTS)
    p.add_argument("--maxiter", type=int, default=MAXITER)
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    models = [m for m in args.models if m in MODEL_ORDER] or list(MODEL_ORDER)
    template = load_data()[0]
    print(
        f"Parameter recovery: models={ [SHORT[m] for m in models] }, n_sim={args.n_sim}, "
        f"design=subject {int(template.subject_id)} n={template.n_trials}"
    )

    payloads = []
    for model_name in models:
        mi = MODEL_ORDER.index(model_name)
        cfg = model_config(model_name)
        rng = np.random.default_rng(args.seed + 100003 * (mi + 1))
        for i in range(args.n_sim):
            payloads.append(
                {
                    "sim_id": i + 1,
                    "model_name": model_name,
                    "true_params": sample_params_uniform(cfg, rng).tolist(),
                    "prob": template.prob.tolist(),
                    "money": template.money.tolist(),
                    "condition": template.condition.tolist(),
                    "n_starts": args.n_starts,
                    "maxiter": args.maxiter,
                    "seed": args.seed + 17000 * (mi + 1) + i * 17,
                }
            )

    rows: list[dict] = []
    n_total = len(payloads)
    if args.n_jobs == 1:
        for i, payload in enumerate(payloads, start=1):
            rows.extend(_one_recovery(payload))
            if i % 50 == 0 or i == n_total:
                print(f"  finished {i}/{n_total}")
    else:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
            futs = {ex.submit(_one_recovery, p): (p["model_name"], p["sim_id"]) for p in payloads}
            done_n = 0
            for fut in as_completed(futs):
                rows.extend(fut.result())
                done_n += 1
                if done_n % 50 == 0 or done_n == n_total:
                    name, sid = futs[fut]
                    print(f"  finished {done_n}/{n_total} (last {SHORT[name]} sim {sid})")

    rec = pd.DataFrame(rows)
    out_dir = ensure_dir(RECOVERY_DIR / "parameter")

    summaries = []
    for model_name in models:
        cfg = model_config(model_name)
        sub = rec[rec["model_name"] == model_name].copy()
        summary = _summarize(sub, cfg.param_names)
        summaries.append(summary)
        short = SHORT[model_name]
        summary.to_csv(out_dir / f"parameter_recovery_summary_{short}.csv", index=False)
        if short == "M2":
            _corr_matrix(sub, cfg.param_names).to_csv(out_dir / "parameter_recovery_corr_heatmap_M2.csv")
        write_recovery_points(sub, cfg.param_names, out_dir / f"parameter_recovery_points_{short}.csv")

    summary_all = pd.concat(summaries, ignore_index=True)
    summary_all["short_name"] = pd.Categorical(summary_all["short_name"], [SHORT[m] for m in MODEL_ORDER], ordered=True)
    summary_all = summary_all.sort_values(["short_name", "parameter"]).reset_index(drop=True)
    if list(models) == list(MODEL_ORDER):
        summary_all.to_csv(out_dir / "parameter_recovery_summary.csv", index=False)
        pd.DataFrame(
            [
                {
                    "models": ",".join(models),
                    "n_sim": args.n_sim,
                    "n_starts": args.n_starts,
                    "maxiter": args.maxiter,
                    "design": f"raw_data subject {int(template.subject_id)}",
                    "simulated_choice": "model_probabilities",
                    "seed": args.seed,
                }
            ]
        ).to_csv(out_dir / "parameter_recovery_meta.csv", index=False)
    print(summary_all.to_string(index=False))


if __name__ == "__main__":
    main()
