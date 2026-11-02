"""Multi-start MLE and 10-fold CVIC for M0–M2.

After the fits are stored, joins each participant's parameters with the
experimental trials and cue-locked EEG and writes results/derived/.
"""

from __future__ import annotations

import argparse
import json
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.optimize import minimize

from model import (
    ANALYSIS_DIR,
    BALANCE34,
    CHANNEL_NAMES,
    DERIVED_DIR,
    DISPLAY,
    FIT_DIR,
    INVALID_SUBJECT_IDS,
    MAXITER,
    MODEL_ORDER,
    N_FOLDS,
    N_STARTS,
    RAW_DIR,
    SEED,
    SHORT,
    WINNER,
    ParticipantData,
    choice_probs,
    clip_probs,
    load_data,
    model_config,
    pro_code_to_prob,
    trial_components,
    valid_participants,
)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def nll(theta: np.ndarray, model_name: str, data: ParticipantData) -> float:
    probs = clip_probs(choice_probs(theta, data, model_name))
    y = data.choice.astype(float)
    value = -np.sum(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs))
    return float(value) if np.isfinite(value) else 1e30


def pointwise_loglik(theta: np.ndarray, model_name: str, data: ParticipantData) -> np.ndarray:
    probs = clip_probs(choice_probs(theta, data, model_name))
    y = data.choice.astype(float)
    return y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs)


def _starts(cfg, n_starts: int, rng: np.random.Generator, lb=None, ub=None) -> list[np.ndarray]:
    lb = cfg.lb if lb is None else np.asarray(lb, float)
    ub = cfg.ub if ub is None else np.asarray(ub, float)
    starts = [0.5 * (lb + ub)]
    n_rand = max(n_starts - 1, 0)
    for i in range(n_rand):
        u = (i + rng.random(cfg.n_params)) / max(n_rand, 1)
        rng.shuffle(u)
        x = np.clip(lb + u * (ub - lb), lb + 1e-8 * (ub - lb), ub - 1e-8 * (ub - lb))
        starts.append(x)
    return starts


def fit_subject(
    model_name: str,
    data: ParticipantData,
    n_starts: int = N_STARTS,
    seed: int = SEED,
    maxiter: int = MAXITER,
    lb=None,
    ub=None,
) -> dict:
    cfg = model_config(model_name)
    lb = cfg.lb if lb is None else np.asarray(lb, float)
    ub = cfg.ub if ub is None else np.asarray(ub, float)
    rng = np.random.default_rng(seed + int(data.subject_id) * 1009)
    bounds = list(zip(lb.tolist(), ub.tolist()))
    best_nll, best_theta, n_ok = np.inf, 0.5 * (lb + ub), 0
    for x0 in _starts(cfg, n_starts, rng, lb, ub):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = minimize(
                    nll,
                    x0=x0,
                    args=(model_name, data),
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": maxiter, "ftol": 1e-10},
                )
            if np.isfinite(res.fun):
                n_ok += 1
                if res.fun < best_nll:
                    best_nll = float(res.fun)
                    best_theta = np.asarray(res.x, float).copy()
        except Exception:  # noqa: BLE001
            continue
    if not np.isfinite(best_nll):
        best_nll = nll(best_theta, model_name, data)
    k, n = cfg.n_params, data.n_trials
    aic = 2.0 * k + 2.0 * best_nll
    bic = k * np.log(n) + 2.0 * best_nll
    row = {
        "subject_id": int(data.subject_id),
        "model_name": model_name,
        "display_name": DISPLAY[model_name],
        "short_name": SHORT[model_name],
        "n_trials": n,
        "n_params": k,
        "nll": best_nll,
        "aic": aic,
        "bic": bic,
        "pseudo_r2": 1.0 - best_nll / (n * np.log(2.0)),
        "n_starts": n_starts,
        "n_success": n_ok,
    }
    eps = 1e-4 * np.maximum(ub - lb, 1e-8)
    for i, name in enumerate(cfg.param_names):
        value = float(best_theta[i])
        row[f"param_{name}"] = value
        row[f"at_bound_{name}"] = bool(value <= lb[i] + eps[i] or value >= ub[i] - eps[i])
    return row


def _fit_job(payload: dict) -> dict:
    data = ParticipantData(
        payload["subject_id"],
        np.asarray(payload["prob"], float),
        np.asarray(payload["money"], float),
        np.asarray(payload["choice"], float),
        np.asarray(payload["condition"], float),
    )
    return fit_subject(payload["model_name"], data, payload["n_starts"], payload["seed"], payload["maxiter"])


def _refit_warmstart(model_name: str, data: ParticipantData, x0: np.ndarray, maxiter: int) -> np.ndarray:
    cfg = model_config(model_name)
    bounds = list(zip(cfg.lb.tolist(), cfg.ub.tolist()))
    x0 = np.clip(np.asarray(x0, float), cfg.lb + 1e-8, cfg.ub - 1e-8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(
            lambda x: -float(np.sum(pointwise_loglik(x, model_name, data))),
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-10},
        )
    return np.asarray(res.x, float) if np.isfinite(res.fun) else x0


def subject_looic_kfold(
    model_name: str,
    data: ParticipantData,
    theta: np.ndarray,
    n_folds: int = N_FOLDS,
    seed: int = SEED,
    maxiter: int = MAXITER,
) -> dict[str, float]:
    n = data.n_trials
    k = model_config(model_name).n_params
    n_folds = min(n_folds, max(n, 2))
    rng = np.random.default_rng(seed + int(data.subject_id) * 17)
    folds = np.array_split(rng.permutation(n), n_folds)
    elpd = 0.0
    for hold in folds:
        if hold.size == 0:
            continue
        train_mask = np.ones(n, dtype=bool)
        train_mask[hold] = False
        if train_mask.sum() < max(k + 1, 2):
            fold_theta = theta
        else:
            fold_theta = _refit_warmstart(model_name, data.subset(np.where(train_mask)[0]), theta, maxiter)
        elpd += float(np.sum(pointwise_loglik(fold_theta, model_name, data.subset(hold))))
    return {"elpd_loo": elpd, "looic": -2.0 * elpd, "loo_n_folds": float(n_folds)}


def _loo_job(payload: dict) -> dict:
    data = ParticipantData(
        payload["subject_id"],
        np.asarray(payload["prob"], float),
        np.asarray(payload["money"], float),
        np.asarray(payload["choice"], float),
        np.asarray(payload["condition"], float),
    )
    metrics = subject_looic_kfold(
        payload["model_name"],
        data,
        np.asarray(payload["theta"], float),
        payload["n_folds"],
        payload["seed"],
        payload["maxiter"],
    )
    metrics["subject_id"] = payload["subject_id"]
    return metrics


def _write_fit(model_name: str, results: pd.DataFrame, extra: dict | None = None) -> Path:
    cfg = model_config(model_name)
    model_dir = ensure_dir(FIT_DIR / model_name)
    param_rows = []
    for _, row in results.iterrows():
        for name in cfg.param_names:
            param_rows.append(
                {
                    "subject_id": int(row["subject_id"]),
                    "model_name": model_name,
                    "display_name": DISPLAY[model_name],
                    "parameter": name,
                    "estimate": float(row[f"param_{name}"]),
                    "at_bound": bool(row.get(f"at_bound_{name}", False)),
                }
            )
    pd.DataFrame(param_rows).to_csv(model_dir / "subject_params.csv", index=False)
    diag_cols = [
        c
        for c in [
            "subject_id", "model_name", "display_name", "short_name", "n_trials", "n_params",
            "nll", "aic", "bic", "pseudo_r2", "n_starts", "n_success", "elpd_loo", "looic", "loo_n_folds",
        ]
        if c in results.columns
    ]
    results[diag_cols].to_csv(model_dir / "fit_diagnostics.csv", index=False)
    results.to_csv(model_dir / "subject_params_wide.csv", index=False)
    meta = {
        "model_name": model_name,
        "display_name": DISPLAY[model_name],
        "n_subjects": int(results["subject_id"].nunique()),
        "param_names": list(cfg.param_names),
        "lb": cfg.lb.tolist(),
        "ub": cfg.ub.tolist(),
        "kappa": "identity" if "kappa" in cfg.param_names else None,
        "metrics": ["AIC", "BIC", "LOOIC"],
    }
    if extra:
        meta.update(extra)
    (model_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved {model_name}: {model_dir} (n={meta['n_subjects']})")
    return model_dir


def fit_model(
    model_name: str,
    n_starts: int = N_STARTS,
    seed: int = SEED,
    maxiter: int = MAXITER,
    n_jobs: int = 8,
) -> pd.DataFrame:
    people = valid_participants()
    payloads = [
        {
            "subject_id": p.subject_id,
            "prob": p.prob,
            "money": p.money,
            "choice": p.choice,
            "condition": p.condition,
            "model_name": model_name,
            "n_starts": n_starts,
            "seed": seed,
            "maxiter": maxiter,
        }
        for p in people
    ]
    rows = []
    if n_jobs == 1:
        for payload in payloads:
            print(f"  Fitting {model_name} subject {payload['subject_id']}")
            rows.append(_fit_job(payload))
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futs = {ex.submit(_fit_job, p): p["subject_id"] for p in payloads}
            for fut in as_completed(futs):
                sid = futs[fut]
                rows.append(fut.result())
                print(f"  Finished {model_name} subject {sid}")
    results = pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)
    _write_fit(model_name, results, {"n_starts": n_starts, "maxiter": maxiter, "seed": seed})
    return results


def compute_looic(
    model_name: str,
    n_folds: int = N_FOLDS,
    seed: int = SEED,
    maxiter: int = MAXITER,
    n_jobs: int = 8,
) -> pd.DataFrame:
    wide = pd.read_csv(FIT_DIR / model_name / "subject_params_wide.csv")
    cfg = model_config(model_name)
    by_id = {p.subject_id: p for p in valid_participants()}
    payloads = []
    for _, row in wide.iterrows():
        sid = int(row["subject_id"])
        if sid not in by_id:
            continue
        payloads.append(
            {
                "model_name": model_name,
                "subject_id": sid,
                "theta": row[[f"param_{n}" for n in cfg.param_names]].to_numpy(float),
                "prob": by_id[sid].prob,
                "money": by_id[sid].money,
                "choice": by_id[sid].choice,
                "condition": by_id[sid].condition,
                "n_folds": n_folds,
                "seed": seed,
                "maxiter": maxiter,
            }
        )
    rows = []
    if n_jobs == 1:
        for payload in payloads:
            print(f"  LOOIC {model_name} subject {payload['subject_id']}")
            rows.append(_loo_job(payload))
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futs = {ex.submit(_loo_job, p): p["subject_id"] for p in payloads}
            for fut in as_completed(futs):
                sid = futs[fut]
                rows.append(fut.result())
                print(f"  Finished LOOIC {model_name} subject {sid}")
    loo = pd.DataFrame(rows)
    for path_name in ("fit_diagnostics.csv", "subject_params_wide.csv"):
        df = pd.read_csv(FIT_DIR / model_name / path_name)
        df = df.drop(columns=[c for c in ("elpd_loo", "looic", "loo_n_folds") if c in df.columns])
        df = df.merge(loo, on="subject_id", how="left")
        df.to_csv(FIT_DIR / model_name / path_name, index=False)
    loo.to_csv(FIT_DIR / model_name / "looic_by_subject.csv", index=False)
    return loo


def compare_ics() -> pd.DataFrame:
    frames = []
    for name in MODEL_ORDER:
        path = FIT_DIR / name / "fit_diagnostics.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    diag = pd.concat(frames, ignore_index=True)
    n = diag["subject_id"].nunique()
    rows = []
    for name, g in diag.groupby("model_name"):
        row = {
            "model_name": name,
            "short_name": SHORT[name],
            "display_name": DISPLAY[name],
            "n_subjects": int(n),
            "n_params": int(g["n_params"].iloc[0]),
            "sum_nll": float(g["nll"].sum()),
            "mean_nll": float(g["nll"].mean()),
            "sum_aic": float(g["aic"].sum()),
            "mean_aic": float(g["aic"].mean()),
            "sum_bic": float(g["bic"].sum()),
            "mean_bic": float(g["bic"].mean()),
            "mean_pseudo_r2": float(g["pseudo_r2"].mean()),
        }
        if "looic" in g.columns and g["looic"].notna().any():
            row["sum_looic"] = float(g["looic"].sum())
            row["mean_looic"] = float(g["looic"].mean())
        for metric, col in [("aic", "n_best_aic"), ("bic", "n_best_bic"), ("looic", "n_best_looic")]:
            if metric not in g.columns:
                continue
            best = diag.loc[diag.groupby("subject_id")[metric].idxmin(), "model_name"]
            row[col] = int((best == name).sum())
            row[f"win_rate_{metric}"] = row[col] / n
        rows.append(row)
    out = pd.DataFrame(rows)
    if "sum_looic" in out.columns:
        out = out.sort_values("sum_looic")
        best = out["sum_looic"].min()
        out["delta_sum_looic"] = out["sum_looic"] - best
        out["delta_sum_aic"] = out["sum_aic"] - out["sum_aic"].min()
        out["delta_sum_bic"] = out["sum_bic"] - out["sum_bic"].min()
        out["recommended_by_sum_looic"] = out["sum_looic"] == best
    ensure_dir(ANALYSIS_DIR)
    out.to_csv(ANALYSIS_DIR / "model_comparison.csv", index=False)
    diag[["subject_id", "model_name", "aic", "bic", "nll", "n_params"] + (["looic"] if "looic" in diag.columns else [])].to_csv(
        ANALYSIS_DIR / "subject_ic.csv", index=False
    )
    return out


def _load_behavior_task(path: Path, experiment: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df[(df["ExperimentName"] == experiment) & (df["Procedure"] == "expe")].copy()
    keep = [c for c in ["Subject", "pro", "mon", "montype", "Choice", "choice.RT", "FB"] if c in df.columns]
    df = df[keep].copy()
    df = df.rename(columns={"choice.RT": "RT"})
    df["RT"] = pd.to_numeric(df["RT"], errors="coerce")
    df["Choice"] = pd.to_numeric(df["Choice"], errors="coerce")
    df["Feedback"] = np.where(df["FB"].astype(str) == "ü", "Good", "Bad")
    df["pro"] = pd.to_numeric(df["pro"], errors="coerce")
    df["mon"] = pd.to_numeric(df["mon"], errors="coerce")
    df["task"] = "single" if experiment == "single" else "double"
    df = df.sort_values(["Subject"], kind="mergesort").copy()
    df["Trial"] = df.groupby("Subject").cumcount() + 1
    df["zpro"] = (df["pro"] - df["pro"].mean()) / df["pro"].std(ddof=0)
    df["zmon"] = (df["mon"] - df["mon"].mean()) / df["mon"].std(ddof=0)
    return df


def _roi_mean(mat_path: Path, key: str, channels: list[str]) -> np.ndarray:
    arr = np.asarray(loadmat(mat_path, squeeze_me=True)[key], dtype=float)
    idx = [CHANNEL_NAMES.index(ch) for ch in channels]
    return arr[:, idx].mean(axis=1)


def build_behavior_eeg() -> pd.DataFrame:
    """Behavior table in EEG row order, with cue-locked P3 and LPP."""
    sin = _load_behavior_task(RAW_DIR / "single.xlsx", "single")
    dou = _load_behavior_task(RAW_DIR / "double.xlsx", "double")
    cols = ["Subject", "pro", "montype", "Choice", "RT", "Trial", "zpro", "zmon", "Feedback", "task", "mon"]
    total = pd.concat([sin[cols], dou[cols]], ignore_index=True)
    coop_first = set(BALANCE34)
    parts = []
    for sid, g in total.groupby("Subject", sort=False):
        if int(sid) in coop_first:
            g = pd.concat([g[g["task"] == "double"], g[g["task"] == "single"]], ignore_index=True)
        else:
            g = pd.concat([g[g["task"] == "single"], g[g["task"] == "double"]], ignore_index=True)
        parts.append(g)
    total = pd.concat(parts, ignore_index=True).sort_values("Subject", kind="mergesort").reset_index(drop=True)
    total["MonP3"] = _roi_mean(RAW_DIR / "Cue2P3400500.mat", "Cue2P3400500", ["P1", "PZ", "P2"])
    total["MonLPP"] = _roi_mean(RAW_DIR / "Cue2LPP600800.mat", "Cue2LPP600800", ["P1", "PZ", "P2"])
    total = total[total["Choice"] != 3]
    total = total[~total["Subject"].isin(INVALID_SUBJECT_IDS)].reset_index(drop=True)
    return total


def param_summary(model_name: str = WINNER) -> pd.DataFrame:
    cfg = model_config(model_name)
    wide = pd.read_csv(FIT_DIR / model_name / "subject_params_wide.csv")
    rows = []
    for name, lo, hi in zip(cfg.param_names, cfg.lb, cfg.ub):
        s = wide[f"param_{name}"].astype(float)
        rows.append(
            {
                "parameter": name,
                "bound_lo": float(lo),
                "bound_hi": float(hi),
                "min": float(s.min()),
                "p25": float(s.quantile(0.25)),
                "median": float(s.median()),
                "mean": float(s.mean()),
                "p75": float(s.quantile(0.75)),
                "max": float(s.max()),
                "sd": float(s.std()),
                "frac_at_lo": float((s <= lo + 1e-6).mean()),
                "frac_at_hi": float((s >= hi - 1e-6).mean()),
            }
        )
    out = pd.DataFrame(rows)
    ensure_dir(ANALYSIS_DIR)
    out.to_csv(ANALYSIS_DIR / f"param_summary_{SHORT[model_name]}.csv", index=False)
    return out


def write_trial_utilities(model_name: str = WINNER) -> pd.DataFrame:
    """Plug stored parameters into each experimental trial and attach cue-locked EEG."""
    cfg = model_config(model_name)
    params = pd.read_csv(FIT_DIR / model_name / "subject_params_wide.csv").set_index("subject_id")
    total = build_behavior_eeg()
    cache: dict[int, dict] = {}
    util_rows = []
    for _, row in total.iterrows():
        sid = int(row["Subject"])
        if sid not in params.index:
            continue
        if sid not in cache:
            g = total.loc[total["Subject"] == sid]
            theta = params.loc[sid, [f"param_{n}" for n in cfg.param_names]].to_numpy(float)
            data = ParticipantData(
                sid,
                pro_code_to_prob(g["pro"].to_numpy(float)),
                g["mon"].to_numpy(float),
                g["Choice"].to_numpy(float),
                (g["task"] == "double").astype(float).to_numpy(),
            )
            cache[sid] = {"index": list(g.index), "values": trial_components(theta, data, model_name), "data": data}
        info = cache[sid]
        i = info["index"].index(row.name)
        util_rows.append(
            {
                "money": float(info["data"].money[i]),
                "loss": float(info["values"]["value_loss_aversion"][i]),
                "defect": float(info["values"]["value_defect_aversion"][i]),
                "gain": float(info["values"]["value_gain"][i]),
                "utility": float(info["values"]["utility"][i]),
            }
        )
    util = pd.DataFrame(util_rows)
    fitted = total[total["Subject"].isin(params.index)].reset_index(drop=True)
    util = util.reset_index(drop=True)
    if len(fitted) != len(util) or not np.allclose(fitted["mon"].to_numpy(float), util["money"].to_numpy(float)):
        raise SystemExit("Trial money does not match the utility table")
    merged = fitted.copy()
    merged["loss"] = util["loss"].to_numpy()
    merged["defect"] = util["defect"].to_numpy()
    merged["gain"] = util["gain"].to_numpy()
    merged["utility"] = util["utility"].to_numpy()
    ensure_dir(DERIVED_DIR)
    merged.to_csv(DERIVED_DIR / "merged_behavior_erp_utilities.csv", index=False)
    print(f"Trial utilities: {len(merged)} rows, {merged['Subject'].nunique()} subjects")
    return merged


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit M0–M2 (identity κ) and compute AIC/BIC/LOOIC.")
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--n-starts", type=int, default=N_STARTS)
    p.add_argument("--maxiter", type=int, default=MAXITER)
    p.add_argument("--skip-fit", action="store_true", help="Reuse existing fits.")
    p.add_argument("--skip-loo", action="store_true", help="Skip the 10-fold CVIC.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_data()
    for name in MODEL_ORDER:
        if not args.skip_fit:
            print(f"Fitting {name} (n_starts={args.n_starts}, maxiter={args.maxiter})")
            fit_model(name, n_starts=args.n_starts, maxiter=args.maxiter, n_jobs=args.n_jobs)
        if not args.skip_loo:
            print(f"LOOIC {name}")
            compute_looic(name, maxiter=args.maxiter, n_jobs=args.n_jobs)
    comp = compare_ics()
    print(comp.to_string(index=False))
    print(param_summary(WINNER).to_string(index=False))
    write_trial_utilities(WINNER)


if __name__ == "__main__":
    main()
