"""Group-level model recovery for M0–M2 (sum of CVIC). Saves the confusion table."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from fitting import ensure_dir, fit_subject, subject_looic_kfold
from model import (
    MAXITER,
    MODEL_ORDER,
    N_FOLDS,
    N_STARTS,
    RECOVERY_DIR,
    SEED,
    SHORT,
    ParticipantData,
    choice_probs,
    model_config,
    sample_params_uniform,
    simulate_choices,
    valid_participants,
)


def _pack(people) -> list[dict]:
    return [
        {
            "subject_id": int(p.subject_id),
            "prob": p.prob.tolist(),
            "money": p.money.tolist(),
            "condition": p.condition.tolist(),
        }
        for p in people
    ]


def _from_pack(pack: dict, choice: np.ndarray) -> ParticipantData:
    return ParticipantData(
        int(pack["subject_id"]),
        np.asarray(pack["prob"], float),
        np.asarray(pack["money"], float),
        np.asarray(choice, float),
        np.asarray(pack["condition"], float),
    )


def _fit_loo_job(payload: dict) -> dict:
    data = _from_pack(payload["pack"], np.asarray(payload["choice"], float))
    fit = fit_subject(payload["cand"], data, payload["n_starts"], payload["seed"], payload["maxiter"])
    cfg = model_config(payload["cand"])
    theta = np.asarray([fit[f"param_{n}"] for n in cfg.param_names], float)
    loo = subject_looic_kfold(payload["cand"], data, theta, payload["n_folds"], payload["seed"], payload["maxiter"])
    return {
        "true_model": payload["true_model"],
        "sim_id": payload["sim_id"],
        "cand_model": payload["cand"],
        "subject_id": int(payload["pack"]["subject_id"]),
        "nll": float(fit["nll"]),
        "aic": float(fit["aic"]),
        "bic": float(fit["bic"]),
        "looic": float(loo["looic"]),
    }


def _summarize(subject_scores: pd.DataFrame, n_subjects: int, candidates: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if subject_scores.empty:
        return pd.DataFrame(), pd.DataFrame()
    need = n_subjects * len(candidates)
    counts = subject_scores.groupby(["true_model", "sim_id"]).size()
    complete = counts[counts >= need].index
    if len(complete) == 0:
        return pd.DataFrame(), pd.DataFrame()
    keys = complete.to_frame(index=False)
    sub = subject_scores.merge(keys, on=["true_model", "sim_id"], how="inner")
    group = sub.groupby(["true_model", "sim_id", "cand_model"], as_index=False).agg(
        sum_nll=("nll", "sum"),
        sum_aic=("aic", "sum"),
        sum_bic=("bic", "sum"),
        sum_looic=("looic", "sum"),
        n_subjects=("subject_id", "nunique"),
    )
    trials = []
    for (true_model, sim_id), g in group.groupby(["true_model", "sim_id"]):
        if not set(candidates).issubset(set(g["cand_model"])):
            continue
        best = g.loc[g["sum_looic"].idxmin()]
        winner = str(best["cand_model"])
        trials.append(
            {
                "true_model": true_model,
                "true_short": SHORT[true_model],
                "sim_id": int(sim_id),
                "selected_model": winner,
                "selected_short": SHORT[winner],
                "correct": winner == true_model,
                "sum_looic_winner": float(best["sum_looic"]),
            }
        )
    return group, pd.DataFrame(trials)


def _write(out_dir, trials, group, true_models, candidates, meta) -> None:
    ensure_dir(out_dir)
    del group
    shorts_t = [SHORT[m] for m in true_models]
    shorts_c = [SHORT[m] for m in candidates]
    if trials.empty:
        conf = pd.DataFrame(0, index=shorts_t, columns=shorts_c)
        rate = pd.DataFrame(columns=["true_model", "true_short", "recovery_rate", "n"])
    else:
        conf = pd.crosstab(trials["true_short"], trials["selected_short"]).reindex(index=shorts_t, columns=shorts_c, fill_value=0)
        rate = trials.groupby(["true_model", "true_short"], as_index=False).agg(recovery_rate=("correct", "mean"), n=("sim_id", "count"))
        rate["true_model"] = pd.Categorical(rate["true_model"], true_models, ordered=True)
        rate = rate.sort_values("true_model").reset_index(drop=True)
    conf.to_csv(out_dir / "model_recovery_confusion.csv")
    rate.to_csv(out_dir / "model_recovery_rates.csv", index=False)
    (out_dir / "model_recovery_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M0–M2 group model recovery (ΣLOOIC).")
    p.add_argument("--n-per-model", type=int, default=50)
    p.add_argument("--n-starts", type=int, default=N_STARTS)
    p.add_argument("--maxiter", type=int, default=MAXITER)
    p.add_argument("--n-folds", type=int, default=N_FOLDS)
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    candidates = list(MODEL_ORDER)
    people = valid_participants()
    templates = _pack(people)
    n_subjects = len(templates)
    out_dir = ensure_dir(RECOVERY_DIR / "model")
    meta = {
        "candidates": list(candidates),
        "n_subjects": n_subjects,
        "n_per_model": args.n_per_model,
        "n_starts": args.n_starts,
        "maxiter": args.maxiter,
        "n_folds": args.n_folds,
        "selection": "sum_looic",
        "seed": args.seed,
    }
    print(f"Model recovery M0–M2: n_subjects={n_subjects}, n_per_model={args.n_per_model}")

    done: set[tuple] = set()
    existing = pd.DataFrame()
    subj_path = out_dir / "model_recovery_subject_scores.csv"
    if args.resume and subj_path.exists():
        existing = pd.read_csv(subj_path)
        done = set(zip(existing["true_model"], existing["sim_id"].astype(int), existing["cand_model"], existing["subject_id"].astype(int)))
        print(f"Resume: {len(done)} jobs already done")

    jobs = []
    for true_model in candidates:
        cfg = model_config(true_model)
        for sim_id in range(1, args.n_per_model + 1):
            sim_seed = int(args.seed + 100000 * (candidates.index(true_model) + 1) + sim_id)
            sim_rng = np.random.default_rng(sim_seed)
            thetas = [sample_params_uniform(cfg, sim_rng) for _ in templates]
            for pack, theta in zip(templates, thetas):
                data = _from_pack(pack, np.zeros(len(pack["prob"])))
                choice = simulate_choices(choice_probs(theta, data, true_model), sim_rng)
                for cand in candidates:
                    key = (true_model, sim_id, cand, int(pack["subject_id"]))
                    if key in done:
                        continue
                    jobs.append(
                        {
                            "true_model": true_model,
                            "sim_id": sim_id,
                            "cand": cand,
                            "pack": pack,
                            "choice": choice.tolist(),
                            "n_starts": args.n_starts,
                            "maxiter": args.maxiter,
                            "n_folds": args.n_folds,
                            "seed": sim_seed + 1000 * candidates.index(cand) + int(pack["subject_id"]),
                        }
                    )
    print(f"Jobs: {len(jobs)}")
    rows: list[dict] = []

    def flush():
        scores = pd.DataFrame(rows)
        if not existing.empty:
            scores = pd.concat([existing, scores], ignore_index=True).drop_duplicates(
                subset=["true_model", "sim_id", "cand_model", "subject_id"], keep="last"
            )
        if not scores.empty:
            scores.to_csv(subj_path, index=False)
            group, trials = _summarize(scores, n_subjects, candidates)
            _write(out_dir, trials, group, candidates, candidates, meta)
        return scores

    if not jobs:
        scores = flush()
    else:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
            futs = {ex.submit(_fit_loo_job, job): i for i, job in enumerate(jobs)}
            done_n = 0
            for fut in as_completed(futs):
                rows.append(fut.result())
                done_n += 1
                if done_n % 50 == 0 or done_n == len(jobs):
                    print(f"  finished {done_n}/{len(jobs)}")
                if done_n % 200 == 0 or done_n == len(jobs):
                    flush()
        scores = flush()

    group, trials = _summarize(scores if not scores.empty else existing, n_subjects, candidates)
    _write(out_dir, trials, group, candidates, candidates, meta)
    print(pd.read_csv(out_dir / "model_recovery_confusion.csv", index_col=0))
    print(pd.read_csv(out_dir / "model_recovery_rates.csv"))


if __name__ == "__main__":
    main()
