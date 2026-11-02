"""M0–M2 choice models. Trial data are read from raw_data/."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PACKAGE_ROOT / "raw_data"
RESULTS_DIR = PACKAGE_ROOT / "results"
FIT_DIR = RESULTS_DIR / "fits"
ANALYSIS_DIR = RESULTS_DIR / "model_comparison"
DERIVED_DIR = RESULTS_DIR / "derived"
RECOVERY_DIR = RESULTS_DIR / "recovery"
SIM_DIR = RESULTS_DIR / "simulation"

SEED = 20260610
N_STARTS = 20
MAXITER = 500
N_SIM_PARAM = 500
N_FOLDS = 10
WINNER = "aversion_linear_model"
INVALID_SUBJECT_IDS = (5, 6, 51)
EPS = 1e-12

PRO_CODE_TO_PROB = {1.0: 0.1, 2.0: 0.3, 3.0: 0.5, 4.0: 0.7, 5.0: 0.9}

# Participants who completed the cooperation block before the risk block.
BALANCE34 = (
    14, 25, 26, 29, 31, 34, 35, 36, 27, 32, 54, 37, 39, 40, 41, 42, 43, 45, 46, 48, 28, 30, 33,
)
CHANNEL_NAMES = [
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ",
    "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4",
    "PO6", "PO8", "O1", "OZ", "O2",
]

DISPLAY = {
    "random_model": "M0 Random",
    "prospect_model": "M1 CPT",
    "aversion_linear_model": "M2 CPT + betrayal",
}
SHORT = {"random_model": "M0", "prospect_model": "M1", "aversion_linear_model": "M2"}
PARAM_PLAIN = {
    "p": "p",
    "alpha_g": "α_g",
    "alpha_l": "α_l",
    "gamma": "γ",
    "lambda": "λ",
    "beta": "β",
    "kappa": "κ",
}


@dataclass(frozen=True)
class ModelConfig:
    name: str
    short: str
    param_names: tuple[str, ...]
    lb: np.ndarray
    ub: np.ndarray

    @property
    def n_params(self) -> int:
        return len(self.param_names)

    def midpoint(self) -> np.ndarray:
        return 0.5 * (self.lb + self.ub)


def _arr(*vals: float) -> np.ndarray:
    return np.asarray(vals, dtype=float)


MODELS: dict[str, ModelConfig] = {
    "random_model": ModelConfig(
        "random_model", "M0", ("p",), _arr(0.0), _arr(1.0)
    ),
    "prospect_model": ModelConfig(
        "prospect_model",
        "M1",
        ("alpha_g", "alpha_l", "gamma", "lambda", "beta"),
        _arr(0.01, 0.01, 0.05, 0.0, 0.01),
        _arr(1.0, 1.0, 1.0, 15.0, 50.0),
    ),
    "aversion_linear_model": ModelConfig(
        "aversion_linear_model",
        "M2",
        ("alpha_g", "alpha_l", "gamma", "lambda", "beta", "kappa"),
        _arr(0.01, 0.01, 0.05, 0.0, 0.01, 0.0),
        _arr(1.0, 1.0, 1.0, 15.0, 50.0, 2.0),
    ),
}
MODEL_ORDER = ("random_model", "prospect_model", "aversion_linear_model")


def model_config(name: str) -> ModelConfig:
    if name not in MODELS:
        raise KeyError(f"Unknown model {name!r}. Registered: {list(MODELS)}")
    return MODELS[name]


@dataclass
class ParticipantData:
    subject_id: int
    prob: np.ndarray
    money: np.ndarray
    choice: np.ndarray
    condition: np.ndarray

    @property
    def n_trials(self) -> int:
        return int(self.choice.size)

    def copy_with_choice(self, choice: np.ndarray) -> "ParticipantData":
        return ParticipantData(
            subject_id=self.subject_id,
            prob=self.prob.copy(),
            money=self.money.copy(),
            choice=np.asarray(choice, dtype=float).copy(),
            condition=self.condition.copy(),
        )

    def subset(self, idx: np.ndarray) -> "ParticipantData":
        idx = np.asarray(idx, dtype=int)
        return ParticipantData(
            self.subject_id,
            self.prob[idx],
            self.money[idx],
            self.choice[idx],
            self.condition[idx],
        )


def pro_code_to_prob(pro) -> np.ndarray:
    x = np.asarray(pro, dtype=float)
    finite = np.isfinite(x)
    if finite.any() and float(np.nanmax(x[finite])) <= 1.0 + 1e-8:
        return x
    out = np.full(x.shape, np.nan, dtype=float)
    for code, prob in PRO_CODE_TO_PROB.items():
        out[np.isclose(x, code)] = prob
    bad = finite & ~np.isfinite(out)
    if np.any(bad):
        raise ValueError(f"Unmapped probability codes: {np.unique(x[bad])}")
    return out


def _load_task_excel(path: Path, experiment: str, condition: int) -> pd.DataFrame:
    df = pd.read_excel(path)
    if "ExperimentName" in df.columns:
        df = df[df["ExperimentName"] == experiment].copy()
    if "Procedure" in df.columns:
        df = df[df["Procedure"] == "expe"].copy()
    if df.empty:
        raise ValueError(f"No experimental trials in {path} ({experiment})")
    out = pd.DataFrame(
        {
            "Subject": pd.to_numeric(df["Subject"], errors="coerce"),
            "pro": pd.to_numeric(df["pro"], errors="coerce"),
            "mon": pd.to_numeric(df["mon"], errors="coerce"),
            "Choice": pd.to_numeric(df["Choice"], errors="coerce"),
            "condition": float(condition),
        }
    )
    out = out.dropna(subset=["Subject", "pro", "mon", "Choice"]).copy()
    out["Subject"] = out["Subject"].astype(int)
    return out.reset_index(drop=True)


def load_data(
    risky_file: str | Path | None = None,
    coop_file: str | Path | None = None,
) -> list[ParticipantData]:
    """Risk then cooperation, in the recorded order within each task."""
    risky = Path(risky_file or (RAW_DIR / "single.xlsx"))
    coop = Path(coop_file or (RAW_DIR / "double.xlsx"))
    if not risky.exists() or not coop.exists():
        raise FileNotFoundError(f"Task files not found under {RAW_DIR}")
    print(f"[data] Excel {risky.name} + {coop.name} (Procedure=expe)")
    combined = pd.concat(
        [
            _load_task_excel(risky, "single", 0),
            _load_task_excel(coop, "double", 1),
        ],
        ignore_index=True,
    )
    participants = []
    for sid, g in combined.groupby("Subject", sort=True):
        participants.append(
            ParticipantData(
                subject_id=int(sid),
                prob=pro_code_to_prob(g["pro"].to_numpy(float)),
                money=g["mon"].to_numpy(float),
                choice=g["Choice"].to_numpy(float),
                condition=g["condition"].to_numpy(float),
            )
        )
    return participants


def valid_participants(participants: list[ParticipantData] | None = None) -> list[ParticipantData]:
    people = participants if participants is not None else load_data()
    drop = set(INVALID_SUBJECT_IDS)
    return [p for p in people if int(p.subject_id) not in drop]


def clip_probs(probs: np.ndarray) -> np.ndarray:
    return np.clip(probs, 1e-6, 1.0 - 1e-6)


def _prob_weight(prob: np.ndarray, gamma: float) -> np.ndarray:
    p = np.clip(prob, EPS, 1.0 - EPS)
    num = np.power(p, gamma)
    den = np.maximum(np.power(num + np.power(1.0 - p, gamma), 1.0 / gamma), EPS)
    return num / den


def _params(theta: np.ndarray, name: str) -> dict[str, float]:
    cfg = model_config(name)
    if len(theta) != cfg.n_params:
        raise ValueError(f"{name} needs {cfg.n_params} params, got {len(theta)}")
    return {k: float(v) for k, v in zip(cfg.param_names, theta)}


def choice_probs(theta: np.ndarray, data: ParticipantData, model_name: str) -> np.ndarray:
    """P(accept). M2 uses identity κ (no link): U -= κ · c · w_-."""
    if model_name == "random_model":
        return clip_probs(np.full(data.n_trials, float(theta[0])))
    p = _params(theta, model_name)
    w_plus, w_minus = _prob_weight(data.prob, p["gamma"]), _prob_weight(1.0 - data.prob, p["gamma"])
    utility = w_plus * np.power(data.money, p["alpha_g"]) + w_minus * (
        -p["lambda"] * np.power(0.5 * data.money, p["alpha_l"])
    )
    if "kappa" in p:
        utility = utility - p["kappa"] * data.condition * w_minus
    logits = np.clip(p["beta"] * utility, -700.0, 700.0)
    return clip_probs(1.0 / (1.0 + np.exp(-logits)))


def trial_components(theta: np.ndarray, data: ParticipantData, model_name: str) -> dict[str, np.ndarray]:
    if model_name == "random_model":
        pr = choice_probs(theta, data, model_name)
        z = np.zeros(data.n_trials)
        return {
            "value_gain": z,
            "value_loss_aversion": z,
            "value_defect_aversion": z,
            "utility": z,
            "invest_probs": pr,
        }
    p = _params(theta, model_name)
    w_plus = _prob_weight(data.prob, p["gamma"])
    w_minus = _prob_weight(1.0 - data.prob, p["gamma"])
    value_gain = w_plus * np.power(data.money, p["alpha_g"])
    value_loss = w_minus * (-p["lambda"] * np.power(0.5 * data.money, p["alpha_l"]))
    kappa = p.get("kappa", 0.0)
    value_defect = -kappa * data.condition * w_minus
    utility = value_gain + value_loss + value_defect
    return {
        "value_gain": value_gain,
        "value_loss_aversion": value_loss,
        "value_defect_aversion": value_defect,
        "utility": utility,
        "invest_probs": choice_probs(theta, data, model_name),
        "gain_weight": w_plus,
        "loss_weight": w_minus,
    }


def simulate_choices(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.binomial(1, clip_probs(probs)).astype(float)


def sample_params_uniform(cfg: ModelConfig, rng: np.random.Generator) -> np.ndarray:
    return cfg.lb + rng.random(cfg.n_params) * (cfg.ub - cfg.lb)
