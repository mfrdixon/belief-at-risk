#!/usr/bin/env python3
"""
LLM-Bayesian POMDP AI Risk Case Study

Uses Massive.com daily aggregate bars with adjusted=true and adjusted close field `c`.
Uses the OpenAI API to infer latent market-regime belief vectors, then combines
those vectors with Bayesian filtering.

Environment:
    MASSIVE_API_KEY=...
    OPENAI_API_KEY=...

Example:
    python src/run_case_study.py --tickers AAPL MSFT GOOGL AMZN JPM SPY \
        --start 2021-01-01 --end 2025-12-31 --sleep 10 --llm-sample-every 20
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from openai import OpenAI

REGIMES = ["risk_on", "neutral", "risk_off", "crisis"]


FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

TABLE_DIR = Path("tables")
TABLE_DIR.mkdir(exist_ok=True)

EVENTS = [
    ("2024-08-05", "2024-08-07", "Yen carry\nunwind", "gray"),
    ("2024-11-05", "2024-11-06", "U.S.\nelection", "gray"),
    ("2025-03-01", "2025-04-15", "Tariff / trade\npolicy shock", "orange"),
    ("2025-06-13", "2025-06-20", "Israel-Iran\nescalation", "red"),
    ("2025-07-30", "2025-07-31", "FOMC\ncaution", "gray"),
    ("2025-10-10", "2025-10-15", "U.S.-China\ntrade tensions", "orange"),
    ("2025-12-18", "2025-12-19", "Fed cut /\nguidance shift", "gray"),
]


@dataclass
class Config:
    tickers: List[str]
    start: str
    end: str
    sleep: float
    llm_sample_every: int
    llm_model: str
    evidence_temperature: float
    massive_api_key: str
    openai_api_key: str
    outdir: Path


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "SPY"])
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--sleep", type=float, default=10.0, help="Seconds to sleep between Massive.com ticker calls.")
    p.add_argument("--llm-sample-every", type=int, default=20, help="Call LLM every N trading days. Use 1 for daily calls.")
    p.add_argument("--llm-model", default="gpt-4.1-mini")
    p.add_argument("--evidence-temperature", type=float, default=0.75, help="Tempering eta for LLM evidence q_t^eta.")
    p.add_argument("--outdir", default=str(Path(__file__).resolve().parents[1]))
    args = p.parse_args()

    massive_key = os.getenv("MASSIVE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    if not massive_key:
        raise RuntimeError("Set MASSIVE_API_KEY in your environment.")
    if not openai_key:
        raise RuntimeError("Set OPENAI_API_KEY in your environment.")

    return Config(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        sleep=args.sleep,
        llm_sample_every=args.llm_sample_every,
        llm_model=args.llm_model,
        evidence_temperature=args.evidence_temperature,
        massive_api_key=massive_key,
        openai_api_key=openai_key,
        outdir=Path(args.outdir),
    )


def ensure_dirs(base: Path) -> None:
    for sub in ["data", "figures", "tables"]:
        (base / sub).mkdir(parents=True, exist_ok=True)


def fetch_massive_daily_adjusted(ticker: str, start: str, end: str, api_key: str) -> pd.DataFrame:
    """Fetch adjusted daily aggregates for one ticker.

    Massive endpoint:
    /v2/aggs/ticker/{ticker}/range/1/day/{from}/{to}?adjusted=true&sort=asc&limit=50000

    The `c` field is the close of the adjusted aggregate bar when adjusted=true.
    """
    url = f"https://api.massive.com/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key,
    }
    rows = []
    while True:
        r = requests.get(url, params=params, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"Massive request failed for {ticker}: {r.status_code} {r.text[:500]}")
        payload = r.json()
        rows.extend(payload.get("results", []))
        next_url = payload.get("next_url")
        if not next_url:
            break
        url = next_url
        params = {"apiKey": api_key}

    if not rows:
        raise RuntimeError(f"No aggregate rows returned for {ticker}.")

    df = pd.DataFrame(rows)
    # t is Unix ms timestamp; c is adjusted close when adjusted=true.
    df["date"] = pd.to_datetime(df["t"], unit="ms").dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.date
    df = df[["date", "c", "o", "h", "l", "v"]].rename(
        columns={"c": ticker, "o": f"{ticker}_open", "h": f"{ticker}_high", "l": f"{ticker}_low", "v": f"{ticker}_volume"}
    )
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")[[ticker]]


def load_prices(cfg: Config) -> pd.DataFrame:
    cache = cfg.outdir / "data" / "adjusted_close_prices.csv"

    if cache.exists():
        print(f"Reading cached prices from {cache}")
        return pd.read_csv(cache, parse_dates=["date"]).set_index("date").sort_index()

    frames = []
    for i, ticker in enumerate(cfg.tickers):
        print(f"Fetching {ticker} adjusted daily bars from Massive.com...")
        frames.append(fetch_massive_daily_adjusted(ticker, cfg.start, cfg.end, cfg.massive_api_key))
        if i < len(cfg.tickers) - 1:
            print(f"Sleeping {cfg.sleep:.1f}s before next ticker call...")
            time.sleep(cfg.sleep)

    prices = pd.concat(frames, axis=1).sort_index().dropna(how="all")
    prices = prices.ffill().dropna()

    prices.reset_index().to_csv(cache, index=False)
    print(f"Cached prices written to {cache}")

    return prices


def compute_features(returns: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    ew = returns.mean(axis=1)
    spy_col = "SPY" if "SPY" in returns.columns else returns.columns[-1]
    feat = pd.DataFrame(index=returns.index)
    feat["ew_return_1d"] = ew
    feat["ew_return_5d"] = ew.rolling(5).sum()
    feat["ew_return_20d"] = ew.rolling(20).sum()
    feat["ew_vol_20d"] = ew.rolling(20).std() * math.sqrt(252)
    feat["ew_vol_60d"] = ew.rolling(60).std() * math.sqrt(252)
    feat["cross_sectional_dispersion_20d"] = returns.rolling(20).std().mean(axis=1) * math.sqrt(252)
    feat["spy_return_20d"] = returns[spy_col].rolling(20).sum()
    rolling_peak = prices[spy_col].cummax()
    feat["spy_drawdown"] = prices[spy_col] / rolling_peak - 1.0
    feat["ew_return_next_1d"] = ew.shift(-1)
    return feat.dropna()


def prompt_for_beliefs(date: pd.Timestamp, row: pd.Series, regimes: List[str]) -> str:
    obs = {k: float(v) for k, v in row.items() if not k.startswith("ew_return_next") and np.isfinite(v)}
    return f"""
You are a quantitative model-risk observer. You are not making a trade. Your task is to infer a latent market regime from observations.

Date: {date.date()}
Regimes: {regimes}

Definitions:
- risk_on: broad positive risk appetite, constructive equity environment.
- neutral: mixed or low-conviction environment.
- risk_off: defensive environment with rising volatility, negative returns, or drawdowns.
- crisis: severe stress, abrupt drawdown, high volatility, or disorderly market behavior.

Observations are trailing daily equity-return features:
{json.dumps(obs, indent=2)}

Return only calibrated probabilities over the four regimes. They must be nonnegative and sum to one.
Also include a one-sentence rationale. Do not recommend a trade.
"""

def normalize_prob_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    x = df[cols].clip(lower=1e-12)
    return x.div(x.sum(axis=1), axis=0)

def normalize_regime_dict(d: Dict[str, float]) -> Dict[str, float]:
    vals = np.array([max(float(d.get(r, 0.0)), 1e-8) for r in REGIMES], dtype=float)
    vals = vals / vals.sum()
    return {r: float(vals[i]) for i, r in enumerate(REGIMES)}

def llm_regime_belief(client: OpenAI, model: str, prompt: str) -> Dict[str, float]:
    """Call OpenAI Responses API with structured JSON output.

    The code uses a strict schema so the output can be consumed by the Bayesian filter.
    """
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "risk_on": {"type": "number", "minimum": 0, "maximum": 1},
            "neutral": {"type": "number", "minimum": 0, "maximum": 1},
            "risk_off": {"type": "number", "minimum": 0, "maximum": 1},
            "crisis": {"type": "number", "minimum": 0, "maximum": 1},
            "rationale": {"type": "string"},
        },
        "required": ["risk_on", "neutral", "risk_off", "crisis", "rationale"],
    }

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": "You are a calibrated Bayesian market-regime observer. Output only valid schema-compliant JSON."},
            {"role": "user", "content": prompt},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "regime_belief",
                "schema": schema,
                "strict": True,
            }
        },
        temperature=0,
    )
    txt = response.output_text
    data = json.loads(txt)
    return normalize_regime_dict(data)


def heuristic_belief(row: pd.Series) -> Dict[str, float]:
    """Fallback and interpolation evidence when not calling the LLM.

    This is intentionally simple; the paper's methodology uses the LLM on sample dates
    and Bayesian filtering between LLM observations.
    """
    r20 = row["ew_return_20d"]
    vol = row["ew_vol_20d"]
    dd = row["spy_drawdown"]

    score_crisis = max(0.0, -r20 * 8 + vol * 1.5 + max(0.0, -dd - 0.10) * 5)
    score_off = max(0.0, -r20 * 5 + vol * 0.8 + max(0.0, -dd) * 2)
    score_on = max(0.0, r20 * 5 + max(0.0, 0.20 - vol))
    score_neutral = 0.6
    raw = np.array([score_on, score_neutral, score_off, score_crisis]) + 1e-3
    raw = raw / raw.sum()
    return {r: float(raw[i]) for i, r in enumerate(REGIMES)}


def bayes_filter_step(prev_b: np.ndarray, q: np.ndarray, transition: np.ndarray, eta: float) -> np.ndarray:
    prior = transition.T @ prev_b
    evidence = np.power(np.maximum(q, 1e-12), eta)
    post = prior * evidence
    return post / post.sum()


def infer_beliefs(cfg: Config, features: pd.DataFrame) -> pd.DataFrame:
    client = OpenAI(api_key=cfg.openai_api_key)
    # Persistent regimes: diagonal dominance with small transition probability.
    transition = np.array([
        [0.88, 0.09, 0.025, 0.005],
        [0.08, 0.84, 0.07, 0.01],
        [0.02, 0.10, 0.80, 0.08],
        [0.005, 0.045, 0.20, 0.75],
    ])
    b = np.ones(len(REGIMES)) / len(REGIMES)
    records = []

    for idx, (date, row) in enumerate(features.iterrows()):
        use_llm = (idx % cfg.llm_sample_every == 0)
        if use_llm:
            prompt = prompt_for_beliefs(date, row, REGIMES)
            try:
                q_dict = llm_regime_belief(client, cfg.llm_model, prompt)
                source = "openai"
            except Exception as e:
                print(f"OpenAI call failed on {date.date()}: {e}. Using heuristic evidence.")
                q_dict = heuristic_belief(row)
                source = "heuristic_fallback"
        else:
            q_dict = heuristic_belief(row)
            source = "heuristic_between_llm_calls"

        q = np.array([q_dict[r] for r in REGIMES], dtype=float)
        b = bayes_filter_step(b, q, transition, cfg.evidence_temperature)
        rec = {"date": date, "source": source}
        rec.update({f"q_{r}": q[i] for i, r in enumerate(REGIMES)})
        rec.update({f"b_{r}": b[i] for i, r in enumerate(REGIMES)})
        records.append(rec)

    out = pd.DataFrame(records).set_index("date")
    return out


def entropy(row: pd.Series) -> float:
    p = np.array([row[f"b_{r}"] for r in REGIMES], dtype=float)
    return float(-(p * np.log(np.maximum(p, 1e-12))).sum() / math.log(len(REGIMES)))


def kl_drift(beliefs: pd.DataFrame) -> pd.Series:
    B = beliefs[[f"b_{r}" for r in REGIMES]].values
    out = [0.0]
    for t in range(1, len(B)):
        p = np.maximum(B[t], 1e-12)
        q = np.maximum(B[t - 1], 1e-12)
        out.append(float((p * np.log(p / q)).sum()))
    return pd.Series(out, index=beliefs.index)


def realized_regime_labels(features: pd.DataFrame) -> pd.Series:
    r = features["ew_return_next_1d"]
    vol = features["ew_vol_20d"]
    labels = []
    for date, row in features.iterrows():
        if row["spy_drawdown"] < -0.20 or (row["ew_return_20d"] < -0.12 and row["ew_vol_20d"] > vol.quantile(0.75)):
            labels.append("crisis")
        elif row["ew_return_20d"] < -0.04 or row["spy_drawdown"] < -0.10:
            labels.append("risk_off")
        elif row["ew_return_20d"] > 0.04 and row["ew_vol_20d"] < vol.quantile(0.75):
            labels.append("risk_on")
        else:
            labels.append("neutral")
    return pd.Series(labels, index=features.index)


def compute_strategy(features: pd.DataFrame, beliefs: pd.DataFrame) -> pd.DataFrame:
    b = beliefs[[f"b_{r}" for r in REGIMES]].copy()
    exposure = (
        1.00 * b["b_risk_on"]
        + 0.60 * b["b_neutral"]
        + 0.25 * b["b_risk_off"]
        + 0.00 * b["b_crisis"]
    )
    next_ret = features.loc[beliefs.index, "ew_return_next_1d"].fillna(0.0)
    strat_ret = exposure * next_ret
    bench_ret = next_ret
    out = pd.DataFrame(index=beliefs.index)
    out["exposure"] = exposure
    out["strategy_return"] = strat_ret
    out["benchmark_return"] = bench_ret
    out["strategy_equity"] = (1 + strat_ret).cumprod()
    out["benchmark_equity"] = (1 + bench_ret).cumprod()
    return out

def macro_only_expected_returns(features, names, config):
    mu = pd.Series(0.0, index=names)

    risk_score = float(features.get("risk_score", 0.0))

    for name in names:
        if name in ["GLD", "TLT"]:
            mu[name] += 0.03 * risk_score
        else:
            mu[name] -= 0.02 * risk_score

    return mu


def belief_only_expected_returns(belief, names, config):
    mu = pd.Series(0.0, index=names)

    ai_boom = float(belief.get("AI_Boom", 0.0))
    soft = float(belief.get("Soft_Landing", 0.0))
    inflation = float(belief.get("Inflation_Shock", 0.0))
    recession = float(belief.get("Recession", 0.0))
    crisis = float(belief.get("Crisis", 0.0))

    for name in names:
        if name in ["NVDA", "MSFT", "GOOGL", "AMZN"]:
            mu[name] += 0.08 * ai_boom + 0.03 * soft
            mu[name] -= 0.04 * recession + 0.05 * crisis

        elif name in ["GLD"]:
            mu[name] += 0.05 * inflation + 0.04 * crisis

        elif name in ["TLT"]:
            mu[name] += 0.05 * recession + 0.03 * crisis
            mu[name] -= 0.03 * inflation

        else:
            mu[name] += 0.02 * soft
            mu[name] -= 0.03 * recession + 0.03 * crisis

    return mu

def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def var_cvar(x: pd.Series, alpha: float = 0.95) -> tuple[float, float]:
    losses = -x.dropna()
    var = float(losses.quantile(alpha))
    cvar = float(losses[losses >= var].mean()) if (losses >= var).any() else var
    return var, cvar


def perf_table(returns: pd.Series, equity: pd.Series) -> Dict[str, float]:
    ann = 252
    mu = returns.mean() * ann
    vol = returns.std() * math.sqrt(ann)
    sharpe = mu / vol if vol > 0 else np.nan
    cagr = float(equity.iloc[-1] ** (ann / len(equity)) - 1)
    var95, cvar95 = var_cvar(returns, 0.95)
    return {
        "CAGR": cagr,
        "Annualized Return": float(mu),
        "Annualized Volatility": float(vol),
        "Sharpe": float(sharpe),
        "Max Drawdown": max_drawdown(equity),
        "VaR 95": var95,
        "CVaR 95": cvar95,
    }

def generate_ablation_study(
    cfg: Config,
    returns: pd.DataFrame,
    features: pd.DataFrame,
    beliefs: pd.DataFrame,
) -> pd.DataFrame:

    out_tables = cfg.outdir / "tables"
    out_data = cfg.outdir / "data"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_data.mkdir(parents=True, exist_ok=True)

    q_cols = [f"q_{r}" for r in REGIMES]
    b_cols = [f"b_{r}" for r in REGIMES]

    raw = beliefs[q_cols].copy()
    raw.columns = b_cols

    filt = beliefs[b_cols].copy()

    labels = realized_regime_labels(features).loc[beliefs.index]

    def normalize(df):
        x = df.clip(lower=1e-12)
        return x.div(x.sum(axis=1), axis=0)

    raw = normalize(raw)
    filt = normalize(filt)

    def entropy_df(probs):
        k = probs.shape[1]
        return -(probs * np.log(probs.clip(lower=1e-12))).sum(axis=1) / np.log(k)

    def drift_df(probs):
        prev = probs.shift(1)
        d = (
            probs
            * np.log(probs.clip(lower=1e-12) / prev.clip(lower=1e-12))
        ).sum(axis=1)
        return d.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def brier_df(probs, labels):
        y = pd.DataFrame(0.0, index=probs.index, columns=probs.columns)
        for r in REGIMES:
            y.loc[labels == r, f"b_{r}"] = 1.0
        return ((probs - y) ** 2).sum(axis=1)

    def portfolio_returns_from_probs(probs):
        exposure = (
            probs["b_risk_on"]
            + 0.5 * probs["b_neutral"]
            - 0.5 * probs["b_risk_off"]
            - probs["b_crisis"]
        )
        return exposure.shift(1) * returns.loc[probs.index, "SPY"]

    def cvar95(x):
        x = pd.Series(x).dropna()
        if len(x) < 20:
            return np.nan
        q = x.quantile(0.05)
        return x[x <= q].mean()

    rows = []

    for name, probs, kappa in [
        ("Raw LLM evidence", raw, 0.0),
        ("LLM + Bayesian filter", filt, 0.0),
        ("LLM + Bayesian filter + controls", filt, 0.50),
    ]:
        H = entropy_df(probs)
        D = drift_df(probs)
        br = brier_df(probs, labels)

        pr = portfolio_returns_from_probs(probs)
        rolling_cvar = pr.rolling(60).apply(cvar95, raw=False).abs()
        bar = H * (1.0 + D) * rolling_cvar
        residual_bar = (1.0 - kappa) * bar

        pred = probs.idxmax(axis=1).str.replace("b_", "", regex=False)
        acc = (pred == labels).mean()

        perf = annualized_performance(pr)

        rows.append({
            "Model specification": name,
            "Mean entropy": H.mean(),
            "Maximum entropy": H.max(),
            "Mean KL drift": D.mean(),
            "Maximum KL drift": D.max(),
            "Brier score": br.mean(),
            "Regime accuracy": acc,
            "Mean BaR": bar.mean(),
            "Maximum BaR": bar.max(),
            "Mean residual BaR": residual_bar.mean(),
            "Maximum residual BaR": residual_bar.max(),
            "Annualized return": perf["Annualized return"],
            "Annualized volatility": perf["Annualized volatility"],
            "Sharpe ratio": perf["Sharpe ratio"],
            "Maximum drawdown": perf["Maximum drawdown"],
            "CVaR 95%": perf["CVaR 95\\%"],
            "Control effectiveness": kappa,
        })

    ablation = pd.DataFrame(rows)
    ablation.to_csv(out_data / "ablation_results.csv", index=False)

    formatted = ablation.copy()
    for c in formatted.columns:
        if c != "Model specification":
            formatted[c] = formatted[c].map(fmt3)

    formatted[
        [
            "Model specification",
            "Mean entropy",
            "Maximum entropy",
            "Mean KL drift",
            "Maximum KL drift",
            "Brier score",
            "Regime accuracy",
        ]
    ].to_latex(
        out_tables / "ablation_uncertainty_calibration.tex",
        index=False,
        escape=False,
        caption="Ablation study comparing uncertainty, belief instability, and calibration for raw LLM evidence, Bayesian-filtered beliefs, and control-adjusted beliefs.",
        label="tab:ablation_uncertainty_calibration",
    )

    formatted[
        [
            "Model specification",
            "Mean BaR",
            "Maximum BaR",
            "Mean residual BaR",
            "Maximum residual BaR",
            "Annualized return",
            "Annualized volatility",
            "Sharpe ratio",
            "Maximum drawdown",
            "CVaR 95%",
            "Control effectiveness",
        ]
    ].to_latex(
        out_tables / "ablation_risk_performance.tex",
        index=False,
        escape=False,
        caption="Ablation study comparing Belief-at-Risk, residual risk, portfolio performance, and downside consequence.",
        label="tab:ablation_risk_performance",
    )

    return ablation
def make_state_specs():
    return {
        "3-state": ["risk_on", "neutral", "stress"],
        "4-state": ["risk_on", "neutral", "risk_off", "crisis"],
        "5-state": ["strong_risk_on", "risk_on", "neutral", "risk_off", "crisis"],
    }


def transition_matrix_for_k(k: int, stay_prob: float = 0.85) -> np.ndarray:
    P = np.zeros((k, k))
    for i in range(k):
        P[i, i] = stay_prob
        if i > 0:
            P[i, i - 1] += (1 - stay_prob) / 2
        else:
            P[i, i] += (1 - stay_prob) / 2
        if i < k - 1:
            P[i, i + 1] += (1 - stay_prob) / 2
        else:
            P[i, i] += (1 - stay_prob) / 2
    return P / P.sum(axis=1, keepdims=True)



def entropy_from_probs(probs: pd.DataFrame) -> pd.Series:
    k = probs.shape[1]
    return -(probs * np.log(probs.clip(lower=1e-12))).sum(axis=1) / np.log(k)


def drift_from_probs(probs: pd.DataFrame) -> pd.Series:
    prev = probs.shift(1)
    d = (probs * np.log(probs.clip(lower=1e-12) / prev.clip(lower=1e-12))).sum(axis=1)
    return d.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def state_action_weights(states: list[str]) -> dict[str, float]:
    """
    Generic ordinal action scale.
    Highest-risk-appetite state maps to +1.
    Most-stressed state maps to -1.
    Intermediate states are evenly spaced.
    """
    vals = np.linspace(1.0, -1.0, len(states))
    return dict(zip(states, vals))


def heuristic_regime_probs(features: pd.DataFrame, states: list[str]) -> pd.DataFrame:
    """
    Robustness proxy for q_t using existing features.
    Replace this with OpenAI calls if you want each state specification
    to use a fresh LLM prompt.
    """

    out = pd.DataFrame(index=features.index)
    k = len(states)

    z_ret = features.get("spy_return_20d", features.iloc[:, 0]).fillna(0.0)
    z_vol = features.get("ew_vol_20d", features.iloc[:, 0]).fillna(0.0)
    z_dd = features.get("spy_drawdown", features.iloc[:, 0]).fillna(0.0)

    # Standardize for stable scoring
    ret = (z_ret - z_ret.mean()) / (z_ret.std() + 1e-12)
    vol = (z_vol - z_vol.mean()) / (z_vol.std() + 1e-12)
    dd = (z_dd - z_dd.mean()) / (z_dd.std() + 1e-12)

    stress_score = -ret + vol - dd

    grid = np.linspace(-1.5, 1.5, k)

    scores = []
    for g in grid:
        scores.append(-((stress_score - g) ** 2))

    scores = np.vstack(scores).T
    scores = scores - scores.max(axis=1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / probs.sum(axis=1, keepdims=True)

    for j, s in enumerate(states):
        out[f"q_{s}"] = probs[:, j]

    return out


def bayesian_filter_probs(q: pd.DataFrame, states: list[str], eta: float = 0.75) -> pd.DataFrame:
    q_cols = [f"q_{s}" for s in states]
    q_mat = normalize_prob_frame(q, q_cols).values

    k = len(states)
    P = transition_matrix_for_k(k)
    b_prev = np.ones(k) / k

    rows = []
    for i in range(len(q_mat)):
        prior = P.T @ b_prev
        post = prior * np.power(np.clip(q_mat[i], 1e-12, 1.0), eta)
        post = post / post.sum()
        rows.append(post)
        b_prev = post

    b = pd.DataFrame(rows, index=q.index, columns=[f"b_{s}" for s in states])
    return b


def realized_labels_for_states(features: pd.DataFrame, states: list[str]) -> pd.Series:
    """
    Creates heuristic ex-post labels by binning a stress score.
    This keeps labels comparable across 3-, 4-, and 5-state specifications.
    """

    z_ret = features.get("spy_return_20d", features.iloc[:, 0]).fillna(0.0)
    z_vol = features.get("ew_vol_20d", features.iloc[:, 0]).fillna(0.0)
    z_dd = features.get("spy_drawdown", features.iloc[:, 0]).fillna(0.0)

    ret = (z_ret - z_ret.mean()) / (z_ret.std() + 1e-12)
    vol = (z_vol - z_vol.mean()) / (z_vol.std() + 1e-12)
    dd = (z_dd - z_dd.mean()) / (z_dd.std() + 1e-12)

    stress_score = -ret + vol - dd

    bins = pd.qcut(stress_score, q=len(states), labels=list(reversed(states)), duplicates="drop")
    return bins.astype(str)


def brier_score_for_states(probs: pd.DataFrame, labels: pd.Series, states: list[str]) -> pd.Series:
    b_cols = [f"b_{s}" for s in states]
    y = pd.DataFrame(0.0, index=probs.index, columns=b_cols)

    for s in states:
        y.loc[labels == s, f"b_{s}"] = 1.0

    return ((probs[b_cols] - y) ** 2).sum(axis=1)


def consequence_returns_from_states(
    probs: pd.DataFrame,
    returns: pd.DataFrame,
    states: list[str],
    asset: str = "SPY",
) -> pd.Series:
    weights = state_action_weights(states)
    exposure = sum(weights[s] * probs[f"b_{s}"] for s in states)
    return exposure.shift(1) * returns.loc[probs.index, asset]


def rolling_cvar_abs(r: pd.Series, window: int = 60, alpha: float = 0.95) -> pd.Series:
    def cvar(x):
        x = pd.Series(x).dropna()
        if len(x) < 20:
            return np.nan
        q = x.quantile(1 - alpha)
        return abs(x[x <= q].mean())

    return r.rolling(window).apply(cvar, raw=False)


def run_state_space_robustness(
    cfg,
    returns: pd.DataFrame,
    features: pd.DataFrame,
    eta: float = 0.75,
    asset: str = "SPY",
) -> pd.DataFrame:
    """
    Runs robustness over 3-, 4-, and 5-state specifications.

    Outputs:
      outputs/data/state_space_robustness.csv
      outputs/tables/state_space_robustness.tex
    """

    out_tables = cfg.outdir / "tables"
    out_data = cfg.outdir / "data"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_data.mkdir(parents=True, exist_ok=True)

    rows = []

    for spec_name, states in make_state_specs().items():
        q = heuristic_regime_probs(features, states)
        b = bayesian_filter_probs(q, states, eta=eta)

        common_idx = returns.index.intersection(features.index).intersection(b.index)
        b = b.loc[common_idx]
        f = features.loc[common_idx]
        r = returns.loc[common_idx]

        labels = realized_labels_for_states(f, states).loc[common_idx]

        H = entropy_from_probs(b)
        D = drift_from_probs(b)
        BS = brier_score_for_states(b, labels, states)

        pred = b.idxmax(axis=1).str.replace("b_", "", regex=False)
        acc = (pred == labels).mean()

        cr = consequence_returns_from_states(b, r, states, asset=asset)
        cvar = rolling_cvar_abs(cr, window=60)
        bar = H * (1.0 + D) * cvar

        rows.append({
            "State specification": spec_name,
            "Number of states": len(states),
            "Mean entropy": H.mean(),
            "Maximum entropy": H.max(),
            "Mean KL drift": D.mean(),
            "Maximum KL drift": D.max(),
            "Brier score": BS.mean(),
            "Regime accuracy": acc,
            "Mean BaR": bar.mean(),
            "Maximum BaR": bar.max(),
        })

    robustness = pd.DataFrame(rows)
    robustness.to_csv(out_data / "state_space_robustness.csv", index=False)

    formatted = robustness.copy()
    for c in formatted.columns:
        if c not in ["State specification", "Number of states"]:
            formatted[c] = formatted[c].map(fmt3)

    latex = formatted.to_latex(
        index=False,
        escape=False,
        caption=(
            "Robustness of uncertainty, belief stability, calibration, "
            "and Belief-at-Risk across alternative latent-state specifications."
        ),
        label="tab:state_space_robustness",
        column_format="lccccccccc",
    )

    (out_tables / "state_space_robustness.tex").write_text(latex)

    return robustness

def write_latex_table(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    tex = df.to_latex(index=True, float_format=lambda x: f"{x:.4f}", escape=False, caption=caption, label=label)
    path.write_text(tex)



def experiment_parameters_table(cfg: Config, transition: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Return a table documenting all numerical-experiment parameters.

    This table is written to LaTeX so the paper discloses the modelling choices
    used in the empirical case study.
    """
    if transition is None:
        transition = np.array([
            [0.88, 0.09, 0.025, 0.005],
            [0.08, 0.84, 0.07, 0.01],
            [0.02, 0.10, 0.80, 0.08],
            [0.005, 0.045, 0.20, 0.75],
        ])
    policy_weights = {
        "risk_on": 1.00,
        "neutral": 0.60,
        "risk_off": 0.25,
        "crisis": 0.00,
    }
    rows = [
        ("Tickers", ", ".join(cfg.tickers), "User-specified equity basket; default is large-cap equities plus SPY."),
        ("Start date", cfg.start, "First date requested from Massive.com."),
        ("End date", cfg.end, "Last date requested from Massive.com."),
        ("Massive endpoint", "/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}", "Daily aggregate bars."),
        ("Adjusted data flag", "adjusted=true", "Uses split-adjusted aggregate bars."),
        ("Adjusted close field", "c", "Close field of the adjusted aggregate bar."),
        ("Sleep between ticker calls", f"{cfg.sleep:g} seconds", "Rate-limit/congestion guard between Massive ticker-history calls."),
        ("Return definition", "log(P_t/P_{t-1})", "Daily log returns from adjusted close prices."),
        ("Latent regimes", ", ".join(REGIMES), "Chosen a priori for interpretability and governance."),
        ("Number of regimes", str(len(REGIMES)), "Dimension of the belief state."),
        ("Initial prior", "Uniform over regimes", "Non-informative prior at the beginning of the run."),
        ("Transition matrix", np.array2string(transition, precision=3, separator=", "), "Stationary, diagonally dominant Markov prior over regimes."),
        ("LLM model", cfg.llm_model, "OpenAI model used as semantic observation model."),
        ("LLM sampling interval", f"Every {cfg.llm_sample_every} trading day(s)", "Use 1 for daily LLM calls; larger values reduce API cost."),
        ("LLM temperature", "0", "Deterministic structured regime-belief extraction."),
        ("Structured output schema", "risk_on, neutral, risk_off, crisis, rationale", "Machine-readable probabilities plus audit rationale."),
        ("Evidence tempering eta", f"{cfg.evidence_temperature:g}", "Trust parameter applied as q_t^eta in the Bayesian update."),
        ("Feature: 1d return", "Equal-weight basket", "Recent return observation."),
        ("Feature: 5d return", "5 trading days", "Short-horizon momentum/stress observation."),
        ("Feature: 20d return", "20 trading days", "Monthly momentum/stress observation."),
        ("Feature: 20d volatility", "Annualized rolling std", "Short-horizon realized risk."),
        ("Feature: 60d volatility", "Annualized rolling std", "Medium-horizon realized risk."),
        ("Feature: dispersion", "20d rolling cross-sectional std", "Cross-sectional stress indicator."),
        ("Feature: drawdown", "SPY / running peak - 1", "Benchmark drawdown state variable."),
        ("Policy weights", json.dumps(policy_weights), "Equity exposure assigned to each posterior regime."),
        ("Evaluation return", "Next-day equal-weight return", "One-step-ahead realized return used for strategy evaluation."),
        ("VaR/CVaR confidence", "0.95", "Tail-loss level for reported risk metrics."),
        ("Calibration proxy", "Brier score against heuristic ex-post labels", "Regime labels derived from realized return/drawdown rules."),
        ("Risk score", "entropy * rolling KL drift * rolling downside loss", "Local uncertainty-to-loss diagnostic."),
    ]
    return pd.DataFrame(rows, columns=["Parameter", "Value", "Role / assumption"])


def write_latex_string_table(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    tex = df.to_latex(index=False, escape=True, caption=caption, label=label, longtable=False)
    path.write_text(tex)


def build_results_df(prices, returns, features, beliefs, portfolio_returns=None):
    belief_cols = [
        "b_risk_on",
        "b_neutral",
        "b_risk_off",
        "b_crisis",
    ]

    beliefs = beliefs.copy()

    beliefs["entropy"] = -(
        beliefs[belief_cols]
        * np.log(beliefs[belief_cols].clip(lower=1e-12))
    ).sum(axis=1)

    beliefs["normalized_entropy"] = beliefs["entropy"] / np.log(len(belief_cols))

    belief_prev = beliefs[belief_cols].shift(1)

    beliefs["belief_drift"] = (
        beliefs[belief_cols]
        * np.log(
            beliefs[belief_cols].clip(lower=1e-12)
            / belief_prev.clip(lower=1e-12)
        )
    ).sum(axis=1)

    beliefs["belief_drift"] = (
        beliefs["belief_drift"]
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )

    if portfolio_returns is not None:
        beliefs["portfolio_return"] = portfolio_returns
    else:
        beliefs["portfolio_return"] = returns.mean(axis=1)

    def cvar95(x):
        if len(x) < 60:
            return np.nan
        q = np.quantile(x, 0.05)
        return x[x <= q].mean()

    beliefs["cvar95"] = (
        beliefs["portfolio_return"]
        .rolling(60)
        .apply(cvar95, raw=False)
        .abs()
    )

    beliefs["local_ai_risk_score"] = (
        beliefs["normalized_entropy"]
        * (1.0 + beliefs["belief_drift"])
        * beliefs["cvar95"]
    )

    beliefs["dominant_regime"] = beliefs[belief_cols].idxmax(axis=1)

    results_df = pd.concat(
        [
            prices.add_suffix("_price"),
            returns.add_suffix("_ret"),
            features,
            beliefs,
        ],
        axis=1,
    ).dropna(subset=[
        "local_ai_risk_score",
        "normalized_entropy",
        "b_risk_on",
        "b_neutral",
        "b_risk_off",
        "b_crisis",
    ])

    results_df = results_df.reset_index()

    if "index" in results_df.columns:
        results_df = results_df.rename(columns={"index": "date"})

    if "date" not in results_df.columns:
        results_df = results_df.rename(columns={results_df.columns[0]: "date"})

    results_df["date"] = pd.to_datetime(results_df["date"])

    return results_df


def add_event_windows(ax, y_top=None):
    for start, end, label, color in EVENTS:
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)

        ax.axvspan(start, end, alpha=0.13, color=color)

        if y_top is not None:
            mid = start + (end - start) / 2
            ax.text(
                mid,
                y_top,
                label,
                ha="center",
                va="top",
                fontsize=8,
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    fc="white",
                    ec="0.45",
                    alpha=0.9,
                ),
            )


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{name}.png", dpi=350, bbox_inches="tight")
    plt.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close()


def plot_ai_risk_score(df):
    df = df.copy().sort_values("date")

    y = df["local_ai_risk_score"]
    ma = y.rolling(20, min_periods=5).mean()

    fig, ax = plt.subplots(figsize=(13, 5.5))

    ax.plot(df["date"], y, linewidth=1.7, label="Local AI risk score")
    ax.plot(df["date"], ma, linewidth=2.2, linestyle="--", label="20-day moving average")

    add_event_windows(ax, y_top=y.max() * 0.97)

    ax.set_title("Local AI Risk Score: Uncertainty × Drift × Consequence", fontsize=15)
    ax.set_xlabel("Date")
    ax.set_ylabel("Risk score")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    ax.text(
        0.01,
        -0.23,
        "Peaks identify periods where posterior uncertainty, belief-state drift, "
        "and market consequence jointly increase.",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        ha="left",
    )

    savefig("figure1_ai_risk_score_journal")


def plot_posterior_entropy(df):
    df = df.copy().sort_values("date")

    y = df["normalized_entropy"]
    ma = y.rolling(20, min_periods=5).mean()

    fig, ax = plt.subplots(figsize=(13, 5.5))

    ax.axhspan(0.0, 0.33, alpha=0.08, label="Low uncertainty")
    ax.axhspan(0.33, 0.66, alpha=0.08, label="Medium uncertainty")
    ax.axhspan(0.66, 1.0, alpha=0.08, label="High uncertainty")

    ax.plot(df["date"], y, linewidth=1.7, label="Posterior entropy")
    ax.plot(df["date"], ma, linewidth=2.2, linestyle="--", label="20-day moving average")

    add_event_windows(ax, y_top=0.96)

    ax.set_ylim(0, 1.02)
    ax.set_title("Normalized Posterior Entropy: Bayesian Uncertainty in the Latent Regime", fontsize=15)
    ax.set_xlabel("Date")
    ax.set_ylabel(r"Entropy / $\log(K)$")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    ax.text(
        0.01,
        -0.23,
        "Entropy measures dispersion of the posterior belief state. "
        "High values indicate that several regimes remain plausible.",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        ha="left",
    )

    savefig("figure2_posterior_entropy_journal")


def plot_regime_probabilities(df):
    df = df.copy().sort_values("date")

    regime_cols = [
        "b_risk_on",
        "b_neutral",
        "b_risk_off",
        "b_crisis",
    ]

    labels = {
        "b_risk_on": "Risk-on",
        "b_neutral": "Neutral",
        "b_risk_off": "Risk-off",
        "b_crisis": "Crisis",
    }

    dominant = df[regime_cols].idxmax(axis=1)

    fig, ax = plt.subplots(figsize=(13, 5.8))

    start_idx = 0

    for i in range(1, len(df)):
        if dominant.iloc[i] != dominant.iloc[start_idx]:
            ax.axvspan(
                df["date"].iloc[start_idx],
                df["date"].iloc[i - 1],
                alpha=0.045,
            )
            start_idx = i

    ax.axvspan(df["date"].iloc[start_idx], df["date"].iloc[-1], alpha=0.045)

    for col in regime_cols:
        ax.plot(df["date"], df[col], linewidth=1.8, label=labels[col])

    add_event_windows(ax, y_top=0.98)

    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Bayesian-Filtered LLM Regime Probabilities", fontsize=15)
    ax.set_xlabel("Date")
    ax.set_ylabel("Posterior probability")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center right")

    transition_dates = []

    for i in range(1, len(df)):
        if dominant.iloc[i] != dominant.iloc[i - 1]:
            transition_dates.append((df["date"].iloc[i], labels[dominant.iloc[i]]))

    if transition_dates:
        step = max(1, len(transition_dates) // 6)

        for dt, lab in transition_dates[::step]:
            ax.axvline(dt, linestyle=":", linewidth=0.8, alpha=0.55)
            ax.text(
                dt,
                0.05,
                lab,
                rotation=90,
                fontsize=8,
                va="bottom",
                ha="right",
            )

    ax.text(
        0.01,
        -0.23,
        "The LLM supplies regime evidence; Bayesian filtering smooths the evidence "
        "through the transition model to produce persistent posterior beliefs.",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        ha="left",
    )

    savefig("figure3_regime_probabilities_journal")


def generate_all_journal_figures(results_df):
    plot_ai_risk_score(results_df)
    plot_posterior_entropy(results_df)
    plot_regime_probabilities(results_df)

# =========================================================
# LaTeX Table Generation for Journal-Grade Case Study
# =========================================================



def fmt3(x):
    """Format numeric values to three significant figures."""
    if pd.isna(x):
        return "--"
    if isinstance(x, (int, np.integer)):
        return f"{x:d}"
    if isinstance(x, (float, np.floating)):
        return f"{x:.3g}"
    return str(x)


def write_latex_table(df, filename, caption, label):
    latex = df.to_latex(
        index=False,
        escape=False,
        column_format="lll",
        caption=caption,
        label=label,
    )

    with open(TABLE_DIR / filename, "w") as f:
        f.write(latex)

def cfg_get(cfg, names, default="--"):
    for name in names:
        if hasattr(cfg, name):
            return getattr(cfg, name)
    return default


def generate_parameter_table(cfg):
    start_date = cfg_get(cfg, ["start_date", "from_date", "date_start", "START_DATE"])
    end_date = cfg_get(cfg, ["end_date", "to_date", "date_end", "END_DATE"])
    tickers = cfg_get(cfg, ["tickers", "TICKERS", "equity_tickers"], [])

    if isinstance(tickers, (list, tuple)):
        tickers = ", ".join(tickers)

    sleep_seconds = cfg_get(cfg, ["sleep_seconds", "sleep_time", "api_sleep_seconds"], 10)

    rows = [
        ["Sample start date", "Start date", start_date],
        ["Sample end date", "End date", end_date],
        ["Equity universe", r"$\mathcal{A}$", tickers],
        ["Massive.com sleep interval", r"$\Delta_{\mathrm{API}}$", f"{float(sleep_seconds):.3g} seconds"],
        ["Adjusted equity prices", "Adjusted close", "True"],
        ["Return frequency", r"$\Delta t$", "Daily"],
        ["Number of latent regimes", r"$K$", "4"],
        ["Latent regime set", r"$\mathcal{S}$", "Risk-on, Neutral, Risk-off, Crisis"],
        ["Initial regime prior", r"$b_0$", "(0.25, 0.25, 0.25, 0.25)"],
        ["Bayesian filter floor", r"$\varepsilon$", f"{1e-12:.3g}"],
        ["Rolling CVaR window", r"$W$", "60 trading days"],
        ["CVaR confidence level", r"$\alpha$", "95\\%"],
        ["Risk-on exposure", r"$w(RO)$", f"{1.0:.3g}"],
        ["Neutral exposure", r"$w(N)$", f"{0.5:.3g}"],
        ["Risk-off exposure", r"$w(RF)$", f"{-0.5:.3g}"],
        ["Crisis exposure", r"$w(C)$", f"{-1.0:.3g}"],
    ]

    df = pd.DataFrame(
        rows,
        columns=[
            "Experimental quantity",
            "Mathematical notation",
            "Numerical specification",
        ],
    )

    write_latex_table(
        df,
        "experiment_parameters.tex",
        "Numerical specifications and modelling assumptions used in the LLM--Bayesian filtering experiment.",
        "tab:experiment_parameters",
    )

    return df        



def annualized_performance(r, periods_per_year=252):
    r = pd.Series(r).dropna()

    if len(r) == 0:
        return {
            "Annualized return": np.nan,
            "Annualized volatility": np.nan,
            "Sharpe ratio": np.nan,
            "Maximum drawdown": np.nan,
            "VaR 95\\%": np.nan,
            "CVaR 95\\%": np.nan,
        }

    equity = (1.0 + r).cumprod()

    ann_ret = equity.iloc[-1] ** (periods_per_year / len(r)) - 1.0
    ann_vol = r.std() * np.sqrt(periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_dd = drawdown.min()

    var95 = np.quantile(r, 0.05)
    cvar95 = r[r <= var95].mean()

    return {
        "Annualized return": ann_ret,
        "Annualized volatility": ann_vol,
        "Sharpe ratio": sharpe,
        "Maximum drawdown": max_dd,
        "VaR 95\\%": var95,
        "CVaR 95\\%": cvar95,
    }


def generate_performance_table(results_df):
    perf = annualized_performance(results_df["portfolio_return"])

    rows = [
        ["Annualized return", r"$\mu_{\mathrm{ann}}$", fmt3(perf["Annualized return"])],
        ["Annualized volatility", r"$\sigma_{\mathrm{ann}}$", fmt3(perf["Annualized volatility"])],
        ["Sharpe ratio", r"$SR$", fmt3(perf["Sharpe ratio"])],
        ["Maximum drawdown", r"$MDD$", fmt3(perf["Maximum drawdown"])],
        ["Daily value-at-risk, 95\\%", r"$VaR_{0.95}$", fmt3(perf["VaR 95\\%"])],
        ["Daily conditional value-at-risk, 95\\%", r"$CVaR_{0.95}$", fmt3(perf["CVaR 95\\%"])],
    ]

    df = pd.DataFrame(
        rows,
        columns=[
            "Performance statistic",
            "Mathematical symbol",
            "Estimated value",
        ],
    )

    write_latex_table(
        df,
        "performance_metrics.tex",
        "Out-of-sample performance diagnostics for the belief-state portfolio induced by the LLM--Bayesian filter.",
        "tab:performance_metrics",
    )

    return df


def generate_regime_summary_table(results_df):
    regime_cols = {
        "b_risk_on": "Risk-on",
        "b_neutral": "Neutral",
        "b_risk_off": "Risk-off",
        "b_crisis": "Crisis",
    }

    rows = []

    for col, name in regime_cols.items():
        prob = results_df[col].dropna()
        dominant_share = (results_df["dominant_regime"] == col).mean()

        rows.append(
            [
                name,
                fmt3(prob.mean()),
                fmt3(prob.std()),
                fmt3(prob.min()),
                fmt3(prob.max()),
                fmt3(dominant_share),
            ]
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "Latent regime",
            "Mean posterior probability",
            "Posterior standard deviation",
            "Minimum posterior probability",
            "Maximum posterior probability",
            "Dominant-regime frequency",
        ],
    )

    latex = df.to_latex(
        index=False,
        escape=False,
        column_format="lccccc",
        caption="Summary statistics for Bayesian-filtered LLM posterior regime probabilities.",
        label="tab:regime_summary",
    )

    with open(TABLE_DIR / "regime_summary.tex", "w") as f:
        f.write(latex)

    return df


def generate_risk_diagnostics_table(results_df):
    diagnostics = [
        [
            "Mean normalized posterior entropy",
            r"$\bar{H}/\log K$",
            results_df["normalized_entropy"].mean(),
        ],
        [
            "Maximum normalized posterior entropy",
            r"$\max_t H_t/\log K$",
            results_df["normalized_entropy"].max(),
        ],
        [
            "Mean belief-state drift",
            r"$\overline{D}_{KL}(b_t\Vert b_{t-1})$",
            results_df["belief_drift"].mean(),
        ],
        [
            "Maximum belief-state drift",
            r"$\max_t D_{KL}(b_t\Vert b_{t-1})$",
            results_df["belief_drift"].max(),
        ],
        [
            "Mean local AI risk score",
            r"$\bar{\mathcal{R}}$",
            results_df["local_ai_risk_score"].mean(),
        ],
        [
            "Maximum local AI risk score",
            r"$\max_t \mathcal{R}_t$",
            results_df["local_ai_risk_score"].max(),
        ],
        [
            "Mean rolling CVaR consequence",
            r"$\overline{CVaR}_{0.95}$",
            results_df["cvar95"].mean(),
        ],
        [
            "Maximum rolling CVaR consequence",
            r"$\max_t CVaR_{0.95,t}$",
            results_df["cvar95"].max(),
        ],
    ]

    rows = [
        [name, symbol, fmt3(value)]
        for name, symbol, value in diagnostics
    ]

    df = pd.DataFrame(
        rows,
        columns=[
            "Risk diagnostic",
            "Mathematical symbol",
            "Estimated value",
        ],
    )

    write_latex_table(
        df,
        "risk_diagnostics.tex",
        "Uncertainty, belief-instability and consequence diagnostics for the proposed local AI risk measure.",
        "tab:risk_diagnostics",
    )

    return df


def generate_all_latex_tables(cfg, results_df):
    parameter_table = generate_parameter_table(cfg)
    performance_table = generate_performance_table(results_df)
    regime_summary_table = generate_regime_summary_table(results_df)
    risk_diagnostics_table = generate_risk_diagnostics_table(results_df)

    return {
        "parameters": parameter_table,
        "performance": performance_table,
        "regime_summary": regime_summary_table,
        "risk_diagnostics": risk_diagnostics_table,
    }




def main() -> None:
    cfg = parse_args()
    ensure_dirs(cfg.outdir)
    prices = load_prices(cfg)
    returns = np.log(prices / prices.shift(1)).dropna()
    features = compute_features(returns, prices).dropna()
    beliefs = infer_beliefs(cfg, features)
    ablation_results = generate_ablation_study(
            cfg=cfg,
            returns=returns,
            features=features,
            beliefs=beliefs,
        )


    state_robustness = run_state_space_robustness(
        cfg=cfg,
        returns=returns,
        features=features,
        eta=cfg.evidence_temperature,
        asset="SPY",
    )

    print(state_robustness)    

    portfolio_returns = (
    (
          beliefs["b_risk_on"]
        + 0.5 * beliefs["b_neutral"]
        - 0.5 * beliefs["b_risk_off"]
        - beliefs["b_crisis"]
    )
    .shift(1)
    * returns["SPY"]
    )

    results_df = build_results_df(
            prices=prices,
            returns=returns,
            features=features,
            beliefs=beliefs,
            portfolio_returns=portfolio_returns
    )

    results_df.to_csv("results_df.csv", index=False)

    generate_all_journal_figures(results_df)
 
    latex_tables = generate_all_latex_tables(cfg, results_df)
    #make_outputs(cfg, prices, returns, features, beliefs)
    print(f"Done. Outputs written under {cfg.outdir}")


if __name__ == "__main__":
    main()
