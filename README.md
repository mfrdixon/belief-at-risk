# Belief at Risk

**Belief at Risk: Quantifying Agentic AI Model Risk with LLM-Inferred Bayesian State Filters**

This package contains a rigorous LaTeX paper and a Python case study showing how to use the OpenAI API as a structured latent-regime observation model, combine those regime probabilities with Bayesian filtering, and quantify agentic AI model risk through entropy, belief drift, calibration, VaR/CVaR, and portfolio outcomes.

## Contents

- `paper/llm_bayesian_agentic_ai_risk.tex` — full academic paper with executive summary cover sheet.
- `src/run_case_study.py` — runnable Python experiment.
- `figures/` — generated figures after running the script.
- `tables/` — generated LaTeX tables after running the script, including `experiment_parameters.tex`.
- `data/` — generated CSV data artifacts.

## Setup

```bash
pip install -r requirements.txt
export MASSIVE_API_KEY="your_massive_key"
export OPENAI_API_KEY="your_openai_key"
```

## Run

```bash
python src/run_case_study.py \
  --tickers AAPL MSFT GOOGL AMZN JPM SPY \
  --start 2021-01-01 \
  --end 2025-12-31 \
  --sleep 10 \
  --llm-sample-every 20
```

The script requests Massive.com daily aggregate bars with `adjusted=true` and uses field `c` as the adjusted close. It sleeps between ticker-history calls according to `--sleep`, defaulting to 10 seconds.

## Outputs

The script writes:

- `figures/regime_probabilities.png`
- `figures/posterior_entropy.png`
- `figures/ai_risk_score.png`
- `figures/equity_curves.png`
- `tables/performance_metrics.tex`
- `tables/risk_diagnostics.tex`
- `tables/regime_summary.tex`
- `tables/experiment_parameters.tex`

## Notes

The empirical strategy is illustrative. The research contribution is the model-risk architecture: the LLM supplies uncertain evidence over latent regimes, the Bayesian filter produces auditable belief states, and risk is measured by the losses induced by actions taken under those beliefs.
