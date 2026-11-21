# CausalAlpha

Research-grade pipeline testing which lead-lag relationships between futures markets survive out-of-sample, across regimes, after costs and multiple-testing correction.

**Project statement:** Causal structure between futures markets is unstable, and most backtests overstate it. This pipeline tests which relationships survive out-of-sample, across market regimes, after costs and multiple-testing correction.

## Problem statement
Most backtests overstate causal structure between futures markets. This project measures which lead-lag edges survive out-of-sample, across regimes, after costs and multiple-testing correction, and reports negative results honestly.

## Method
1. Download and clean daily futures data (yfinance), align calendars, compute log returns, ADF tests.
2. Rolling Granger causality (252-day windows, step 21, lags 1-5) with Benjamini-Hochberg FDR per window; edge stability.
3. Gaussian HMM (2-3 states, BIC) fit on initial training window only; filtered (causal) states; regime-conditional edge stability.
4. Walk-forward strategy using edges stable in the past training window; baselines; costs; metrics with bootstrap CIs; DSR and PBO.
5. Optional LLM sentiment ablation via Ollama (not available in this environment).

## Leakage prevention
- Every feature at time t uses data up to t only.
- Scalers, HMMs, and causal graphs are fit on the training window only.
- Filtered (forward) HMM states; no smoothed states that peek into the future.
- Walk-forward splits with expanding training windows; no shuffling.
- Purged k-fold with embargo provided for any CV.
- Transaction costs and slippage deducted; no leverage.

## Limitations
- yfinance continuous futures have roll gaps and occasional bad prints; WTI went negative in April 2020 (masked).
- Survivorship and data-quality limitations of free data.
- Regime sample sizes are small; HMM states are unstable.
- Sentiment ablation not run (no Ollama / no reliable historical headline source in this environment).
- Strategy did not beat baselines out-of-sample (see verdict).

## Reproduce
```
make run        # Phases 1-5
make test       # pytest
streamlit run app/streamlit_app.py
```

<!-- AUTO-GENERATED:START -->
## Auto-generated results (do not edit by hand)

### Data summary

| market   | ticker   |   n_prices | start      | end        |   price_mean |   price_min |   price_max |   return_mean |   return_std |   return_min |   return_max |   n_returns |
|:---------|:---------|-----------:|:-----------|:-----------|-------------:|------------:|------------:|--------------:|-------------:|-------------:|-------------:|------------:|
| Brent    | BZ=F     |       4255 | 2010-01-04 | 2026-11-20 |     78.4227  |     19.33   |    127.98   |   3.39538e-05 |   0.0234359  |   -0.279761  |    0.190774  |        4254 |
| WTI      | CL=F     |       4255 | 2010-01-04 | 2026-11-20 |     72.2713  |     10.01   |    123.7    |   7.01763e-06 |   0.0277907  |   -0.601676  |    0.319634  |        4254 |
| Gold     | GC=F     |       4255 | 2010-01-04 | 2026-11-20 |   1816.96    |   1050.8    |   5318.4    |   0.000269365 |   0.0109343  |   -0.11406   |    0.0589264 |        4254 |
| Copper   | HG=F     |       4255 | 2010-01-04 | 2026-11-20 |      3.56024 |      1.9395 |      6.8035 |   0.00011291  |   0.0152476  |   -0.251712  |    0.124437  |        4254 |
| NatGas   | NG=F     |       4255 | 2010-01-04 | 2026-11-20 |      3.36604 |      1.482  |      9.68   |  -0.000128169 |   0.0376563  |   -0.53846   |    0.381727  |        4254 |
| SP500    | ES=F     |       4255 | 2010-01-04 | 2026-11-20 |   3182.87    |   1014.25   |   7874      |   0.000426768 |   0.0109132  |   -0.109552  |    0.0934456 |        4254 |
| US10Y    | ZN=F     |       4255 | 2010-01-04 | 2026-11-20 |    123.364   |    104.203  |    140.328  |  -1.83765e-05 |   0.003427   |   -0.0217677 |    0.0197272 |        4254 |
| Dollar   | DX-Y.NYB |       4255 | 2010-01-04 | 2026-11-20 |     93.1158  |     72.93   |    114.11   |   6.86171e-05 |   0.00441801 |   -0.0239877 |    0.0203203 |        4254 |
| VIX      | ^VIX     |       4255 | 2010-01-04 | 2026-11-20 |     18.401   |      9.14   |     82.69   |   0.000136984 |   0.0772156  |   -0.442449  |    0.549246  |        4254 |

### ADF stationarity tests (5% level)

| market   |   adf_statistic |     p_value | stationary_5pct   |
|:---------|----------------:|------------:|:------------------|
| Brent    |        -10.4483 | 1.46058e-18 | True              |
| WTI      |        -10.6348 | 5.10355e-19 | True              |
| Gold     |        -66.9199 | 0           | True              |
| Copper   |        -27.8643 | 0           | True              |
| NatGas   |        -19.8005 | 0           | True              |
| SP500    |        -13.7896 | 8.97841e-26 | True              |
| US10Y    |        -47.8675 | 0           | True              |
| Dollar   |        -64.4345 | 0           | True              |
| VIX      |        -21.1201 | 0           | True              |

### Top 10 edges by stability (BH)

| edge            |   n_windows_sig_bh |   stability_bh |   stability_raw |
|:----------------|-------------------:|---------------:|----------------:|
| SP500 -> Dollar |                 25 |      0.13089   |        0.287958 |
| VIX -> Dollar   |                 21 |      0.109948  |        0.335079 |
| Brent -> WTI    |                 20 |      0.104712  |        0.146597 |
| VIX -> SP500    |                 17 |      0.0890052 |        0.366492 |
| US10Y -> Gold   |                 16 |      0.0837696 |        0.350785 |
| SP500 -> US10Y  |                 15 |      0.078534  |        0.240838 |
| WTI -> SP500    |                 15 |      0.078534  |        0.314136 |
| WTI -> Brent    |                 15 |      0.078534  |        0.272251 |
| Brent -> SP500  |                 14 |      0.0732984 |        0.314136 |
| Dollar -> SP500 |                 13 |      0.0680628 |        0.267016 |

### Multiple-testing inflation

- Windows: 191
- Mean edges/window (raw): 13.95
- Mean edges/window (BH): 1.96

### Regime counts

|   regime |   count |
|---------:|--------:|
|        0 |    1413 |
|        1 |    2244 |
|        2 |     577 |

### Strategy vs baselines (out-of-sample)

| strategy        |   ann_return |   ann_vol |     sharpe |   max_drawdown |   turnover |   hit_rate |
|:----------------|-------------:|----------:|-----------:|---------------:|-----------:|-----------:|
| causal_strategy |   0.00348168 |  0.107026 |  0.0325311 |      -0.475801 |   0.532876 |   0.421955 |
| buy_hold        |   0.0277368  |  0.167169 |  0.16592   |      -0.384605 |   0        |   0.494282 |
| momentum        |  -0.158256   |  0.200946 | -0.787555  |      -0.928068 |   0        |   0.469125 |
| mean_reversion  |   0.13756    |  0.200964 |  0.684498  |      -0.429868 |   0        |   0.52773  |
| random          |   0.0455004  |  0.170282 |  0.267206  |      -0.265703 |   0        |   0.505432 |

### Validation summary

|       dsr |   dsr_p_value |   n_trials |       pbo | best_baseline   |   best_baseline_sharpe |   strategy_sharpe |
|----------:|--------------:|-----------:|----------:|:----------------|-----------------------:|------------------:|
| -0.494637 |      0.310428 |         72 | 0.0714286 | mean_reversion  |               0.684498 |         0.0325311 |

### Verdict

```
Strategy did NOT beat baselines out-of-sample: causal Sharpe 0.033 vs best baseline mean_reversion 0.684. DSR p-value=0.310; PBO=0.071. Negative result: the lead-lag signal does not survive costs/OOS under this protocol.
```

<!-- AUTO-GENERATED:END -->
