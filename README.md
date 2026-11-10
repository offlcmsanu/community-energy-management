# ComErgy 2026 – Fair, Intelligent Community Energy Market

Solution for the ComErgy Grand Challenge 2026 benchmark (9 households, rooftop PV, 14 battery modules,
Sydney time-of-use tariff).  Question answered: *how can a local energy market maximise community benefit
while sharing it fairly?*

## Quick start (Windows / VS Code)
Requires Python 3.10+ (tested up to 3.14) - check with `python --version`.
1. Unzip so the folder is e.g. `E:\Documents\AIML\ComErgy`, then in VS Code: File > Open Folder > that folder
   (the folder that directly contains `app.py`).  Open a terminal with Ctrl+` .
2. **One-time setup**: `.\setup_windows.bat` (creates `.venv` and installs `requirements.txt`; a few minutes).
3. **Tests** (15-40 s, expect `14 passed`): `.\.venv\Scripts\python.exe -m pytest -q tests`
4. **Quick run** (~2-4 min): `.\.venv\Scripts\python.exe main.py --fast`
5. **Full run, six models** (first time ~7-15 min, cached afterwards): `.\.venv\Scripts\python.exe main.py`
   add `--rl` to also train the deep-RL battery agent.
6. **Dashboard**: `.\.venv\Scripts\python.exe -m streamlit run app.py`  (opens http://localhost:8501; press *Run / refresh model*).
The `.bat` files (`run_cli.bat`, `run_dashboard.bat`, `run_tests.bat`) do the same things by double-click.
Always call Python as `.\.venv\Scripts\python.exe ...` (not the `.exe` launchers in `.venv\Scripts`) so the project still works if the folder is moved or renamed.

The dataset is already in `data\`.  Use `--data <path>` for another copy.
Useful options: `--rl`, `--fast` (only trees + MLP), `--models gbm,mlp,gru`, `--dl-epochs 10`, `--forecaster auto|transformer|...`,
`--no-cache`, `--no-ml`, `--pv-mode realistic`, `--equity-lambda 0.8`, `--rep-days 60`, `--peak-weight 0.5`.
Results: `outputs\comergy_results.xlsx` + PNG figures.  Trained forecasts and the Shapley table are cached in `outputs\cache`.

## Architecture
| File | Role |
|---|---|
| `comergy/config.py` | constants + **assumptions** (battery efficiency, C-rate, emission factor) |
| `comergy/data_loader.py` | Excel loader, household table parser, PV model (`report` = identical to the PDF) |
| `comergy/forecasting.py` | **Deep learning + ML**: neural-network (MLP, 2 hidden layers, ReLU, Adam, L2) and gradient-boosting day-ahead forecasts of demand (9 homes) and GHI, out-of-fold, plus an ensemble |
| `comergy/nn.py` | **Deep-learning library written from scratch in NumPy**: Adam, MLP, **LSTM, GRU, temporal CNN, Transformer** (manual back-propagation, all gradient-checked) |
| `comergy/dl_forecaster.py` | day-ahead forecaster that plugs any of the four sequence encoders into the same pipeline (24 h history window + calendar/weather context, residual targets, early stopping, out-of-fold) |
| `comergy/rl_agent.py` | **Deep reinforcement learning**: Double-DQN battery controller trained on random days, evaluated out-of-sample |
| `comergy/optimizer.py` | **Optimisation**: LP battery dispatch (HiGHS) + rolling-horizon MPC with real-time reactive layer |
| `comergy/scenarios.py` | baseline, standalone (greedy / optimised), community pooled dispatch |
| `comergy/allocation.py` | **KMeans** representative days, exact **Shapley** values (511 coalitions), price-based LEM settlement, equity QP, no-worse-off top-up |
| `comergy/metrics.py` | cost, self-sufficiency, peak, CO2, Gini, Jain, energy burden, no-worse-off checks |
| `comergy/pipeline.py` | orchestration (used by CLI and UI) |
| `comergy/reporting.py` | Excel + figures |
| `main.py` / `app.py` | CLI / Streamlit dashboard |

## Method in one paragraph
Physical layer: all batteries form one *virtual community battery*, dispatched by MPC (48 h LP re-solved daily,
driven by ML forecasts, plus a real-time layer that absorbs actual PV surplus and covers actual peak-tier deficits).
Financial layer: the realised saving versus each household's **best standalone operation** is shared by
(a) a price-based local market (report-style mid-market, supply/demand price), (b) **Shapley value** of the cooperative
saving game (never negative → no household is worse off), or (c) the **recommended equity-adjusted Shapley**:
a small QP that moves savings towards low-income households (lower energy burden = bill/income) while every member keeps
at least α × their Shapley saving.  Sliders λ (need weight) and α are live in the dashboard.

## Results with default settings (AUD / year)
| | Cost |
|---|---|
| Baseline (grid only) | 22,613 |
| Standalone, each home optimised | ≈ 10,005 |
| Community (Transformer-driven MPC) | ≈ 5,916 (perfect foresight: 5,428) |

Self-sufficiency 33 % → 60 %; peak-tier grid imports −94 % vs baseline.  Recommended mechanism: nobody worse off,
M5 (lowest income, no DER) saves ≈ AUD 1,534 vs standalone, energy-burden Gini 0.35 (mid-market LEM: 0.45; the
supply/demand-price market leaves 2 households worse off without the top-up).

## Deep-learning components
All deep learning is implemented **from scratch in NumPy** (`comergy/nn.py`) - no PyTorch/TensorFlow, no extra install.
`tests/test_nn.py` verifies every hand-written backward pass (MLP, LSTM, GRU, temporal CNN, Transformer with multi-head attention
and LayerNorm) against numerical finite-difference gradients - the standard correctness test for manual back-propagation.

**Model comparison** - every forecaster predicts the next day for all 9 households + solar irradiance and is scored out-of-fold
(each hour predicted by a model that never saw it).  `Model ranking` table (CLI, Excel, dashboard) reports parameters, training time,
RMSE, skill vs "same as yesterday" and the community bill obtained when that model drives the battery controller.

| Model | Family | Params | Community RMSE [kW] |
|---|---|---|---|
| gbm | decision trees (non-deep baseline) | - | 2.50 |
| mlp | feed-forward NN (scikit-learn) | - | 2.31 |
| lstm | recurrent (NumPy) | 10.6 k | 2.56 |
| gru | recurrent (NumPy) | 9.2 k | 2.55 |
| tcn | 1-D convolutional (NumPy) | 11.2 k | 2.45 |
| transformer | self-attention, 2 heads (NumPy) | 15.7 k | 2.40 |
| naive "yesterday" | persistence | - | 2.76 |

Bill when each model drives the controller: transformer 5,916 | ensemble 5,930 | mlp 5,938 | tcn 5,942 | gru 5,953 | lstm 5,960 | gbm 5,965
(perfect foresight 5,428).  The `--forecaster auto` default picks the model with the lowest bill.  The exact numbers are printed by `python main.py` and saved in `outputs/comergy_results.xlsx`, sheet `Model ranking`.

**Deep RL**: `--rl` trains a Double-DQN (replay buffer, target network, 16-dim state, 7 actions) to control the shared battery; each of
two agents is tested only on weeks it never trained on.  Result: bill ~1-2 % above the optimiser (daily training cost falls from ~AUD 38 to ~10).

**What these results say (be upfront about them in a report)**
* Among the sequence models the Transformer and TCN beat the recurrent nets (LSTM/GRU); all deep sequence models beat "yesterday" but the
  simple MLP with engineered lag features is still the most accurate single model - one year of data (8,760 samples) favours strong
  inductive bias / hand-made features over learning from raw history.  More regularisation made the LSTM worse (hyper-parameter sweep).
* Better forecasts help only a little (tens of AUD per year): household loads are bursty and the ~AUD 500 gap to perfect foresight is
  missing information, not a modelling weakness.
* With this much solar and storage a greedy rule is already near-optimal, so the DQN can match but not beat the optimiser.

## Honest notes and assumptions
* The PDF's numbers are reproduced exactly for the baseline and PV output.  Standalone/LEM figures differ because my standalone
  reference is *price-optimised* (battery arbitrage), a stronger and fairer "no-worse-off" benchmark than the PDF's.
* Battery efficiency (95 % one-way), 10 % reserve, 0.5C power and the 0.66 kg/kWh emission factor are **assumptions**
  (`config.py`); the dataset does not provide them.
* The Shapley game uses 24 KMeans medoid days (cyclic SOC) instead of 365 days; shares are then applied to the realised
  year-long saving.  The grand-coalition cost error of this approximation is ≈ AUD 120 (shown in `shapley_diag`).
  Use `--rep-days 365` for the closest-to-exact result (slow).
* Temperature in the dataset stands in for a day-ahead weather forecast.
