# GridironIQ

NFL fantasy football analytics and machine learning platform using SQL, Python, Azure Machine Learning, and historical NFL data.

## Project Goal

GridironIQ will turn historical NFL data into weekly fantasy-football predictions and matchup-aware lineup recommendations.

The first milestone is a quarterback regression model that predicts custom fantasy points using only information that would have been available before kickoff.

## Initial Architecture

```text
nflverse data
     |
     v
Python ingestion
     |
     v
data/raw/
     |
     v
SQL + Python feature engineering
     |
     v
data/processed/
     |
     v
Regression models
     |
     v
Azure Machine Learning
     |
     v
Weekly predictions / fantasy decision engine
```

## Phase 1 — QB Model v1

Completed milestones:

1. Download weekly player/team stats, snap counts, schedules, depth charts, and supporting metadata.
2. Reconstruct the league's custom fantasy scoring from weekly nflverse data.
3. Build leakage-safe rolling QB and opponent-defense features.
4. Add role/snap-share, home/away, rest, venue/weather, betting-market, and pregame depth-chart context.
5. Compare Linear Regression, Random Forest, and Gradient Boosting with walk-forward validation.
6. Hold 2026 out of model selection as the live/demo season.

### QB Model v1 Benchmark

The selected model is a Random Forest evaluated with walk-forward validation:

| Fold | Train | Test | MAE |
| --- | --- | --- | ---: |
| 1 | 2021-2022 | 2023 | 8.560 |
| 2 | 2021-2023 | 2024 | 8.534 |
| 3 | 2021-2024 | 2025 | 8.316 |
| **Average** | — | **2023-2025** | **8.470** |

The simple Last-3 fantasy-points baseline averaged **9.639 MAE**, so QB Model v1 reduced MAE by **12.1%** across the three unseen-season folds.

The current 2026 live/demo snapshot produced **8.187 MAE on 109 QB-games** after fitting the selected model on 2021-2025. That number is not used for model selection and will change as the season grows.

## Repository Structure

```text
GridironIQ/
├── data/
│   ├── raw/
│   └── processed/
├── sql/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   └── models/
├── azure/
├── docs/
├── requirements.txt
└── README.md
```

## Data

Initial data comes from the nflverse project:

- Weekly player statistics
- Weekly team statistics
- Snap counts
- Schedules and game context
- Depth charts
- Injury/practice reports
- Player metadata

Raw downloaded data is intentionally excluded from Git. The ingestion script reproduces it locally.

## First Model

The first target will be:

```text
ACTUAL_FANTASY_POINTS
```

for quarterbacks.

Example features will include:

- previous-week fantasy points
- rolling 3-game and 5-game fantasy averages
- completions and attempts
- passing yards and passing touchdowns
- interceptions
- rushing yards and rushing touchdowns
- opponent passing yards allowed
- opponent passing touchdowns allowed
- opponent sacks and interceptions
- home/away status
- rest and neutral-site context
- roof, surface, temperature, and wind
- pregame spread and game total
- QB depth-chart rank / QB1 status

All predictive features must be calculated from games that happened before the game being predicted.

## Phase 2 — Availability and Personnel

The first injury/practice experiment did **not** improve the active-QB model: walk-forward MAE moved from **8.472** to **8.478**. Those features are therefore not part of the current v1 model.

That result is expected to be limited by the current training table, which only contains QBs who recorded game activity. An initial depth-chart candidate audit exposed an important data-quality issue: older weekly depth charts also contain entries during team bye weeks. Those rows must not become zero-point fantasy training examples.

The candidate audit now filters depth-chart rows against the actual regular-season schedule before measuring zero activity. After removing bye-week contamination, the 2021-2025 audit found **6,716 historical QB candidate rows**. Among **2,702 QB1 candidate rows**, **12.1% recorded zero QB activity**.

A deeper validation checked those 327 zero-activity QB1 rows: only **3.4%** still showed offensive snaps, while **98.8%** had another QB active for the same team/week. That supports using these rows as real availability/replacement examples for the v2 pregame candidate dataset rather than treating them as simple join failures.

### QB Model v2

QB v2 uses a two-stage architecture:

1. A Random Forest classifier estimates the probability that each listed QB receives a meaningful role (at least 50% of offensive snaps).
2. A Random Forest regressor predicts fantasy points conditional on a meaningful role.
3. The production gate selects the **highest role-probability QB on each team/week** and assigns that QB the conditional fantasy projection; other listed QBs receive zero.

The role classifier averaged **0.937 ROC AUC**, **0.870 accuracy**, **0.818 precision**, and **0.869 recall** in walk-forward validation across 2023-2025.

| Candidate scoring strategy | All-candidate MAE | Listed-QB1 MAE | Actual-role MAE | Played-QB MAE |
| --- | ---: | ---: | ---: | ---: |
| **Top role per team** | **5.321** | 9.915 | 9.690 | **8.512** |
| Listed QB1 only | 5.425 | 10.009 | 9.883 | 8.666 |
| 50% hard role gate | 5.927 | 9.995 | 9.778 | 8.755 |
| Direct candidate regression | 6.567 | **9.631** | 9.712 | 8.680 |
| Soft probability-weighted two-stage | 7.224 | 9.706 | **9.425** | 8.618 |

The production top-role-per-team gate is used because it performed best on the full pregame candidate pool and on QBs who actually played, while also enforcing the normal one-primary-QB-per-team structure. The table also preserves the tradeoffs: direct regression was best on the listed-QB1 subset, and soft probability weighting was best on the actual meaningful-role subset.

The v1 **8.470 MAE** and v2 **5.321 all-candidate MAE** are not directly comparable because v1 evaluates only QBs who recorded game activity, while v2 includes the entire pregame depth-chart candidate pool, including inactive and zero-point rows.

## Weekly Workflow

After the QB v2, RB v2, WR v2, and TE v2 models have each been trained once, the normal weekly workflow is a single command:

```cmd
py run_weekly.py
```

That command:

1. refreshes the live 2026 nflverse files and schedule/results data;
2. rebuilds the processed historical tables;
3. rebuilds the QB v2 pregame candidate dataset;
4. rebuilds the RB v2 pregame candidate dataset;
5. rebuilds the WR v2 pregame candidate dataset;
6. rebuilds the TE v2 pregame candidate dataset;
7. loads the saved QB v2 model bundle and generates the earliest upcoming week's QB rankings;
8. loads the saved RB v2 model bundle and generates the earliest upcoming week's RB rankings;
9. loads the saved WR v2 model bundle and generates the earliest upcoming week's WR rankings;
10. loads the saved TE v2 model bundle and generates the earliest upcoming week's TE rankings;
11. saves immutable, timestamped predictions for every candidate whose game has not kicked off.

The main weekly outputs are:

```text
data/processed/qb_v2_weekly_rankings.csv
data/processed/rb_v2_weekly_rankings.csv
data/processed/wr_v2_weekly_rankings.csv
data/processed/te_v2_weekly_rankings.csv
```

All future candidates and role probabilities are also saved to:

```text
data/processed/qb_v2_all_future_candidates.csv
data/processed/rb_v2_all_future_candidates.csv
data/processed/wr_v2_all_future_candidates.csv
data/processed/te_v2_all_future_candidates.csv
```

The weekly rankings file also includes `key_positives` and `key_negatives` columns. These are descriptive context signals built from recent fantasy form, betting environment, opponent pass-defense trends, pass rush, rest, home/away status, weather, injury status, and role confidence. They are intentionally labeled as context signals rather than exact Random Forest feature-attribution values.

The command-line report prints the top 10 QBs with a short explanation of why GridironIQ likes or dislikes the matchup.

Use `py run_weekly.py --retrain` only when intentionally refitting the official QB v2, RB v2, WR v2, and TE v2 models. Normal weekly refreshes do not need to retrain the 2021-2025 models.

## QB Start / Sit Comparison

After the weekly rankings have been generated, compare any two current-week quarterbacks with:

```cmd
py compare_qbs.py "Dak Prescott" "Bryce Young"
```

The comparison shows each QB's current GridironIQ rank, official projection, role confidence, matchup positives/negatives, and side-by-side context for recent fantasy form, opponent passing yards allowed, opponent sack pressure, game total, and role confidence.

It also includes a SHAP attribution section for the conditional Random Forest. Those values are additive fantasy-point contributions relative to the model's baseline and show which features actually pushed that QB's conditional projection up or down. SHAP explains the fantasy-points regressor only; the role classifier and team-selection gate remain separate model stages.

It finishes with a projection-based GridironIQ lean:

```text
Clear lean: START Dak Prescott over Bryce Young.
Projection gap: 4.65 FP.
```

The comparison tool accepts full names, unique partial names, and close name matches. If a player is present in the future candidate pool but was not selected as that team's projected primary QB, the tool will show that status rather than silently treating the player as a ranked starter.

## RB Model v1

The first running-back benchmark uses RBs who recorded fantasy-relevant game activity and evaluates only pregame information.

The RB feature table includes prior fantasy production, carries, targets, receptions, touches, opportunities, carry/target/opportunity share, offensive snap count/share, depth-chart rank, injury/practice status, game context, and opponent defensive form.

Walk-forward validation:

| Fold | Train | Test | Best MAE |
| --- | --- | --- | ---: |
| 1 | 2021-2022 | 2023 | 4.984 (Gradient Boosting) |
| 2 | 2021-2023 | 2024 | 5.014 (Random Forest) |
| 3 | 2021-2024 | 2025 | 5.105 (Linear Regression) |

Average 2023-2025 results:

| Model | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| **Gradient Boosting** | **5.043** | **6.963** | **0.337** |
| Random Forest | 5.073 | 6.992 | 0.331 |
| Linear Regression | 5.081 | 7.001 | 0.329 |
| Last-3 baseline | 5.391 | 7.564 | 0.217 |
| Mean baseline | 6.554 | 8.551 | -0.001 |

Gradient Boosting is the current RB v1 benchmark because it produced the lowest average walk-forward MAE. It improved on the Last-3 baseline by **6.5%**.

RB v1 is intentionally an active-game benchmark. It does not yet solve the harder pregame workload problem for inactive backs, committees, injury replacements, or multiple fantasy-relevant RBs on the same team.

The RB candidate audit found **9,601** historical pregame depth-chart rows. **75.8%** played at least one offensive snap, while **33.7%** recorded zero fantasy-relevant activity. RB1s averaged 11.92 fantasy points, but RB2s still averaged 6.31 points and reached 10+ opportunities in 30.6% of rows, confirming that RB production should not use a one-player-per-team gate.

### RB Model v2

RB v2 starts from the full pregame depth-chart candidate pool and adds player workload history, snap share, carry/target/opportunity share, RB-room competition, injury context, game environment, and opponent defense.

Three role definitions were tested with 2023-2025 walk-forward validation:

- `snap_35_role`: at least 35% offensive snaps
- `opp_10_role`: at least 10 carries + targets
- `meaningful_workload`: either 35% snaps or 10+ opportunities

The best all-candidate result was the **35% snap-share classifier + soft expected-points projection**, with **4.167 MAE**. The role classifier averaged **0.916 ROC AUC**, **0.109 Brier score**, **0.791 precision**, **0.821 recall**, and **0.852 accuracy**.

The production RB v2 formula is:

```text
P(35%+ offensive snaps) × fantasy points conditional on that role
```

No one-RB-per-team gate is applied because NFL backfields commonly support multiple fantasy-relevant players.

RB v1's 5.043 MAE and RB v2's 4.167 all-candidate MAE are not directly comparable because their evaluation populations differ.

Train the official RB v2 models with:

```cmd
py src\models\train_rb_v2.py
```

Then generate current-week RB rankings with:

```cmd
py src\models\predict_rb_v2.py
```

Compare two current-week running backs with:

```cmd
py compare_rbs.py "Bucky Irving" "Rachaad White"
```

The RB comparison shows official rank, soft expected projection, probability of reaching the 35% snap-share role threshold, conditional fantasy points, depth-chart position, recent workload/snap share, rushing matchup, RB-room competition, game environment, and descriptive positives/negatives.

It also uses SHAP to explain the conditional Gradient Boosting fantasy-points model. Those additive SHAP values explain the conditional points prediction only; the separate workload classifier then scales that prediction by the player's probability of reaching 35% offensive snaps.

## WR Model v1

The first wide-receiver benchmark uses WRs who recorded fantasy-relevant game activity and evaluates only pregame information.

The WR feature table includes recent fantasy production, targets, receptions, receiving yards/TDs, target share, reception share, receiving-yard share, yards per target, catch rate, offensive snap count/share, depth-chart rank, injury/practice status, game context, opponent passing-defense form, and available air-yards/YAC/first-down history.

The active-game dataset contains **11,692 historical WR-games from 2021-2025**. Historical depth-chart coverage is **93.4%** and prior-snap coverage is **96.1%**.

Walk-forward validation:

| Fold | Train | Test | Best MAE |
| --- | --- | --- | ---: |
| 1 | 2021-2022 | 2023 | 5.185 (Linear Regression) |
| 2 | 2021-2023 | 2024 | 5.193 (Gradient Boosting) |
| 3 | 2021-2024 | 2025 | 4.868 (Gradient Boosting) |

Average 2023-2025 results:

| Model | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| **Gradient Boosting** | **5.093** | **7.032** | **0.285** |
| Linear Regression | 5.124 | 7.065 | 0.278 |
| Random Forest | 5.126 | 7.050 | 0.281 |
| Last-3 baseline | 5.486 | 7.663 | 0.150 |
| Mean baseline | 6.370 | 8.325 | -0.002 |

Gradient Boosting is the WR v1 active-game benchmark. It reduced MAE by **7.2%** versus the Last-3 fantasy-points baseline.

WR v1 does not yet solve the full pregame participation problem. WR v2 starts from the depth-chart candidate pool and tests snap participation, target volume, target share, and combined role definitions rather than assuming one receiver per team.

The historical WR candidate audit found **15,540** pregame depth-chart rows. **34.7%** recorded zero fantasy-relevant activity, while **81.6%** played at least one offensive snap. WR1s averaged **10.14 fantasy points**, but WR2s and WR3s still averaged **4.27** and **3.19**, confirming that WR production should not use a one-player-per-team gate.

Role prevalence in the candidate pool:

- 50%+ offensive snaps: **44.7%**
- 65%+ offensive snaps: **35.3%**
- 5+ targets: **30.3%**
- 20%+ target share: **20.0%**
- 50% snaps OR 5+ targets: **46.6%**
- 65% snaps OR 5+ targets: **40.7%**

### WR Model v2

WR v2 evaluated six candidate role definitions with 2023-2025 walk-forward validation:

```text
snap_50_role
snap_65_role
target_5_role
target_share_20_role
snap50_or_target5_role
snap65_or_target5_role
```

For each role definition, GridironIQ compared a role classifier + soft expected-points model, a 50% hard role gate, and a direct candidate Gradient Boosting control.

The selected production architecture is the **65% snap-share classifier + soft expected-points projection** because it produced the lowest average all-candidate MAE:

| Role / scoring method | All MAE | WR1 MAE | WR2 MAE | WR3 MAE | Played MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| **65% snaps + soft expected points** | **4.190** | 6.538 | 3.944 | **2.942** | 4.725 |
| 50% snaps + soft expected points | 4.198 | **6.519** | **3.940** | 2.948 | 4.691 |
| 20% target share + soft expected points | 4.205 | 6.559 | 3.957 | 2.988 | 4.692 |
| 50% snaps OR 5+ targets + soft expected points | 4.208 | 6.521 | **3.940** | 2.969 | **4.673** |
| Direct candidate Gradient Boosting | 4.387 | 6.634 | 4.112 | 3.156 | 4.686 |

The 50% snap-role model remains an important near-tie: its classifier had slightly stronger discrimination (**0.910 AUC** vs **0.906**) and it performed slightly better on WR1, WR2, and played-WR subsets. The production choice follows the predeclared all-candidate MAE criterion rather than switching metrics after seeing the results.

The production WR v2 formula is:

```text
P(65%+ offensive snaps) × fantasy points conditional on that role
```

No one-WR-per-team gate is applied.

WR v1's 5.093 MAE and WR v2's 4.190 all-candidate MAE are not directly comparable because the evaluation populations differ.

Train the official WR v2 models with:

```cmd
py src\models\train_wr_v2.py
```

Then generate current-week WR rankings with:

```cmd
py src\models\predict_wr_v2.py
```

Compare two current-week wide receivers with:

```cmd
py compare_wrs.py "Amon-Ra St. Brown" "Puka Nacua"
```

The WR comparison includes rank, role probability, conditional points, soft expected projection, target/snap context, WR-room competition, game environment, calibrated uncertainty, and SHAP attribution for the conditional Gradient Boosting model.

## WR Two-Conditional Experiment (Research Only)

The current WR v2 production model scores receivers using
`P(65%+ snaps) × E(FP | 65%+ snaps)`. This assumes zero production
from the *below-threshold conditional branch*, although a receiver's
pregame projection remains positive whenever P(65%+ snaps) is positive.
Receivers can still earn targets and points in a lower-snap game, so we
tested whether estimating that branch explicitly improves predictions.

The **research-only** experiment tests:

```text
Current:  P(65%+ snaps) × E(FP | 65%+ snaps)

Proposed: P(65%+ snaps) × E(FP | 65%+ snaps)
        + P(below 65%) × E(FP | below 65%)

Control:  Direct candidate Gradient Boosting
```

It uses the same 2023-2025 walk-forward folds and pregame features as
the official WR v2 benchmark. It measures overall MAE as well as WR1,
WR2, WR3, played receivers, actual 65%+ role, actual below-65% role,
played receivers below 65%, and a **fixed pregame top-24 cohort**
selected from the current model's projection. It saves fold results
and individual out-of-fold predictions for inspection.

Run independently (no production model is changed):

```cmd
py src\models\experiment_wr_two_conditional.py
```

Outputs:

```text
data/processed/wr_two_conditional_results.csv
data/processed/wr_two_conditional_oof_predictions.csv
```

### Result: retain current WR v2 production

The two-conditional experiment **did not improve** walk-forward accuracy
over the same 2023-2025 test seasons. Mean absolute errors:

| Cohort | Current (zero below-threshold branch) | Two conditionals | Direct GB |
| --- | ---: | ---: | ---: |
| All candidates | **4.190** | 4.429 | 4.387 |
| WR1 | **6.538** | 6.604 | 6.634 |
| WR2 | **3.944** | 4.233 | 4.112 |
| WR3 | **2.942** | 3.254 | 3.156 |
| Played receivers | 4.725 | 4.722 | **4.686** |
| Actual 65%+ snap role | 6.800 | **6.541** | 6.591 |
| Actual below-65% role | **2.824** | 3.326 | 3.234 |
| Played, below 65% | **3.206** | 3.394 | 3.297 |
| Fixed baseline-projected top 24 | **8.084** | 8.092 | 8.087 |

The mixture raised predicted points on the below-65% group enough to
move mean signed error from **-0.492** (current model) to **+1.283**
(two conditionals), worsening that group's MAE by **0.502 FP**.
The all-candidate MAE deterioration also occurred in every test year.

**Decision:** keep the existing WR v2 role classifier, conditional
regressor, saved bundle, prediction intervals, ranking pipeline,
and start/sit comparison unchanged. The research script and its local
CSV outputs remain available for future investigation. In future, test
target/route opportunity or a more carefully calibrated below-threshold
branch rather than promoting this mixture based only on its formula.

## TE Model v1 (Benchmark)

TE v1 follows the active-game baseline used for RB and WR. It builds a
historical tight-end dataset from players who recorded fantasy-relevant
game activity, then uses 2023, 2024, and 2025 as consecutive unseen-season
walk-forward validation folds. The 2026 season is excluded from model
selection.

The model table includes pregame rolling fantasy production, targets,
receptions, receiving yards and touchdowns, receiving efficiency, team
receiving shares, offensive snap count/share, depth-chart TE rank,
injury/practice reports, game environment, and opponent passing-defense
trends. It also includes **prior TE-specific defensive trends** for targets,
receptions, receiving yards, and custom fantasy points allowed to tight
ends. These are descriptive opponent statistics, not verified route or
coverage assignments.

The v1 benchmarks are the training-population mean and last-three-game
fantasy average versus Linear Regression, Random Forest, and Gradient
Boosting. All rolling statistics are shifted to prior games, and
the validation also reports error on the **listed TE1** subgroup.

Build the TE v1 dataset and evaluate it with:

```cmd
py src\features\build_te_model_dataset.py
py src\models\walk_forward_te_v1.py
```

Outputs:

```text
data/processed/te_model_dataset.csv
data/processed/te_v1_walk_forward_results.csv
```

The active-game dataset contains **5,460 historical TE-games from
2021-2025**, with **94.9% depth-chart coverage** and **95.9% prior-snap
coverage**. The model evaluated **94 pregame features**.

Walk-forward validation:

| Fold | Train | Test | Gradient Boosting MAE | Listed TE1 MAE |
| --- | --- | --- | ---: | ---: |
| 1 | 2021-2022 | 2023 | 3.768 | 4.464 |
| 2 | 2021-2023 | 2024 | 3.929 | 4.845 |
| 3 | 2021-2024 | 2025 | 3.904 | 5.171 |

Average 2023-2025 results:

| Model | MAE | Listed TE1 MAE | RMSE | R² |
| --- | ---: | ---: | ---: | ---: |
| **Gradient Boosting** | **3.867** | **4.827** | **5.341** | **0.270** |
| Random Forest | 3.883 | 4.858 | 5.360 | 0.265 |
| Linear Regression | 3.985 | 4.901 | 5.471 | 0.233 |
| Last-3 baseline | 4.230 | 5.323 | 5.897 | 0.110 |
| Mean baseline | 4.700 | 5.410 | 6.252 | -0.001 |

Gradient Boosting is the TE v1 active-game benchmark, improving on
Last-3 MAE by **8.6%**. These are errors on TEs with recorded
fantasy-relevant activity; they are not full pregame-candidate errors.

### TE pregame candidate audit

The schedule-filtered 2021-2025 TE depth-chart pool contains **9,274**
historical candidates. **44.2%** recorded zero fantasy-relevant
activity and **82.6%** played at least one offensive snap.

Selected usage thresholds:

| Role or receiving threshold | Share of all candidates |
| --- | ---: |
| 35%+ offensive snaps | 50.2% |
| 50%+ offensive snaps | 35.2% |
| 1+ targets | 55.5% |
| 3+ targets | 29.7% |
| 5+ targets | 16.2% |
| 15%+ target share | 15.8% |
| 35%+ snaps OR 3+ targets | 51.8% |
| 50%+ snaps OR 3+ targets | 40.4% |

Snap share is not a receiving-role proxy on its own: **206**
candidates played 50%+ offensive snaps but earned zero targets,
and **2,047** played 35%+ snaps yet earned fewer than three
targets. Meanwhile, **109** earned five or more targets on
fewer than 50% of snaps.

Average fantasy points by TE depth-chart rank were TE1 **6.98**,
TE2 **2.58**, and TE3 **1.19**. This is a descriptive audit,
not a production role decision.

Reproduce the audit with:

```cmd
py src\features\audit_te_candidates.py
```

### TE Model v2 (Walk-forward experiments)

TE v2 starts from the complete pregame depth-chart candidate pool,
including inactive players and blocking-focused tight ends. It includes
pregame player usage, competition within the TE room, injury and game
context, prior opponent-wide receiving trends, and **opponent
TE-specific fantasy/target/reception/yard/TD trends**.

We retain eight role labels: `snap_35_role`, `snap_50_role`,
`target_1_role`, `target_3_role`, `target_5_role`,
`target_share_15_role`, `snap35_or_target3_role`,
and `snap50_or_target3_role`.

Each definition is evaluated with a Random Forest role classifier and a
conditional Gradient Boosting fantasy regressor, comparing probability-
weighted projections, a hard 50% probability gate, and a direct
candidate Gradient Boosting control.

Validation uses 2023, 2024, and 2025 as strictly forward test seasons
and excludes 2026 from model selection. In addition to overall MAE,
it evaluates listed TE1/TE2/TE3, TEs who played, TEs who received
at least three targets, 10+ FP games, and a **fixed pregame top-12 TE
cohort** selected by the direct control for each test week.

Build the candidate dataset and run the benchmark:

```cmd
py src\features\build_te_v2_candidate_dataset.py
py src\models\walk_forward_te_v2.py
```

Outputs:

```text
data/processed/te_v2_candidate_dataset.csv
data/processed/te_v2_walk_forward_results.csv
```

### TE v2 production selection

The candidate build reproduced **9,274 historical TE rows**, **115
pregame features**, **145 future/unplayed 2026 candidates**, and
**100% future opponent-defense feature coverage** as reported by the
local build. We tested all eight role labels on the same 2023-2025
walk-forward seasons.

Selected architecture: the **50% offensive-snap role classifier +
conditional Gradient Boosting regressor**, using a soft
probability-weighted fantasy projection. It had the lowest average
all-candidate MAE, and the role classifier had the highest AUC of
the eight definitions (0.906).

| Scoring method | All-candidate MAE | TE1 MAE | Played MAE | Fixed top-12 MAE |
| --- | ---: | ---: | ---: | ---: |
| **50% snaps + soft expected points** | **2.603** | **4.594** | 2.921 | 5.879 |
| 15% target share + soft expected points | 2.621 | 4.664 | **2.907** | 6.044 |
| 35% snaps + soft expected points | 2.636 | 4.609 | 2.951 | **5.870** |
| Direct candidate Gradient Boosting | 2.810 | 4.705 | 3.049 | 5.992 |

The 35% snap model was 0.009 FP better on the fixed projected top-12
cohort, while the 15% target-share model had slightly lower played-TE
MAE. The production choice follows the all-candidate MAE objective;
the small top-12 difference is a reason to keep evaluating TE
lineup decisions on fresh games. TE v1's 3.867 active-game MAE is
**not directly comparable** to TE v2's all-candidate MAE.

Official production projection:

```text
P(50%+ offensive snaps) × fantasy points conditional on that role
```

No one-TE-per-team gate is applied. Snap share includes blocking;
it must **not** be interpreted as the likelihood of running a route
or earning a target.

Train the new TE production bundle (including historical 80%
walk-forward residual uncertainty) and generate weekly rankings:

```cmd
py src\models\train_te_v2.py
py src\models\predict_te_v2.py
```

Compare two tight ends:

```cmd
py compare_tes.py "George Kittle" "Brock Bowers"
```

After TE has been trained once, `py run_weekly.py` runs
QB, RB, WR, and TE together. `py run_weekly.py --retrain` refits
all four position models. TE-specific opponent features have not
yet been isolated in a feature ablation experiment.



## Prediction Uncertainty

QB v2, RB v2, WR v2, and TE v2 include empirical **80% historical prediction intervals**. These are calibrated from out-of-season 2023-2025 walk-forward residuals using each position's actual production scoring rule.

The calibration is role-aware: residual ranges are estimated separately for low, medium, and high predicted role-confidence buckets when enough historical rows are available. This lets a high-confidence starter use a different historical error distribution than a low-confidence backup or committee player.

Weekly ranking files include:

```text
prediction_low_80
prediction_high_80
```

Start/sit comparison tools display the same interval and flag when the two players' historical 80% ranges overlap.

These ranges are empirical uncertainty estimates, not guaranteed floors or ceilings. They describe how the production models missed on comparable historical walk-forward predictions; future seasons can behave differently.

Because the uncertainty calibration is stored inside the saved model bundles, existing local bundles must be retrained once after pulling this update:

```cmd
py run_weekly.py --retrain
```

## Prospective Weekly Performance Tracker

GridironIQ now records **genuine prekickoff forecasts**. The final step
of `py run_weekly.py` snapshots the four positions' **all-future
candidate files** (not only the visible top-ranked players, which would
omit backup QBs and inflate all-candidate metrics).

A snapshot contains every candidate's original projection, role
probability and role definition, 80% historical interval, baseline
last-three fantasy average, game and model provenance, UTC capture
time, and the scheduled UTC kickoff. The snapshot script:

- accepts only games whose recorded kickoff is still in the future
  and whose schedule has no final score;
- requires explicit `gameday` and `gametime` values in `games.csv`;
  the nflverse schedule's `gametime` is treated as **Eastern local
  time** and converted to UTC with the `tzdata` package on Windows;
- refuses to mislabel stale prediction CSVs as newly generated
  forecasts (default maximum age 60 minutes);
- uses exclusive file creation in `data/snapshots/`, preserving
  previous snapshots rather than replacing them.

This is automatic in the normal weekly workflow. For a one-off
snapshot immediately after running all four position predictors:

```cmd
py src\evaluation\snapshot_predictions.py
```

To run the weekly pipeline without creating a snapshot (for example
for a local development experiment), use:

```cmd
py run_weekly.py --no-snapshot
```

**After games finish**, refresh the nflverse results and processed
tables (the normal weekly pipeline does both), then run:

```cmd
py src\evaluation\evaluate_predictions.py
```

The evaluator independently rechecks prekickoff timestamps against
the refreshed schedule. It selects the **latest verified prekickoff
prediction per player/game** by default, never retrains historical
forecasts or backfills predictions for games that have already started.
The alternative `--selection earliest` evaluates each player's
earliest valid saved forecast.

Only games with a final schedule score **and downloaded team boxscore**
are evaluated; a completed game with no player stat row is scored as
zero using GridironIQ's existing custom fantasy scoring. Missing team
boxscores are deferred, not silently converted to zeros. If a player's
offensive snap share cannot be verified, their fantasy error can still
be scored, but their role outcome is omitted from role metrics.

Reports in `data/processed/evaluation/`:

```text
prediction_results.csv
weekly_summary.csv
cumulative_summary.csv
role_calibration.csv
```

Metrics include MAE, RMSE, mean signed error, **projected-starter
MAE**, last-three baseline error on the *same eligible rows*, role
Brier score and 50%-threshold accuracy, genuinely prospective
80%-interval coverage, and correct ordering of projected-starter
pairs. Starter groups are the pregame **QB top 12, RB/WR top 24 and
TE top 12**. Pairwise ordering is a start/sit *ranking proxy*,
not a record of the user's real fantasy lineup decisions. Pairs
are compared only within the same season, week and position.

The snapshots are **ignored by Git**. Keep or back up
`data/snapshots/`: if that directory is lost, its pregame evidence
cannot be reconstructed honestly from refreshed predictions.
Historical walk-forward errors and new prospective errors should
be reported separately, with partial-week sample sizes shown.

Run the offline safety tests from the repository root:

```cmd
py -m unittest discover -s tests -v
```

## Long-Term Roadmap

Later phases will add travel distance, defensive personnel changes, supporting-cast availability, tracking/charting features when available, player correlation, and matchup-level win-probability recommendations. The proposed offensive-scheme, receiver-route, defensive-coverage and individual-defender matchup research is outlined in [Scheme-aware matchup roadmap](docs/scheme_aware_matchups.md).
