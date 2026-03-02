---
layout: post
title: Predicting PGA Tour Outcomes with XGBoost and Monte Carlo Simulation
subtitle: Building a Full-Stack Golf Prediction System with 91 Features, Rolling Form Signals, and Live Betting Integration
thumbnail-img: ../assets/img/shap_top20.png 
share-img:
tags: Sports-Analytics, Golf, Machine-Learning, XGBoost, Feature-Engineering, Sports-Betting, Monte-Carlo
author: Charles Benfer
---

## Results First

Across seven completed 2026 PGA Tour events — Sony Open, American Express, Farmers Insurance Open, WM Phoenix Open, AT&T Pebble Beach Pro-Am, Genesis Invitational, and Cognizant Classic — the model placed the actual top-20 finishers in its top-10 picks at a rate of **37.1%**. Given that a random baseline would produce around 13% (20 top-20 finishers out of 150 players), that represents roughly a 2.8x lift over chance.

The global XGBoost classifier achieved a **top-20 AUC of 0.730** and **top-10 AUC of 0.732** on a held-out test set of 2024–2025 events — held out temporally, never seen during training. The model correctly ranked eventual winners inside its top-10 predictions in three of seven 2026 events.

Those numbers won't win a Kaggle competition. But in a sport defined by variance — 156-player fields, a cut that eliminates half the field, and where the world's best player wins roughly 20% of the time — they represent meaningful predictive signal extracted from a genuinely hard problem.

This post walks through how I built the system end-to-end: data scraping, 91-feature engineering, four XGBoost models, Monte Carlo simulation, and a live betting dashboard with Bovada odds integration.

---

## Why Golf Is Hard to Predict

Golf presents a prediction problem unlike most team sports. A typical PGA Tour event has 150+ players, 72 holes played over four days, and a cut after 36 holes that eliminates roughly half the field. A player who finishes T-2 at one tournament might miss the cut at the next — not because they played worse, but because golf's variance is enormous.

Three structural challenges stand out:

**1. Field size dilutes signal.** In baseball, a team plays 162 games against 30 opponents, generating thousands of data points per season. In golf, the best players enter 25–30 tournaments per year, generating a few hundred individual tournament-week observations. There isn't enough per-player data to train individual models.

**2. Course specificity matters — but is hard to measure.** Augusta National rewards long drivers who can shape the ball right-to-left. Pebble Beach rewards iron accuracy and approach play. A player perfectly suited for one course may be mismatched for another, but quantifying "course fit" from historical results requires years of data and careful feature design.

**3. Form is everything and nothing.** A player's recent form — last 3 events — matters enormously. But a player ranked 80th in the world can beat the field one week with hot putting. Separating genuine form from noise is the core challenge.

The solution I settled on: a global model trained across all tournaments and players, using carefully engineered features that capture player quality, course fit, and recent form.

---

## The Data Pipeline

### Leaderboard Scraping (ESPN API)

The foundation is per-round leaderboard data for every PGA Tour event from 2015 through the present 2026 season, scraped from ESPN's public sports API. Each row represents one player in one tournament-year, with final position, round scores, and score-to-par.

One significant data quality issue: the 2020 season contains duplicate rows due to the COVID-interrupted schedule. These were deduplicated on `(playerName, position, year)`. The Zurich Classic — a team event using a two-man scramble format — was filtered out entirely, since team-format finish positions carry no individual predictive signal.

Final dataset: **~58,000 player-tournament rows** across 2016–2025.

### Statistics Scraping (PGA Tour GraphQL API)

Season-level statistics came from the PGA Tour's internal GraphQL API, which provides per-player, per-season averages for ~80 statistics: Strokes Gained components (Off the Tee, Approach, Around the Green, Putting, Total), driving distance, fairway percentage, GIR, scrambling, and more.

A second scraper fetches per-event SG data — round-by-round Strokes Gained for each tournament — enabling rolling form calculations across the actual tournament schedule. This was the more complex scrape: roughly 2.5 hours of runtime to fetch all events from 2015 onward with checkpoint resumption.

### Data Merge and Master Training Dataset

The leaderboard CSV and stats CSV were merged on `(playerName, stats_year)` where `stats_year = tournament_year - 1`. This is the critical design choice: **we always use the prior season's statistics as features**, so there's no leakage from using in-season stats to predict in-season results. A player's 2025 tournament predictions are made using their 2024 SG and accuracy statistics.

---

## Feature Engineering: 91 Features Across 6 Groups

### Group 1: Core SG and Accuracy Stats (Prior Season)

The backbone of the model. For each player-tournament row, we attach their prior season's Strokes Gained Total, Strokes Gained components (OTT, APP, ATG, Putting), driving distance, driving accuracy, GIR percentage, scrambling, and about 15 other accuracy-based statistics. Where a player has no prior season stats (rookies, partial seasons), these features are left as NaN — XGBoost handles missing values natively by learning the optimal split direction for each null.

**SG Total** turned out to be the single most predictive individual statistic, consistently appearing in the top-3 features by importance. This aligns with the established golf analytics consensus: SG Total is the best single measure of overall player quality.

### Group 2: Prior Season Form Features

Seven features derived from the player's prior season tournament results: number of starts, cut rate, top-10 rate, top-20 rate, average percentile finish, average finish position, and win count. These complement the per-stroke statistics by capturing competitive outcomes — a player can have excellent SG numbers but also have the temperament to convert them into top-10 finishes (or not).

`prev_season_top20_rate` and `prev_season_top10_rate` both ranked in the model's top-15 features, suggesting that prior-season results carry independent signal beyond the raw SG metrics.

### Group 3: Course History Features

Four features capturing each player's historical performance at the specific course being played:

- `course_hist_avg_finish`: rolling mean of prior finishes (3-year window, minimum 1 appearance)
- `course_hist_appearances`: number of times the player has played this event
- `course_hist_best_finish`: best historical finish at this course
- `course_hist_made_cut_rate`: historical cut rate at this course

These were computed with strict temporal care: for a 2023 tournament row, only 2015–2022 finishes contribute. No look-ahead.

### Group 4: Course SG Signature + Player Fit Score

This was the most technically interesting feature group. For each tournament, I computed Pearson correlations between each SG component and `percentile_finish` using all historical players at that event — yielding a "SG signature" for each course. Augusta, for example, has a high correlation between Approach SG and finish position; courses with tight fairways show high correlations for Off-the-Tee accuracy.

From these per-course correlations, I computed a `course_fit_sg_score` for each player: a weighted sum of the player's SG components scaled by how much each SG dimension matters at that particular course. A player with elite approach play gets a higher fit score at approach-heavy courses.

The course SG signatures were saved as a separate artifact (`data/models/course_sg_signatures.pkl`) and reused at inference time, enabling forward predictions for 2026 events.

### Group 5: YoY SG Change

A single feature: `sg_total_yoy_change = SG_Total(year) - SG_Total(year-1)`. Players improving their SG Total year-over-year are on an upward trajectory; declining players may be over-ranked by the backward-looking SG Total average.

This feature ranked 31st out of 91 with importance 0.00862 in the top-20 model. It provided modest but consistent signal — confirming that trajectory matters beyond the level.

### Group 6: Rolling Form Features (Last N Events Played)

The most iteratively refined feature group. Thirty-one features capturing each player's performance in their last 1, 3, and 5 tournament appearances:

- `rollN_avg_finish`, `rollN_avg_percentile`, `rollN_cut_rate`, `rollN_events_played`
- `rollN_sg_total`, `rollN_sg_ott`, `rollN_sg_app`, `rollN_sg_atg`, `rollN_sg_putt` (from per-event data)
- `rollN_sg_delta`: rolling SG Total minus full-season average (hot/cold streak indicator)
- `roll5_sg_total_slope`: linear trend of SG Total over last 5 events
- `sg_ema_14w`: exponentially weighted SG Total with 14-week half-life

**One critical design decision that significantly impacted results**: the initial implementation defined "last N events" as the N most recently *scheduled* PGA Tour events, regardless of whether the player entered them. A player who skipped four tournaments would show NaN for `roll1_avg_finish` even if they had played the week before, simply because four events had been scheduled in the interim.

Switching to **last N events the player actually played in** — using rank-based filtering on event sequence numbers — reduced the rolling feature NaN rate from **50.5% to 8.9%** overall. The impact on feature importance was immediate and dramatic: `roll5_avg_finish` became the **single most important feature** in the model (8.9% importance), up from its previous position outside the top-5. Top-20 AUC improved by 0.008 and P@10 on 2026 events improved by 2.8 percentage points.

The lesson: when your features are half NaN, the model can't use them. The data quality fix mattered more than any algorithmic change.

---

## What Didn't Work

### Per-Tournament XGBoost Models

The intuitive idea: train a separate XGBoost model for each tournament, so the model for Augusta specifically learns which features predict Masters performance. In theory, this would capture course-specific patterns that the global model misses.

In practice, it failed comprehensively. Each tournament has ~100 players × 8–10 historical years = ~800–1,000 training rows. The global model had ~38,000 rows. Data starvation produced models that couldn't generalize:

Numbers below are averaged across six early-season 2024–25 events (Sony Open, American Express, Farmers Insurance, WM Phoenix Open, AT&T Pebble Beach, Genesis Invitational):

| Target | Global AUC (avg) | Per-Tournament AUC (avg) | Δ |
|--------|-----------------|--------------------------|---|
| made_cut | 0.584 | 0.580 | −0.005 |
| top_20 | 0.673 | 0.610 | **−0.064** |
| top_10 | 0.654 | 0.551 | **−0.103** |
| percentile (ρ) | 0.294 | 0.244 | **−0.050** |

The global model's `tournament_encoded` feature already captures average difficulty differences between events. Course-specific player fit is better handled through the course SG signature features than through separate models. The per-tournament experiment is archived in `notebooks/archive/`.

### XGBRanker (Learning-to-Rank)

I also experimented with XGBoost's `rank:ndcg` objective, which directly optimizes for ranked list quality rather than binary classification. The idea was that ranking players *relative to each other within a field* might be better suited to the tournament prediction problem than independent binary classifiers.

Results were disappointing across the board. The ranker produced a score with standard deviation of only 0.14 — nearly constant across the field — compared to probability outputs from the classifiers that spanned a much wider range. AUC dropped from 0.718 to 0.698, and Precision@K was worse at every K ≥ 10.

The root cause: 48% of training rows are grade-0 (missed cut), which created a coarse 4-grade scale that the ranker couldn't effectively discriminate. Calibrated classifier probabilities, which can assign continuous scores from 0 to 1, provided much richer ranking signal than the NDCG-optimized ranker.

---

## The Model Architecture

### Four XGBoost Models

The final system trains four global XGBoost models on the same feature set:

1. **`model_made_cut`** — binary classifier predicting whether a player makes the 36-hole cut
2. **`model_top20`** — binary classifier predicting top-20 finish
3. **`model_top10`** — binary classifier predicting top-10 finish
4. **`model_percentile`** — regressor predicting percentile finish position within the field

Each model was trained on 2016–2022 data, validated on 2023, and tested on 2024–2025. All four classifiers were calibrated using `CalibratedClassifierCV` with Platt scaling (sigmoid), which significantly improved probability calibration without sacrificing discrimination. sklearn 1.6's `FrozenEstimator` wrapper was used to calibrate on held-out data without retraining.

The percentile regressor was trained on a **logit-transformed target**: `logit(percentile_finish)` rather than raw percentile. At inference, a sigmoid is applied to recover the probability scale. This improved Spearman correlation from 0.271 to 0.297 by concentrating model capacity on the top and bottom of the field where the signal is richest.

Tournament names were encoded using `OrdinalEncoder` as a `tournament_encoded` feature, allowing the model to learn per-event difficulty offsets without requiring separate models.

### Final Test Set Performance (2024–25)

| Model | AUC / ρ |
|-------|---------|
| made_cut | 0.683 |
| top_20 | 0.730 |
| top_10 | 0.732 |
| percentile (Spearman ρ) | 0.297 |

### Monte Carlo Simulation

The four models produce probabilities, but bettors need win probabilities. No classifier naturally produces win probabilities across a 150-player field.

I added a Monte Carlo simulation layer on top of the models:

1. For each player, the percentile model predicts their expected finish percentile.
2. 100,000 simulations are run per field. In each simulation:
   - Each player's cut is drawn from a Bernoulli with `p = p_made_cut`
   - Each player who makes the cut draws a performance score from `N(pred_pct, σ²)`
   - Players who miss the cut receive a penalty score of 100.0
3. The player with the lowest simulated score wins that simulation. Win probability = fraction of simulations won.

The noise parameter σ = 0.2121 was estimated empirically as the standard deviation of `(predicted_percentile − actual_percentile)` on 2024–25 cut-makers. This calibrates the simulation to match the observed prediction uncertainty. Spearman ρ = 0.33 between predicted and actual percentile on the test set.

The MC simulation doesn't improve AUC (it's a transformation on top of existing model outputs), but it adds something the classifiers can't provide: **win probabilities**. These are essential for expected value calculations in betting.

---

## The Betting Dashboard

The prediction system is wrapped in a Streamlit application that runs locally and integrates live odds from Bovada.

### Architecture

- **`predict_week.py`** contains all inference logic: `load_artifacts()`, `predict_field()`, `_compute_rolling_for_field()`, and `simulate_field_mc()`. The notebook pipeline calls these functions directly; the dashboard also imports from them.
- **`betting_dashboard/app.py`** auto-detects the next upcoming tournament from the ESPN schedule, loads the field (using the prior year's field as a proxy until official fields are announced), and calls `predict_field()`.
- **`bovada_client.py`** fetches live odds from Bovada's sports API for winner outrights and make/miss cut markets, caching results for 30 minutes.
- **`name_matcher.py`** handles the name normalization problem: Bovada uses player names with different formatting than the ESPN/PGA Tour data. Unicode normalization plus rapidfuzz fuzzy matching (threshold 85) resolves most mismatches.
- **`ev_calculator.py`** computes expected value: `EV = (model_prob × decimal_odds) - 1`, with Quarter Kelly (25%) bet sizing by default.

### Odds Coverage

Bovada covers winner outrights and make/miss cut markets for most weekly PGA Tour events. The make cut market is structured as one event per player ("Scottie Scheffler - Make Cut / Miss Cut"), which requires parsing ~40 individual events per tournament. Top-10 and Top-20 markets are not consistently available for standard events, which limits the EV analysis to outright winner and cut bets.

The Odds API — a popular aggregator — was evaluated but found to only cover the four major championships (Masters, PGA Championship, The Open, U.S. Open) for golf. Bovada's direct API has broader event coverage.

---

## 2026 Retroactive Validation

The model was retroactively evaluated against seven completed 2026 events. Predictions were made using the actual 2026 field (players who entered the tournament) and 2025 statistics as features — the same pipeline used for forward predictions.

### Per-Tournament Validation Results

P@K (Precision at K) measures what fraction of the model's top-K picks by predicted win probability were actual top-20 finishers. Winner Rank is where the actual tournament winner appeared in the model's ranked output — lower is better.

| Tournament | P@10 | P@20 | Winner Rank |
|-----------|------|------|-------------|
| WM Phoenix Open | 50% | 40% | 3 |
| Farmers Insurance Open | 50% | 40% | 53 |
| Sony Open in Hawaii | 40% | 35% | 14 |
| Genesis Invitational | 40% | 30% | 41 |
| AT&T Pebble Beach Pro-Am | 30% | 35% | 33 |
| Cognizant Classic | 30% | 25% | 17 |
| American Express | 20% | 25% | 4 |

**Aggregate: P@10 = 37.1%, P@20 = 32.9%, Mean Winner Rank = 23.6**

WM Phoenix Open and Farmers Insurance Open stand out as the model's strongest results — both with 50% P@10, meaning 5 of the actual top-20 finishers appeared in the model's top-10 picks. The Farmers Insurance winner ranked 53rd, illustrating the irreducible variance in a sport where a single hot putting week can propel a mid-tier player to victory.

The per-tournament diagnostics notebook also runs the predictions through XGBoost models fit on each tournament's historical data separately (kept for comparison). The global model consistently outperforms the per-tournament models, confirming that the global approach with course-specific features is the right architecture.

### Feature Importance Highlights

The top features by gain importance in the top-20 model after all engineering:

1. `roll5_avg_finish` — last 5 played events average finish (8.9%)
2. `Scoring_Average_Avg` — prior season scoring average (6.4%)
3. `SG_Total_Avg` — prior season SG Total (5.2%)
4. `sg_ema_14w` — exponentially weighted 14-week SG Total (3.5%)
5. `SG_Tee_to_Green_Avg` — prior season tee-to-green SG (2.8%)

Recent form (`roll5_avg_finish`) being the single most important feature validates the decision to fix the rolling window semantics. The model most values *recent results at tournaments* over any individual statistical category — which makes intuitive sense for a sport with as much weekly variance as professional golf.

---

## Reflections and Lessons Learned

### Data Quality Over Model Complexity

The two most impactful changes across the entire project were both data quality fixes:

1. **Rolling window semantics**: switching from "last N scheduled events" to "last N events played" halved the NaN rate (50% → 9%) and elevated `roll5_avg_finish` to the most important feature in the model.
2. **Zurich Classic exclusion**: removing the team-format event prevented ~400 team-format rows from corrupting individual-player training data.

Neither of these required a new model or algorithm. They required understanding the data.

### The Variance Problem Is Real and Persistent

Mean winner rank of 23.6 across seven events means the actual tournament winner ranked, on average, outside the model's top-20 picks. This isn't a modeling failure — it reflects the irreducible variance in professional golf. A player who ranks 40th by long-run SG quality can and does win on any given week. The model is trying to rank players by expected performance; golf rewards peak performance in a specific week.

This has direct implications for betting: the model is better suited to identifying consistent value in top-20 and cut markets than to picking weekly winners. The MC win probability output is useful for understanding relative expected value, not for finding lock picks.

### Course Fit Is Hard to Measure

The course SG signature features (correlations between SG components and finishing position) capture which statistical skills matter most at each course. But they require several years of data to be reliable (minimum 30 player-events), and they describe the *average* player's course fit, not specific players who might have asymmetric relationships with a course (e.g., a player who grew up near Augusta).

Adding actual player-specific course history features (`course_hist_avg_finish`, `course_hist_appearances`) helps, but players who haven't played a course before have no history signal. Some of the most interesting prediction failures involve new course configurations or players competing at a specific course for the first time.

### Prior Season Stats as Features Have Limits

Using the prior year's statistics as features creates an inherent lag. A player who dramatically improved their game in the current season (new coach, technical change) is still being evaluated on last year's numbers. The YoY SG change feature (`sg_total_yoy_change`) partially addresses this, but it only captures the difference between the two most recent completed seasons. In-season rolling SG data (from per-event scraping) helps bridge the gap, but mid-season predictions remain harder than end-of-season retrospectives.

---

## Technical Stack and Reproducibility

The full pipeline runs in Python 3.11 with XGBoost, scikit-learn 1.6, pandas, and Streamlit. The notebooks are numbered sequentially and self-contained:

1. `01_data_prep.ipynb` — merges raw data into master training dataset
2. `01b_feature_engineering.ipynb` — adds prior season form, course signatures, SG features
3. `01c_rolling_features.ipynb` — computes rolling and EMA form features from per-event data
4. `03_modeling.ipynb` — trains all four XGBoost models
5. `04_diagnostics.ipynb` — ROC/PR curves, SHAP, calibration, P@K
6. `05_predictions_2026.ipynb` — retroactive and forward predictions for 2026

The code is available at [https://github.com/charlesbenfer/PGA_Prediction_Tools](https://github.com/charlesbenfer/PGA_Prediction_Tools).

---

## What's Next: Improving the Percentile Finish Regressor

The classifier models (made_cut, top_20, top_10) are performing reasonably well. The weakest link in the system is the **percentile finish regressor** — Spearman ρ = 0.297 on the test set, meaning the model explains less than 10% of the variance in where a player actually finishes within the field. This is the input that drives the Monte Carlo simulation, so improving it would directly improve win probability estimates.

A few directions worth exploring:

- **Better target engineering**: percentile finish compresses a lot of information into a single number. Separating the problem into "will this player finish in the top half of the field?" and "given they're in the top half, how high?" may give the model more tractable sub-problems to solve.
- **Quantile regression**: rather than predicting a single expected percentile, training separate models for the 25th, 50th, and 75th quantiles would give a fuller picture of each player's outcome distribution — useful for bet sizing under uncertainty.
- **Tournament-week context features**: field strength varies significantly week to week. A player's expected finish percentile against a major field is different from the same player against a low-key fall event, even controlling for `tournament_encoded`. Explicit field-strength features (mean SG of the field, number of top-50 players entered) may sharpen predictions.

The regressor is where the most headroom remains in the system, and it's the next area of focus.

---

*If you're working in sports analytics and want to discuss the methodology or compare notes, feel free to reach out. Predicting golf remains a genuinely hard problem — the variance humbles you every week.*
