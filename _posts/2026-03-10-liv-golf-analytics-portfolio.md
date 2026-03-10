---
layout: post
title: Building a Full-Stack Analytics System for LIV Golf
subtitle: Player Valuation, SG Estimation, Team Construction, and an Interactive Dashboard — Built for a League Without Strokes Gained Data
thumbnail-img: ../assets/img/liv_2025_sg_ranking.png
share-img:
tags: Sports-Analytics, Golf, Machine-Learning, LIV-Golf, Streamlit, Dashboard, Player-Valuation, XGBoost
author: Charles Benfer
---

## The Problem

LIV Golf does not publish Strokes Gained data.

That single fact makes it one of the more interesting analytics problems in professional sports. Strokes Gained — the shot-value framework pioneered by Mark Broadie — is the gold standard for measuring player skill in golf. Every serious evaluation of a professional golfer, from PGA Tour betting models to caddie strategy to front office decisions, runs through SG in some form. And LIV, despite having a roster that includes some of the best players in the world, publishes only traditional counting stats: driving distance, fairway percentage, GIR, scrambling, and putting average.

This project started with a simple question: can you reconstruct a meaningful skill picture for LIV players using only what LIV publishes? And from there, can you build the analytics infrastructure that a front office would actually need — player valuation, acquisition rankings, team construction tools, and a dashboard for decision-makers who don't live in notebooks?

The answer, it turns out, is yes — with some methodological creativity and a lot of careful engineering.

The full system is live at [charlesbenfer.github.io/LIV_Portfolio](https://charlesbenfer.github.io/LIV_Portfolio) and the interactive dashboard is embedded below.

---

## The Data Pipeline

### Scraping LIV Golf

LIV's public website (`livgolf.com/stats`, `livgolf.com/schedule`) provides the traditional stats and event leaderboards, but not in a clean API format. The site is a Next.js single-page application that hydrates data client-side, which means standard HTTP scraping fails — you get the shell of the page, not the data.

The solution was a Playwright-based headless browser scraper that launches a full Chromium instance, waits for the SPA to hydrate, and then parses the `<main>` inner text. The stat pages require UI interaction: clicking a season dropdown to select the year, then clicking an event dropdown to filter to a specific event. Both interactions had to be handled programmatically with explicit waits for DOM state changes.

The scraper collects three data sources:

1. **Season stats** (`/stats/{category}?season=YYYY`) — full-season averages for seven stat categories across all players for 2022–2026
2. **Per-event stats** — the same categories filtered to a single event, requiring two sequential dropdown interactions per stat per event
3. **Event leaderboards** (`/schedule/{slug}/leaderboard`) — round-by-round scores (R1/R2/R3 and total-to-par) for every event from 2022 through the present 2026 season

Across 50+ events and four complete seasons, the scraper produces three output CSVs:

| File | Description |
|------|-------------|
| `liv_season_stats.csv` | One row per player per season, wide on stat category |
| `liv_event_stats.csv` | One row per player per event, wide on stat category |
| `liv_event_results.csv` | One row per player per event, with R1/R2/R3/total |

The scraper is designed for incremental updates — a `save_and_merge` function upserts new rows into existing CSVs keyed on `(event_slug, playerName)`, so adding a new event (e.g. Hong Kong 2026 after it concluded) requires running `python liv_scraper.py --event hong-kong-2026` rather than re-scraping the entire history.

### PGA Tour Historical Data

Every current LIV player has a PGA Tour career. That career is the analytical foundation of the project. Player historical SG data — Off the Tee, Approach, Around the Green, Putting, and Total — was pulled from the PGA Tour's public statistics API for all seasons where these players competed on tour. This gives us a decade or more of actual SG measurements per player, which serves two purposes:

1. **Training data** for the SG estimation model (paired PGA seasons where both traditional stats and SG values are known)
2. **Career trajectory data** for the player profiles and aging analysis

---

## Estimating Strokes Gained from Traditional Stats

The core methodological challenge: LIV publishes traditional stats, PGA Tour publishes both traditional stats and SG. The two tours share the same players. Can we learn the relationship from the PGA Tour data and transfer it to LIV?

This is defensible on physical grounds. The relationship between driving distance and shot value, or between GIR percentage and approach SG, reflects the underlying geometry of the golf course. That relationship doesn't change because the tour logo is different.

### The Model

A regression ensemble — Ridge and Gradient Boosting blended via `VotingRegressor` — is trained on paired PGA Tour player-seasons (2015–2024) where both traditional stats and SG values are observed. Five separate models are trained, one per SG target:

| Target | CV R² | Key Predictors |
|--------|-------|----------------|
| SG: Off the Tee | ~0.72 | Driving distance, driving accuracy |
| SG: Approach | ~0.68 | GIR %, proximity to hole |
| SG: Around Green | ~0.55 | Scrambling %, sand save % |
| SG: Putting | ~0.61 | Putts per GIR, putting average |
| SG: Total | ~0.78 | Ensemble of component predictions |

The models are then applied to LIV's published traditional stats to produce estimated SG values per player per season — with bootstrap confidence intervals to communicate uncertainty. The SG Total R² of 0.78 means the estimates are meaningful but imperfect, which is appropriate: we're honest in the dashboard about the fact that these are model-derived estimates, not measured values.

### Key Finding

Bryson DeChambeau and Jon Rahm carry the highest estimated SG Totals on the current LIV roster, which aligns with their world rankings and recent performance. More interestingly, the model identifies several players whose results — tournament finishes, scoring average — diverge significantly from their estimated skill profile in either direction. Players who consistently outperform their SG estimate are likely variance beneficiaries; players who underperform are candidates for a correction toward the mean. Both signals have direct relevance for roster construction.

---

## Player Valuation and ROI

With estimated SG values in hand, the next question is: what are these players worth relative to the prize money they've earned?

### The Framework

The ROI tab of the dashboard builds a value scoring system across three components:

1. **Skill score** — estimated SG Total relative to the field average
2. **Results score** — actual tournament finish percentile over their LIV career
3. **Prestige score** — a composite of career PGA Tour wins, major titles, and world ranking peak, representing the marquee value a player brings beyond on-course performance

The prestige component is important for LIV specifically. The league's business model depends not just on who wins, but on who's playing. A player who finished 50th in every event but won two majors on the PGA Tour is generating TV interest and fan engagement that a statistically equivalent but less decorated player is not. The valuation model attempts to quantify both dimensions.

### Prestige vs. Performance 2x2

One of the more useful outputs is a four-quadrant classification of every LIV player on two axes: prestige score (horizontal) and value score (vertical). The quadrants tell you something analytically useful:

- **High prestige, high value**: the franchise anchors — players who both deliver on-course and sell the product
- **High prestige, low value**: marquee names whose best golf is behind them
- **Low prestige, high value**: underpriced skill — players whose performance exceeds their star billing
- **Low prestige, low value**: developmental or roster-filling players

This framework is more useful than a single ranking because it separates two genuinely different questions: "who is playing well?" and "who is worth paying for?"

---

## Team Analytics

LIV's team format has no real precedent in professional golf. Teams of four players compete across 54-hole strokeplay events, with the team score determined by combining individual results. Building analytics for this structure required thinking carefully about what actually drives team outcomes.

### What Predicts Team Wins?

A regression analysis of team results against SG component profiles finds that **approach play and off-the-tee performance** are the strongest predictors of team wins, with putting and around-the-green play contributing more weakly. This has practical implications: a team optimizing for wins should prioritize approach and tee-to-green skill over short game variance, which is more volatile week-to-week anyway.

### Floor vs. Ceiling

A structural question in LIV team construction: does your worst player matter more than your best? In a four-person team, a single player who catastrophically underperforms can sink the team score. Analysis of historical results shows that teams with a **high floor** — defined as a consistently competitive minimum contributor — outperform teams that rely on elite star power but have a weak link in the lineup. The weakest team member's SG estimate is a stronger predictor of team results than the strongest member's.

This is the opposite of the intuition you'd get from individual sports, and it has direct implications for acquisition strategy.

### Team Heatmaps

The dashboard includes per-season event heatmaps for all LIV teams, showing event-by-event results as a color-coded grid. These make it immediately obvious which teams are consistent vs. streaky, and which events expose specific teams' weaknesses.

---

## The Dashboard

All of this analysis is packaged into an interactive Streamlit application with four tabs:

**Player Profiles** — Select any LIV player to see their full SG career trajectory (PGA Tour actuals + LIV estimates), their skill breakdown across SG components, their LIV event-by-event results in chronological order, and a stat trend chart for each of the seven tracked stat categories. A second player can be added for head-to-head comparison with a radar chart overlay.

**Team Analysis** — Team-level skill profiles, season summary tables, performance heatmaps by event, and the floor vs. ceiling analysis for the current 2025 roster.

**Head-to-Head** — Any two players on the roster compared directly across every available skill dimension, with historical career context.

**ROI & Acquisition** — The valuation model outputs: the prestige vs. performance 2x2, the skill vs. results scatter, and individual player value scores. Useful for framing acquisition discussions around players who would move the needle in either dimension.

Events are sorted chronologically using an explicit canonical event order derived from the scraper's schedule, so charts and tables reflect the actual season progression rather than alphabetical artifact.

---

## Reflections

### The Transfer Learning Bet

The most consequential methodological decision in this project is betting that the traditional-stats-to-SG relationship is stable across tours. If it isn't — if LIV's courses, formats, or competitive dynamics meaningfully change what a given driving distance percentage is worth — then the estimated SG values are systematically off. I believe the assumption is reasonable, and the R² values on PGA Tour holdout data support it, but it's a bet worth flagging. Any consumer of these estimates should understand they're model-derived, not measured.

### Counting Stats Are Noisier Than SG

Working with traditional stats as features rather than SG reminded me how much information SG throws away. Driving distance and accuracy tell you something, but they don't tell you how much better or worse a player is than the field — they're absolute measurements, not relative ones. The SG estimation model can only recover so much of the SG signal from these inputs, which is why the R² values in the 0.55–0.78 range are honest but imperfect. This is the fundamental information constraint the project operates under.

### Team Analytics Is Genuinely Underexplored

Most golf analytics research, including my own PGA Tour prediction work, focuses on individual outcomes. The team format at LIV creates a genuinely different optimization problem that the industry hasn't fully worked through. The floor vs. ceiling finding — that weak links matter more than star peaks — is the kind of counterintuitive result that comes from actually looking at the data rather than assuming golf teams work like basketball teams. There's more work to do here.

---

## Technical Stack

| Layer | Tools |
|-------|-------|
| Data Collection | Python (Playwright), livgolf.com scraping |
| Data Processing | pandas, numpy, scikit-learn |
| Machine Learning | Ridge, GradientBoostingRegressor, VotingRegressor |
| Visualization | plotly, matplotlib, seaborn |
| Dashboard | Streamlit |
| Notebooks | Jupyter |

The full codebase is available at [github.com/charlesbenfer/LIV_Portfolio](https://github.com/charlesbenfer/LIV_Portfolio). The live dashboard is deployed at [livportfolio.streamlit.app](https://livportfolio.streamlit.app).

---

*If you work in golf analytics or are building something similar for a data-sparse sports environment, I'd be glad to compare notes.*
