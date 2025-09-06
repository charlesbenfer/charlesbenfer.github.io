---
layout: post
title: Building an XGBoost Win Probability Calculator for Baseball
subtitle: Creating a Foundational Metric for Advanced Baseball Analytics
thumbnail-img: ../assets/img/calibration_comparison_sigmoid.png
share-img: ../assets/img/calibration_comparison_isotonic.png
tags: Sports-Analytics, Baseball, Machine-Learning, XGBoost, Win-Probability
author: Charles Benfer
---

## Project Goals

Win probability is one of the most fundamental metrics in sports analytics - the foundation upon which many advanced analyses are built. In September 2025, I developed a comprehensive win probability model for baseball that will serve as a critical component for many of my future projects, particularly my reinforcement learning-based bullpen management system. The model needed to be accurate, fast, and most importantly, free from data leakage issues that plague many amateur implementations.

The primary objective wasn't just to calculate win probability, but to build a production-ready system that could evaluate the impact of strategic decisions in real-time. When should a manager pull their starter? How much does bringing in a specific reliever change the team's chances? These questions require precise, contextual win probability calculations that account for the specific pitchers and batters involved.

## Project Methods

Building a reliable win probability model requires careful attention to both the machine learning pipeline and the baseball domain knowledge. I implemented two complementary approaches: a Bayesian model for uncertainty quantification and an XGBoost model for fast, accurate predictions.

### The Data Pipeline Challenge

The most critical challenge in this project wasn't model selection or hyperparameter tuning - it was avoiding data leakage. Many win probability models inadvertently use future information when making predictions, leading to artificially inflated performance metrics that fail catastrophically in production.

#### The Leakage Problem

Consider this scenario: It's the top of the 5th inning, score tied 2-2. To calculate win probability, we need to know the current game state. But here's where things get tricky - when exactly do we capture that state?

Most naive implementations use the state *after* an at-bat completes. This introduces subtle but devastating leakage. If a player hits a home run, the post-at-bat state includes that run, and the model learns to associate certain at-bat features with scoring that already happened. The model appears brilliant in testing but fails miserably in production when it has to predict *before* knowing the at-bat outcome.

#### The Solution: Pre-At-Bat State Capture

I rebuilt the entire data pipeline to capture state at the *first pitch* of each at-bat:

```python
def process_game_with_leak_free_timing(game_data):
    states = []
    for at_bat_id, at_bat_data in game_data.groupby('at_bat_number'):
        # Get the FIRST pitch (pre-at-bat state)
        first_pitch = at_bat_data.iloc[0]
        
        # Calculate running scores from COMPLETED at-bats only
        completed_at_bats = game_data[game_data['at_bat_number'] < at_bat_id]
        
        if len(completed_at_bats) == 0:
            current_score = (0, 0)  # Game start
        else:
            last_completed = completed_at_bats.iloc[-1]
            current_score = (last_completed['home_score'], 
                           last_completed['away_score'])
        
        # Build state using only past information
        state = build_state(first_pitch, current_score)
        states.append(state)
```

This seemingly simple change had profound effects. The model's performance dropped from an unrealistic 0.95 ROC-AUC to a more honest 0.885 - still excellent, but grounded in reality.

### Feature Engineering

The model uses 24 carefully selected features that capture the multifaceted nature of baseball game states:

#### Game Situation Features
- **Inning and outs**: Where we are in the game structure
- **Score differential**: The current competitive state
- **Base runners**: Runners on base and scoring position situations
- **Leverage index**: How critical is this moment?

#### Pitcher Performance Metrics
- **Season stats**: ERA, WHIP, FIP, K/9, BB/9
- **Fatigue indicators**: Pitch count, times through the order
- **Recent performance**: Rolling averages over last 10 batters faced

#### Batter Metrics
- **Season stats**: AVG, OBP, SLG, OPS
- **Platoon splits**: Performance vs. left/right pitchers
- **Clutch metrics**: Performance in high-leverage situations

#### Contextual Factors
- **Save situation**: Is this a save opportunity?
- **High leverage**: Critical game moments
- **RISP**: Runners in scoring position pressure

### The XGBoost Approach

After experimenting with various algorithms, XGBoost emerged as the clear winner for the primary model. Gradient boosting excels at capturing the complex, non-linear relationships inherent in baseball:

```python
class FastWinProbabilityModel:
    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='logloss'
        )
```

The model builds an ensemble of decision trees, each learning from the mistakes of its predecessors. This approach naturally handles interactions - for instance, a 3-run lead means something very different in the 2nd inning versus the 9th.

### Bayesian Alternative

I also implemented a Bayesian model using PyMC. From what I understand, the Bayesian approach is the industry standard, and was my original idea when starting this project:

```python
with pm.Model() as bayesian_model:
    # Hierarchical structure for pitcher effects
    pitcher_variance = pm.HalfNormal('pitcher_variance', sigma=0.5)
    pitcher_effects = pm.Normal('pitcher_effects', 
                                mu=0, 
                                sigma=pitcher_variance, 
                                shape=n_pitchers)
    
    # Main effects with informative priors
    beta = pm.Normal('beta', mu=prior_means, sigma=prior_sds)
    
    # Logistic regression
    logit_p = pm.math.dot(X, beta) + pitcher_effects[pitcher_ids]
    p = pm.Deterministic('p', pm.math.sigmoid(logit_p))
    
    # Likelihood
    y_obs = pm.Bernoulli('y_obs', p=p, observed=y)
```

The Bayesian approach provides full posterior distributions, allowing us to say not just "65% win probability" but "65% with a standard deviation of 8%." This uncertainty quantification is invaluable for risk-sensitive decision-making.

## Addressing the Calibration Challenge

One persistent challenge in win probability modeling is calibration at the extremes. Real baseball games can produce seemingly impossible comebacks - the 2011 Cardinals' Game 6 World Series comeback, down to their last strike twice, comes to mind. Yet machine learning models tend to be conservative, rarely predicting below 10% or above 90% probability.

### Why Models Avoid Extremes

This conservatism stems from how models are trained. Loss functions like log loss severely punish overconfident wrong predictions. Predicting 99% probability and being wrong incurs massive penalty, while predicting 85% is much safer. The model learns to hedge its bets.

Additionally, extreme situations are rare in training data. A 9th inning, 7-run deficit might occur only a handful of times in a season. Without sufficient examples, the model can't learn that these situations truly have near-zero win probability.

### Calibration Techniques

I explored two approaches to improve calibration:

#### Platt Scaling (Sigmoid Calibration)
The primary approach I used was Platt scaling, which fits a logistic regression to map predicted probabilities to actual probabilities:

```python
from sklearn.linear_model import LogisticRegression

# Fit sigmoid calibration
platt_cal = LogisticRegression()
platt_cal.fit(val_predictions.reshape(-1, 1), val_outcomes)

# Apply to test set
calibrated_probs = platt_cal.predict_proba(test_predictions.reshape(-1, 1))[:, 1]
```

Sigmoid calibration provided the best overall performance, improving log loss from 0.411 to 0.357 (13% improvement) and reducing Brier score from 0.133 to 0.118. The calibrated model maintained excellent discrimination while providing much more reliable probability estimates.

#### Isotonic Regression Alternative
I also tested isotonic regression as a non-parametric alternative:

```python
from sklearn.isotonic import IsotonicRegression

iso_cal = IsotonicRegression(out_of_bounds='clip')
iso_cal.fit(val_predictions, val_outcomes)
calibrated_probs = iso_cal.transform(test_predictions)
```

However, sigmoid calibration proved superior for this dataset, achieving better log loss (0.357 vs 0.378) while providing reliable probability estimates across the full range of game situations.

## Performance and Validation

The model achieved strong performance metrics on held-out 2024 test data:

### Classification Metrics
- **ROC-AUC**: 0.915 (XGBoost), 0.617 (Bayesian)
- **Log Loss**: 0.411 (XGBoost), 0.657 (Bayesian)
- **Brier Score**: 0.133 (XGBoost), 0.233 (Bayesian)

The XGBoost model clearly dominates in raw predictive performance, while the Bayesian model provides valuable uncertainty estimates.

### Calibration Quality
After sigmoid calibration:
- **Final log loss**: 0.357 (13% improvement from 0.411)
- **Final Brier score**: 0.118 (11% improvement from 0.133) 
- **ROC-AUC maintained**: 0.915 (discrimination preserved)
- Reliable probability estimates across all game situations

### Extreme Situation Handling
The model correctly identifies extreme situations:
- 9th inning, 5+ run leads: 94.3% average predicted win probability
- 9th inning, down 3+ runs: 8.7% average predicted win probability
- Early game tied situations: 49-51% predictions (appropriate uncertainty)

### Data Quality Validation
Critically, the leak-free pipeline was validated:
- Early inning correlations with outcome: 0.15-0.25 (expected: low)
- Late inning correlations with outcome: 0.45-0.55 (expected: high)
- No evidence of future information leakage


## Lessons Learned

This project reinforced several critical principles in sports analytics:

1. **Data leakage is insidious**: The initial "amazing" results were too good to be true. Always be suspicious of exceptional performance and audit your pipeline thoroughly.

2. **Domain knowledge matters**: Understanding baseball helped identify which features matter and which model behaviors were unrealistic.

3. **Simple models can excel**: XGBoost outperformed complex neural networks and sophisticated Bayesian hierarchical models.

4. **Calibration requires attention**: Raw model outputs often need post-processing to match real-world probabilities.

5. **Validation on future data is essential**: Testing on 2024 data (not available during 2023 training) provided honest assessment.

## Future Enhancements

While the current model performs well, several enhancements could push it further:

### 1. Pitcher-Specific Modeling
Instead of using aggregate stats, model individual pitcher tendencies:
- Velocity trends within games
- Performance under pressure
- Pitch arsenal effectiveness by count

### 2. Momentum and Clutch Factors
Baseball has psychological elements the current model ignores:
- Recent scoring runs
- Clutch performance history
- Home field advantage in late innings

### 3. Weather and Park Effects
Environmental factors affect win probability:
- Wind effects at Wrigley Field
- Altitude at Coors Field
- Temperature impacts on ball flight

### 4. Real-Time Updates
Implement online learning to update the model during the season:
- Adjust for injured players
- Capture hot/cold streaks
- Respond to roster changes

### 5. Uncertainty-Aware RL Integration
Use the Bayesian model's uncertainty estimates to make risk-adjusted decisions in the bullpen management system.

## Conclusion

Building a production-ready win probability model required more than just applying XGBoost to baseball data. The critical innovation was identifying and eliminating data leakage through careful pre-at-bat state capture.

The model now serves as the foundation for multiple downstream projects. My reinforcement learning bullpen management system (post to come soon) uses it to evaluate every potential pitcher change. Future projects will leverage it for in-game strategy optimization, player valuation adjustments, and even fan engagement applications.

Most importantly, this project demonstrated that rigorous attention to data quality and proper validation trumps algorithmic complexity. A simple XGBoost model with clean data outperforms sophisticated approaches built on flawed foundations. In sports analytics, as in many domains, getting the fundamentals right is the key to building systems that actually work when it matters - during the game, when decisions have real consequences.

The win probability calculator is more than just another model - it's a fundamental building block that enables a new generation of baseball analytics tools. By providing accurate, contextual probability estimates, it transforms vague intuitions about game situations into quantifiable insights that can drive better decisions. This foundation will support many exciting projects to come.