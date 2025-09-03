---
layout: post
title: Predicting Pitcher Injuries with Bayesian Survival Analysis
subtitle: A Deep Dive into Model Development and the Quest for Better Features
cover-img: ../assets/img/pitcher_injury_header.png
thumbnail-img: ../assets/img/pitcher_injury_thumbnail.png
share-img: ../assets/img/pitcher_injury_thumbnail.png
tags: Sports-Analytics, Baseball, Bayesian-Statistics, Survival-Analysis, Machine-Learning
author: Charles Benfer
---

## Project Goals

In late 2024, I embarked on a project to predict pitcher injuries using Bayesian survival analysis. The motivation was clear: pitcher injuries, particularly elbow injuries requiring Tommy John surgery, have reached epidemic proportions in baseball. Teams lose millions of dollars and precious wins when their aces hit the injured list. If we could identify high-risk pitchers before catastrophic injuries occur, teams could make better roster decisions, manage workloads more effectively, and potentially prevent some injuries altogether.

The goal wasn't just to build another injury prediction model - it was to create a production-ready system that could provide actionable insights to teams. This meant not only achieving good predictive performance but also building an interpretable model with uncertainty quantification and a user-friendly interface for exploring risk factors.

## Project Methods

The foundation of this project was survival analysis, specifically Accelerated Failure Time (AFT) models implemented in a Bayesian framework using PyMC. Unlike traditional classification approaches that predict whether an injury will occur, survival models predict *when* an injury might occur while properly handling censored data (pitchers who haven't been injured yet).

The data pipeline started with MLB pitch-level data from 2019-2024, aggregated to season-level statistics for each pitcher. I tracked injury events using transaction data, identifying when pitchers went on the injured list and for what type of injury. The dataset included 1,284 pitcher-seasons with a 36.8% injury rate, providing a robust foundation for modeling.

### The Bayesian Approach

Let me explain why Bayesian statistics was perfect for this problem, especially for those less familiar with the approach.

#### What Makes Bayesian Different?

Traditional (frequentist) statistics asks: "Given this data, what's the most likely value for our parameters?" It gives you a single best estimate, like "pitchers over 30 have 1.5x higher injury risk."

Bayesian statistics asks a richer question: "Given this data AND what we already know about baseball, what range of values are plausible for our parameters?" Instead of one number, you get a full distribution - maybe that risk factor is most likely 1.5x, but it could plausibly be anywhere from 1.2x to 1.8x.

Think of it like scouting a pitcher. A frequentist approach is like watching one game and declaring "this pitcher throws 95 mph." A Bayesian approach is like combining that game with the scouting report, last season's data, and your knowledge that velocity varies - concluding "this pitcher probably throws 93-97 mph, most likely around 95."

#### Why This Matters for Injury Prediction

1. **Uncertainty is Crucial**: When telling a team their ace has high injury risk, you need to convey confidence. Saying "60% injury risk" is different from "40-80% injury risk" - the uncertainty changes decisions.

2. **Small Sample Sizes**: Only 62 pitchers had elbow injuries in my data. Frequentist methods struggle here, but Bayesian methods can borrow strength from prior knowledge about sports injuries in general.

3. **Prior Knowledge Integration**: We know things before looking at data - older players generally have higher injury risk, extreme workloads are dangerous. Bayesian methods formally incorporate this knowledge.

4. **Hierarchical Structure**: Pitchers aren't independent - they share teams, coaching, training facilities. Bayesian hierarchical models naturally handle this structure.

#### How Bayesian Survival Analysis Works

Survival analysis predicts time until an event (injury) while handling "censoring" - pitchers who haven't been injured *yet*. We don't know when they'll get injured, just that it hasn't happened by the end of our observation period.

The Bayesian approach models this probabilistically:
1. Start with prior beliefs about injury risk factors (e.g., "age probably matters, but we're not sure how much")
2. Update these beliefs using the data through Bayes' theorem
3. Get posterior distributions - our updated beliefs after seeing the data

Here's the mathematical intuition without getting too technical:
- **Prior**: Age effect on injury ~ Normal(mean=0, uncertainty=large)
- **Data**: Observed injuries and their timing
- **Posterior**: Age effect ~ Normal(mean=0.02, uncertainty=small)

The posterior tells us: "After seeing the data, we believe age increases injury risk by about 2% per year, and we're fairly confident about this."

The core model was a Weibull AFT (Accelerated Failure Time) model:

```python
with pm.Model() as weibull_model:
    # Priors - our initial beliefs before seeing data
    beta = pm.Normal('beta', mu=0, sigma=1, shape=len(features))
    # This says: "I think most features have small effects near zero, 
    # but I'm uncertain - they could be positive or negative"
    
    alpha = pm.Exponential('alpha', lam=1)  # Shape parameter
    # Controls whether injury risk increases over time (wear and tear)
    # or decreases (survivor bias)
    
    # Linear predictor - combining all features into a risk score
    eta = pm.math.dot(X_scaled, beta)
    
    # The Weibull distribution models time to failure (injury)
    # It's flexible - can model increasing or decreasing hazard over time
    y_obs = pm.Weibull('y_obs', alpha=alpha, beta=pm.math.exp(eta),
                        observed=times, censored=~events)
```

#### What the Model Learns

Through MCMC (Markov Chain Monte Carlo) sampling - essentially a sophisticated way of exploring all plausible parameter values - the model learns:

1. **Effect Sizes with Uncertainty**: Not just "age increases risk by 2%" but the full distribution of plausible values
2. **Relative Importance**: Which factors matter most for injury risk
3. **Individual Predictions**: For each pitcher, a full distribution of injury risk, not just a point estimate

The beauty is that we get honest uncertainty. When the model says "Gerrit Cole has 65% injury risk," it can also tell us how confident it is in that assessment based on how similar pitchers have performed.

### The Critical Bug Fix

Early in the project, I encountered a frustrating issue: the model was performing terribly with a concordance index (C-index) of 0.361 - worse than random chance. 

First, let me explain what the C-index measures. Imagine you have two pitchers - one gets injured after 50 games, another after 100 games. A good model should assign higher risk to the pitcher who gets injured sooner. The C-index measures how often the model gets these pairwise comparisons correct. A C-index of 0.5 is random guessing (coin flip), 0.7+ is good, and 1.0 is perfect.

My model scored 0.361 - it was getting the comparisons wrong more often than right! This was deeply puzzling because the model coefficients made sense and the convergence diagnostics looked good.

After extensive debugging, I discovered a critical error in how I was calculating the C-index for AFT models. The bug was subtle but devastating. 

Here's the issue: There are two main types of survival models:
1. **Cox models**: Model the hazard (instantaneous risk). Higher hazard = shorter survival
2. **AFT models**: Model the survival time directly. Higher predicted time = longer survival

I was using AFT models but calculating the C-index as if I had Cox models. It's like measuring temperature in Celsius but interpreting it as Fahrenheit - the numbers are completely wrong!

The fix was simple but crucial:
```python
# WRONG (Cox-style):
if risk_scores[i] > risk_scores[j] and times[i] < times[j]:
    concordant += 1

# CORRECT (AFT-style):
if risk_scores[i] > risk_scores[j] and times[i] > times[j]:
    concordant += 1
```

This single change boosted the C-index from 0.361 to 0.607 - a massive improvement that brought the model into the realm of clinical utility.

## Model Evolution and Feature Engineering

With the core model working, I embarked on an iterative process to improve predictive performance. Each iteration taught valuable lessons about what works and what doesn't in injury prediction.

### Iteration 1: Kitchen Sink Approach
Started with every available feature: age, games played, ERA, innings pitched, WHIP, strikeouts, walks, WAR, and more. The model achieved a C-index of 0.607, but many features had overlapping credible intervals with zero, suggesting overfitting.

### Iteration 2: Feature Selection
Used LASSO-inspired horseshoe priors to perform automatic feature selection. The model identified seven key features:
- **Age** (HR: 0.98): Slight protective effect, surprisingly
- **Games played** (HR: 1.015): More games = higher risk
- **Veteran status** (HR: 1.008): Experienced pitchers at higher risk
- **ERA** (HR: 0.991): Better performance = lower risk
- **Innings pitched** (HR: 1.001): Workload matters
- **WAR** (HR: 1.004): Higher value players at slightly higher risk
- **High workload flag** (HR: 0.995): Counterintuitively protective

### Iteration 3: Elbow-Specific Model
Given the Tommy John surgery epidemic, I built a specialized model for elbow injuries. With only 62 elbow injuries in the dataset, this was challenging. The model achieved a C-index of 0.464 - not great, but the sample size was limiting. Interestingly, the risk factors differed from general injuries, with workload playing a larger role.

### Iteration 4: Non-Linear Effects (Failed)
I hypothesized that age might have a non-linear effect - young pitchers still developing and older pitchers breaking down might both be at higher risk. I tried polynomial terms and splines:

```python
age_squared = (X_scaled[:, age_idx] ** 2).reshape(-1, 1)
X_with_poly = np.hstack([X_scaled, age_squared])
```

The result was catastrophic - C-index dropped to 0.38. The model couldn't handle the added complexity with the available data.

### Iteration 5: Interaction Terms (Failed)
Next, I tried modeling interactions between age and workload, thinking older pitchers might handle high workloads differently:

```python
age_workload_interaction = (X_scaled[:, age_idx] * X_scaled[:, workload_idx]).reshape(-1, 1)
```

This led to convergence failures and divergent transitions in MCMC sampling. The posterior geometry became too complex for efficient sampling.

### Key Learning: Simpler is Better
After all these attempts, the simple linear model with carefully selected features performed best. This is a common pattern in applied statistics - fancy methods often fail to beat simple, well-executed basics.

## Building the Production System

With a solid model in hand, I built a production-ready web application using Streamlit. The app provides:

1. **Individual Risk Assessment**: Search any pitcher and see their injury risk score with uncertainty bounds
2. **Team Analysis**: Filter by team to identify high-risk pitchers on the roster
3. **2025 Projections**: Forward-looking predictions based on 2024 performance
4. **Model Validation**: Transparent display of model performance metrics

The risk scoring system uses calibrated quartiles from the training data:
- **Low Risk** (🟢): Bottom 25% of risk scores
- **Moderate Risk** (🟡): 25th-50th percentile
- **High Risk** (🟠): 50th-75th percentile
- **Very High Risk** (🔴): Top 25% of risk scores

Critically, I validated that these categories showed monotonic injury rates:
- Low Risk: 28.8% injury rate
- Moderate Risk: 34.6% injury rate
- High Risk: 39.7% injury rate
- Very High Risk: 49.1% injury rate

## The Missing Piece: Biomechanical Data

Throughout this project, I kept bumping into a fundamental limitation: I only had performance statistics, not biomechanical data. Modern baseball generates incredible biomechanical metrics:

- **Release point consistency**: Variation might indicate mechanical issues
- **Velocity trends**: Sudden drops often precede injuries
- **Spin rate changes**: Can indicate grip or arm slot problems
- **Pitch movement profiles**: Deviations from baseline suggest compensation
- **Arm slot variations**: Dropping arm slot is a classic injury precursor
- **Extension and stride length**: Mechanical efficiency indicators

These features are almost certainly more predictive than ERA or wins. A pitcher might maintain good performance statistics while his body is breaking down - velocity drops of 2-3 mph, release point variations, or spin rate changes would be immediate red flags that performance stats miss entirely.

The challenge is data availability. While MLB teams have access to Trackman, Rapsodo, and high-speed cameras generating this data for every pitch, public researchers must work with what's available. This creates a ceiling on model performance that better data could break through.

## External Validation and Real-World Performance

The true test came with 2024 holdout data - pitchers the model had never seen during training.

### Understanding Model Performance Metrics

Before diving into results, let me explain what these metrics mean in plain terms:

- **C-index (Concordance Index)**: As explained earlier, this measures how often the model correctly ranks injury risk. Our 0.592 means that given any two pitchers, the model correctly identifies which one will get injured first about 59% of the time. While this might not sound impressive, in medical prediction (which injury prediction resembles), C-indices of 0.6-0.7 are considered clinically useful.

- **Brier Score**: This measures how close predicted probabilities are to actual outcomes. If I predict 60% injury risk for 10 pitchers, about 6 should actually get injured. Our score of 0.229 beats the baseline of 0.241 (which you'd get by always predicting the population average of 40.7% injury rate for everyone).

- **Calibration**: This checks if predicted probabilities match reality. When the model says "30% risk," do 30% of those pitchers actually get injured? Good calibration means teams can trust the risk percentages.

### The Results

The model achieved:
- **C-index: 0.592** on completely unseen 2024 data
- **Brier Score: 0.229** (beating baseline of 0.241)
- **Proper calibration**: Risk categories showed monotonic injury rates:
  - Low Risk: 28.8% actual injury rate
  - Moderate Risk: 34.6% actual injury rate  
  - High Risk: 39.7% actual injury rate
  - Very High Risk: 49.1% actual injury rate

This monotonic increase is crucial - it means the risk categories actually mean something. A "Very High Risk" pitcher is genuinely about 1.7x more likely to get injured than a "Low Risk" pitcher.

### What This Means for Teams

A C-index of 0.592 might not sound amazing, but it's genuinely useful for decision-making. Think of it this way:
- Without the model: Teams are guessing based on intuition and basic rules of thumb
- With the model: Teams correctly identify higher-risk pitchers 59% of the time

That 9% improvement over random chance, when applied to roster decisions worth millions of dollars, provides real value. It's the difference between keeping your ace healthy through October or watching him rehab during the playoffs.

## Lessons Learned

This project reinforced several important principles:

1. **Debug systematically**: The concordance calculation bug could have killed the entire project. Always verify your evaluation metrics match your model type.

2. **Start simple**: Complex models with interactions and non-linearities failed where simple linear models succeeded.

3. **Domain knowledge matters**: Understanding baseball helped identify which features made sense and which results were suspicious.

4. **Data quality beats model complexity**: Better features (biomechanical data) would likely improve performance more than any modeling trick.

5. **Validate ruthlessly**: Internal cross-validation isn't enough - true holdout testing on future data is essential.

6. **Make it usable**: The best model in the world is worthless if stakeholders can't use it. The Streamlit dashboard makes insights accessible.

## Future Directions

If I could extend this project, the priorities would be:

1. **Biomechanical data integration**: Partner with a team or facility that has Trackman/Rapsodo data
2. **Temporal dynamics**: Model how risk evolves during a season, not just between seasons
3. **Recovery modeling**: Predict not just injury occurrence but recovery time
4. **Pitch-level modeling**: Use granular pitch data rather than season aggregates
5. **Causal inference**: Move beyond prediction to understand what interventions might prevent injuries

## Code Availability

All code, notebooks, and the interactive dashboard from this project are available on GitHub:

**[github.com/charlesbenfer/pitcher-injury-analysis](https://github.com/charlesbenfer/pitcher-injury-analysis)**

The repository includes:
- The complete Bayesian survival analysis notebook with all model iterations
- Production-ready Streamlit dashboard for risk assessment
- Data processing pipelines and validation scripts
- 2025 season projections for all active MLB pitchers

Feel free to clone, modify, and improve upon this work. If you have access to biomechanical data, I'd love to see how much the model improves!

## Conclusion

Building a pitcher injury prediction system taught me that real-world data science is messy, iterative, and humbling. The model that works is rarely the fanciest one - it's the one that correctly handles the fundamentals. While the current system provides value with a C-index of 0.607 and proper calibration, the ceiling for improvement is clear: biomechanical data would transform this from a decent statistical model into a truly powerful injury prevention tool.

The journey from a broken model with C-index 0.361 to a production system serving real predictions illustrates the importance of rigorous debugging, iterative development, and focusing on usability. Sometimes the biggest improvements come not from complex methods but from fixing fundamental errors and making insights accessible to decision-makers.

For teams looking to protect their pitching investments, even our statistics-only model provides actionable intelligence. But the real revolution will come when biomechanical monitoring becomes standard, allowing models to see not just what happened on the field, but how the pitcher's body produced those results. Until then, we work with what we have - and what we have is good enough to start making better decisions.