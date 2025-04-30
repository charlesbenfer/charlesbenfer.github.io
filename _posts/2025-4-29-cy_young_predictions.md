---
layout: post
title: Combining Pitch-Level, Game-Level, and Season Level Data to Quantify Pitcher Performance
subtitle: A meta-model approach to Cy Young award prediction
thumbnail-img: ../assets/img/cy_young_award.jpg
share-img:
tags: Sports-Analytics, Baseball, Pitching, Machine-Learning
author: Charles Benfer
---

### What Makes a Cy Young Winner?

Ask yourself, what qualities make a Cy Young winner? If you asked anyone in the days of old, it was wins and a low ERA. While these statistics are still quite important, and a player with a poor ERA is unlikely to have a Cy Young caliber season in the underlying metrics, there are plenty of other ways to evaluate the effectiveness of a pitcher.

Let's take a look at the Cy Young award winners in both the American and National Leagues in the last 3 seasons. 

- 2024: Tarik Skubal, Chris Sale
- 2023: Gerritt Cole, Blake Snell
- 2022: Justin Verlander, Sandy Alcantara

What, if anything, do all 6 of these pitchers have in common? In 2024, Skubal relied a lot on his high 90's 4-Seam to blow by guys, and nasty breaking stuff to miss bats and confuse timing, combining to a 5-pitch mix of 4-seam, Change, Sinker, Slider, Curve with 33%, 27%, 20%, 15%, and 4% usage, respectively. In 2022, Sandy Alcantara threw 4 pitches with almost identical usage - Change, 4-seam, Sinker, Slider at 27%, 25%, 25%, 22% respectively. So a primary 4-seam with peppered off-speed and breaking pitches is not the formula, even though this is exactly what Blake Snell featured in 2023, throwing about 50%  4-seams. 

Pitch mix was never the answer, was it? If there was a 3 pitch sequence that resulted in a Cy Young winner, that would be the only 3-pitch sequence ever thrown. So perhaps, it boils down to command? Or velocity? When following this path of logic, the answer remains unclear. In 2022, Justin Verlander was in the top 94th percentile for walk-rate, limiting free bases with the best of them, while 2023 Blake Snell was in the bottom 4th percentile!!! of walk-rate, adopting a "well you aren't biting on my pitches, I bet the next guy will" mentality. 

The reality is, there is not one single metric or aspect of a pitcher's game that results in an effective season, let alone a Cy Young season. If we want to predict who will the Cy Young award in any given year, the problem arises: how do we quantify pitcher effectiveness? 

### Quantifying the Pitching Game

So our model has a clear output: Did you win the Cy Young? Whether this is a binary 0 if you didn't 1 if you did, multiclass where the label is the place you finish, or regressing on total points in the final standing, these are all essentially equivalent.

What is not finalized, however, is the set of predictors. As discussed in the previous section, there is no one feature or set of features that seems to translate linearly, or non-linearly, to Cy Young ranking. So this sparks an idea: how can we quantify the different parts of a pitchers game? This makes total sense when taking a step back. Maybe a guy limits a lot of hard contact and doesn't issue free passes, but also doesn't make guys whiff a lot. This is the plan for a very effective pitcher. What about a guy who enduces more whiffs than not, but lets guys on via the walk a little more than desireable? This should sound familiar to the first section. Clearly, some combination of player characteristics can result in a very successful season, and I think we can predict those combinations. 

#### Limiting Hard Contact



### A Meta-Model Approach


### Results


### Future Work









