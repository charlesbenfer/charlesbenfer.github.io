---
layout: post
title: Influencing Pitcher Improvement and Transfer Portal Acquisitons
subtitle: Predictive Modeling for advanced Pitcher Metrics  
cover-img: ../plainsman.jpeg
thumbnail-img: ../chase.jpeg
share-img:
tags: Sports-Analytics, Baseball, Pitching, Machine-Learning
author: Charles Benfer
---
#### Acknowlegement
*Shared work with Ethan Sax, Auburn University*

## Project Goals

This year I was given the opportunity by a friend and colleague of mine to join a project working with the baseball team at Auburn University, specifically working with the pitching staff. The main goals for this project were simple to state, but less simple to implement. First, we wanted to develop a way to evaluate pitchers from other college teams, especially teams outside of the SEC, with the ultimate of understanding of their "stuff" will translate to SEC play. Second, we wanted to develop a way to motivate pitcher improvement plans, trying to understand which aspects of a pitch make it more or less effective than others. This sort of framework is not a new idea in baseball, but if properly implemented, can be invaluable when it come to team performance. If a school like Auburn, who ranked very poorly in the SEC in the 2023-2024 season in terms of pitching, can pick up some good pitchers from the transfer portal, and fine-tune the guys on the squad already, the team will be one step closer to the success seen in recent years. 

## Project Methods

At this stage in the project, the team has focused on generating a Stuff+ metric. Though Stuff+ takes on several definitions and nuances, our Stuff+ metric quantifies a pitcher's ability to produce a swing and a miss on any given pitch. Using some predictive modeling, we can quantify a specific pitch's % chance for a whiff, and compare it to the league average chance for a whiff. This definition has seen use in the past, the team encountered it first from Kai Franke's article found [here](https://medium.com/@kaifranke3/building-a-stuff-model-using-xgboost-8c548fbab8f2), a source of inspiration for some of out methods. 

The data used for this project is the main difference between our results and the results shown in most other article. As stated in the first paragraph, the main motivation lies in understanding how a pitcher will translate to SEC play. To this end, the data considered involves only SEC pitchers. Specifically, the team was provided with the Trackman pitch data from SEC games in the 2020 to 2024 seasons. The data was filtered down to only contain pitch characteristics, such as spin rate, velocity, etc. (All variables with descriptions can be found [here](https://support.trackmanbaseball.com/hc/en-us/articles/5089413493787-V3-FAQs-Radar-Measurement-Glossary-Of-Terms))      


