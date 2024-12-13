---
layout: post
title: Influencing Pitcher Improvement and Transfer Portal Acquisitons
subtitle: Predictive Modeling for advanced Pitcher Metrics  
cover-img: ../assets/plainsman.jpeg
thumbnail-img: ../chase.jpeg
share-img:
tags: Sports Analytics, Baseball, Pitching, Machine Learning
author: Charles Benfer
---
##  Motivation
The game of baseball, at its core, boils down to repeated competition between a pitcher and a batter. If either team can win a large enough number of these battles over the course of 9 innings, that team will likely win the game. To this end, it is important for a team to understand their strengths and weaknesses at a deep level to put their team in the best position to win those battles. Sports analytics has been a booming industry for many years at this point, and new methodologies in the fields of machine learning and artificial intelligence have only propelled the possibilities to new heights. 
College athletics is divided into 3 divisions, D1, D2, and D3 with D1 schools having the greatest size and resources. Due to these differences in resources, D1 athletic programs tend to have the most talent. Each of these divisions are then divided into conferences. At the D1, the Southeastern Conference (SEC) is as good as it gets when it comes to baseball. The best prospects come to play at those 16 schools, most of whom have strong desires to compete at the next level in the major leagues. Since 2021, college athletes at all levels have been able to earn monetary gains from their name, image, and likeness (NIL). This opportunity further allowed the large, resource rich D1 schools to pursue, even poach, the most talented prospects from smaller schools by offering monetary gain beyond a scholarship. Before NIL gains were allowed, it was fairly important to select the right students to give scholarships, but scholarships can be revoked and placed on other students fairly freely, and most great D1 athletes do not stay the entire four school years anyway. With the dawn of NIL, however, huge sums of monetary investment are being thrown around to ensure that the best teams secure the best prospects or transfers. When huge sums of money are involved, specific reasoning and motivation is required before deciding to ink contracts that can be harder to deal with if the player does not achieve the originally predicted levels. 
Accurately predicting player performance is a very difficult task in this situation for a few different reasons. First, player regression or improvement is almost impossible to predict unless you have inside information about their training plans and methods in the offseason. That is not what we focus on in this project. Second, player performance in one division or conference may not directly correlate to performance in another conference. If a hitter performs very well in the SEC, it is likely that that hitter will also perform well, probably very well, in a lower division or conference: moving down in skill is not in question. What may not be as clear is the change in performance as a player moves up in division or conference. This is what we are mostly focused on capturing in this project. The issue is, however, traditional baseball statistics only measure outcomes for the most part. For pitchers, specifically, we can quantify how well a pitcher performed over the course of a season by examining traditional metrics like his earned run average (ERA), the average number of runs he lets up over 9 innings, K/9, the average number of strikeouts he gets over 9 innings, etc. In recent years, more advanced metrics such as fielder independent pitching (FIP), hard-hit rate (HH%), wins above replacement (WAR), etc., attempt to quantify pitcher ability beyond basic performance, but metrics like these are still combinations of outcomes that occurred throughout the course of that pitcher's appearances. Since these metrics are outcome based, a grain of salt is required when comparing a pitcher from a worse division or conference with one from the SEC, since the batters they are competing against are also of a lower caliber. 
The goal of this project is to develop a metric which takes the outcome out of the equation, and only quantifies a pitcher’s ability based on the characteristics of his pitches. To do so, we must analyse the characteristics of a pitch. This data comes to us in the form of Trackman readings for SEC pitchers from the 2020-2024 seasons. The predictor variables we take into account are metrics regarding pitch speed, pitch spin, pitch trajectory, as well as metrics about the pitcher’s release point. In total, we take into account 36 predictor variables. For this project, we highlight a pitcher’s ability to induce whiffs, when a batter swings and misses at a pitch. To this end, our response variable is the outcome of the pitch, whether it is a whiff or not. This boils the problem down into a binary classification problem. While this does seem to go against the motivation of our project, isolating the pitch and disregarding the outcome, it is important to realize that this outcome is only taken into account in the training of the model. This model can then be applied to pitchers from other conferences and divisions to produce a statistic for that pitcher, disregarding those outcomes. For this reason, it still fits our goals. 

## Methods

This is the methods section

## Results

This is the results section

##

Future Work

This is the future work section

