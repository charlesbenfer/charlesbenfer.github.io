---
layout: post
title: (Work in Progress) Predicting the Phenotypic Properties of Zea Mays (Field Corn) Using Genetic and Environmental Predictors
subtitle: Master's Capstone Project
tags: Machine-Learning, Prediction, Breeding
comments: true
mathjax: true
author: Charles Benfer  
---

# Project Description

## Collaboration Details
In the Fall 2024 semester, I completed my master's capstone project for the withthe Bayer Corporation (specifically Dr. Katiana Kontolati and Dr. Fabiana Freitas Moreira). The my partner in the project was Yasin Fatemi, Ph.D. student in the Auburn Engineering Department, and our faculty mentors were Dr. Roberto Molinari and Dr. Nedret Billor.

## Project Motivation
As indicated by the title of the project, the overall goals involved influencing the breeding practices for corn plants. When corn plants, or pretty much any plant or animal, is created from two distinct parent lines, it tends tends to display favorable charactaristics: this is the concept of "heterosis", and is the basic motivation behind most breeding practices. The main project that is being explored at the Bayer corporation is called "hybrid mining", which is the practice of modeling the phenotypic qualities of the corn plants based on their genetic charactaristics, and motivating breeding practices based on those qualities. For our capstone project, the goal was to incorporate the environmental characteristics in which the corn was grown, along with the genetic characteristics, and see if this would result in better model performance than either set of predictors by themselves. 

## Provided Data

Before any modeling took place, the team needed to understand the provided data sets. The data came to us in 3 different "sets." The first data set consisted of the phenotypic features of the corn plants. This data was split between two different sets, depending on the cluster of breeding experiments the corn plant was a result of. This data consisted of the year and location the corn was grown in, the genetic line of the corn (indicated by a number, not the actual genetic information), and 8 phenotypic features which will be explained at a later time. There was also some irrelevant information to our project included. 

The second set of data included the environmental data for a specific location and year, each of which corresponded to the specific year and location provided in the phenotypic data. This data included some variables that were split monthly, such as monthly rain totals, monthly average temperature, as well as some general inforamtion such as mineral content of the soil. 

The last set of data was the most complex, which was the genetic information for each line of corn progeny. Each of these lines correspond to the line that was present in the phenotypic feature data set, and depends on the parents that were crossed to create the progeny. The genetic data was in the form of single nucleotide polymorphisms (SNPs) which represent differences at specific locations along the genome in comparison to some reference genome. These differences result in various differences in the corn plants, but we are unsure which values in which locations result in which differences.
 
