
# Womens-fashion-recommendation-system
---
## Table of contents
---
* [Project highlights](#Project-highlights)
* [Introduction](#Introduction)
* [System description](#System description)
* [Recommendation process](#MRecommendation process)
* [Model training && Evaluation](#Model-training-&&-Evaluation)
* [Input-Output examples](#Input-Output-examples)
* [Reference](#Reference)


## Project highlights
---
-	Content-Based recommendation system using ResNet and AnnoyIndex the system recommends clothing items in one of the following categories: __Skirts, Dresses, Shorts, Blouses.__ 
-	Train ResNet 50 model on GCP and TensorFlow for clothing classification and image clustering   
-	Build a k-nearest neighbor model in python using AnnoyIndex for finding
  the similarity between users items and new items in the shop
-	Design and implemented ML pipeline for feature extraction and generate a recommendation for users based on favorite items ratings
-	Visualiz all recommendation process in Jupyter Notebook


## Introduction
---
Content-Based recommendation system using ResNet and AnnoyIndex 
This project is the final assignment in the Recommendation systems course.
the project's target:
- Our recommendation system is Content-Based
- The system recommends clothing items in one of the following categories: Skirts, Dresses, Shorts, Blouses.
  based on the similarity between the new products in the store, and the items that the user liked most in that category.

- In order to represent each item, we trained a [FashioNet] model based on the ResNet architecture to classify each image into one of the categories.
- To represent the items as vectors, we took the output from FashioNet and used Image embedding as input to AnnoyIndex
  To find the similarities between the items we used AnnoyIndex with imaginary-angular metrics

## System description
---
we simulate a woman's fashion shop with the following category for recommendation:
  1-Skirts
  2-Dresses
  3-Shorts
  4-Blouses

- load the pre-train ResNet model with weights from  __ImageNet__  
- transfer learning,  we freeze all layer except the last 4 layer
  train the model on DeepFashion dataset http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html 
- The goal in training this model is to get a classification between different clothing. We will then use the output results of the model to generate Image embeddings for each       category. 

## Recommendation process
---
•To simulate user data for the recommendation, we toke from (https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews/home)
 the recommendation step works as follow:
    given a user data
    - Search for the most favorite items the user rate
    - Fashion shops get the category and create image embeddings on all the new items the store has to offer in that category.
    - create inventory in this category by using __AnnoyIndex__ 
    - Find the most similar items in the fashion shop that match the user's favorite item. 	


## Model training && Evaluation
---
Attempt | #1 | #2 | #3 | #4 
--- | --- | --- | --- |--- |
Seconds | 301 | 283 | 290 |120

source	Train	Val	Test
Skirts	493	209	160
Dresses	1829	638	545
Shorts	847	373	267
Blouses	2063	694	694


## Input-Output examples
---

## Reference
---
