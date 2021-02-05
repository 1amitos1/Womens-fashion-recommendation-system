# Womens-fashion-recommendation-system
Women's fashion recommendation system using ResNet and AnnoyIndex 

This project is the final assignment in the Recommendation systems course.
the project's target:
• Our recommendation system is Content-Based
• The system recommends clothing items in one of the following categories: Skirts, Dresses, Shorts, Blouses.
Based on the similarity between the new products in the store, and the items that the user liked most in that category.

• In order to represent each item, we trained a [FashioNet] model based on the ResNet architecture to classify each image into one of the categories.
• To represent the items as vectors, we took the output from FashioNet and used Image embedding as input to AnnoyIndex
  To find the similarities between the items we used AnnoyIndex with imaginary-angular metrics

System description
we simulate a woman's fashion shop with the following category for recommendation:
1-Skirts
2-Dresses
3-Shorts 
4-Blouses 

• we load the pre-train ResNet model with weights from  ImageNet  
• transfer learning,  we freeze all layer except the last 4 
  train the model on DeepFashion dataset http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html
• The goal in training this model is to get a classification between different clothing. We will then use the output results of the model to generate Image embeddings for each       category. 

•To simulate user data for the recommendation, we toke from https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews/home
   the recommendation step works as follow:
    given a user data
    1-Search for the most favorite items the user rate
    2-Fashion shops get the category and create image embeddings on all the new items the store has to offer in that category.
    3-create inventory in this category by using AnnoyIndex 
    4- Find the most similar items in the fashion shop that match the user's favorite item. 	
