# Object-Detection

## ABOUT

The goal of the project was to detect features from various YouTube thumbnails, build subsequent hypothesis and test their significance to provide profitable recommendations to potential digital content creators.

Majority of the image-based features were extracted by using Google Vision. These include things like - area_covered_by_faces, number_of_objects, number_of_people and so on. Textual data was also extracted from these images and put through NLP techniques. This gave access to further features such as sentiments, emoji_count and so on. Lastly, additional techniques such as KMeans, HoughLines wer3e used to detect face color category, chyron bands and so on. 

Having built our corpus of features, various hypothesis were developed and tested using ANOVA and Pearson's Correlation. These tests were developed to build low-level insights such as significane of hashtags, emojis, skin color of the faces and so on. 

The end goal was to build a recommendation engine as well that digital content creators could refer to for digital decisions. As such I decided to create a Decision Tree model that would predict the performance (impressions/engagements) based on various rules. In order to reduce the dimensionality of the features I used an ExtraTreesClassifier which gave the best features in a model. 

I am also in progress to perform market basket analysis on these features and thus get an idea on which kind of features work well together. 

## BUILT WITH / MODULES USED

Built with Python 3.7. Modules used:
 - Python 3.7
 - GoogleVision API
 - NLP
 - VADER
 - CV2
 - ExtraTreesClassifier
 - ANOVA
 - Decision Trees

## USE CASES
 - By leveraging the ability to detect features in images/thumbnails and test their significance, content creators can get a better picture of what works (drives engagements/impressions) and what does not.
 - A decision tree with its rules can also give a blueprint of the list of features that a thumbnail can/should utilise
 - By building a market basket analysis on top of this, one can also see what kind of features work well together and what don't.
