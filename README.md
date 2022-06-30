# Object-Detection

## ABOUT

The project's objective was to identify attributes from diverse YouTube subtitles and thumbnails in order to develop and test pertinent hypotheses as well as a framework that can guide decision-making for those who generate digital content. 
The framework would provide a summary or blueprint of the many features used in the videos, their relationships, and guidelines for combining them to increase engagement.


Following is an example of a thumbnail that was used:
![6EN7A23YtHI](https://user-images.githubusercontent.com/69982245/161476440-a494b83c-c997-4eb4-9511-a67684df4c16.jpg)

## STEPS

- Feature Extraction / Engineering
     - Majority of the image-based features were extracted by using Google Vision. These include things like - area_covered_by_faces, number_of_objects, number_of_people and so on. 
     - Textual data was also extracted from these images and put through NLP techniques. This gave access to further features such as sentiments, emoji_count and so on. 
     - Lastly, additional techniques such as KMeans, HoughLines were used to detect face color category, chyron bands and so on. 

- Hypothesis Testing
     - Having built our corpus of features, various hypothesis were developed and tested using ANOVA and Pearson's Correlation. 
     - These tests were developed to build low-level insights such as the statistical significane of hashtags, emojis, skin color of the faces and so on. 

- Model Build
     - The end goal was to build a sort of recommendation engine that digital content creators could refer to for digital decisions. As such I decided to create a Decision Tree model that would predict the performance (impressions/engagements) based on various rules. 
     - In order to reduce the dimensionality of the features I used an ExtraTreesClassifier which gave the best (top N) features in a model. 
     - Further, separate decision trees were built for just 'caption' and 'thumbnail' variables in order to get further information. 
     - Following are examples of some insights that we garnered from the Decision Trees below
          - The 1st Decision Tree shows that videos with use of '#' as well as mention of 'sneak peek' tend to perform better in terms of CTR
          - Similarly, the 2nd Decision Tree shows that videos with '#' as well as with a positive sentiments get higher CTR
     - I am also in progress to perform market basket analysis on these features and thus get an idea on which kind of features work well together. 

## RESULTS

Following are the final decision trees for All (Caption + Thumbnail) variables, Caption variables and Thumbnail variables:

![All Variables](https://user-images.githubusercontent.com/69982245/161476198-4263bdb9-6b86-4bf2-8535-e6f74166c50f.png)

![Caption Variables](https://user-images.githubusercontent.com/69982245/161476235-e0460a2b-17be-47a1-9747-f02bb25cc840.png)

![Thumbnail Variables](https://user-images.githubusercontent.com/69982245/161476241-3bb7f3fd-8b26-4a64-bafd-c7e4cb0df9b6.png)


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
 - A decision tree with its rules can also give a blueprint of the list of features that a thumbnail can/should utilise to drive high click-through rates
 - By building a market basket analysis on top of this, one can see what kind of features work well together and what don't, thus making optimal marketing decisions
