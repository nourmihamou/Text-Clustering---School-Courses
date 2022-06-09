PROJECT TITLE: Text Clustering - School Descriptions

PURPOSE OF PROJECT: The goal of this project is to cluster the descriptions of Siena College's courses. 

VERSION or DATE: April 27, 20222

AUTHORS: Nour Mihamou


USER INSTRUCTIONS: 
The first step will be vectorizing the text appropriately. Set max_df parameter of vectorizer to eliminate extremely common words (e.g. "course").

Four different groupings will be created:

- clustering with three groups (corresponding to Siena's three schools)
- clustering with 33 groups (corresponding to departments)
- clustering with 57 groups (corresponding to course prefixes)
- clustering with the optimal number of groups, based on silhouette score

The first three of these have a fixed number of clusters, which suggests using k-means, agglomerative clustering, or LDA. For each of these, adjust parameters to maximize the ARI when compared to the ground truth.

(Note: There are a few courses with "0" labels for department and course prefix, and thus 34 & 58 ground truth labels in these settings. These course do NOT actually group together, and ARI will mark them "wrong" no matter how your model treats them - and that's OK.) 

For the fourth case, use the best parameter settings from above and try several different numbers of groups. (Suggestion: start with 2,4,6,10,15,20,30.) You should also try DBSCAN with a few different parameter settings and see how many groups it suggests as well. Your optimal grouping will be the one with the highest silhouette score.