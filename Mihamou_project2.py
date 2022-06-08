# -*- coding: utf-8 -*-
"""
Project 2: Text Clustering using different types of clustering methods, then finding
the optimal method as well as optimal number of clusters for the data. 
The data used are course descriptions from Siena College's course catalogs.

@author: Nour Mihamou
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer


#Functions
def lda_ari_ss(n_comp, data, codes = []):
    lda = LatentDirichletAllocation(n_components = n_comp)
    topic_list = lda.fit_transform(data)
    #Find max value on each row, store in an array
    arr = np.argmax(topic_list, axis = 1)
    #Calculate ARI score and/or SS
    if len(codes) < 1 :
       return silhouette_score(data, arr)
    else:
        return adjusted_rand_score(arr, codes)

def kmeans_ari_ss(n_clusters, data, codes = []):
    clf = KMeans(n_clusters = n_clusters, max_iter = 600)
    clf.fit(data)
    labels = clf.labels_
    #ARI Score
    if len(codes) < 1 :
        return silhouette_score(data, labels)
    else:
        print("Adjusted Rand Index Score ( k = ", n_clusters, ") = ", adjusted_rand_score(labels, codes))

def elbow_graph(K, data):
    Sum_of_squared_distances = []
    for k in K:
        km = KMeans(n_clusters=k, max_iter=600)
        km = km.fit(data)
        Sum_of_squared_distances.append(km.inertia_)
    plt.plot(K, Sum_of_squared_distances,  'bx-', c = 'red', marker = '.')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.grid(True)
    plt.show()
    
def agglom_clustering(nclusters, data):
    data = data.toarray()
    clf = AgglomerativeClustering(n_clusters = nclusters, affinity = "manhattan", linkage="average" )
    clf.fit(data)
    silhouette_score(data, clf.labels_)
  
def dbscan_ss(db_e, min_sample, data):
    clf = DBSCAN(eps = db_e, min_samples = min_sample).fit(data)
    dblab = clf.labels_
    for i in range(0,len(dblab)):
        dblab[i] += 1000
    #SS Score
    return silhouette_score(data, dblab)
    #print("no_clusters =", len(np.unique(clf.labels_)))
    
    


#importing data
descr = np.loadtxt("Z:/CSIS-320/Project2/descriptions.txt", dtype="str", delimiter="\t", skiprows=1)
d_codes = np.loadtxt("Z:/CSIS-320/Project2/dept_codes.txt", dtype="str", delimiter="\t", skiprows=1)
p_codes = np.loadtxt("Z:/CSIS-320/Project2/prefix_codes.txt", dtype="str", delimiter="\t", skiprows=1)
s_codes = np.loadtxt("Z:/CSIS-320/Project2/school_codes.txt", dtype="str", delimiter="\t", skiprows=1)

#Clean the data by eliminatiing the first column (the indices)
ind = descr[:,0]
ind1 = d_codes[:,0]
ind2 = p_codes[:,0]
ind3 = s_codes[:,0]

clndescr = descr[:,1]
clndc = d_codes[:,1]
clnp = p_codes[:,1]
clns = s_codes[:,1]

#Vectorize the text and generate a scaled bag of words
cv = CountVectorizer(max_df=0.75, stop_words=ENGLISH_STOP_WORDS)
descr_cv = cv.fit_transform(clndescr)
tfv = TfidfVectorizer(max_df=0.75, stop_words=ENGLISH_STOP_WORDS)
descr_tfv = tfv.fit_transform(clndescr)


###################### LDA
print("Latent Dirichlet Allocation ")
#ARI scores 
lda1 = lda_ari_ss(3, descr_cv, clns)
#ARI= 0.1417

lda2 = lda_ari_ss(33, descr_cv, clndc)
#ARI= 0.0405

lda3 = lda_ari_ss(57, descr_cv, clnp)
#ARI= 0.0415

#Get top ten words
for i in [lda1, lda2, lda3]:
    sorting = np.argsort(i.components_, axis = 1)[:,::-1]
feature_names = np.array(descr_cv.get_feature_names())


#####################  KMeans
print("\nKMeans")
kmeans_ari_ss(3, descr_tfv, clns)
#ARI= 0.1314

kmeans_ari_ss(33, descr_tfv, clndc)
#ARI= 0.1502

kmeans_ari_ss(57, descr_tfv, clnp)
#ARI= 0.1224



#####################  Agglomerative Clustering
print("\nAgglomerative Clustering")
for i in [3, 33, 57]:
    print("Silhouette Score (", i, " ) = ", agglom_clustering(i, descr_cv))
#SS = 0.4737, 0.2056, 0.1974



#####################   PART 4 LDA, KMeans, Agglomerative, DBSCAN

#####  LDA
print("\nPart 4")
print("LDA")
#Silhouette SCores (components = 40, 30, 25, 20, 15, 10, 5)
# for i in [40, 30, 10, 3]:
#     lda_ari_ss(i, descr_cv)

#SS = -0.1075, -0.0695, -0.04534, -0.0205

S_arr1=[]
ind1 = []
for j in range(2, 102):
    s = lda_ari_ss(j, descr_cv)
    S_arr1.append(s)
    ind1.append(j)

#location of max value
ymax1 = max(S_arr1)
xmax1 = ind[0]

#first value
ymin1 = min(S_arr1)
xmin1 = ind1[43]

#plot
plt.plot(ind1, S_arr1, c = 'purple', marker = '.')
plt.xlabel('clusters')
plt.ylabel('Silhouette Scores')
plt.title('Latent Dirichlet Allocation Classifier')
plt.grid(True)
plt.ylim([1,1])
plt.annotate('cluster = 2; SS = -0.002', xy = (xmax1, ymax1), xytext=(0, 10), 
             arrowprops=dict(facecolor='red', arrowstyle='->'))
plt.annotate('cluster = 45; SS = -0.194', xy = (xmin1, ymin1), xytext=(1, 1), 
             arrowprops=dict(facecolor='red', arrowstyle='->'))
plt.show()




#####  KMeans
print("\nKMeans")
###  First Elbow MEthod Analysis
elbow_graph([1, 2, 4, 6, 10, 15, 20, 30], descr_tfv)

###  Second Elbow Graph Analysis
elbow_graph(range(60,70), descr_tfv)

###  Third Elbow Analysis
elbow_graph(range(80,90), descr_tfv)

### Fourth Elbow Analysis
elbow_graph(range(88,98), descr_tfv)

#all of the silhouette scores from 2 to 101
S_arr=[]
ind = []
for i in range(2, 102):
    s = kmeans_ari_ss(i, descr_tfv)
    S_arr.append(s)
    ind.append(i)
    
#location of max value
ymax = max(S_arr)
xmax = ind[99]

#first value
ymin = min(S_arr)
xmin = ind[0]

plt.plot(ind, S_arr, c = 'blue', marker = '.')
plt.xlabel('k')
plt.ylabel('Silhouette Scores')
plt.title('KMeans Clustering')
plt.grid(True)
plt.ylim([0,0.07])
plt.annotate('k = 101; SS = 0.058', xy = (xmax, ymax), xytext=(70, 0.035), 
             arrowprops=dict(facecolor='red', arrowstyle='->'))
plt.annotate('k = 2; SS = 0.006', xy = (xmin, ymin), xytext=(15, 0.004), 
             arrowprops=dict(facecolor='red', arrowstyle='->'))
plt.show()



#####  Agglomerative Clustering
print("\nAgglomerative")
for i in [2, 4, 6, 10, 15, 20, 30]:
    agglom_clustering(i, descr_cv)


#all of the silhouette scores from 2 to 31
S_arr2=[]
ind2 = []
for k in range(2, 32):
    s = agglom_clustering(i, descr_cv)
    S_arr2.append(s)
    ind2.append(k)
    
#location of max value
ymax2 = max(S_arr2)
xmax2 = ind2[10]

#first value
ymin2 = min(S_arr2)
xmin2 = ind2[0]

plt.plot(ind2, S_arr2, c = 'green', marker = '.')
plt.xlabel('clusters')
plt.ylabel('Silhouette Scores')
plt.title('Agglomerative Clustering')
plt.grid(True)
plt.ylim([0,0.2])
plt.annotate('clusters = 12; SS = 0.122', xy = (xmax2, ymax2), xytext=(15, 0.127), 
             arrowprops=dict(facecolor='red', arrowstyle='->'))
plt.annotate('clusters = 2; SS = 0.038', xy = (xmin2, ymin2), xytext=(2, 0.015), 
             arrowprops=dict(facecolor='red', arrowstyle='->'))
plt.show()


#####  DBSCAN
print("\nDBSCAN")
for i in range(1,4):
    dbscan_ss(1, i, descr_tfv)

# min sample = 4
#SS= 0.008
#no_clusters = 26

# min sample = 3
#SS= 0.032
#no_clusters = 51

# min sample = 2
#SS= 0.172
#no_clusters = 203

# min sample = 1
#SS= 0.212
#no_clusters = 968

#all of the silhouette scores from eps = 1 to 4
S_arr3=[]
ind3 = []
for x in range(1, 5):
    s = dbscan_ss(1, x, descr_tfv)
    S_arr3.append(s)
    ind3.append(x)
    
#location of max value
ymax3 = max(S_arr3)
xmax3 = ind3[0]

#first value
ymin3 = min(S_arr3)
xmin3 = ind3[3]


plt.plot(ind3, S_arr3, c = 'orange', marker = '.')
plt.xlabel('min_samples')
plt.ylabel('Silhouette Scores')
plt.title('DBSCAN')
plt.grid(True)
plt.ylim([0,0.25])
plt.annotate('min_sample = 1; SS = 0.212', xy = (xmax3, ymax3), xytext=(1.5, 0.23), 
             arrowprops=dict(facecolor='red', arrowstyle='->'))
plt.annotate('min_sample = 4; SS = 0.008', xy = (xmin3, ymin3), xytext=(3.3, 0.04), 
             arrowprops=dict(facecolor='red', arrowstyle='->'))
plt.show()

