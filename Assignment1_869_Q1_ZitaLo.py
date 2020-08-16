# Name: Zita Lo
# Student Number: 20196119
# Program: MMA
# Cohort: Winter 2021
# Course Number: MMA 869
# Date: August 13, 2020
#

# Answer to Question 1


# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_profiling import ProfileReport
import matplotlib
matplotlib.use('Agg')

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Read in data
df = pd.read_csv("jewelry_customers.csv")

# # -----------------------
# # EDA
# # -----------------------

# Understand the data
df.head()
df.info()

# To print out feature column names by categorical vs numerics
n_features = df.shape[1]
cat_col_names = list(df.select_dtypes(include=np.object).columns)
num_col_names = list(df.select_dtypes(include=np.number).columns)

print('cat_col_names: {}'.format(cat_col_names))
print('num_col_names: {}'.format(num_col_names))


# Descriptive stats of the numeric features
df.describe(include=[np.number]).transpose()

# Generate Profile Report on the Jewelry DataFrame
# This part only has to generate it once 

# profile = ProfileReport(df)
# profile.to_file('Jewelry Customer Profiling Report.html')

# Check if there are null values
df.isnull().sum()

# Plot graph to understand data - Age distribution
sns.distplot(df['Age'],kde=False,bins=20)

# Plot graph to understand data - Income distribution
sns.distplot(df['Income'],kde=False,bins=20)

# Plot graph to understand data - Spending Score distribution
sns.distplot(df['SpendingScore'],kde=False,bins=20)


# Plot graph to understand data - Spending Score vs Age
sns.lmplot(data = df, y='SpendingScore',x='Age')

# Plot some graphs to understand all data in pairs
sns.pairplot(df)

# Plot Age vs Income
sns.jointplot(data=df,x='Age',y='Income')

# Plot Age vs Spending Score
sns.jointplot(data=df,x='Age',y='SpendingScore')

# Plot heat map to look at correlations between features
sns.heatmap(df.corr())

# # -----------------------
# # Scale and normalize data
# # -----------------------


from sklearn.preprocessing import StandardScaler

# To standardize features by removing the mean and scaling to unit variance. Especially income and savings features
# Most clustering algorithms e.g. Kmeans require standardize features to begin with
col_names = df.columns
col_names

scaler = StandardScaler()
X = scaler.fit_transform(df)
X


# Convert it from series to a data frame and include headings for viewing purpose
X_feat = pd.DataFrame(X,columns=col_names)
X_feat.head()

# # -----------------------
# # Hierarchcial Clustering
# # -----------------------

import scipy.cluster
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Perform hierarchical/agglomerative clustering - with 'ward'method and 'euclidean' metric
aggl = scipy.cluster.hierarchy.linkage(X, method='ward', metric='euclidean')

# Define 5 clusters
labels = scipy.cluster.hierarchy.fcluster(aggl, 5, criterion="maxclust")

# Performance measure using silhouette score, calinski harabasz score and davies bouldin score
sil = silhouette_score(X , labels)
cal = calinski_harabasz_score(X, labels)
dav = davies_bouldin_score(X, labels)
print("Silhouette Score ={:.3},calinski_harabasz={},davies_bouldin={:.3}".format(sil, cal, dav))

# # -----------------------
# # Parameter Exploration
# # -----------------------

# Compare performance using different paramters including the linkages, metrics and number of clusters

import itertools
def plot_agg(X, linkage, metric, num):
    aggl = scipy.cluster.hierarchy.linkage(X, method=linkage, metric=metric)    
    labels = scipy.cluster.hierarchy.fcluster(aggl, num, criterion="maxclust")
    
    sil = 0
    n = len(set(labels))
    if n > 1:
        #sil = silhouette_score(X , labels, metric=metric)
        sil = silhouette_score(X , labels)
        cal = calinski_harabasz_score(X, labels)
        dav = davies_bouldin_score(X, labels)

    print("Linkage={}, Metric={}, Clusters={}, Silhouette={:.3},calinski_harabasz={},davies_bouldin={:.3}".format(linkage, metric, n, sil, cal, dav))
    
linkages = ['complete', 'ward', 'single', 'centroid', 'average']
metrics = ['euclidean', 'minkowski', 'cityblock', 'cosine', 'correlation', 'chebyshev', 'canberra', 'mahalanobis']
num = [3,4,5,6]
for prod in list(itertools.product(linkages, metrics, num)):
    
    # Some combos are not allowed
    if (prod[0] in ['ward', 'centroid']) and prod[1] != 'euclidean':
        continue
        
    plot_agg(X, prod[0], prod[1], prod[2])

# # -----------------------
# # Plot Dendogram based on Finalized Parameters
# # -----------------------

# Define K=5 clusters (which gives the best evaluation result) as our final version
K=5
aggl = scipy.cluster.hierarchy.linkage(X, method='ward', metric='euclidean')
labels = scipy.cluster.hierarchy.fcluster(aggl, K, criterion="maxclust")


# Plot the dendogram with better labels
plt.figure(figsize=(16, 8))
plt.grid(False)
plt.title("Jewelry Store Customers Dendogram")
dend = scipy.cluster.hierarchy.dendrogram(aggl, color_threshold=5)
plt.savefig('jewelry_dendro_hcluster.jpg')


# Plot the dendogram with better labels

# Custom function to give each leaf of the dendrogram a label -
# Printing out feature values for each isntance in a pretty way
def llf(id):
    Xr = [int(x) for x in scaler.inverse_transform(X[id, :])]
    return "{:>04d}: [{:>2d}, {:>2d}, {:>3d}, {:>2d}]".format(id, Xr[0], Xr[1], Xr[2], Xr[3])


# Plot the dendogram with better labels
plt.figure(figsize=(12, 35))
plt.grid(False)
plt.title("Jewelry Store Customers Dendogram")  
dend = scipy.cluster.hierarchy.dendrogram(aggl, color_threshold=5, orientation="left", leaf_font_size=10, leaf_label_func=llf)


# Generate statistics (count and mean) for each cluster

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

means = np.zeros((K, X.shape[1]))

for i, label in enumerate(set(labels)):
    means[i,:] = X[labels==label].mean(axis=0)
    print('\nCluster {} (n={}):'.format(label, sum(labels==label)))
    print(scaler.inverse_transform(means[i,:]))
    
means


# Generate summarized statistics ('count', 'Min', 'Mean', 'Max', 'Variance', 'Skewness', 'Kurtosis') for each cluster

from scipy import stats

pd.set_option("display.precision", 2)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

def stats_to_df(d, scaler):
    tmp_df = pd.DataFrame(columns=col_names)
    
    tmp_df.loc[0] = scaler.inverse_transform(d.minmax[0])
    tmp_df.loc[1] = scaler.inverse_transform(d.mean)
    tmp_df.loc[2] = scaler.inverse_transform(d.minmax[1])
    tmp_df.loc[3] = scaler.inverse_transform(d.variance)
    tmp_df.loc[4] = scaler.inverse_transform(d.skewness)
    tmp_df.loc[5] = scaler.inverse_transform(d.kurtosis)
    tmp_df.index = ['Min', 'Mean', 'Max', 'Variance', 'Skewness', 'Kurtosis'] 
    
    return tmp_df.T

print('All Data:')
print('Number of Instances: {}'.format(X.shape[0]))
d = stats.describe(X, axis=0)
#display(stats_to_df(d, scaler))
print(stats_to_df(d, scaler))

for i, label in enumerate(set(labels)):
    d = stats.describe(X[labels==label], axis=0)
    print('\nCluster {}:'.format(label))
    print('Number of Instances: {}'.format(d.nobs))
    #display(stats_to_df(d, scaler))
    print(stats_to_df(d, scaler))


# Generate a dataframe with headings and labelled cluster and export it as csv 
X_df = pd.DataFrame(scaler.inverse_transform(X), columns=col_names)
X_df['cluster'] = labels
X_df.head()
X_df.to_csv('Question1_HClust_Results.csv')

# Used panda's group-by function, generate statistics on different features
cl_group = X_df.groupby(['cluster']).agg('describe')
cl_group

cl_group.info()
print('\n\n','Group by Cluster - Means of Age: ')
cl_group['Age']

print('\n\n','Group by Cluster - Means of Spending Score: ')
cl_group['SpendingScore']

print('\n\n','Group by Cluster - Means of Income: ')
cl_group['Income']

print('\n\n','Group by Cluster - Means of Savings: ')
cl_group['Savings']

# # -----------------------
# # Examplars
# # -----------------------

# Generate examplar for each cluster
from scipy.spatial import distance

for i, label in enumerate(set(labels)):
    X_tmp= X
    exemplar_idx = distance.cdist([means[i]], X).argmin()
   
    print('\nCluster {}:'.format(label))
    #display(df.iloc[[exemplar_idx]])
    print(df.iloc[[exemplar_idx]])

# # -----------------------
# # Relative Importance Plots
# # -----------------------
# The idea here is to understand how does each feature compared to overall customers
# Greener = higher value than population
# Redder = lower value than population
# Yellow = same value as population

all_means = X.mean(axis=0)
all_means
relative_imp = means - all_means
relative_imp.shape
plt.figure(figsize=(8, 4));
plt.title('Relative importance of features');
sns.heatmap(data=relative_imp, 
            annot=scaler.inverse_transform(relative_imp), 
            fmt='.2f', 
            cmap='RdYlGn', 
            robust=True, 
            square=False,
            xticklabels=col_names, 
            yticklabels=['Cluster {}'.format(x) for x in range(K)]);
plt.savefig('jewelry_store_feature_importance_heatmap.png')

# # -----------------------
# # Performance Metrics
# # -----------------------
# Performance measure for our final results using silhouette score, calinski harabasz score and davies bouldin score

# Calculate silhouette score
# The Silhouette Coefficient is calculated using the mean intra-cluster distance and the mean nearest-cluster distance for each sample. 
# The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters.
# Negative values generally indicate that a sample has been assigned to the wrong cluster, as a different cluster is more similar.
sil = silhouette_score(X , labels)

# Calculate calinski harabasz score
# The score is defined as ratio between the within-cluster dispersion and the between-cluster dispersion.
# A higher Calinski-Harabasz score relates to a model with better defined clusters.
# The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster.
cal = calinski_harabasz_score(X, labels)

# Calculate davies bouldin score
# The score is defined as the average similarity measure of each cluster with its most similar cluster, 
# where similarity is the ratio of within-cluster distances to between-cluster distances. 
# Thus, clusters which are farther apart and less dispersed will result in a better score.
# he minimum score is zero, with lower values indicating better clustering.
dav = davies_bouldin_score(X, labels)
print("Silhouette Score ={:.3},calinski_harabasz={},davies_bouldin={:.3}".format(sil, cal, dav))

