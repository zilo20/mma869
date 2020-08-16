# Name: Zita Lo
# Student Number: 20196119
# Program: MMA
# Cohort: Winter 2021
# Course Number: MMA 869
# Date: August 13, 2020


# Answer to Question 1
# See Assignment1_869_Q1_ZitaLo.py for Final Result
# This is supplementary code - Pycaret 2.0 (HClust and Kmeans)

# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Read in data
df = pd.read_csv("jewelry_customers.csv")

# # -------------------------
# # EDA
# # -------------------------
df.head()
df.info()

# # -------------------------
# # PyCaret
# # -------------------------
# Set up data for clustering
from pycaret.clustering import *

clu1 = setup(df, 
             remove_multicollinearity = True, multicollinearity_threshold = 0.9, 
             session_id=123, log_experiment=True, log_plots = True, 
             transformation=True,          
             ignore_low_variance = True )

# Display the data after set up
clu1

# Call out the transformed data

type(clu1)
transformedData = clu1[5][2][1]
print(transformedData)


# Create models
models()

# Create kmeans model
kmeans = create_model('kmeans', num_clusters = 5)

# Assign kmeans model and export to csv
kmeans_results = assign_model(kmeans)
kmeans_results.head()
kmeans_results.to_csv('kmeans_results_pycaret.csv')

# Plot silhouette graph
plot_model(kmeans, plot = 'silhouette')

# Plot kmeans elbow graph
plot_model(kmeans, plot = 'elbow')

# Plot kmeans cluster graph with PCA components
plot_model(kmeans)

# Create hierarchical clustering
hclust = create_model('hclust', num_clusters = 5)

# Print out to see hclust parameters
hclust

# Set up different parameter for hierarchical clustering
# Score the same as hclust
hclust1 = create_model('hclust', affinity='correlation',linkage='complete',num_clusters = 5)

# Print out to see hclust1 parameters
hclust1

# View labels of hclust
hclust.labels_

# Plot model in a 2d chart using PCA components
plot_model(hclust)

# Assign model to data and explort results to csv file
hclust_results = assign_model(hclust)
hclust_results.head()
hclust_results['Cluster'].value_counts()
hclust_results.reset_index(inplace = True, drop = True) 
hclust_results.to_csv('hclust_results_pycaret.csv')

# # -------------------------
# # Intepreting the clusters
# # -------------------------
# # Means of each cluster and Examplar

from scipy.spatial import distance

#  Step 1 - Calulcate the mean of each cluster first
labels = hclust_results['Cluster'].str.replace('Cluster','')
#labels

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

K = 5
X= hclust_results.drop(columns =['Cluster'])
#hclust_results.shape[1]

# hclust_results.shape
# shape is 5 x 505
means = np.zeros((K, X.shape[1]))

for i, label in enumerate(set(labels)):
    means[i,:] = X[labels==label].mean(axis=0)
    print('\nCluster {} (n={}):'.format(label, sum(labels==label)))
    print(means[i,:])
    
means
pd.DataFrame(means).to_csv('means_byCluster.csv')

# -----------------------------------
# Step 2 - Based on the means from step 1, calulcate examplar for each cluster

i=0
unique_id_list = list()


for i, label in enumerate(set(labels)):
    X_tmp= X[labels ==label].copy()
    
    # Euclidean distance 
    #exemplar_idx = distance.cdist([means[i]], X_tmp).argmin()
    
    #cosine distance
    exemplar_idx = distance.cdist([means[i]], X_tmp,metric='cosine').argmin()   
    
 
    print('exemplar start')
    print('\nCluster {}:'.format(label))
    display(X_tmp.iloc[[exemplar_idx]])
    print('exemplar end ************' + '\n')
    
    

# # -------------------------------------------------
# # Means of each cluster (another way to calculate)
# # -------------------------------------------------

# Used panda's group-by function to calculate means
import pandas as pd

X_df = pd.DataFrame(hclust_results, columns=hclust_results.columns)
cl_group = X_df.groupby(['Cluster']).agg('describe')
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


# # -------------------------
# # Relative Importance Plots
# # -------------------------
# The idea here is to understand how does each feature compared to overall customers
# Greener = higher value than population
# Redder = lower value than population
# Yellow = same value as population

all_means = X.mean(axis=0)

relative_imp = means - all_means.to_numpy()
col_names = hclust_results.columns

plt.figure(figsize=(8, 4));
plt.title('Relative importance of features');
sns.heatmap(data=relative_imp, 
            annot=relative_imp, 
            #annot=scaler.inverse_transform(relative_imp), 
            fmt='.2f', 
            cmap='RdYlGn', 
            robust=True, 
            square=False,
            xticklabels=col_names, 
            yticklabels=['Cluster {}'.format(x) for x in range(K)]);
#plt.savefig('heatmap_pycaret_hclust.png')



