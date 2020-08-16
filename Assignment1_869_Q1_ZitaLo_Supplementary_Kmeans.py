# Name: Zita Lo
# Student Number: 20196119
# Program: MMA
# Cohort: Winter 2021
# Course Number: MMA 869
# Date: August 13, 2020
#

# Answer to Question 1
# See Assignment1_869_Q1_ZitaLo.py for Final Result
# This is supplementary code - Kmeans

# Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read in data
df = pd.read_csv("jewelry_customers.csv")

# # ----------------------
# # EDA
# # ----------------------
# Understand the data
df.head()
df.info()
df.shape
df.describe().transpose()

# Check if there are null values
df.isnull().any()

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

# # ----------------------
# # Scale and normalize data
# # ----------------------
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df)
scaled_features = scaler.transform(df)

# Convert it from series to a data frame and include headings for viewing purpose
df_feat = pd.DataFrame(scaled_features,columns=df.columns)
df_feat.head()

# # ----------------------
# # K Means Cluster
# # ----------------------
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5,init='k-means++', n_init=10,random_state=101)

# Fit K means model
kmeans.fit(df_feat)

# Display K means centers
kmeans.cluster_centers_

# Display K means labels
kmeans.labels_

# Plot k menas cluster results by age and spending score
f = plt.figure(figsize=(10,6))
ax1 = f.add_axes([0,0,1,1])
ax1.set_title('K Means')
ax1.scatter(df_feat['Age'],df_feat['SpendingScore'],c=kmeans.labels_,cmap='rainbow')

# Plot k menas cluster results by income and spending score
f = plt.figure(figsize=(10,6))
ax1 = f.add_axes([0,0,1,1])
ax1.set_title('K Means')
ax1.scatter(df_feat['Income'],df_feat['SpendingScore'],c=kmeans.labels_,cmap='rainbow')

# Inverse transform cluster centers
scaler.inverse_transform(kmeans.cluster_centers_)

# Look at some example rows in each cluster
for label in set(kmeans.labels_):
    print('\nCluster {}:'.format(label))
    print(scaler.inverse_transform(df_feat[kmeans.labels_==label].head()))

# Visualize Silhouette for the 5 clusters
from yellowbrick.cluster import SilhouetteVisualizer, InterclusterDistance, KElbowVisualizer
visualizer = SilhouetteVisualizer(kmeans)
visualizer.fit(df_feat)
visualizer.poof()
fig = visualizer.ax.get_figure()
fig.savefig('kmeans-5-silhouette.png', transparent=False)

# # ----------------------
# # Hyperparameter Tuning
# # ----------------------
from sklearn.metrics import silhouette_score, silhouette_samples

# load the function
def do_kmeans(X, k):
    kmeans = KMeans(init='k-means++', n_clusters=k, n_init=10, random_state=101)
    kmeans.fit(X)

    plt.figure();
    plt.scatter(df_feat.iloc[:, 0], df_feat.iloc[:, 1], c=kmeans.labels_)
    plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], marker='x', c="black")
    plt.title("K-Means (K={})".format(k));
    plt.xlabel('Income (K)');
    plt.ylabel('Spending Score');
    #plt.savefig('simple_kmeans_k{}.png'.format(k))
    plt.show()
    
    wcss = kmeans.inertia_
    sil = silhouette_score(df_feat, kmeans.labels_)
    print("K={}, WCSS={:.2f}, Sil={:.2f}".format(k, wcss, sil))

# Loop through K from 2 to 8 and generate plots
for k in range(2, 8):
    do_kmeans(df_feat, k)

# # ----------------------
# # Elbow Method
# # ----------------------
# Display plots using Elbow Method and find the optimal cluster number
inertias = {}
silhouettes = {}
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(df_feat)
    inertias[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
    silhouettes[k] = silhouette_score(df_feat, kmeans.labels_, metric='euclidean')
    

plt.figure();
plt.plot(list(inertias.keys()), list(inertias.values()));
plt.title('K-Means, Elbow Method')
plt.xlabel("Number of clusters, K");
plt.ylabel("Inertia");


plt.figure();
plt.plot(list(silhouettes.keys()), list(silhouettes.values()));
plt.title('K-Means, Elbow Method')
plt.xlabel("Number of clusters, K");
plt.ylabel("Silhouette");

