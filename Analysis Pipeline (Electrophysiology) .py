#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import sys
sys.path
from ams_paq_utilities import *
from ams_utilities import *
import scipy
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')
import pyabf
import paq2py
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.interpolate
import itertools
from copy import deepcopy
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
import scipy.stats as ss 
from itertools import product
from sklearn.cluster import AffinityPropagation
from itertools import cycle


# In[2]:


# Import data from spreadsheet 
df = GS_to_df('https://docs.google.com/spreadsheets/d/1ziOx80em0ZhmMmSjKePYbOq3K6sHcOapfHjy9S4oDbk/edit#gid=1828339698')
df


# In[3]:


# Eliminate rows containing redundant, uninformative or incomplete (NaN) data

df = df.drop([1,33,34], axis=0)
df = df.replace([np.inf, -np.inf], np.nan)
dattab = df.dropna(axis=1, how='any')
celllabels = dattab.iloc[0,1:]
dattab = dattab.drop([0], axis=0)
metriclabels = dattab.iloc[:,0]

# Convert to float array and standardise data ((x - mean)/std)
dattab = str_flt(dattab.iloc[:,1:])
dattab -= np.mean(dattab)
dattab /= np.std(dattab)

# Transpose array so variables arranged column-wise 
data = dattab.T
data.columns = metriclabels 
data.index.names = ["Cell Number"]
allcelldata = pd.DataFrame(data)
allcelldata


# In[4]:


# SANITY CHECK - do we observe expected correlations between different properties? 

# Generate a heat map 

plt.figure(figsize=(20,12))
plt.title('Correlation of Electrophysiological Properties', fontsize=20, y = 1.03);
cor = data.corr()
sns.heatmap(cor, annot=False, cmap=plt.cm.seismic)
plt.show()


# In[5]:


# Compute correlations between properties
def corrank(X):
        df = pd.DataFrame([[(i,j),X.corr().loc[i,j]] for i,j in list(itertools.combinations(X.corr(), 2))],columns=['pairs','corr'])    
        print(df.sort_values(by='corr',ascending=False))

corrank(data)


# In[6]:


# PRINCIPAL COMPONENT ANALYSIS 

# The following analysis will attempt to identify clusters of related cell based on a lower dimensional data-set established via PCA 


# In[7]:


# Perform PCA 

pca = PCA(n_components = 10).fit(data)
X_pca = pca.transform(data)
PCA_components = pd.DataFrame(X_pca)
print(data.shape, X_pca.shape)

# Can determine % variance explained to extract appropraite number of components

plt.subplots()
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='red')
plt.xlabel('PCA dimensions')
plt.ylabel('Variance %')
plt.title('PCA: % Explained Variance')

# Eigenvalues greater than one explain more variance than the original variables 

print(pca.explained_variance_)


# In[8]:


# Create a scree plot to establish components for which eigenvalue > 1 

plt.subplots()
plt.scatter(features, pca.explained_variance_, color = 'blue')
plt.plot(features, pca.explained_variance_, color = 'blue')
plt.title('Scree Plot')
plt.xlabel('Dimensions')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()


# In[9]:


# Cumulative % variance explained by extracted components  
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.axvline(x=1, color='r', linestyle='--')
plt.xlabel('Number of Dimensions')
plt.ylabel('Variance (%)') #for each component
plt.title('Cumulative Explained Variance')
plt.show()


# In[10]:


# Re run PCA on appropraite number of components 
pca = PCA(n_components=2)
pca.fit_transform(allcelldata)

# Plot weightings of original metrics on extracted components 
plt.figure(figsize=(20,5))
plt.title('PCA: Metric Weights', fontsize=20, y = 1.03);
PCAweights = pd.DataFrame(pca.components_,columns=allcelldata.columns,index = ['PC1','PC2'])
sns.heatmap(PCAweights, annot = True, cmap=plt.cm.seismic)


# In[11]:


# Create plot of PC space (components 1/2)
P1 = PCA_components[0]
P2 = PCA_components[1]
labels = celllabels


fig, ax = plt.subplots(figsize=(10,10))
plt.scatter(P1,P2, color='blue')
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('Two Component PCA Space (All Data)', fontsize = 20)
ax.grid()

for i,type in enumerate(labels):
    
    x = P1[i]
    y = P2[i]
    plt.text(x+0.03, y+0.03, type, fontsize=10)
    
    
plt.show()


# In[12]:


# Assess appropriate number of clusters by identifying 'elbow' in KMeans Inertia plot 

ks = range(1, 10)
inertias = []
for k in ks:
    
    model = KMeans(n_clusters=k)

    model.fit(PCA_components.iloc[:,0:1])
    
    inertias.append(model.inertia_)
    
fig, ax = plt.subplots()
plt.plot(ks, inertias, '-o', color='black')
plt.xlabel('Number of Clusters, K')
plt.ylabel('Sum of Squared Distance')
plt.title('Elbow Plot')
plt.xticks(ks)
plt.show()


# In[13]:


P1 = np.array(X_pca[:,0])
P2 = np.array(X_pca[:,1])


X = np.column_stack((P1,P2))


# In[15]:


# Clustering method 1: K-Means + Silhouette analysis 
# Sihouette analysis enables you to compare results of K-means for different number of clusters
# Highest silhouette value indicates most variance explained 

range_n_clusters = [2,3,4,5,6,7,8,9,10]

for n_clusters in range_n_clusters:
  
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(10, 4)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)
    
 
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):

        ith_cluster_silhouette_values =             sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        y_lower = y_upper + 10 

    ax1.set_title("Silhouette pot")
    ax1.set_xlabel("Silhouette Cofficient Values")
    ax1.set_ylabel("Cluster label")

  
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    centers = clusterer.cluster_centers_
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("Clustered data")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering:  "
                  "n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

plt.show()


# In[ ]:


# Clustering Method 2: Ward's method
# Hierarchical clustering (bottom-up, agglomerative clustering) 

import scipy.cluster.hierarchy as shc
plt.figure(figsize=(11, 10))
plt.title("Clustering Dendrogram (Ward's Method)")
plt.xlabel("Cell")
# plt.axhline(y=10, color='r', linestyle='--')
plt
dend = shc.dendrogram(shc.linkage(X, method='ward'))


# In[ ]:


# Calinski Harabasz Score = the ratio between the wthin cluster dispersion and between cluster dispersion
# Alternative means of assessing most appopriate cutoff in Ward's dendrogram

CHS = pd.DataFrame()

for k in range(2,10):
    cluster = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward')
    cluster.fit_predict(X)  
    df = pd.DataFrame([metrics.calinski_harabasz_score(data,cluster.labels_)],[k])
    CHS = CHS.append(df)
    
CHS.columns = ['CHS Value']
CHS.index.name = 'Cluster'

CHS


# In[ ]:


# Plot clustered data in principal component space 

plt.figure(figsize=(10,10))
cluster = AgglomerativeClustering(n_clusters =3, affinity = 'euclidean', linkage = 'ward')
cluster.fit_predict(X)
plt.scatter(X[:,0],X[:,1], c=cluster.labels_, cmap='rainbow')
plt.xlabel('Feature Space for 1st Feature')
plt.ylabel('Feature Space for 2nd Feature')
plt.title('Clustering: Wards Method (n=3)')

P1 = X[:,0]
P2 = X[:,1]

for i,type in enumerate(labels):
    
    x = P1[i]
    y = P2[i]
    plt.text(x+0.03, y+0.03, type, fontsize=8)
    
    
plt.show()


# In[ ]:


# Identify cells in each cluster 

df = pd.DataFrame(cluster.labels_, labels)
df.columns = ['Cluster']
df.index.name = 'Cell'

Group1 = df.loc[df['Cluster'] == 0]

Group2 = df.loc[df['Cluster'] == 1]

Group3 = df.loc[df['Cluster'] == 2]


a = pd.DataFrame(Group1.index)
b = pd.DataFrame(Group2.index)
c = pd.DataFrame(Group3.index)

clusters = pd.concat([a,b,c], ignore_index=True, axis=1)
clusters.columns = ['Cluster 1', 'Cluster 2', 'Cluster 3']

clusters


# In[ ]:


# Clustering Method 3: Affinity Propagation 


af = AffinityPropagation(preference = None).fit(X)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

n_clusters_ = len(cluster_centers_indices)

print('Estimated number of clusters: %d' % n_clusters_)


plt.close('all')
plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    class_members = labels == k
    cluster_center = X[cluster_centers_indices[k]]
    plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
    for x in X[class_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.xlabel('Feature Space for 1st feature')
plt.ylabel('Feature Space for 2nd feature')
plt.show()

