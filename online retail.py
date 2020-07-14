#!/usr/bin/env python
# coding: utf-8

# # Overview

# 
# Online retail is a transnational data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail. The company mainly sells unique all-occasion gifts. Many customers of the company are wholesalers.

# # Business Goal

# We aim to segement the Customers based on RFM so that the company can target its customers efficiently.

# The steps are broadly divided into: 
#     
# 1: Reading and Understanding the Data Step 
#         
# 2: Data Cleansing  
#             
# 3: Data Preparation 
#                 
# 4: Model Building 
#                     
# 5: Final Analysis

# # Step 1 : Reading and Understanding Data
# 

# In[7]:


# import required libraries for dataframe and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt 
# import required libraries for clustering
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree


# In[8]:


# Reading the data on which analysis needs to be 
retail= pd.read_excel(r"C:\Users\user\Downloads\OneDrive_2020-05-19\Assignment Clustering\Online Retail.xlsx")
retail.head()


# In[9]:


# shape of df

retail.shape


# In[10]:


# df info

retail.info()


# In[11]:


# df description

retail.describe()


# # Step 2 : Data Cleansing
# 

# In[12]:


# Calculating the Missing Values % contribution in DF

df_null = round(100*(retail.isnull().sum())/len(retail), 2)
df_null


# In[13]:


# Droping rows having missing values

retail = retail.dropna()
retail.shape


# In[14]:


# Changing the datatype of Customer Id as per Business understanding

retail['CustomerID'] = retail['CustomerID'].astype(str)


# # Step 3 : Data Preparation

# We are going to analysis the Customers based on below 3 factors
# 
# 
# R (Recency): Number of days since last purchase
# 
# 
# F (Frequency): Number of tracsactions
# 
# 
# M (Monetary): Total amount of transactions (revenue contributed)

# In[15]:


# New Attribute :Frequency
rfm_f = retail.groupby('CustomerID')['InvoiceNo'].count()
rfm_f = rfm_f.reset_index()
rfm_f.columns = ['CustomerID', 'Frequency']
rfm_f.head()


# In[16]:


# New Attribute: Monetary
retail['Amount'] = retail['Quantity']*retail['UnitPrice']
rfm_m = retail.groupby('CustomerID')['Amount'].sum()
rfm_m = rfm_m.reset_index()
rfm_m.head()


# In[17]:


# Merging the two dfs

rfm = pd.merge(rfm_m, rfm_f, on='CustomerID', how='inner')
rfm.head()


# In[18]:


# New Attribute : Recency

# Convert to datetime to proper datatype

retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'],format='%d-%m-%Y %H:%M')


# In[19]:


# Compute the maximum date to know the last transaction date

max_date = max(retail['InvoiceDate'])
max_date


# In[14]:


# Compute the difference between max date and transaction date

retail['Diff'] = max_date - retail['InvoiceDate']
retail.head()


# In[15]:


# Compute last transaction date to get the recency of customers

rfm_p = retail.groupby('CustomerID')['Diff'].min()
rfm_p = rfm_p.reset_index()
rfm_p.head()


# In[16]:


# Extract number of days only

rfm_p['Diff'] = rfm_p['Diff'].dt.days
rfm_p.head()


# In[17]:


# Merge tha dataframes to get the final RFM dataframe

rfm = pd.merge(rfm, rfm_p, on='CustomerID', how='inner')
rfm.columns = ['CustomerID', 'Amount', 'Frequency', 'Recency']
rfm.head()


# There are 2 types of outliers and we will treat outliers as it can skew our dataset
# *Statistical
# *Domain specific

# In[40]:


# Outlier Analysis of Amount Frequency and Recency

attributes = ['Amount','Frequency','Recency']
plt.rcParams['figure.figsize'] = [10,8]
sns.boxplot(data = rfm[attributes], orient="v", palette="Set2" ,whis=1.5,saturation=1, width=0.7)
plt.title("Outliers Variable Distribution", fontsize = 14, fontweight = 'bold')
plt.ylabel("Range", fontweight = 'bold')
plt.xlabel("Attributes", fontweight = 'bold')


# In[19]:


# Removing (statistical) outliers for Amount
Q1 = rfm.Amount.quantile(0.05)
Q3 = rfm.Amount.quantile(0.95)
IQR = Q3 - Q1
rfm = rfm[(rfm.Amount >= Q1 - 1.5*IQR) & (rfm.Amount <= Q3 + 1.5*IQR)]

# Removing (statistical) outliers for Recency
Q1 = rfm.Recency.quantile(0.05)
Q3 = rfm.Recency.quantile(0.95)
IQR = Q3 - Q1
rfm = rfm[(rfm.Recency >= Q1 - 1.5*IQR) & (rfm.Recency <= Q3 + 1.5*IQR)]

# Removing (statistical) outliers for Frequency
Q1 = rfm.Frequency.quantile(0.05)
Q3 = rfm.Frequency.quantile(0.95)
IQR = Q3 - Q1
rfm = rfm[(rfm.Frequency >= Q1 - 1.5*IQR) & (rfm.Frequency <= Q3 + 1.5*IQR)]


# Rescaling the Attributes
# It is extremely important to rescale the variables so that they have a comparable scale.| There are two common ways of rescaling:
# 
# 1.Min-Max scaling
# 2.Standardisation (mean-0, sigma-1)
# Here, we will use Standardisation Scaling.

# In[20]:


# Rescaling the attributes

rfm_df = rfm[['Amount', 'Frequency', 'Recency']]

# Instantiate
scaler = StandardScaler()

# fit_transform
rfm_df_scaled = scaler.fit_transform(rfm_df)
rfm_df_scaled.shape


# In[21]:


rfm_df_scaled = pd.DataFrame(rfm_df_scaled)
rfm_df_scaled.columns = ['Amount', 'Frequency', 'Recency']
rfm_df_scaled.head()


# # Step 4 : Building the Model

# K-Means Clustering K-means clustering is one of the simplest and popular unsupervised machine learning algorithms.
# 
# The algorithm works as follows:
# 
# 1.First we initialize k points, called means, randomly. 
# 
# 2.We categorize each item to its closest mean and we update the mean’s coordinates, which are the averages of the items categorized in that mean so far. 
# 
# 3.We repeat the process for a given number of iterations and at the end, we have our clusters.
# 
# 

# In[42]:


# k-means with some arbitrary k

kmeans = KMeans(n_clusters=4, max_iter=50)
kmeans.fit(rfm_df_scaled)


# In[23]:


kmeans.labels_


# Finding the Optimal Number of Clusters
# 

# Elbow Curve to get the right number of Clusters
# A fundamental step for any unsupervised algorithm is to determine the optimal number of clusters into which the data may be clustered. The Elbow Method is one of the most popular methods to determine this optimal value of k.
# 
# 

# In[24]:


# Elbow-curve/SSD

ssd = []
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
for num_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(rfm_df_scaled)
    
    ssd.append(kmeans.inertia_)
    
# plot the SSDs for each n_clusters
plt.plot(ssd)


# Silhouette Analysis
# silhouette score=p−q/max(p,q)
#  
# p  is the mean distance to the points in the nearest cluster that the data point is not a part of
# 
# q  is the mean intra-cluster distance to all the points in its own cluster.
# 
# 1.The value of the silhouette score range lies between -1 to 1.
# 
# 2.A score closer to 1 indicates that the data point is very similar to other data points in the cluster,
# 
# 3.A score closer to -1 indicates that the data point is not similar to the data points in its cluster.
# 
# 

# In[25]:


# Silhouette analysis
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

for num_clusters in range_n_clusters:
    
    # intialise kmeans
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(rfm_df_scaled)
    
    cluster_labels = kmeans.labels_
    
    # silhouette score
    silhouette_avg = silhouette_score(rfm_df_scaled, cluster_labels)
    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))
    


# In[26]:


# Final model with k=3
kmeans = KMeans(n_clusters=3, max_iter=50)
kmeans.fit(rfm_df_scaled)


# In[27]:


kmeans.labels_


# In[28]:


# assign the label
rfm['Cluster_Id'] = kmeans.labels_
rfm.head()


# In[29]:



# Box plot to visualize Cluster Id vs Frequency

sns.boxplot(x='Cluster_Id', y='Amount', data=rfm)


# In[30]:


# Box plot to visualize Cluster Id vs Frequency

sns.boxplot(x='Cluster_Id', y='Frequency', data=rfm)


# In[31]:


# Box plot to visualize Cluster Id vs Frequency

sns.boxplot(x='Cluster_Id', y='Frequency', data=rfm)


# Hierarchical Clustering
# Hierarchical clustering involves creating clusters that have a predetermined ordering from top to bottom. For example, all files and folders on the hard disk are organized in a hierarchy. There are two types of hierarchical clustering,
# 
# 1.Divisive
# 2.Agglomerative.

# Single Linkage:
# 
# In single linkage hierarchical clustering, the distance between two clusters is defined as the shortest distance between two points in each cluster. For example, the distance between clusters “r” and “s” to the left is equal to the length of the arrow between their two closest points.

# In[32]:


# single linkage
mergings = linkage(rfm_df_scaled, method="single", metric='euclidean')
dendrogram(mergings)
plt.show()


# Complete Linkage
# 
# In complete linkage hierarchical clustering, the distance between two clusters is defined as the longest distance between two points in each cluster. For example, the distance between clusters “r” and “s” to the left is equal to the length of the arrow between their two furthest points.

# In[33]:


# Complete linkage

mergings = linkage(rfm_df_scaled, method="complete", metric='euclidean')
dendrogram(mergings)
plt.show()


# Average Linkage:
# 
# In average linkage hierarchical clustering, the distance between two clusters is defined as the average distance between each point in one cluster to every point in the other cluster. For example, the distance between clusters “r” and “s” to the left is equal to the average length each arrow between connecting the points of one cluster to the other.

# In[34]:


# Average linkage

mergings = linkage(rfm_df_scaled, method="average", metric='euclidean')
dendrogram(mergings)
plt.show()


# Cutting the Dendrogram based on K

# In[35]:


# 3 clusters
cluster_labels = cut_tree(mergings, n_clusters=3).reshape(-1, )
cluster_labels


# In[36]:


# Assign cluster labels

rfm['Cluster_Labels'] = cluster_labels
rfm.head()


# In[37]:


# Plot Cluster Id vs Amount

sns.boxplot(x='Cluster_Labels', y='Amount', data=rfm)


# In[38]:


# Plot Cluster Id vs Frequency

sns.boxplot(x='Cluster_Labels', y='Frequency', data=rfm)


# In[39]:


# Plot Cluster Id vs Recency

sns.boxplot(x='Cluster_Labels', y='Recency', data=rfm)


# # Step 5 : Final Analysis

# Inference: K-Means Clustering with 3 Cluster Ids
# 
# 1.Customers with Cluster Id 1 are the customers with high amount of transactions as compared to other customers.
# 
# 2.Customers with Cluster Id 1 are frequent buyers.
# 
# 3.Customers with Cluster Id 2 are not recent buyers and hence least of importance from business point of view.
# 
# 

# Hierarchical Clustering with 3 Cluster Labels
# 
# 1.Customers with Cluster_Labels 2 are the customers with high amount of transactions as compared to other customers.
# 
# 2.Customers with Cluster_Labels 2 are frequent buyers.
# 
# 3.Customers with Cluster_Labels 0 are not recent buyers and hence least of importance from business point of view.
