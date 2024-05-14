#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[3]:


df = pd.read_csv('ECommerceCustomer.csv', encoding='ISO-8859-1')
df


# In[4]:


duplicate_rows = df[df.duplicated()]
df


# In[5]:


df.isnull().sum()


# In[6]:


df.dropna()
df.shape


# In[7]:


df.notnull().count()


# In[8]:


df.describe


# In[9]:


bool_series = pd.notnull(df)
bool_series


# In[10]:


df.dropna()


# In[11]:


# using dropna() function     
df.dropna(how = 'all',inplace=True)
df


# In[12]:


import pandas as pd

df = pd.read_csv('ECommerceCustomer.csv', encoding='ISO-8859-1')


print(df.info())


print(df.head())


print("Missing values:\n", df.isnull().sum())



# In[13]:


cust_country = df[['Country', 'CustomerID']].drop_duplicates()
cust_country.groupby(['Country'])['CustomerID'].aggregate('count').reset_index().sort_values('CustomerID', ascending=False)
#df_data.groupby(['Country'])['Customer ID'].head()


# In[14]:


#keep only UK customer data
df = df.query("Country== 'United Kingdom'").reset_index(drop=True)


# In[15]:


#From the df_info() command we can see that the 'Customer ID' feature has some null values
print('Number of customers having no ID: ',df['CustomerID'].isnull().sum())


# In[16]:


#Ignore the transactions where there is no Customer ID
df_data = df[df['CustomerID'].notna()==True]
df_data


# In[17]:


#Check if there are any negative values in Price
df[df['UnitPrice']<0]


# In[18]:


index_name = df[(df['UnitPrice']==0)].index
df.drop(index_name, inplace = True)


# In[19]:


#Create a new feature named TotalPrice by multiplying Quantity and Price feature
df_data['TotalPrice'] = df_data['Quantity'] * df_data['UnitPrice']


# In[20]:


#Get the top customers
df_data.groupby("CustomerID")["TotalPrice"].sum()


# In[21]:


#for datetime operations
import datetime as dt
from datetime import datetime


# In[22]:


#We will assign the next day after the last date recorded in the dataset as 'today_date'
today_date = dt.datetime(2011, 12, 11)


# In[23]:


#data after cleaning
df_data2 = df_data.copy(deep=True)
df_data2.head()


# In[24]:


df_data2['InvoiceDate'] = df_data2['InvoiceDate'].astype('datetime64[ns]')


# In[25]:


rfm = df_data2.groupby('CustomerID').agg({'InvoiceDate': lambda date: (today_date - date.max()).days,
                                     'InvoiceNo': lambda num: num.nunique(),
                                     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})


# In[26]:


rfm=rfm.reset_index()
rfm.head(20)


# In[27]:


rfm.shape


# In[28]:


rfm.size


# In[29]:


#renaming the column names
rfm.columns = ['CustomerID','Recency', 'Frequency', 'Monetary']
rfm.head()


# In[30]:


import pandas as pd
import numpy as np

# Sample RFM dataset (replace with your actual dataset)
rfm.columns = ['CustomerID','Recency', 'Frequency', 'Monetary']
rfm.head()


# In[31]:


#Split into four segments using quantiles
quantiles = rfm.quantile(q=[0.25,0.5,0.75])
quantiles = quantiles.to_dict()


# In[34]:


quantiles


# In[34]:


def score_rfm(x, quantiles, segment, is_reverse=False):
    """
    Score RFM segments based on defined quartiles.

    Parameters:
    - x (float): The RFM value to score.
    - quantiles (dict): A dictionary containing the quartiles for each RFM segment.
    - segment (str): The segment key ('Recency', 'Frequency', 'Monetary').
    - is_reverse (bool): Flag to determine scoring direction. True for Frequency and Monetary.

    Returns:
    - int: The score for the segment.
    """
    if x <= quantiles[segment][0.25]:
        return 4 if is_reverse else 1
    elif x <= quantiles[segment][0.50]:
        return 3 if is_reverse else 2
    elif x <= quantiles[segment][0.75]:
        return 2 if is_reverse else 3
    else:
        return 1 if is_reverse else 4

# Ensure the quantiles dictionary is properly defined as shown previously
# Then, apply the scoring function like this:

rfm['R'] = rfm['Recency'].apply(score_rfm, args=(quantiles, 'Recency', False))
rfm['F'] = rfm['Frequency'].apply(score_rfm, args=(quantiles, 'Frequency', True))
rfm['M'] = rfm['Monetary'].apply(score_rfm, args=(quantiles, 'Monetary', True))


# In[35]:


#Calculate Add R, F and M segment value columns in the existing dataset to show R, F and M segment values
rfm['R'] = rfm['Recency'].apply(RScoring, args=('Recency',quantiles,))
rfm['F'] = rfm['Frequency'].apply(FnMScoring, args=('Frequency',quantiles,))
rfm['M'] = rfm['Monetary'].apply(FnMScoring, args=('Monetary',quantiles,))
rfm.head(20)


# In[39]:


pip install minisom


# # SOM Neural Network

# In[44]:


from sklearn.preprocessing import MinMaxScaler


# In[45]:


scaler = MinMaxScaler()
rfm_normalized = scaler.fit_transform(rfm)


# In[47]:


# Example of calculating a simple RFMScore (you might have a different formula)
rfm['RFMScore'] = rfm['Recency'] + rfm['Frequency'] + rfm['Monetary']

# Placeholder for clustering, assuming clustering hasn't been done yet
# You should replace this with actual clustering logic if needed
rfm['cluster'] = 'ClusterID'  # Replace 'ClusterID' with actual cluster IDs if available


# In[49]:


from minisom import MiniSom

# Ensure rfm_normalized is a numpy array or convert it if it's a DataFrame
if isinstance(rfm_normalized, pd.DataFrame):
    rfm_data = rfm_normalized.values
else:
    rfm_data = rfm_normalized  # Assuming rfm_normalized is already a numpy array

# Adjust parameters such as the number of nodes, learning rate, and neighborhood size
som_rows = 20  # Number of rows in the SOM grid
som_cols = 20  # Number of columns in the SOM grid
input_len = rfm_data.shape[1]  # Number of features in the input data

som = MiniSom(som_rows, som_cols, input_len, sigma=1.0, learning_rate=0.7)
som.random_weights_init(rfm_data)
som.train_random(data=rfm_data, num_iteration=200)

# Find the best-matching unit (BMU) for each customer in the SOM
bmu_coordinates = [som.winner(x) for x in rfm_data]

# Add the BMU information to the original DataFrame
rfm['bmu_row'] = [index[0] for index in bmu_coordinates]
rfm['bmu_col'] = [index[1] for index in bmu_coordinates]

# Display the results (ensure all these columns exist or adjust accordingly)
print(rfm[['Recency', 'Frequency', 'Monetary', 'RFMScore', 'cluster', 'bmu_row', 'bmu_col']].iloc[0:10])


# In[60]:


from minisom import MiniSom
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Assuming rfm is your DataFrame containing the Recency, Frequency, Monetary columns
scaler = MinMaxScaler()
rfm_normalized = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# Initializing and training the SOM
som = MiniSom(10, 10, 3, sigma=1.0, learning_rate=0.5, random_seed=42)
som.random_weights_init(rfm_normalized)
som.train_random(rfm_normalized, 500)

# Using k-means to cluster the SOM weight vectors
num_clusters = 5  # Define the number of clusters you want to identify
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

# Correctly reshape the weights
weights = som.get_weights().reshape((-1, som.get_weights().shape[-1]))
kmeans.fit(weights)

# Mapping each data point to its closest cluster
bmu_indices = np.array([som.winner(x) for x in rfm_normalized])
cluster_labels = kmeans.labels_[bmu_indices[:, 0] * som.get_weights().shape[1] + bmu_indices[:, 1]]

# Adding the cluster labels to the original DataFrame
rfm['cluster'] = cluster_labels

# Count the number of data points per cluster
cluster_counts = rfm['cluster'].value_counts()

# Display the cluster counts
print("Number of data points per cluster:\n", cluster_counts)

# Optionally, you can print the first few rows to inspect the clusters assigned to each entry
print(rfm.head())


# In[61]:


from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Calculate Quantization Error
qe = som.quantization_error(rfm_normalized)

# Calculate Topographic Error
te = som.topographic_error(rfm_normalized)

print("Quantization Error: {:.4f}".format(qe))
print("Topographic Error: {:.4f}".format(te))

# Assuming we have used k-means for clustering on SOM as earlier
# We will use cluster labels from k-means to calculate cluster validity measures

# Check if the number of clusters is > 1 and less than the number of samples
if 1 < len(np.unique(cluster_labels)) < len(rfm_normalized):
    silhouette_avg = silhouette_score(rfm_normalized, cluster_labels)
    davies_bouldin = davies_bouldin_score(rfm_normalized, cluster_labels)
    calinski_harabasz = calinski_harabasz_score(rfm_normalized, cluster_labels)

    print("Silhouette Score: {:.4f}".format(silhouette_avg))
    print("Davies-Bouldin Index: {:.4f}".format(davies_bouldin))
    print("Calinski-Harabasz Index: {:.4f}".format(calinski_harabasz))
else:
    print("Not enough clusters or data points to compute cluster validity measures.")


# # Hybrid Model (Gausssian Mixture + Hybrid Model)

# In[63]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Assuming rfm_scaled is your scaled RFM data
n_components = np.arange(1, 21)
models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(rfm_scaled)
          for n in n_components]

plt.plot(n_components, [m.bic(rfm_scaled) for m in models], label='BIC')
plt.plot(n_components, [m.aic(rfm_scaled) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components')
plt.ylabel('Information Criterion')
plt.title('AIC and BIC for Different Number of Components')
plt.show()


# In[66]:


from sklearn.mixture import GaussianMixture
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Example: Load your RFM DataFrame
# rfm = pd.read_csv('path_to_your_data.csv')

# Assuming 'rfm' DataFrame contains the RFM features: Recency, Frequency, Monetary
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# Initialize the Gaussian Mixture Model
# Choose the number of components (clusters) based on your analysis
gmm = GaussianMixture(n_components=5, random_state=42)

# Fit the GMM to the scaled data
gmm.fit(rfm_scaled)

# Predict the cluster for each data point
cluster_labels = gmm.predict(rfm_scaled)

# Assign these labels back to your original DataFrame
rfm['cluster'] = cluster_labels

# Optionally, you might want to calculate the probabilities of cluster assignments
# which can be insightful especially when deciding borderline cases
cluster_probabilities = gmm.predict_proba(rfm_scaled)

# Display the first few entries to see the cluster assignment
print(rfm.head())



# In[69]:


import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Assuming 'rfm' is your DataFrame with 'Recency', 'Frequency', 'Monetary'
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# Fit the Gaussian Mixture Model
gmm = GaussianMixture(n_components=5, random_state=42)
gmm.fit(rfm_scaled)
clusters = gmm.predict(rfm_scaled)

# Apply t-SNE to the RFM scaled data
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=3000)
rfm_tsne = tsne.fit_transform(rfm_scaled)

# Visualization of the clusters using t-SNE
plt.figure(figsize=(10, 8))
colors = ['navy', 'turquoise', 'darkorange', 'red', 'green']

for color, i in zip(colors, range(5)):
    plt.scatter(rfm_tsne[clusters == i, 0], rfm_tsne[clusters == i, 1], color=color, alpha=0.8, lw=2, label=f'Cluster {i + 1}')

plt.title('t-SNE visualization of GMM clusters')
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.show()


# In[70]:


from sklearn.metrics import silhouette_score

# Compute the silhouette score to quantify the quality of clustering
silhouette_avg = silhouette_score(rfm_scaled, clusters)
print(f'Silhouette Score: {silhouette_avg:.2f}')


# In[71]:


from sklearn.metrics import davies_bouldin_score

# Assuming 'clusters' are your GMM cluster assignments and 'rfm_scaled' is your scaled RFM data
db_index = davies_bouldin_score(rfm_scaled, clusters)
print(f'Davies-Bouldin Index: {db_index:.3f}')


# In[72]:


from sklearn.metrics import calinski_harabasz_score

ch_index = calinski_harabasz_score(rfm_scaled, clusters)
print(f'Calinski-Harabasz Index: {ch_index:.3f}')


# In[73]:


import numpy as np

# Calculate the mean distance within clusters (cohesion)
def mean_distance_within_cluster(X, labels, cluster):
    cluster_points = X[labels == cluster]
    centroid = np.mean(cluster_points, axis=0)
    distances = np.linalg.norm(cluster_points - centroid, axis=1)
    return np.mean(distances)

# Calculate the mean distance between cluster centroids (separation)
def mean_distance_between_clusters(centroids):
    n_clusters = len(centroids)
    distances = []
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            distances.append(np.linalg.norm(centroids[i] - centroids[j]))
    return np.mean(distances)

# Compute centroids
centroids = [np.mean(rfm_scaled[clusters == k], axis=0) for k in range(np.max(clusters) + 1)]

# Cohesion and Separation
cohesion = np.mean([mean_distance_within_cluster(rfm_scaled, clusters, k) for k in range(np.max(clusters) + 1)])
separation = mean_distance_between_clusters(centroids)

print(f'Mean Cohesion: {cohesion:.3f}')
print(f'Mean Separation: {separation:.3f}')


# # Hybrid Model (DBSCAN + Kmeans)

# In[40]:


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as pl


# In[86]:


from sklearn.cluster import KMeans, DBSCAN
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Assuming 'rfm_scaled' is your scaled RFM data from earlier steps

# Step 1: Run K-Means to find initial cluster centers
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(rfm_scaled)
kmeans_labels = kmeans.labels_
initial_centers = kmeans.cluster_centers_

# Step 2: Run DBSCAN using the entire dataset (not just the centroids)
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(rfm_scaled)

# Combine the labels, handling the potential merging and noise identification
max_kmeans_label = np.max(kmeans_labels)
for i, label in enumerate(dbscan.labels_):
    if label != -1:  # If DBSCAN found a cluster (not noise)
        kmeans_labels[i] = label + max_kmeans_label + 1  # Offset DBSCAN labels to keep them distinct from K-Means labels

# Assign the adjusted labels back to the original DataFrame
rfm['cluster'] = kmeans_labels

# Calculate quantiles for RFM values
quantiles = {
    'Recency': rfm['Recency'].quantile([0.25, 0.5, 0.75]),
    'Frequency': rfm['Frequency'].quantile([0.25, 0.5, 0.75]),
    'Monetary': rfm['Monetary'].quantile([0.25, 0.5, 0.75])
}

# Function to dynamically assign labels based on calculated quantiles
def assign_cluster_labels(row, quantiles):
    high_spend_threshold = quantiles['Monetary'][0.75]  # 75th percentile of Monetary
    frequent_buyers_threshold = quantiles['Frequency'][0.75]  # 75th percentile of Frequency

    if row['Monetary'] >= high_spend_threshold:
        return 'High Spend Customers'
    elif row['Frequency'] >= frequent_buyers_threshold:
        return 'Frequent Buyers'
    elif row['Recency'] <= quantiles['Recency'][0.25]:
        return 'Recent Customers'
    else:
        return 'Average Customers'

# Apply labels based on the dynamic function
rfm['cluster_label'] = rfm.apply(assign_cluster_labels, axis=1, args=(quantiles,))

# Print results
print("Count of customers in each cluster:")
print(rfm['cluster'].value_counts())
print(rfm[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'cluster', 'cluster_label']].head(30))


# In[87]:


from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


# In[88]:


silhouette_avg = silhouette_score(rfm_scaled, rfm['cluster'])
davies_bouldin = davies_bouldin_score(rfm_scaled, rfm['cluster'])
calinski_harabasz = calinski_harabasz_score(rfm_scaled, rfm['cluster'])

# Print the evaluation scores
print(f"Silhouette Score: {silhouette_avg:.2f}")
print(f"Davies-Bouldin Index: {davies_bouldin:.2f}")
print(f"Calinski-Harabasz Index: {calinski_harabasz:.2f}")


# # RUN THIS CODE SEPERATELY. 

# In[ ]:


import pandas as pd

# Example data: Replace these values with your actual computed metrics
data = {
    'Model': ['SOM', 'GMM', 'K-Means + DBSCAN'],
    'Silhouette Score': [0.55, 0.45, 0.65],  # Replace with your scores
    'Davies-Bouldin Index': [0.75, 0.60, 0.50],  # Replace with your scores
    'Calinski-Harabasz Index': [300, 450, 350]  # Replace with your scores
}

# Create a DataFrame
comparison_df = pd.DataFrame(data)

# Display the DataFrame for analysis
print(comparison_df)


# In[ ]:




