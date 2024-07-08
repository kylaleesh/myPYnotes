# ML models - Unsupervised:
## Clustering
* Partitioning Based Clustering Techniques
    * We need to predefine the number of clusters we are going to form from the data. K stands for the number of clusters.
        1. K-means Clustering
        2. K-Model Clustering
        3. K-Mediod Clustering
* Hierachical Clustering Technique_
    1. Agglomerative Hierachical Clustering --> Bottom-up approach
    2. Divisive Hierachical Clustering --> Top-down approach
You do not need to predefine the number of clusters.
* Density Based Clustering Technique
    * You do not need to predefine the number of clusters. It comes up with the clusters based on density of the data points within the data. Eg. DBSCAN

Difference between Partitioning Algorithm, Hierachical Clustering  and Density-based Clustering Algorithms
|Technique| Difference|
|---|---|
|Partitioning algos|the number of clusters must be known. These algorithms are fasters|
|Hierachical algos|the number of clusters might not be known. Comparatively slower and computationally expensive|
|Density algos|No need to specify clusters. Good with outliers in your data|

### Data Preprocessing 
Since we are using distance based metrics to calculate close proximity and similar datapoints, we must use numerical data.
If we have categorical data, we should convert them to numeric data by performing encoding.
* Perform encoding for categorical data
* Perform feature scaling (since we are looking at distance...the distance/scale is very important)
* Outlier treatment (if working with partition based algo) to ensure that your clusters' prediction aren't skewed due to outliers

#### K-means 
```py
from sklearn.cluster import KMeans 
'''
n_clusters: The number of clusters that we are going to create
n_init: 10 by default
init : random or kmeans++ (default)
'''
# Fit Model
kmeans = KMeans(n_clusters = 4,
               n_init = 1, # only 1 round of random selected centriods
               init = 'random', # can also use 'k-means++'
               random_state=170,
               verbose = False).fit(X)

# Retrieve the prediction, (4 clusters and 4 centriods)
prediction = display(kmeans.labels_)
clusters4 = display(kmeans.cluster_centers_)

# Plotting the clusters, with white crosses as centriods
plt.scatter(X[:,0], X[:,1], c=kmeans.labels_)

plt.scatter(kmeans.cluster_centers_[:,0], 
            kmeans.cluster_centers_[:,1], 
            c='w', marker='x', linewidths=2)
```
##### Choosing the number of K 
###### using Elbow Plot - SSE
Sum of Squared Errors or Inertia: n_clusters is inversely proportionaly to SSE/inertia. SO WE JUST LOOK AT THE BEND
```py
sse = {}
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(X)
    #print(data["clusters"])
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()),color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()
```
###### using Silhouette Analysis
Silhouette Coefficient > 0.6 is good
```py
# List of number of clusters
import numpy as np
range_n_clusters = [2, 3, 4, 5, 6]

# For each number of clusters, perform Silhouette analysis and visualize the results.
for n_clusters in range_n_clusters:
    
    # Perform k-means.
    kmeans = KMeans(n_clusters=n_clusters, random_state=10)
    y_pred = kmeans.fit_predict(X)
    
    # Compute the Silhouette Coefficient for each sample.
    s = metrics.silhouette_samples(X, y_pred)
    # Compute the mean Silhouette Coefficient of all data points.
    s_mean = metrics.silhouette_score(X, y_pred) 
    
    # For plot configuration -----------------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    
    # Configure plot.
    plt.suptitle('Silhouette analysis for K-Means clustering with n_clusters: {}'.format(n_clusters),
                 fontsize=14, fontweight='bold')
    
    # Configure 1st subplot.
    ax1.set_title('Silhouette Coefficient for each sample')
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.set_xlim([-1, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    
    ax2.set_title('Mean Silhouette score: {}'.format(s_mean))
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")
    
    
# Plot Silhouette Coefficient for each sample
    y_lower = 10
    for i in range(n_clusters):
        ith_s = s[y_pred == i]  #s score for all the cluster 2,3,4,5,6
        ith_s.sort()
        size_cluster_i = ith_s.shape[0]
        y_upper = y_lower + size_cluster_i
        cmap = cm.get_cmap("Spectral")
        color = cmap(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_s,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    # Plot the mean Silhouette Coefficient using red vertical dash line.
    ax1.axvline(x=s_mean, color="red", linestyle="--")
    
    # Plot the predictions
    colors = cmap(y_pred.astype(float) / n_clusters)
    ax2.scatter(X[:,0], X[:,1], c=colors)

```
