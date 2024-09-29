# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset with cluster assignments
dataset = pd.read_csv('Mall_Customers_with_Clusters.csv')

# Getting the cluster counts and sorting them
cluster_counts = dataset['Cluster'].value_counts().sort_index()

# Define explode to create space between the slices
explode = [0.1] * len(cluster_counts)  # Explode each slice a bit

# Plot Pie Chart of Customer Clusters
plt.figure(figsize=(10, 10))
plt.pie(cluster_counts, 
        labels=[f'Cluster {i}' for i in cluster_counts.index], 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=plt.cm.Paired(range(len(cluster_counts))),  # Better color scheme
        explode=explode,  # Space between slices
        textprops={'fontsize': 12},  # Increase label font size
        labeldistance=1.2)  # Move labels further from the center

plt.title('Customer Distribution in SOM Clusters', fontsize=16)
plt.axis('equal')  # Equal aspect ratio ensures that pie chart is a circle.
plt.show()
