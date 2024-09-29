import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Mall_Customers_with_Clusters.csv')

cluster_counts = dataset['Cluster'].value_counts().sort_index()

explode = [0.1] * len(cluster_counts) 

plt.figure(figsize=(10, 10))
plt.pie(cluster_counts, 
        labels=[f'Cluster {i}' for i in cluster_counts.index], 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=plt.cm.Paired(range(len(cluster_counts))), 
        explode=explode, 
        textprops={'fontsize': 12},  
        labeldistance=1.2)  
plt.title('Customer Distribution in SOM Clusters', fontsize=16)
plt.axis('equal')  
plt.show()
