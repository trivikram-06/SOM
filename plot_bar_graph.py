# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset with cluster assignments
dataset = pd.read_csv('Mall_Customers_with_Clusters.csv')

# Calculate average spending score by cluster
avg_spending_by_income = dataset.groupby('Cluster')['Spending Score (1-100)'].mean()

# Plot Bar Graph for Spending Score by Income Bracket
avg_spending_by_income.plot(kind='bar', figsize=(10, 6))
plt.title('Average Spending Score by Income Cluster')
plt.ylabel('Average Spending Score')
plt.xlabel('Income Cluster (Grid Position)')
plt.xticks(rotation=0)
plt.show()
