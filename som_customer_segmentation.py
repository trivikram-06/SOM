# Import necessary libraries
import pandas as pd
import numpy as np
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
import pickle

# Load the dataset
dataset = pd.read_csv('Mall_Customers.csv')

# Select relevant features for SOM (Annual Income and Spending Score)
X = dataset[['Annual Income (k$)', 'Spending Score (1-100)']].values

# Data normalization (Scaling values between 0 and 1)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the SOM (10x10 grid, 2 input features)
som = MiniSom(x=10, y=10, input_len=2, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X_scaled)

# Train the SOM with 100 iterations
som.train_random(X_scaled, 100)

# Save the trained SOM model and dataset with cluster assignments
dataset['Cluster'] = [som.winner(x)[0] * 10 + som.winner(x)[1] for x in X_scaled]  # Assign clusters
dataset.to_csv('Mall_Customers_with_Clusters.csv', index=False)

# Save the trained SOM model
with open('som_model.pkl', 'wb') as f:
    pickle.dump(som, f)

print("SOM segmentation completed and data saved.")
