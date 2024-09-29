import pandas as pd
import numpy as np
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
import pickle

dataset = pd.read_csv('Mall_Customers.csv')

X = dataset[['Annual Income (k$)', 'Spending Score (1-100)']].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

som = MiniSom(x=10, y=10, input_len=2, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X_scaled)

som.train_random(X_scaled, 100)

dataset['Cluster'] = [som.winner(x)[0] * 10 + som.winner(x)[1] for x in X_scaled]  # Assign clusters
dataset.to_csv('Mall_Customers_with_Clusters.csv', index=False)

with open('som_model.pkl', 'wb') as f:
    pickle.dump(som, f)

print("SOM segmentation completed and data saved.")
