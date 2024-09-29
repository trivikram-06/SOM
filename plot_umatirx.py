import matplotlib.pyplot as plt
import pickle
from minisom import MiniSom

with open('som_model.pkl', 'rb') as f:
    som = pickle.load(f)

plt.figure(figsize=(10, 7))
plt.pcolor(som.distance_map().T, cmap='coolwarm')  # Distance map as background
plt.colorbar(label='Distance')
plt.title('SOM U-Matrix')
plt.show()
