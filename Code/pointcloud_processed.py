import numpy as np
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
import torch
import os
import json

#Label che voglio prendere in considerazione
model_cat = ["Vase", "Lamp","Knife", "Bottle", "Laptop","Faucet", "Chair", "Table"]
num_samples = 1024


#Traslazione
def translate_to_origin(point_cloud):
    # Calcola il centroide
    centroid = np.mean(point_cloud, axis=0)
    
    # Sottrai il centroide da ogni punto
    point_cloud_centered = point_cloud - centroid
    
    return point_cloud_centered

#Scalatura
def scale_to_unit_sphere(point_cloud):
    # Calcola la distanza massima (norma euclidea massima)
    max_distance = np.max(np.linalg.norm(point_cloud, axis=1))
    
    # Scala tutti i punti in modo che la distanza massima sia 1
    point_cloud_normalized = point_cloud / max_distance
    
    return point_cloud_normalized

#Normalizzazione
def normalize_point_cloud(point_cloud):
    # Step 1: Traslazione verso l'origine
    point_cloud_centered = translate_to_origin(point_cloud)
    
    # Step 2: Scalatura
    point_cloud_normalized = scale_to_unit_sphere(point_cloud_centered)
    
    return point_cloud_normalized

# Carica i dati
def load_point_cloud(file_path):
    points = np.loadtxt(file_path)
    points[:, :3] = normalize_point_cloud(points[:, :3]) # Traslo solo le coordinate spaziali xyz
    return points

# Funzione per il campionamento dei punti più lontani
def farthest_point_sampling(points, num_samples):
    N, D = points.shape
    centroids = np.zeros((num_samples,), dtype=int)
    distances = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    
    for i in range(num_samples):
        centroids[i] = farthest
        centroid = points[farthest, :]
        dist = np.sum((points - centroid) ** 2, axis=1)
        # Aggiorna le distanze minime
        distances = np.minimum(distances, dist)
        farthest = np.argmax(distances)
        
    sampled_points = points[centroids.astype(int)]
    return sampled_points, centroids

class PointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, model_cat, num_samples=num_samples):
        self.file_path = file_path
        self.model_cat = model_cat
        self.num_samples = num_samples
        self.data = self.load_dataset()

    def load_dataset(self):
        dataset = []
        for root, directories, files in os.walk("sample"):
            if "sample-points-all-pts-nor-rgba-10000.txt" in files:
                percorso_file = os.path.join(root, "sample-points-all-pts-nor-rgba-10000.txt")
                percorso_meta = os.path.join(root, "meta.json")
                with open(percorso_meta, 'r') as meta_file:
                    meta_data = json.load(meta_file)
                model_cat_value = meta_data.get("model_cat")
                if model_cat_value not in self.model_cat:
                    print(f"Categoria non riconosciuta: {model_cat_value}")
                    continue
                label = self.model_cat.index(model_cat_value)
                print(f"File: {percorso_file} con etichetta {label} ({model_cat_value})")
                point_cloud_data = load_point_cloud(percorso_file)
                points = point_cloud_data[:, :3]
                sampled_points, _ = farthest_point_sampling(points, self.num_samples)
                points_tensor = torch.tensor(sampled_points, dtype=torch.float32).transpose(0, 1)
                dataset.append({'points': points_tensor, 'label': torch.tensor(label, dtype=torch.long)})
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
