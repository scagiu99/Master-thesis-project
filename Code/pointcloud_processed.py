import numpy as np
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
import torch
import os
import json

#Label che voglio prendere in considerazione
model_cat = ["Knife", "Bag", "Earphone"]
num_samples = 1024

# Rotazione point cloud
def rotate_pointcloud_3d(pointcloud, normals):
    # Genera angoli casuali
    theta_x = np.pi * 2 * np.random.uniform()
    theta_y = np.pi * 2 * np.random.uniform()
    theta_z = np.pi * 2 * np.random.uniform()

    # Matrici di rotazione per ciascun asse
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta_x), -np.sin(theta_x)],
                   [0, np.sin(theta_x), np.cos(theta_x)]])
                   
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                   [0, 1, 0],
                   [-np.sin(theta_y), 0, np.cos(theta_y)]])
                   
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                   [np.sin(theta_z), np.cos(theta_z), 0],
                   [0, 0, 1]])

    # Composizione delle rotazioni
    R = Rz.dot(Ry).dot(Rx)
    
    # Applica la rotazione alla point cloud
    pointcloud = pointcloud.dot(R)
    normals = normals.dot(R)
    return pointcloud, normals

# Traslazione point cloud
def translate_pointcloud(pointcloud, translate_range=(-0.5, 0.5)):
     # Applica traslazione casuale
    translation_factors = np.random.uniform(low=translate_range[0], high=translate_range[1], size=[3])
    
    # Applica la traslazione alla point cloud
    translated_pointcloud = pointcloud + translation_factors
    return translated_pointcloud

# Carica i dati
def load_point_cloud(file_path):
    points = np.loadtxt(file_path)
    points[:, :3], points[:, 3:6] = rotate_pointcloud_3d(points[:, :3], points[:, 3:6])  # Normalizza le coordinate spaziali xyz e le normali
    points[:, :3] = translate_pointcloud(points[:, :3]) # Traslo solo le coordinate spaziali xyz
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
