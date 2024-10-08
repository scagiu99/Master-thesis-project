import numpy as np
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
import torch
from torch_geometric.utils import from_networkx
import os
import json
import pickle
import torch.nn.functional as F
import networkx as nx
from sklearn.decomposition import PCA

#Label che voglio prendere in considerazione
model_cat = ["Vase","Lamp", "Knife", "Bottle", "Laptop", "Faucet", "Chair", "Table"]
num_samples = 512

# Rotazione point cloud
def rotate_pointcloud_3d(pointcloud):
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
    return pointcloud

# Traslazione point cloud
def translate_pointcloud(pointcloud, translate_range=(-0.5, 0.5)):
     # Applica traslazione casuale
    translation_factors = np.random.uniform(low=translate_range[0], high=translate_range[1], size=[3])
    
    # Applica la traslazione alla point cloud
    translated_pointcloud = pointcloud + translation_factors
    return translated_pointcloud

def data_augmentation(pointcloud):
    # Rotazione casuale
    point_cloud_rotated = rotate_pointcloud_3d(pointcloud)
    
    # Traslazione casuale
    augmentated_pointcloud = translate_pointcloud(point_cloud_rotated)

    return augmentated_pointcloud

def translate_to_origin(point_cloud):
    # Calcola il centroide
    centroid = np.mean(point_cloud, axis=0)
    
    # Sottrai il centroide da ogni punto
    point_cloud_centered = point_cloud - centroid
    
    return point_cloud_centered

# PCA
def apply_pca(point_cloud):
    pca = PCA()
    point_cloud_pca = pca.fit_transform(point_cloud)
    return point_cloud_pca

# Normalizzazione
def normalize_point_cloud(point_cloud):
    # Traslazione verso l'origine
    point_cloud_centered = translate_to_origin(point_cloud)
    
    # Applicazione della PCA
    point_cloud_normalized = apply_pca(point_cloud_centered)
    
    return point_cloud_normalized

# Carica i dati
def load_point_cloud(file_path):
    points = np.loadtxt(file_path)
   # points[:, :3] = data_augmentation(points[:, :3])  # Data Augmentation
   # points[:, :3] = normalize_point_cloud(points[:, :3]) # Normalizzo
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

# Funzione per creare patch locali attorno ai punti campionati
def create_local_patches(points, normals, sampled_points, k=30):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree', n_jobs=-1).fit(points)
    _, indices = nbrs.kneighbors(sampled_points)
    patches_points = points[indices]
    patches_normals = normals[indices]
    return patches_points, patches_normals

# Funzione per calcolare il FPFH su una point cloud Open3D
def compute_fpfh(point_cloud, voxel_size=0.05):
    # Definisci i raggi per la stima delle normali e il calcolo delle feature
    radius_normal = voxel_size * 2
    radius_feature = voxel_size * 5

    # Controlla se la point cloud ha normali; se non le ha, stima le normali
    if not point_cloud.has_normals():
        point_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius_normal, max_nn=30))

    # Calcolo del FPFH
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        point_cloud,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature, max_nn=100))

    return np.array(fpfh.data).T

# Converte ogni patch in una point cloud di Open3D e calcola il FPFH
def process_patches(patches_points, patches_normals):
    fpfh_descriptors = []
    
    for patch_points, patch_normals in zip(patches_points, patches_normals):
        # Crea un oggetto PointCloud per ogni patch
        patch_point_cloud = o3d.geometry.PointCloud()
        patch_point_cloud.points = o3d.utility.Vector3dVector(patch_points[:, :3])  # Usa solo le coordinate xyz

        # Aggiungi normali alla point cloud
        patch_point_cloud.normals = o3d.utility.Vector3dVector(patch_normals)

        # Calcola il FPFH per la patch corrente
        fpfh = compute_fpfh(patch_point_cloud)
        
        # Aggiungi i descriptor FPFH alla lista
        fpfh_descriptors.append(fpfh[0]) # Prendi solo il descrittore del punto centrale della patch
    
    # Concatena tutti i descriptor FPFH in un singolo array numpy
    fpfh_all_patches = np.vstack(fpfh_descriptors)
    return fpfh_all_patches

# Funzione per costruire un grafo dai punti
def build_graph(points, normals, colors, fpfh_descriptors, k):
    G = nx.Graph()
    
    # Concatenate diverse feature (coordinate, normali, colore, FPFH)
    node_features = np.concatenate((points, normals, colors, fpfh_descriptors), axis=1)

    # Aggiunge i nodi al grafo con le feature concatenate
    for i in range(len(points)):
        G.add_node(i, x=node_features[i])
    
    # Usa NearestNeighbors per trovare i k-nearest neighbors per ogni punto
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(points)  # k+1 perché il punto stesso è incluso
    distances, indices = nbrs.kneighbors(points)

    # Aggiungi archi per i k vicini più vicini di ogni punto
    for i in range(len(points)):
        for j in range(1, k + 1):  # Inizia da 1 per evitare di considerare il punto stesso come vicino
            neighbor_idx = indices[i, j]
            dist = distances[i, j]
            G.add_edge(i, neighbor_idx, weight=dist)

    return G

# Funzione per creare il dataset
def create_dataset(output_file):
    dataset = []
    
    # Itero su tutte le cartelle nella cartella principale
    for root, directories, files in os.walk("sample"):

        if "sample-points-all-pts-nor-rgba-10000.txt" in files:
            percorso_file = os.path.join(root, "sample-points-all-pts-nor-rgba-10000.txt")
            
            # Leggo il file meta.json
            percorso_meta = os.path.join(root, "meta.json")
            with open(percorso_meta, 'r') as meta_file:
                meta_data = json.load(meta_file)
            
            # Ottiengo l'etichetta corrispondente a model_cat
            model_cat_value = meta_data.get("model_cat")
            if model_cat_value not in model_cat:
                print(f"Categoria non riconosciuta: {model_cat_value}")
                continue
            
            label = model_cat.index(model_cat_value)  # Ottieni l'indice della categoria

            print(f"File: {percorso_file} con etichetta {label} ({model_cat_value})")
            point_cloud_data = load_point_cloud(percorso_file)

            points = point_cloud_data[:, :3] # Coordinate spaziali (x, y, z)
            normals = point_cloud_data[:, 3:6] # Coordinate normali (x, y, z)
            colors = point_cloud_data[:, 6:9]  
        
            # Eseguo il campionamento di punti
            # ho point cloud da 10mila punti quindi il campione va scelto tra 1000 e 5000 campioni, 
            #quindi questo valore per avere un valore medio che preservi i dettagli e bilanci la riduzione
            sampled_points, indices = farthest_point_sampling(points, num_samples)

            # Filtra le normali e i colori dei punti campionati
            sampled_normals = normals[indices]
            sampled_colors = colors[indices]

            # Creo patch locali attorno ai punti campionati
            patches_points, patches_normals = create_local_patches(points, normals, sampled_points, k=30)

            # Calcolo il FPFH per ciascuna patch
            fpfh_descriptors = process_patches(patches_points, patches_normals)
            #print(fpfh_descriptors)

            k = 10
            # Costruisco il grafo a partire dai punti campionati
            graph = build_graph(sampled_points, sampled_normals, sampled_colors, fpfh_descriptors, k=k)
            
            # Converto il grafo in torch_geometric
            graph_torch_geometric = from_networkx(graph)
            
            # Codifica one-hot dell'etichetta
            y_one_hot = F.one_hot(torch.tensor([label]), num_classes=len(model_cat)).to(torch.float32)
            
            # Assegna l'etichetta one-hot al grafo
            graph_torch_geometric.y = y_one_hot
            #print(graph_torch_geometric.y)
           
            # Assegna le feature dei nodi
            graph_torch_geometric.x = torch.tensor(np.concatenate((sampled_normals, sampled_colors, fpfh_descriptors), axis=1), dtype=torch.float32)
            #print(graph_torch_geometric.x)
            
            # Aggiungo il grafo al dataset
            dataset.append(graph_torch_geometric)
    
    # Salvo il dataset come file pickle
    with open(output_file, 'wb') as file:
        pickle.dump(dataset, file)
    
    return dataset

#dataset = create_dataset("graph_dataset.pkl")
