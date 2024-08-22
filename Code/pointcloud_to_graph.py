import numpy as np
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
import torch
from torch_geometric.utils import from_networkx
import os
import json
import shutil
import pickle
import torch.nn.functional as F
import networkx as nx

#Label che voglio prendere in considerazione
model_cat = ["Knife", "Bag", "Earphone", "Laptop", "Hat"]

def organize_dataset():
    # Specifica il percorso della cartella principale
    cartella_principale = "Data_sample"
    # Specifica la destinazione dove copiare le cartelle filtrate
    cartella_destinazione = "Data"

    # Crea la cartella di destinazione se non esiste
    os.makedirs(cartella_destinazione, exist_ok=True)

    # Itera su tutte le cartelle e sottocartelle nella cartella principale
    for root, directories, files in os.walk(cartella_principale):
            # Verifica se "meta.json" è presente nella cartella principale dell'oggetto
            if "meta.json" in files and "point_sample" in directories:
                percorso_meta = os.path.join(root, "meta.json")
                
                # Legge il file meta.json
                with open(percorso_meta, 'r') as meta_file:
                    meta_data = json.load(meta_file)
                
                # Verifica se il model_cat è tra quelli desiderati
                if meta_data.get("model_cat") in model_cat:
                    # Crea la cartella per l'oggetto nella cartella di destinazione
                    nome_cartella_oggetto = os.path.basename(root)
                    destinazione_oggetto = os.path.join(cartella_destinazione, nome_cartella_oggetto)
                    os.makedirs(destinazione_oggetto, exist_ok=True)
                    
                    # Copia il file meta.json nella nuova cartella
                    shutil.copy(percorso_meta, destinazione_oggetto)
                    print(f"Copiato {percorso_meta} in {destinazione_oggetto}")
                    
                    # Cerca il file sample-points-all-pts-nor-rgba-10000.txt nella sottocartella point_sample
                    percorso_point_sample = os.path.join(root, "point_sample")
                    file_sample_points = "sample-points-all-pts-nor-rgba-10000.txt"
                    percorso_sample = os.path.join(percorso_point_sample, file_sample_points)
                    
                    if os.path.exists(percorso_sample):
                        shutil.copy(percorso_sample, destinazione_oggetto)
                        print(f"Copiato {percorso_sample} in {destinazione_oggetto}")

# Carica i dati
def load_point_cloud(file_path):
    return np.loadtxt(file_path)

# Funzione per il campionamento dei punti più lontani
def farthest_point_sampling(points, num_samples):
    N, D = points.shape
    centroids = np.zeros((num_samples,))
    distances = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    
    for i in range(num_samples):
        centroids[i] = farthest
        centroid = points[farthest, :]
        dist = np.sum((points - centroid) ** 2, axis=1)
        mask = dist < distances
        distances[mask] = dist[mask]
        farthest = np.argmax(distances)
        
    sampled_points = points[centroids.astype(int)]
    return sampled_points

# Funzione per creare patch locali attorno ai punti campionati
def create_local_patches(points, sampled_points, k=30):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points)
    _, indices = nbrs.kneighbors(sampled_points)
    patches = [points[idx] for idx in indices]
    return patches

# Funzione per calcolare il FPFH su una point cloud Open3D
def compute_fpfh(point_cloud, voxel_size=0.05):
    radius_normal = voxel_size * 2
    radius_feature = voxel_size * 5
    
    # Stima delle normali (se non presenti)
    if not point_cloud.has_normals():
        point_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius_normal, max_nn=30))
    
    # Calcolo del FPFH
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        point_cloud,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature, max_nn=100))
    
    return fpfh

# Converte ogni patch in una point cloud di Open3D e calcola il FPFH
def process_patches(patches):
    fpfh_descriptors = []
    for patch in patches:
        # Crea un oggetto PointCloud per ogni patch
        patch_point_cloud = o3d.geometry.PointCloud()
        patch_point_cloud.points = o3d.utility.Vector3dVector(patch)
        
        # Calcola il FPFH per la patch corrente
        fpfh = compute_fpfh(patch_point_cloud)
        
        # Converte FPFH in un array numpy e aggiungilo alla lista dei descriptor
        fpfh_numpy = np.array(fpfh.data).T  # Trasponi per ottenere (N_points, N_features)
        fpfh_descriptors.append(fpfh_numpy)
    
    # Concatena tutti i descriptor FPFH in un singolo array numpy
    fpfh_all_patches = np.vstack(fpfh_descriptors)
    return fpfh_all_patches

# Funzione per costruire un grafo dai punti
def build_graph(points, fpfh_descriptors, radius):
    G = nx.Graph()
    
    # Aggiunge i nodi al grafo con le feature FPFH
    for i, (point, fpfh) in enumerate(zip(points, fpfh_descriptors)):
        # Aggiunge il nodo con la feature FPFH
        G.add_node(i, x=point, fpfh=fpfh)
    
    # Aggiunge gli archi basati sulla distanza tra i punti
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = np.linalg.norm(points[i] - points[j])
            if dist < radius:
                G.add_edge(i, j, weight=dist)  
    
    return G


# Funzione per creare il dataset
def create_dataset(output_file):
    dataset = []

    # Itero su tutte le cartelle nella cartella principale
    for root, directories, files in os.walk("Data"):
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

            print(f"Utilizzando il file: {percorso_file} con etichetta {label} ({model_cat_value})")
            point_cloud_data = load_point_cloud(percorso_file)

            points = point_cloud_data[:, :3]  # Coordinate spaziali (x, y, z)
        
            # Eseguo il campionamento di punti
            num_samples = 2048 # ho point cloud da 10mila punti quindi il campione va scelto tra 1000 e 5000 campioni, 
                                #quindi questo valore per avere un valore medio che preservi i dettagli e bilanci la riduzione
            sampled_points = farthest_point_sampling(points, num_samples)

            # Creo patch locali attorno ai punti campionati
            patches = create_local_patches(points, sampled_points, k=30)

            # Calcolo il FPFH per ciascuna patch
            fpfh_descriptors = process_patches(patches)

            # Costruisco il grafo a partire dai punti campionati
            graph = build_graph(sampled_points, fpfh_descriptors, radius=0.1)
            
            # Converto il grafo in torch_geometric
            graph_torch_geometric = from_networkx(graph)
            
            # Codifica one-hot dell'etichetta
            y_one_hot = F.one_hot(torch.tensor([label]), num_classes=len(model_cat)).to(torch.float32)
            
            # Assegna l'etichetta one-hot al grafo
            graph_torch_geometric.y = y_one_hot
            #print(graph_torch_geometric.y)
            # Assegna le coordinate spaziali campionate a x
            graph_torch_geometric.x = torch.tensor(fpfh_descriptors, dtype=torch.float32)
            #print(graph_torch_geometric.x)
            
            # Aggiungo il grafo al dataset
            dataset.append(graph_torch_geometric)
    
    # Salvo il dataset come file pickle
    with open(output_file, 'wb') as file:
        pickle.dump(dataset, file)
    
    return dataset

#dataset = create_dataset("graph_dataset.pkl")