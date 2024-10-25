import numpy as np
from torch_geometric.data import DataLoader
import os
import shutil
import json
import pickle
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pointcloud_processed import *

# Funzione per caricare il dataset o crearlo se non esiste
def load_or_create_dataset(file_path, model_cat, num_samples=num_samples):
    if os.path.exists(file_path):  # Se il dataset è già memorizzato caricalo da pickle
        print('Loading dataset...')
        with open(file_path, 'rb') as file:
            dataset = pickle.load(file)
    else:
        print('Creating new dataset...')
        dataset = PointCloudDataset('sample', model_cat, num_samples=num_samples)
        with open(file_path, 'wb') as file:
            pickle.dump(dataset, file)
    return dataset


def print_dataset_summary(dataset):
    label_counts = {}

    for data in dataset:
        # Accedi all'etichetta dal dizionario
        labels = data['label']
        
        # Se le etichette sono in formato one-hot encoded, trova l'indice dell'etichetta attiva
        if isinstance(labels, torch.Tensor) and labels.dim() > 1:
            label = torch.argmax(labels).item()
        else:
            # Se le etichette sono già sotto forma di indici
            label = labels.item()

        # Aggiorna il conteggio per quella label nel dizionario
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1

    # Stampa il numero di grafi per ogni label
    print("Numero di dati per label:")
    for label, count in label_counts.items():
        print(f"Label {label}: {count} dati")

def plot_fold_metrics(metrics, fold, model):
    num_metrics = len(metrics)
    num_rows = num_metrics // 2 + num_metrics % 2
    fig, axs = plt.subplots(num_rows, 2, figsize=(16, 9 * num_rows))
    
    for i, (metric_values, metric_name) in enumerate(metrics):
        row = i // 2
        col = i % 2
        ax = axs[row, col]
        ax.plot([metric[0] for metric in metric_values], label='Training')
        ax.plot([metric[1] for metric in metric_values], label='Validation')
        if len(metric_values[0]) > 2:
            ax.plot([metric[2] for metric in metric_values], label='Test')
        ax.set_xlabel('Epochs', fontsize=25)
        ax.set_ylabel(metric_name, fontsize=25)
        ax.legend(fontsize=14)

    if num_metrics % 2 != 0:  # Check se c'è un numero dispari di metriche 
        axs[-1, -1].axis('off')  # Ritorna l'ultimo subplot se c'è un numero dispari 

    plt.suptitle(f'Metrics for Fold {fold+1}, Model {model}', fontsize=35)
    plt.tight_layout()

    # Check che la directory 'plots' esista
    os.makedirs('./point_plots', exist_ok=True)
    
    # Aumento i margini e il padding per evitare la sovrapposizione
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.4, wspace=0.4)
    
    # Salvo l'immagine
    plt.savefig(f"./point_plots/model_{model}fold{fold+1}_metrics.png")

def plot_final_curve(losses, accuracies, f1s, aurocs, auprcs, model):
    metrics = [
        (losses, 'Loss'),
        (accuracies, 'Accuracy'),
        (f1s, 'F1-score'),
        (aurocs, 'AUROC'),
        (auprcs, 'AUPRC')
    ]
    
    num_metrics = len(metrics)
    num_rows = num_metrics // 2 + num_metrics % 2
    fig, axs = plt.subplots(num_rows, 2, figsize=(16, 9 * num_rows))
    
    for i, (metric_values, metric_name) in enumerate(metrics):
        row = i // 2
        col = i % 2
        ax = axs[row, col]

        # Estrai i valori finali per ogni fold (training, validation, test)
        train_values = [fold[0] for fold in metric_values]
        val_values = [fold[1] for fold in metric_values]
        test_values = [fold[2] for fold in metric_values]

        # Plotta le curve
        ax.plot(range(1, len(train_values) + 1), train_values, label='Training', marker='o')
        ax.plot(range(1, len(val_values) + 1), val_values, label='Validation', marker='o')
        ax.plot(range(1, len(test_values) + 1), test_values, label='Test', marker='o')

        ax.set_xlabel('Fold', fontsize=20)
        ax.set_ylabel(metric_name, fontsize=20)
        ax.set_title(f'{metric_name} over Folds', fontsize=24)
        ax.legend(fontsize=14)

    if num_metrics % 2 != 0:  # Se c'è un numero dispari di metriche, nasconde l'ultimo subplot
        axs[-1, -1].axis('off')

    plt.suptitle(f'Final Metrics Folds for Model {model}', fontsize=30, y=1.02)
    plt.tight_layout()

    # Margini aggiuntivi per evitare sovrapposizioni
    plt.subplots_adjust(top=0.88, bottom=0.1, left=0.07, right=0.93, hspace=0.4, wspace=0.3)

    # Check della directory 'plots'
    os.makedirs('./point_plots', exist_ok=True)

    # Salva il plot
    plt.savefig(f"./point_plots/final_model_{model}_fold_curves.png")
    plt.show()

    
def print_cv_summary(losses, accuracies, f1s, aurocs, auprcs, model_name):
    
    def print_metric_summary(metric_name, data):
        mean = np.mean(data)
        std = np.std(data)
        print(f" {metric_name}: Mean: {mean:.4f} ± {std:.4f}")
    
    train_losses = [loss[0] for loss in losses]
    val_losses = [loss[1] for loss in losses]
    test_losses = [loss[2] for loss in losses]
    
    train_accuracies = [acc[0] for acc in accuracies]
    val_accuracies = [acc[1] for acc in accuracies]
    test_accuracies = [acc[2] for acc in accuracies]

    train_f1s = [f1[0] for f1 in f1s]
    val_f1s = [f1[1] for f1 in f1s]
    test_f1s = [f1[2] for f1 in f1s]

    train_aurocs = [auroc[0] for auroc in aurocs]
    val_aurocs = [auroc[1] for auroc in aurocs]
    test_aurocs = [auroc[2] for auroc in aurocs]

    train_auprcs = [auprc[0] for auprc in auprcs]
    val_auprcs = [auprc[1] for auprc in auprcs]
    test_auprcs = [auprc[2] for auprc in auprcs]

    print(f'-------- CV SUMMARY -------- MODEL {model_name} --------')
    print_metric_summary('Train Loss', train_losses)
    print_metric_summary("Val Loss", val_losses)
    print_metric_summary("Test Loss", test_losses)
    print_metric_summary("Train Accuracy", train_accuracies)
    print_metric_summary("Val Accuracy", val_accuracies)
    print_metric_summary("Test Accuracy", test_accuracies)
    print_metric_summary("Train F1", train_f1s)
    print_metric_summary("Val F1", val_f1s)
    print_metric_summary("Test F1", test_f1s)
    print_metric_summary("Train AUROC", train_aurocs)
    print_metric_summary("Val AUROC", val_aurocs)
    print_metric_summary("Test AUROC", test_aurocs)
    print_metric_summary("Train AUPRC", train_auprcs)
    print_metric_summary("Val AUPRC", val_auprcs)
    print_metric_summary("Test AUPRC", test_auprcs)
    

def print_confusion_matrix(confusion_matrix, model_name, class_labels):
    print('----------- CUMULATIVE CONFUSION MATRIX -----------')
    print(confusion_matrix)
    print('--------------------------------------------------------')

    # Configuro la dimensione della figura
    plt.figure(figsize=(8, 6))
    
    # Creo una heatmap per la matrice di confusione
    sns.heatmap(confusion_matrix, annot=True, fmt=".1f", cmap="Blues", cbar=False, linewidths=1, linecolor='white', xticklabels=class_labels, yticklabels=class_labels)
    
    # Aggiungo titoli ed etichette
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    
    # Salvo la figura
    filename = f"./point_plots/{model_name}_confusion_matrix.png"
    plt.savefig(filename)
    print(f"Confusion matrix saved as {filename}")


# Funzione per PointNet, DGCNN e TransformerNet
def compute_loaders(dataset, train_indices, val_indices, test_indices):
    # Seleziona il subset di dataset per training, validation, e test
    train_dataset = [dataset[i] for i in train_indices]
    val_dataset = [dataset[i] for i in val_indices]
    test_dataset = [dataset[i] for i in test_indices]

    # Crea i DataLoader per i vari set con batch size 32
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader
