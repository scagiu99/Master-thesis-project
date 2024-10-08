import numpy as np
import torch
import torch.nn as nn
import os
import pickle
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
from graph_classifier import *
from general_utils import *
from pointcloud_to_graph import *

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU disponibile")
else:
    device = torch.device("cpu")
    print("GPU non disponibile, utilizzo della CPU")


def stratified_cross_validation(dataset, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    train_loaders = []
    val_loaders = []
    test_loaders = []

    for train_val_indices, test_indices in skf.split(range(len(dataset)), [ graph.y.argmax().item() for graph in dataset]):
        train_val_labels = [dataset[idx].y.argmax().item() for idx in train_val_indices]
        train_indices, val_indices = train_test_split(train_val_indices, test_size=0.4, stratify=train_val_labels,random_state=random_state)

        train_loader, val_loader, test_loader = compute_loaders(dataset, train_indices, val_indices, test_indices)
        
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
        test_loaders.append(test_loader)

    return train_loaders, val_loaders, test_loaders

def train(train_loader, val_loader):
    model.train()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    all_preds = []
    all_labels = []

    for data in train_loader:
        data.to(device)
        optimizer.zero_grad()

        x, edge_index, batch, batch_size, y = data.x, data.edge_index, data.batch, data.batch_size, data.y
        out = model(x, edge_index, batch, batch_size)
        y = torch.argmax(y, dim=1) 

        loss = criterion(out, y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        all_preds.append(out.detach().cpu().numpy())
        all_labels.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    acc = accuracy_score(all_labels, np.argmax(all_preds, axis=1))
    f1 = f1_score(all_labels, np.argmax(all_preds, axis=1), average='weighted')
    
   # Apply softmax to convert logits to probabilities for AUROC and AUPRC
    all_preds_prob = torch.softmax(torch.tensor(all_preds), dim=1).numpy()

    # Calculate AUROC and AUPRC
    auroc = roc_auc_score(label_binarize(all_labels, classes=np.arange(all_preds_prob.shape[1])), all_preds_prob, multi_class='ovr')
    auprc = average_precision_score(label_binarize(all_labels, classes=np.arange(all_preds_prob.shape[1])), all_preds_prob, average='weighted')

    loss = total_loss / len(train_loader.dataset)

    val_loss, val_f1, val_auroc, val_auprc, val_acc, _ = test(val_loader)

    return loss, f1, auroc, auprc, acc, val_loss, val_f1, val_auroc, val_auprc, val_acc



@torch.no_grad()
def test(test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    all_preds = []
    all_labels = []
    for data in test_loader:
        
        data = data.to(device)

        x, edge_index, batch, batch_size, y = data.x, data.edge_index, data.batch, data.batch_size, data.y
        out = model(x, edge_index, batch, batch_size)
        y = torch.argmax(y, dim=1) 

        loss = criterion(out, y)

        total_loss += loss.item()
        
        all_preds.append(out.detach().cpu().numpy())
        all_labels.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    acc = accuracy_score(all_labels, np.argmax(all_preds, axis=1))
    f1 = f1_score(all_labels, np.argmax(all_preds, axis=1), average='weighted')

  # Apply softmax to convert logits to probabilities for AUROC and AUPRC
    all_preds_prob = torch.softmax(torch.tensor(all_preds), dim=1).numpy()

    # Calculate AUROC and AUPRC
    auroc = roc_auc_score(label_binarize(all_labels, classes=np.arange(all_preds_prob.shape[1])), all_preds_prob, multi_class='ovr')
    auprc = average_precision_score(label_binarize(all_labels, classes=np.arange(all_preds_prob.shape[1])), all_preds_prob, average='weighted')

    loss = total_loss / len(test_loader.dataset)
    cm = confusion_matrix(all_labels, np.argmax(all_preds, axis=1))
    cr = classification_report(all_labels, np.argmax(all_preds, axis=1))

    return loss, f1, auroc, auprc, acc, cm, cr


#################################################################

file_path = 'graph_dataset.pkl'
sbj_number = 0

print('Loading dataset...')

if os.path.exists(file_path): # Se il dataset è già memorizzato caricalo da pickle
    with open(file_path, 'rb') as file:
        dataset = pickle.load(file)
else:
   # organize_dataset()
    dataset = create_dataset(file_path)

# Dataset Labels Summary
print_dataset_summary(dataset)

n_splits = 5
train_loaders, val_loaders, test_loaders = stratified_cross_validation(dataset=dataset, n_splits = n_splits)

# Creao un modello GNN
num_features = dataset[0].num_node_features
print("Numero di feature: ",num_features)
num_classes = len(model_cat)

models = [ v1M1(num_features=num_features, num_classes=num_classes),
            M1(num_features=num_features, num_classes=num_classes), 
            oldM1(num_features=num_features, hidden_channels=64, num_classes=num_classes),
            M3(num_features=num_features, num_classes=num_classes),
            oldM3(num_features=num_features, hidden_channels=64, num_classes=num_classes),
            v1M3(num_features=num_features, num_classes=num_classes),
            GAT(num_features=num_features, hidden_channels=64, num_classes=num_classes),
            GAT2(num_features=num_features, hidden_channels=64, num_classes=num_classes)]


for model in models:

    model.to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    num_epochs = 250

    #scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-4)

    # Inizializzo la somma cumulativa delle matrici di confusione
    cumulative_confusion_matrix = np.zeros((num_classes, num_classes))
    final_losses, final_accuracies, final_f1s, final_aurocs, final_auprcs = [], [], [], [], []

    all_true_labels = []
    all_pred_labels = []

    for fold, (train_loader, val_loader, test_loader) in enumerate(zip(train_loaders, val_loaders, test_loaders)):
        
        fold_losses, fold_accuracies, fold_f1s, fold_aurocs, fold_auprcs = [], [], [], [], []

        # Early stopping parameters
        best_val_loss = float('inf')
        counter = 0
        early_stopping_patience = 15

        print(f"Fold {fold + 1}/{n_splits}")
        for epoch in tqdm(range(1, num_epochs + 1)):
            train_loss, train_f1, train_auroc, train_auprc, train_acc, val_loss, val_f1, val_auroc, val_auprc, val_acc = train(train_loader, val_loader)
            test_loss, test_f1, test_auroc, test_auprc, test_acc, test_confusion_matrix, test_classification_report  = test(test_loader)
            
           # scheduler.step()

            print(f"""\nEpoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f},
                    Train Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}, Test Accuracy: {test_acc:.4f}, 
                    Train F1: {train_f1:.4f}, Validation F1: {val_f1:.4f}, Test F1: {test_f1:.4f}, 
                    Train AUROC: {train_auroc:.4f}, Validation AUROC: {val_auroc:.4f}, Test AUROC: {test_auroc:.4f}, 
                    Train AUPRC: {train_auprc:.4f}, Validation AUPRC: {val_auprc:.4f}, Test AUPRC: {test_auprc:.4f}""")

            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                counter = 0
            else:
                counter += 1
                if counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            fold_losses.append((train_loss, val_loss, test_loss))
            fold_accuracies.append((train_acc, val_acc, test_acc))
            fold_f1s.append((train_f1, val_f1, test_f1))
            fold_aurocs.append((train_auroc, val_auroc, test_auroc))
            fold_auprcs.append((train_auprc, val_auprc, test_auprc))

            fold_metrics = [ (fold_losses, 'Loss'), (fold_accuracies, 'Accuracy'), (fold_f1s, 'F1-score'), (fold_aurocs, 'AUROC'), (fold_auprcs, 'AUPRC') ]

        plot_fold_metrics( fold_metrics, fold, model.__class__.__name__ )

        # Memorizzo le ultime metriche calcolate
        final_losses.append(fold_losses[-1])
        final_accuracies.append(fold_accuracies[-1])
        final_f1s.append(fold_f1s[-1])
        final_aurocs.append(fold_aurocs[-1])
        final_auprcs.append(fold_auprcs[-1])
        cumulative_confusion_matrix += test_confusion_matrix
        classification_report += test_classification_report

    # Plotting delle metriche per tutti i modelli
    print_cv_summary( final_losses, final_accuracies, final_f1s, final_aurocs, final_auprcs, model.__class__.__name__ )
    print_confusion_matrix(cumulative_confusion_matrix, model.__class__.__name__, model_cat)