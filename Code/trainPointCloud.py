import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from pointcloud_classifier import *
from pointcloud_utils import *
from pointcloud_processed import *

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

    # Estrai le etichette dal dataset
    labels = [data['label'].item() if isinstance(data['label'], torch.Tensor) else data['label'] for data in dataset]

    for train_val_indices, test_indices in skf.split(range(len(dataset)), labels):
        train_val_labels = [dataset[idx]['label'].item() if isinstance(dataset[idx]['label'], torch.Tensor) else dataset[idx]['label'] for idx in train_val_indices]
        
        train_indices, val_indices = train_test_split(train_val_indices, test_size=0.4, stratify=train_val_labels, random_state=random_state)

        train_loader, val_loader, test_loader = compute_loaders(dataset, train_indices, val_indices, test_indices)
        
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
        test_loaders.append(test_loader)

    return train_loaders, val_loaders, test_loaders


def train(model, train_loader, val_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in train_loader:
        points, labels = batch['points'].to(device), batch['label'].to(device)
        optimizer.zero_grad()
        batch_size = points.size()[0]
        outputs = model(points)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        all_preds.append(outputs.detach().cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    acc = accuracy_score(all_labels, np.argmax(all_preds, axis=1))
    f1 = f1_score(all_labels, np.argmax(all_preds, axis=1), average='weighted')

    # One-hot encode le etichette per il calcolo dell'AUROC
    all_labels_one_hot = label_binarize(all_labels, classes=np.arange(all_preds.shape[1]))
    
    auroc = roc_auc_score(all_labels_one_hot, all_preds, multi_class='ovr')
    auprc = average_precision_score(all_labels_one_hot, all_preds, average='weighted')
    loss = total_loss / len(train_loader.dataset)

    val_loss, val_f1, val_auroc, val_auprc, val_acc, _ = test(model, val_loader, criterion, device)

    return loss, f1, auroc, auprc, acc, val_loss, val_f1, val_auroc, val_auprc, val_acc

@torch.no_grad()
def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in test_loader:
        points, labels = batch['points'].to(device), batch['label'].to(device)
        outputs = model(points)
        loss = criterion(outputs, labels)
        batch_size = points.size()[0]
        total_loss += loss.item()

        all_preds.append(outputs.detach().cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    acc = accuracy_score(all_labels, np.argmax(all_preds, axis=1))
    f1 = f1_score(all_labels, np.argmax(all_preds, axis=1), average='weighted')

    # One-hot encode le etichette per il calcolo dell'AUROC
    all_labels_one_hot = label_binarize(all_labels, classes=np.arange(all_preds.shape[1]))
    
    auroc = roc_auc_score(all_labels_one_hot, all_preds, multi_class='ovr')
    auprc = average_precision_score(all_labels_one_hot, all_preds, average='weighted')
    loss = total_loss / len(test_loader.dataset)
    cm = confusion_matrix(all_labels, np.argmax(all_preds, axis=1))

    return loss, f1, auroc, auprc, acc, cm


#################################################################

file_path = 'pointcloud_dataset.pkl'

# Carica o crea il dataset
dataset = load_or_create_dataset(file_path, model_cat, num_samples=num_samples)

# Dataset Labels Summary
print_dataset_summary(dataset)

n_splits = 5
train_loaders, val_loaders, test_loaders = stratified_cross_validation(dataset=dataset, n_splits=n_splits)

num_classes = len(model_cat)
models = [PointNet(embedding_dimension=num_samples, num_classes=num_classes),
            DGCNN(embedding_dimension=num_samples, num_classes=num_classes)]

for model in models:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 250
    cumulative_confusion_matrix = np.zeros((num_classes, num_classes))
    final_losses, final_accuracies, final_f1s, final_aurocs, final_auprcs = [], [], [], [], []

    for fold, (train_loader, val_loader, test_loader) in enumerate(zip(train_loaders, val_loaders, test_loaders)):
        fold_losses, fold_accuracies, fold_f1s, fold_aurocs, fold_auprcs = [], [], [], [], []
        best_val_loss = float('inf')
        counter = 0
        early_stopping_patience = 15

        print(f"Fold {fold + 1}/{n_splits}")
        for epoch in tqdm(range(1, num_epochs + 1)):
            train_loss, train_f1, train_auroc, train_auprc, train_acc, val_loss, val_f1, val_auroc, val_auprc, val_acc = train(model, train_loader, val_loader, optimizer, criterion, device)
            test_loss, test_f1, test_auroc, test_auprc, test_acc, test_confusion_matrix  = test(model, test_loader, criterion, device)

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

        plot_fold_metrics([(fold_losses, 'Loss'), (fold_accuracies, 'Accuracy'), (fold_f1s, 'F1-score'), (fold_aurocs, 'AUROC'), (fold_auprcs, 'AUPRC')], fold, model.__class__.__name__)

        final_losses.append(fold_losses[-1])
        final_accuracies.append(fold_accuracies[-1])
        final_f1s.append(fold_f1s[-1])
        final_aurocs.append(fold_aurocs[-1])
        final_auprcs.append(fold_auprcs[-1])
        cumulative_confusion_matrix += test_confusion_matrix

    # Alla fine del ciclo su tutti i fold, plotta i risultati
    plot_final_curve(final_losses, final_accuracies, final_f1s, final_aurocs, final_auprcs, model.__class__.__name__)

    print_cv_summary(final_losses, final_accuracies, final_f1s, final_aurocs, final_auprcs, model.__class__.__name__)
    print_confusion_matrix(cumulative_confusion_matrix, model.__class__.__name__, model_cat)
