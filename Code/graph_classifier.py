from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, SAGEConv
from torch.nn import Linear, BatchNorm1d
import torch
import torch.nn.functional as F

class M1(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(M1, self).__init__()
        self.conv1 = GCNConv(num_features, 64)  
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.relu1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(p=0.5)

        self.conv2 = GCNConv(64, 128)  
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.relu2 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(p=0.5)

        self.conv3 = GCNConv(128, 256) 
        self.bn3 = torch.nn.BatchNorm1d(256)
        self.relu3 = torch.nn.ReLU()
        self.dropout3 = torch.nn.Dropout(p=0.5)

        self.pooling = global_mean_pool
        self.lin1 = Linear(256, 512)  
        self.bn4 = BatchNorm1d(512)
        self.dp4 = torch.nn.Dropout(p=0.5)

        self.lin2 = Linear(512, num_classes)

    def forward(self, x, edge_index, batch, batch_size):
        x = self.relu1(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout1(x)
        x = self.relu2(self.bn2(self.conv2(x, edge_index)))
        x = self.dropout2(x)
        x = self.relu3(self.bn3(self.conv3(x, edge_index)))
        x = self.dropout3(x)

        x = self.pooling(x, batch=batch, size=batch_size)
        x = F.relu(self.bn4(self.lin1(x)))
        x = self.dp4(x)
        x = self.lin2(x)
        return x

class M2(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(M2, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels, heads=4, dropout=0.6)  
        self.conv2 = GATConv(hidden_channels * 4, hidden_channels * 2, dropout=0.6)
        self.conv3 = GCNConv(hidden_channels * 2, hidden_channels)
        self.dropout3 = torch.nn.Dropout(p=0.6) 
        self.dropout4 = torch.nn.Dropout(p=0.6) 
        self.relu = torch.nn.LeakyReLU()
        self.pooling = global_mean_pool  
        self.norm1 = BatchNorm1d(hidden_channels * 4)
        self.norm2 = BatchNorm1d(hidden_channels * 2)
        self.norm3 = BatchNorm1d(hidden_channels * 2)
        self.lin1 = Linear(hidden_channels, 128) 
        self.lin2 = Linear(128, num_classes)

    def forward(self, x, edge_index, batch, batch_size):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.norm1(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.norm2(x)
        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.pooling(x, batch)
        x = self.lin1(x)
        x = self.norm3(x)
        x = self.relu(x)
        x = self.dropout4(x)
        x = self.lin2(x)
        return x
    

class M3(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(M3, self).__init__()
        
        # Primo strato: GCNConv con 64 hidden channels
        self.conv1 = GCNConv(num_features, 64)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.relu1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(p=0.5)
        
        # Secondo strato: GCNConv con 128 hidden channels
        self.conv2 = GCNConv(64, 128)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.relu2 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(p=0.5)
        
        # Terzo strato: combinazione di GCNConv e SAGEConv con 256 hidden channels
        self.conv3 = GCNConv(128, 128)
        self.conv4 = SAGEConv(128, 256)
        self.bn3 = torch.nn.BatchNorm1d(256)
        self.relu3 = torch.nn.ReLU()
        self.dropout3 = torch.nn.Dropout(p=0.5)
        
        # Pooling globale
        self.pooling = global_mean_pool
        
        # Strato lineare e normalizzazione
        self.lin1 = Linear(256, 512)  # Lineare con 512 hidden units
        self.bn4 = torch.nn.BatchNorm1d(512)
        self.dropout4 = torch.nn.Dropout(p=0.5)
        
        self.lin2 = Linear(512, num_classes)

    def forward(self, x, edge_index, batch, batch_size):
        # Primo blocco convolutivo
        x = self.relu1(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout1(x)
        
        # Secondo blocco convolutivo
        x = self.relu2(self.bn2(self.conv2(x, edge_index)))
        x = self.dropout2(x)
        
        # Terzo blocco convolutivo (combinazione di GCN e SAGE)
        x = self.relu3(self.bn3(self.conv4(self.conv3(x, edge_index), edge_index)))
        x = self.dropout3(x)
        
        # Pooling globale
        x = x.to(torch.float32)
        x = self.pooling(x, batch=batch, size=batch_size)
        
        # Strati finali completamente connessi
        x = F.relu(self.bn4(self.lin1(x)))
        x = self.dropout4(x)
        x = self.lin2(x)
        
        return x

class GAT(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GAT, self).__init__()
        self.conv = GATConv(num_features, hidden_channels, heads=4, dropout=0.6)
        self.relu = torch.nn.LeakyReLU()
        self.pooling = global_mean_pool
        self.lin = Linear(hidden_channels*4, num_classes)

    def forward(self, x, edge_index, batch, batch_size):
        x = self.conv(x, edge_index)
        x = self.relu(x)
        x = self.pooling(x, batch)
        x = self.lin(x)
        return x