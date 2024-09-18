from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, SAGEConv
from torch.nn import Linear, BatchNorm1d
import torch

class M1(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(M1, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.relu1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(p = 0.6)

        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.relu2 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(p = 0.6)
        
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.relu3 = torch.nn.ReLU()
        self.dropout3 = torch.nn.Dropout(p = 0.6)
        
        self.pooling = global_mean_pool
        self.norm = BatchNorm1d(hidden_channels)
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch, batch_size):
        x = self.relu1(self.conv1(x, edge_index))
        x = self.dropout1(x)
        x = self.relu2(self.conv2(x, edge_index))
        x = self.dropout2(x)
        x = self.relu3(self.conv3(x, edge_index))
        x = self.dropout3(x)
        x = x.to(torch.float32)
        x = self.pooling(x, batch=batch, size=batch_size)
        x = self.norm(x)
        x = self.lin1(x)
        x = self.lin2(x)
        return x


class M2(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(M2, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels, dropout=0.6)
        self.relu1 = torch.nn.ReLU()

        self.conv2 = GATConv(hidden_channels, hidden_channels, dropout=0.6)
        self.relu2 = torch.nn.ReLU()
        
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.relu3 = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p = 0.6)
        
        self.pooling = global_mean_pool
        self.norm = BatchNorm1d(hidden_channels)
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch, batch_size):
        x = self.relu1(self.conv1(x, edge_index))
        x = self.relu2(self.conv2(x, edge_index))
        x = self.relu3(self.conv3(x, edge_index))
        x = self.dropout(x)
        x = x.to(torch.float32)
        x = self.pooling(x, batch=batch, size=batch_size)
        x = self.norm(x)
        x = self.lin1(x)
        x = self.lin2(x)
        return x


class M3(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(M3, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.relu1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(p = 0.6)

        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.relu2 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(p = 0.6)
        
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = SAGEConv(hidden_channels, hidden_channels)
        self.relu3 = torch.nn.ReLU()
        self.dropout3 = torch.nn.Dropout(p = 0.6)

        self.pooling = global_mean_pool
        self.norm = BatchNorm1d(hidden_channels)
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch, batch_size):
        x = self.relu1(self.conv1(x, edge_index))
        x = self.dropout1(x)
        x = self.relu2(self.conv2(x, edge_index))
        x = self.dropout2(x)
        x = self.relu3(self.conv3( self.conv4(x, edge_index) , edge_index))
        x = self.dropout3(x)
        x = x.to(torch.float32)
        x = self.pooling(x, batch=batch, size=batch_size)
        x = self.norm(x)
        x = self.lin1(x)
        x = self.lin2(x)
        return x
    

class GAT(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GAT, self).__init__()
        self.conv = GATConv(num_features, hidden_channels, dropout=0.6)
        self.relu = torch.nn.LeakyReLU()
        self.pooling = global_mean_pool
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch, batch_size):
        x = self.conv(x, edge_index)
        x = self.relu(x)
        x = self.pooling(x, batch)
        x = self.lin(x)
        return x


class GAT2(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GAT2, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels,dropout=0.6)
        self.conv2 = GATConv(hidden_channels, hidden_channels, dropout=0.6)
        self.relu = torch.nn.LeakyReLU()
        self.pooling = global_mean_pool
        self.norm = BatchNorm1d(hidden_channels)
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch, batch_size):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.norm(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.pooling(x, batch)
       # x = self.norm(x)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        return x


class GAT3(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GAT3, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels, dropout=0.6)
        self.conv2 = GATConv(hidden_channels, hidden_channels, dropout=0.6)
        self.relu = torch.nn.LeakyReLU()
        self.pooling = global_mean_pool
        self.norm = BatchNorm1d(hidden_channels)
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch, batch_size):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.norm(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.norm(x)
        x = self.pooling(x, batch)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        return x