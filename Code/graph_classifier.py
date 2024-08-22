from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, SAGEConv
from torch.nn import Linear, BatchNorm1d
import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        # Calcolo delle probabilità e delle log probabilità
        probs = torch.sigmoid(input)
        log_probs = torch.log(probs.clamp(min=1e-7))

        # Calcolo della focal loss
        focal_loss = -((1 - probs) ** self.gamma) * log_probs * target - \
                     (probs ** self.gamma) * log_probs * (1 - target)

        # Applicazione dei pesi di bilanciamento alpha
        focal_loss = self.alpha * focal_loss

        # Applicazione della riduzione
        if self.reduction == 'mean':
            focal_loss = torch.mean(focal_loss)
        elif self.reduction == 'sum':
            focal_loss = torch.sum(focal_loss)

        return focal_loss


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
        return torch.softmax(x, dim=-1)


class M2(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(M2, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels)
        self.relu1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(p = 0.6)

        self.conv2 = GATConv(hidden_channels, hidden_channels)
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
        return torch.softmax(x, dim=-1)



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
        return torch.softmax(x, dim=-1)

