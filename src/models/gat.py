import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GATChromatinGNN(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=128, output_dim=4, heads=4):
        super().__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False)
        self.gat3 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        x = F.elu(self.gat3(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.mlp(x)
