import numpy as np
import torch
from torch_geometric.data import Data

def create_enhanced_graph(sample):
    hic = sample['hic_matrix']
    chip = sample['chip_signals']
    gt = sample['ground_truth']
    positions = sample['positions']
    
    n = hic.shape[0]
    
    # 8 features per node
    chip_features = chip
    pos_features = positions[:, :3]
    
    threshold = 0.15
    degrees = np.sum(hic > threshold, axis=1)
    degree_features = (degrees / n).reshape(-1, 1)
    
    density = np.zeros(n)
    for i in range(n):
        start = max(0, i-10)
        end = min(n, i+10)
        density[i] = np.sum(hic[i, start:end] > threshold) / (end - start)
    density_features = density.reshape(-1, 1)
    
    x = np.hstack([chip_features, pos_features, degree_features, density_features])
    x = torch.tensor(x, dtype=torch.float)
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if hic[i, j] > threshold:
                edges.append([i, j])
    
    if len(edges) == 0:
        for i in range(n-1):
            edges.append([i, i+1])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    y = torch.tensor([
        gt['spring_k'],
        gt['attraction'],
        gt['noise'],
        gt['compaction']
    ], dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, y=y)
