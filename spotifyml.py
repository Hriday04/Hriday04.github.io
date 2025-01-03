# -*- coding: utf-8 -*-
"""
SpotifyML
This script preprocesses Spotify dataset JSON files and implements a LightGCN model for recommendation.
Dataset: https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge
"""

import torch
import networkx as nx
import random
import torch_geometric
import numpy as np
from pathlib import Path
from JSON_Classes import JSONFile 
# Helper functions for PyTorch version formatting
def format_pytorch_version(version):
    return version.split('+')[0]

def format_cuda_version(version):
    return 'cu' + version.replace('.', '')

TORCH_version = torch.__version__
CUDA_version = torch.version.cuda
TORCH = format_pytorch_version(TORCH_version)
CUDA = format_cuda_version(CUDA_version)

# Install PyTorch Geometric dependencies
try:
    !pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
    !pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
    !pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
    !pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
    !pip install torch-geometric
except Exception as e:
    print(f"Error during PyTorch Geometric installation: {e}")

# Preprocess JSON files
json_list = []
folder_path = Path('sampleJSONs2')

for file in folder_path.iterdir():
    json_file = JSONFile(folder_path, file.name, 0)
    json_file.process_files()
    json_list.append(json_file)

# Extract playlists and tracks
playlist_data = {}
playlist_list = []
track_list = []

for index, obj in enumerate(json_list):
    playlist_list.extend([x.name for x in obj.playlists.values()])
    track_list.extend([track.uri for playlist in obj.playlists.values() for track in playlist.tracks.values()])
    
    for key, playlist in obj.playlists.items():
        unique_key = f"file_{index}_{key}"
        playlist_data[unique_key] = playlist

# Create a graph
G = nx.Graph()
G.add_nodes_from([(p, {'node_type': 'playlist'}) for p in playlist_data])
G.add_nodes_from([(t, {'node_type': 'track'}) for t in track_list])

edges = [(p_name, t.uri) for p_name, playlist in playlist_data.items() for t in playlist.tracks.values()]
G.add_edges_from(edges)

# Filter nodes with degree >= k
k = 20
G = nx.k_core(G, k)

# Convert to PyTorch Geometric Data format
n_nodes = G.number_of_nodes()
edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
num_nodes = len(G.nodes)
graph_data = torch_geometric.data.Data(edge_index=edge_index, num_nodes=num_nodes)

# Separate playlist and track indices
node2id = {node: idx for idx, node in enumerate(G.nodes())}
id2node = {idx: node for node, idx in node2id.items()}

playlists_idx = [idx for idx, node in enumerate(node2id.keys()) if "playlist" in node]
tracks_idx = [idx for idx, node in enumerate(node2id.keys()) if "track" in node]

# Define LightGCN Model
class LightGCNConv(torch_geometric.nn.MessagePassing):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=(x, x))

    def message(self, x_j):
        return x_j

    def aggregate(self, inputs, index):
        return torch_scatter.scatter(inputs, index, dim=0, reduce='mean')

class LightGCN(torch.nn.Module):
    def __init__(self, num_nodes, embedding_dim, num_layers):
        super().__init__()
        self.embeddings = torch.nn.Embedding(num_nodes, embedding_dim)
        self.layers = torch.nn.ModuleList([LightGCNConv(embedding_dim, embedding_dim) for _ in range(num_layers)])

    def forward(self, edge_index):
        x = self.embeddings.weight
        for layer in self.layers:
            x = layer(x, edge_index)
        return x

# Training and Evaluation Functions
def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    pred = model(data.edge_index)
    loss = torch.nn.functional.mse_loss(pred, data.y)
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(model, data):
    model.eval()
    pred = model(data.edge_index)
    loss = torch.nn.functional.mse_loss(pred, data.y)
    return loss.item()

# Initialize and Train the Model
embedding_dim = 64
num_layers = 3
model = LightGCN(num_nodes=num_nodes, embedding_dim=embedding_dim, num_layers=num_layers).to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1, 101):
    loss = train(model, optimizer, graph_data)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
