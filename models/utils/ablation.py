import torch
from utils.graph_builder import build_graph

def remove_edges(data, percent=0.5):
    num_edges = data.edge_index.shape[1]
    keep = int(num_edges * percent)
    data.edge_index = data.edge_index[:, :keep]
    return data

if __name__ == "__main__":
    data = build_graph("data/transactions.csv")

    print("Original edges:", data.edge_index.shape[1])

    new_data = remove_edges(data, 0.5)

    print("After ablation:", new_data.edge_index.shape[1])
