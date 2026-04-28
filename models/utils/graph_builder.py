import pandas as pd
import torch
from torch_geometric.data import Data

def build_graph(file):
    df = pd.read_csv(file)

    users = list(set(df['user_id']).union(set(df['receiver_id'])))
    user_map = {u: i for i, u in enumerate(users)}

    edges = []
    for _, row in df.iterrows():
        edges.append([user_map[row['user_id']], user_map[row['receiver_id']]])

    edge_index = torch.tensor(edges).t().contiguous()

    # Node features (random for now)
    x = torch.rand((len(users), 16))

    # Labels
    y = torch.zeros(len(users))
    for _, row in df.iterrows():
        if row['label'] == 1:
            y[user_map[row['user_id']]] = 1

    return Data(x=x, edge_index=edge_index, y=y)
