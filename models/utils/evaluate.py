import torch
from sklearn.metrics import accuracy_score
from models.gcn_model import FraudGCN
from utils.graph_builder import build_graph

data = build_graph("data/transactions.csv")

model = FraudGCN(16, 32, 2)
model.load_state_dict(torch.load("model.pth"))

model.eval()
pred = model(data.x, data.edge_index).argmax(dim=1)

acc = accuracy_score(data.y, pred)
print("Accuracy:", acc)
