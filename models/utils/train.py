import torch
from models.gcn_model import FraudGCN
from utils.graph_builder import build_graph

data = build_graph("data/transactions.csv")

model = FraudGCN(16, 32, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(50):
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)
    loss = loss_fn(out, data.y.long())

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")
torch.save(model.state_dict(), "model.pth")
