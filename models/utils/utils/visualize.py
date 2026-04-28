import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph(data):
    G = nx.Graph()

    edge_index = data.edge_index.numpy()

    # Add edges
    for i in range(edge_index.shape[1]):
        u = int(edge_index[0][i])
        v = int(edge_index[1][i])
        G.add_edge(u, v)

    # Node colors (fraud = red, normal = blue)
    colors = []
    for label in data.y:
        if label == 1:
            colors.append("red")
        else:
            colors.append("blue")

    plt.figure(figsize=(8, 6))
    nx.draw(G, with_labels=True, node_color=colors, node_size=500)

    plt.title("Fraud Detection Graph\nRed = Fraud, Blue = Normal")
    plt.show()

def plot_loss(losses):
    import matplotlib.pyplot as plt

    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.show()
