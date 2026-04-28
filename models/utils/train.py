from utils.visualize import plot_loss

losses = []

for epoch in range(50):
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)
    loss = loss_fn(out, data.y.long())

    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    print(f"Epoch {epoch}, Loss: {loss.item()}")

# Save model
torch.save(model.state_dict(), "model.pth")

# Plot loss
plot_loss(losses)
