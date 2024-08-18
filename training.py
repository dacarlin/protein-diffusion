import torch
import torch.optim as optim
import torch.nn as nn

def train(model, dataloader, epochs=100, lr=1e-4):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for batch in dataloader:
            noisy_batch = batch + torch.randn_like(batch) * 0.1  # Adding noise
            optimizer.zero_grad()
            output = model(noisy_batch)
            loss = loss_fn(output, batch)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
