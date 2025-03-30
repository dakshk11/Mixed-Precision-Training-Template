import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
# Other imports remain unchanged

def train(FLAGS):
    # Dataset loading, model setup, etc. unchanged until training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_size, n_layers, drop_prob).to(device)
    criterion = nn.BCELossWithLogits()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler()

    counter = 0
    for epoch in range(epochs):
        model.train()
        h = model.init_hidden(batch_size)
        for inputs, labels in train_loader:
            counter += 1
            h = tuple([e.data for e in h])
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast():
                output, h = model(inputs, h)
                loss = criterion(output.squeeze(), labels.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if counter % print_every == 0:
                val_h = model.init_hidden(batch_size)
                val_losses = []
                model.eval()
                for inputs, labels in valid_loader:
                    val_h = tuple([each.data for each in val_h])
                    inputs, labels = inputs.to(device), labels.to(device)
                    with autocast():
                        output, val_h = model(inputs, val_h)
                        val_loss = criterion(output.squeeze(), labels.float())
                    val_losses.append(val_loss.item())
                # Rest of validation and printing logic...
