import torch

def train_model(train_loader, test_loader, model, optimizer, loss_fn, epochs, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(epochs):
        model.train()
        for _input, _output in train_loader:
            _input = _input.to(device)
            _output = _output.to(device)
            optimizer.zero_grad()
            output = model(_input, _output)
            loss = loss_fn(output.view(-1, output.size(-1)), _output.view(-1))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            total_loss = 0
            for _input, _output in test_loader:
                _input = _input.to(device)
                _output = _output.to(device)
                output = model(_input, _output)
                loss = loss_fn(output.view(-1, output.size(-1)), _output.view(-1))
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(test_loader)}")
    return model