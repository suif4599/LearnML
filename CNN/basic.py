import torch
from mnist import get_loader

class BasicCNN(torch.nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.pool1 = torch.nn.MaxPool2d(2)
        self.pool2 = torch.nn.MaxPool2d(2)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.fc1 = torch.nn.Linear(32 * 7 * 7, 128)
        self.fc2 = torch.nn.Linear(128, 10)
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 32 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def start_train(self, batch_size=32, epochs=10):
        train_loader, test_loader = get_loader(batch_size)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.parameters())
        train_loss = 0
        for i in range(epochs):
            self.train()
            for x, y in train_loader:
                optimizer.zero_grad()
                y_hat = self(x)
                loss = criterion(y_hat, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= epochs
            self.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for x, y in test_loader:
                    y_hat = self(x)
                    _, predicted = torch.max(y_hat, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
            print(f"Epoch {i + 1}, Test Accuracy: {correct / total} , Train Loss: {train_loss}")


model = BasicCNN()
model.start_train()
