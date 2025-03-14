import torch

class VGG16(torch.nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(512, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout)
        )
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout)
        )
        self.fc3 = torch.nn.Linear(4096, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(-1, 512)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
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

model = VGG16()
model.start_train()