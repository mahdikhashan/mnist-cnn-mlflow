import mlflow
assert mlflow.__version__ >= "2.0.0"

mlflow.set_tracking_uri("http://localhost:8083")

import torch
assert torch.version.__version__ >= '2.0.0'

device = ("cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
torch.set_default_device(device)

from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

import torchvision
assert torchvision.version.__version__ >= "0.18.0"

from torchvision import datasets
from torchvision import transforms

from torchinfo import summary


class MNISTCNN(nn.Module):
    def __init__(self, num_channels=1, num_filters=16, num_classes=10):
        super(MNISTCNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(num_channels, num_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(num_filters * num_channels * 28 * 28, num_classes)
        )

    def forward(self, x):
        o = F.softmax(self.layers(x), dim=1)
        return o

def train(data_loader, model: MNISTCNN, loss_fn, optimizer, epoch, logger):
    model = model.train()

    total_loss = 0.0

    for (data, target) in data_loader:
        data, target = data.to(device), target.to(device)

        output = model(data)

        optimizer.zero_grad()

        loss = loss_fn(output, target)
        logger.log_metric('loss', loss.item(), step=epoch)

        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    logger.log_metric("train_loss", total_loss / len(data_loader), step=epoch)

    return total_loss / len(data_loader)

def test():
    pass

batch_size = 200

data_dir = './data'
print(data_dir)

train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(data_dir, train=False, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

if __name__ == '__main__':
    model = MNISTCNN()
    model.to(device)

    model_info = summary(model)

    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr)

    loss_fn = nn.CrossEntropyLoss()

    epochs = 5

    with mlflow.start_run():
        params = {
            "epochs": epochs,
            "learning_rate": lr,
        }
        mlflow.log_params(params)

        for epoch in range(epochs):
            print("Epoch {}/{}".format(epoch, epochs))
            loss = train(train_loader, model, loss_fn, optimizer, epoch, mlflow)

        test()

        mlflow.pytorch.log_model(model, registered_model_name="mnist_cnn", artifact_path="mnist_cnn")
