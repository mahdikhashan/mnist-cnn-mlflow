from tqdm import tqdm

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
    def __init__(self, num_channels=1, num_filters=32, num_classes=10):
        super(MNISTCNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(num_channels, num_filters, kernel_size=3, padding="valid"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(32 * 13 * 13, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        o = F.softmax(self.layers(x), dim=1)
        return o

def correct(output, target):
    predicted_digits = output.argmax(1)                            # pick digit with largest network output
    correct_ones = (predicted_digits == target).type(torch.float)  # 1.0 for correct, 0.0 for incorrect
    return correct_ones.sum().item()                               # count number of correct ones

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

def test(test_loader, model, criterion, _logger):
    model.eval()

    num_batches = len(test_loader)
    num_items = len(test_loader.dataset)

    test_loss = 0
    total_correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            # Copy data and targets to GPU
            data = data.to(device)
            target = target.to(device)

            # Do a forward pass
            output = model(data)

            # Calculate the loss
            loss = criterion(output, target)
            test_loss += loss.item()

            # Count number of correct digits
            total_correct += correct(output, target)

    test_loss = test_loss/num_batches
    accuracy = total_correct/num_items

    _logger.log_metric('testset-accuracy', 100 * accuracy)

batch_size = 32

data_dir = './data'
print(data_dir)

# a simple data augmentation
train_transforms = transforms.Compose([
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=train_transforms)
test_dataset = datasets.MNIST(data_dir, train=False, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

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

        for epoch in tqdm(range(epochs)):
            print("Epoch {}/{}".format(epoch, epochs))
            loss = train(train_loader, model, loss_fn, optimizer, epoch, mlflow)

        test(test_loader, model, loss_fn, _logger=mlflow)

        mlflow.pytorch.log_model(model, registered_model_name="mnist_cnn", artifact_path="mnist_cnn")
