import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from matplotlib import pyplot as plt


def load_data():
    batch_size = 100
    # Image Preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.247, 0.2434, 0.2615)),
    ])

    # CIFAR-10 Dataset
    train_dataset = dsets.CIFAR10(root='./data/',
                                  train=True,
                                  transform=transform,
                                  download=True)

    test_dataset = dsets.CIFAR10(root='./data/',
                                 train=False,
                                 transform=transform,
                                 download=True)

    # Data Loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    return train_dataset, test_dataset, train_loader, test_loader


class CNN(nn.Module):
    def __init__(self, learning_rate=0.02):
        super(CNN, self).__init__()
        self.learning_rate = learning_rate
        self.padding = 1
        self.filter_size = 3
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=self.filter_size, padding=self.padding),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(20, 40, kernel_size=self.filter_size, padding=self.padding),
            nn.BatchNorm2d(40),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(40, 80, kernel_size=self.filter_size, padding=self.padding),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(4 * 4 * 80, 10)
        self.input_dropout = nn.Dropout(p=0.2)
        self.dropout = nn.Dropout(p=0.5)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.input_dropout(x)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return self.logsoftmax(out)


def evaluate_model_q1():
    train_dataset, test_dataset, train_loader, test_loader = load_data()
    cnn = CNN()
    cnn.load_state_dict(torch.load('cnn.pkl'))
    if torch.cuda.is_available():
        cnn = cnn.cuda()
    cnn.eval()
    correct, total = 0, 0
    for images, labels in test_loader:
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        outputs = cnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    accuracy = correct / total
    print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * accuracy))


if __name__ == '__main__':
    evaluate_model_q1()
