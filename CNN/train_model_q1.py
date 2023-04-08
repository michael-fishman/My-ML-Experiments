import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import itertools


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


def create_augmented_data():
    """
    Create additional data by augmenting the CIFAR-10 database.
    """
    batch_size = 100
    # Augmentation Transform
    augmentation_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.247, 0.2434, 0.2615)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.RandomCrop(32, padding=4)], p=0.1),
        transforms.RandomRotation(15)
    ])
    augmented_dataset = dsets.CIFAR10(root='./data/',
                                      train=True,
                                      transform=augmentation_transform,
                                      download=True)

    # Data Loader for Augmented Data
    augmented_data_loader = torch.utils.data.DataLoader(dataset=augmented_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True)

    return augmented_data_loader


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


def train_model_q1():
    # Hyper Parameters
    learning_rate = 0.02
    cnn = CNN()
    train_dataset, test_dataset, train_loader, test_loader = load_data()
    augmented_loader = create_augmented_data()
    augmented_loader2 = create_augmented_data()
    augmented_loader3 = create_augmented_data()
    if torch.cuda.is_available():
        cnn = cnn.cuda()

    # Loss and Optimizer
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    # Train the Model
    train_loss, train_error, test_loss, test_error = [], [], [], []
    accuracy, epoch = 0, 0
    decay = lambda epoch: 0.92 ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=decay)
    while accuracy < 0.5:
        epoch += 1
        correct, total = 0, 0
        for i, ((images, labels), (aug_images, aug_labels), (aug_images2, aug_labels2), (aug_images3, aug_labels3)) \
                in enumerate(zip(train_loader, augmented_loader, augmented_loader2, augmented_loader3)):
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
                aug_images = aug_images.cuda()
                aug_labels = aug_labels.cuda()
                aug_images2 = aug_images2.cuda()
                aug_labels2 = aug_labels2.cuda()
                aug_images3 = aug_images3.cuda()
                aug_labels3 = aug_labels3.cuda()

            # Forward + Backward + Optimize
            images = torch.cat((images, aug_images), axis=0)
            images = torch.cat((images, aug_images2), axis=0)
            images = torch.cat((images, aug_images3), axis=0)
            labels = torch.cat((labels, aug_labels), axis=0)
            labels = torch.cat((labels, aug_labels2), axis=0)
            labels = torch.cat((labels, aug_labels3), axis=0)
            outputs = cnn(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            if (i + 1) % 100 == 0:
                print(optimizer.param_groups[0]["lr"])
                print('Epoch [%d], Iter [%d/%d] Loss: %.4f'
                      % (epoch, i + 1,
                         len(train_dataset) // 100, loss.data))

            # Train on augmented CIFAR-10 database:
        scheduler.step()
        train_error.append(1 - correct / total)
        train_loss.append(loss.data)
        print('Epoch [%d] Train Accuracy: %.4f' % (epoch, correct / total))
        cnn.eval()
        correct, total, current_loss = 0, 0, 0
        for images, labels in test_loader:
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            outputs = cnn(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            loss = criterion(outputs, labels)
            current_loss += loss.data
        accuracy = correct / total
        print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * accuracy))
        test_error.append(1 - accuracy)
        test_loss.append(current_loss / 100)
    torch.save(cnn.state_dict(), 'out_cnn.pkl')
    return train_loss, train_error, test_loss, test_error, epoch


def draw_graphs(train_loss, train_error, test_loss, test_error, epochs):
    epoch_range = range(1, epochs + 1)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(epoch_range, train_loss, 'b', label='Training loss')
    ax[0].plot(epoch_range, test_loss, 'r', label='Test loss')
    ax[0].set_title('Training and Test loss')
    ax[0].legend()
    ax[1].plot(epoch_range, train_error, 'b', label='Training error')
    ax[1].plot(epoch_range, test_error, 'r', label='Test error')
    ax[1].set_title('Training and Test error')
    ax[1].legend()
    plt.show()


def evaluate_model_q1():
    train_dataset, test_dataset, train_loader, test_loader = load_data()
    # test_dataset = dsets.CIFAR10(root='./data/',
    #                              train=False,
    #                              transform=transforms.ToTensor())
    # test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
    #                                           batch_size=100,
    #                                           shuffle=False)
    cnn = CNN()
    cnn.load_state_dict(torch.load('out_cnn.pkl'))
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
    train_loss, train_error, test_loss, test_error, epochs = train_model_q1()
    draw_graphs(train_loss, train_error, test_loss, test_error, epochs)
    evaluate_model_q1()
