import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC

random_forest = RandomForestClassifier
linear_svm = LinearSVC
svm = SVC
resnet18 = torchvision.models.resnet18


class OverfitResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super(OverfitResNet18, self).__init__()
        self.resnet18 = torchvision.models.resnet18(num_classes=1000)
        self.fc1 = nn.Linear(1000, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.resnet18(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def create_mlp(input_size, output_size=2):
    return nn.Sequential(nn.Linear(input_size, 100),
                         nn.ReLU(),
                       #  nn.Linear(128, 128),
                       #  nn.ReLU(),
                         nn.Linear(100, output_size))


class LeNetTIMIT(nn.Module):
    def __init__(self, input_channels=1, output_size=2):
        super(LeNetTIMIT, self).__init__()
        # 3 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=3)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(1408, 128)  # 6*6 from image dimension
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class LeNetTIMIT2(nn.Module):
    def __init__(self, input_channels=1, output_size=2):
        super(LeNetTIMIT2, self).__init__()
        # 3 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=3)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3)
        # an affine operation: y = Wx + b
        # row count 50 => 1408 fc1
        # row count 100 => 2944 fc1
        self.fc1 = nn.Linear(2944, 128)  # 6*6 from image dimension
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class LeNetMNIST(nn.Module):
    def __init__(self, input_channels=3, output_size=2):
        super(LeNetMNIST, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, output_size)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    """
        def __init__(self, input_channels=3, output_size=2):
            super(LeNetMNIST, self).__init__()
            # input is 28x28
            # padding=2 for same padding
            self.conv1 = nn.Conv2d(input_channels, 20, 5, padding=2)
            # feature map size is 14*14 by pooling
            # padding=2 for same padding
            self.conv2 = nn.Conv2d(20, 20, 5, padding=2)
            # feature map size is 7*7 by pooling
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, output_size)
    
        def forward(self, x):
            x = F.max_pool2d(F.relu(self.conv1(x)), 2)
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = x.view(-1, 320)  # reshape Variable
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    """