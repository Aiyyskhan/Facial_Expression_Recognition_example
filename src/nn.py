import torch.nn as nn
import torch

class NN(nn.Module):
    def __init__(self, groups=1, device=None, dtype=None):
        super(NN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, groups=groups, device=device, dtype=dtype)
        self.cnn2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, groups=groups, device=device, dtype=dtype)
        self.cnn3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, groups=groups, device=device, dtype=dtype)
        self.cnn4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, groups=groups, device=device, dtype=dtype)
        self.cnn5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, groups=groups, device=device, dtype=dtype)
        self.cnn6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, groups=groups, device=device, dtype=dtype)
        self.cnn7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, groups=groups, device=device, dtype=dtype)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.cnn1_bn = nn.BatchNorm2d(8, device=device, dtype=dtype)
        self.cnn2_bn = nn.BatchNorm2d(16, device=device, dtype=dtype)
        self.cnn3_bn = nn.BatchNorm2d(32, device=device, dtype=dtype)
        self.cnn4_bn = nn.BatchNorm2d(64, device=device, dtype=dtype)
        self.cnn5_bn = nn.BatchNorm2d(128, device=device, dtype=dtype)
        self.cnn6_bn = nn.BatchNorm2d(256, device=device, dtype=dtype)
        self.cnn7_bn = nn.BatchNorm2d(256, device=device, dtype=dtype)
        self.fc1 = nn.Linear(1024, 512, device=device, dtype=dtype)
        self.fc2 = nn.Linear(512, 256, device=device, dtype=dtype)
        self.fc3 = nn.Linear(256, 7, device=device, dtype=dtype)
        self.dropout = nn.Dropout(0.3)
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = self.relu(self.pool1(self.cnn1_bn(self.cnn1(x))))
        x = self.relu(self.pool1(self.cnn2_bn(self.dropout(self.cnn2(x)))))
        x = self.relu(self.pool1(self.cnn3_bn(self.cnn3(x))))
        x = self.relu(self.pool1(self.cnn4_bn(self.dropout(self.cnn4(x)))))
        x = self.relu(self.pool2(self.cnn5_bn(self.cnn5(x))))
        x = self.relu(self.pool2(self.cnn6_bn(self.dropout(self.cnn6(x)))))
        x = self.relu(self.pool2(self.cnn7_bn(self.dropout(self.cnn7(x)))))

        x = x.view(x.size(0), -1)

        x = self.relu(self.dropout(self.fc1(x)))
        x = self.relu(self.dropout(self.fc2(x)))
        x = self.log_softmax(self.fc3(x))
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    bn_model = NN()
    x = torch.randn(1,1,48,48)
    print('Shape of output = ',bn_model(x).shape)
    print('No of Parameters of the BatchNorm-CNN Model =',bn_model.count_parameters())