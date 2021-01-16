import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1, stride=1),  # 8x28x28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 8x14x14
            nn.Dropout(0.5),
        )

        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(5, 5), padding=2, stride=1),  # 8x14x14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 8x7x7
            nn.Dropout(0.5),
            nn.Flatten(),
        )

        self.lin = nn.Sequential(
            nn.Linear(392, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.lin(x)
        return x
