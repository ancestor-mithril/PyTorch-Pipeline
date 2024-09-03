from torch import nn


class CNN(nn.Module):
    def __init__(self, num_classes: int, input_channels: int):
        super(CNN, self).__init__()

        self.block1 = self.create_conv_block(input_channels, 32, 0.25)
        self.block2 = self.create_conv_block(32, 64, 0.25)
        self.block3 = self.create_conv_block(64, 128, 0.25)
        self.flatten = nn.Flatten()
        self.dense1 = self.create_dens_block(input_channels * 1152, 512, 0.5)
        self.dense2 = self.create_dens_block(512, 256, 0.4)
        self.dense3 = self.create_dens_block(256, 64, 0.3)
        self.fc = nn.Linear(64, num_classes)

    @staticmethod
    def create_dens_block(in_features, out_features, dropout_p):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_features),
            nn.Dropout(dropout_p),
        )

    @staticmethod
    def create_conv_block(in_channels, out_channels, dropout_p):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding="same"),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(2),
            nn.Dropout(dropout_p),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.fc(x)


def CNN_MNIST(num_classes):
    return CNN(num_classes=num_classes, input_channels=1)
