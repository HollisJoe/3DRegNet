import torch
import torch.nn as nn


# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
class ResNetBlock1d(nn.Module):
    def __init__(self, channels=128, kernel_size=1):
        super(ResNetBlock1d, self).__init__()

        self.conv1 = nn.Conv1d(channels, channels, 1)
        self.conv2 = nn.Conv1d(channels, channels, 1)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.relu(out + identity)

        return out


class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()

    def forward(self, x):
        return torch.max(x, dim=-1).values.unsqueeze(-1)


class RegNetClassifier(nn.Module):
    def __init__(self, resnet_blocks=12, feature_channels=128):
        super(RegNetClassifier, self).__init__()
        self.conv_first = nn.Conv1d(6, feature_channels, 1)
        self.conv_weight = nn.Conv1d(feature_channels, 1, 1)

        self.resnet_blocks = []
        for i in range(0, resnet_blocks):
            self.resnet_blocks.append(ResNetBlock1d(channels=feature_channels))

        self.global_maxpool1d = GlobalMaxPool1d()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # input: (b, 6, N)
        # (b, 128, N)
        x = self.conv_first(x)

        # append (b, 128, 1) for (1 + 12) times
        features = []
        features.append(self.global_maxpool1d(x))

        for resnet_block in self.resnet_blocks:
            x = resnet_block(x)
            features.append(self.global_maxpool1d(x))

        # output weight: (b, 1, N)
        weight = self.conv_weight(x)
        weight = self.relu(self.tanh(x))

        # output feature: (b, 128, 13)
        feature = torch.cat(features, dim=-1)

        return weight, feature


class RegNetRegressor(nn.Module):
    def __init__(self,
                 resnet_blocks=12,
                 feature_channels=128,
                 conv_channels=8,
                 fc2_channels=256):
        super(RegNetRegressor, self).__init__()
        self.conv = nn.Conv2d(1, conv_channels, 3, stride=(2, 1))
        self.relu = nn.ReLU(inplace=True)

        self.fc1_channels = conv_channels \
            * (resnet_blocks - 1) \
            * (feature_channels // 2 - 1)
        self.fc1 = nn.Linear(self.fc1_channels, fc2_channels)
        self.fc2 = nn.Linear(fc2_channels, 6)

    def forward(self, x):
        # input: (b, 128, 13) -> unsqueeze -> (b, 1, 128, 13)
        x = x.unsqueeze(1)

        # (b, 8, 63, 11)
        x = self.conv(x)
        x = self.relu(x)
        x = x.view(-1, self.fc1_channels)

        # (b, 5544)
        x = self.fc1(x)
        x = self.relu(x)

        # (b, 256)
        x = self.fc2(x)

        # output pose: (b, 6)
        return x


class RegNet3D(nn.Module):
    def __init__(self,
                 classifier_resnet_blocks=12,
                 classifier_feature_channels=128,
                 regressor_conv_channels=8,
                 regressor_fc2_channels=256):
        super(RegNet3D, self).__init__()
        self.classifier = RegNetClassifier(
            resnet_blocks=classifier_resnet_blocks,
            feature_channels=classifier_feature_channels)

        self.regressor = RegNetRegressor(conv_channels=regressor_conv_channels,
                                         fc2_channels=regressor_fc2_channels)

    def forward(self, x):
        # input: (b, 6, N)

        # (b, 1, N), (b, 128, 13)
        weight, feature = self.classifier(x)

        # (b, 6)
        pose = self.regressor(feature)

        return weight, pose


if __name__ == '__main__':
    # Separate, for easier debugging
    regnet_classifier = RegNetClassifier()
    regnet_regressor = RegNetRegressor()
    x = torch.rand(3, 6, 2000)
    weight, feature = regnet_classifier(x)
    pose = regnet_regressor(feature)
    print(weight.size())
    print(pose.size())

    # Aggregate, for overall test
    regnet3d = RegNet3D()
    weight, pose = regnet3d(x)
    print(weight.size())
    print(pose.size())
