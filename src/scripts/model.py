import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary


class ResidualEncoder(nn.Module):
    """
    IAEmu Encoder with residual connections, batch normalization, leaky ReLU, and dropout.
    """

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 128,
        bottleneck_dim: int = 128,
        dropout: float = 0.2,
    ):
        super(ResidualEncoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
        )
        # Dimension matching from 2*hidden_dim to 4*hidden_dim
        self.match_dim1 = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(2 * hidden_dim),
            nn.Dropout(dropout),
        )
        self.layer4 = nn.Sequential(
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(2 * hidden_dim),
            nn.Dropout(dropout),
        )
        # Final layer
        self.final_layer = nn.Sequential(
            nn.Linear(2 * hidden_dim, bottleneck_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(bottleneck_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.layer1(x)
        x2 = self.layer2(x)
        identity1 = self.match_dim1(
            x2
        )  # Match dimensions for the first residual connection
        x3 = self.layer3(x2)
        x3 += identity1  # Add the first residual connection
        x4 = self.layer4(x3)
        identity2 = x3  # Use output from layer3 as the second skip connection
        x4 += identity2  # Add the second residual connection
        x_final = self.final_layer(x4)
        return x_final

    def forward_with_dropout(self, x):
        # Enable dropout
        for layer in [
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.final_layer,
        ]:
            for sub_layer in layer:
                if isinstance(sub_layer, nn.Dropout):
                    sub_layer.train(True)
        return self.forward(x)


class ResidualDecoder(nn.Module):
    """
    IAEmu Decoder with residual connections, batch normalization, leaky ReLU, and dropout.
    """

    def __init__(self, input_dim: int = 128, output_dim: int = 2, dropout: float = 0.2):
        super(ResidualDecoder, self).__init__()

        self.initial_layer = nn.Linear(input_dim, 700)
        self.initial_bn = nn.BatchNorm1d(700)
        self.initial_relu = nn.LeakyReLU()
        self.initial_dropout = nn.Dropout(dropout)

        # Define the convolutional layers with potential for residual connections
        self.conv1 = nn.Conv1d(in_channels=7, out_channels=14, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(14)
        self.relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            in_channels=14, out_channels=28, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm1d(28)
        self.relu2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.conv3 = nn.Conv1d(
            in_channels=28, out_channels=28, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm1d(28)
        self.relu3 = nn.LeakyReLU()
        self.dropout3 = nn.Dropout(dropout)

        self.conv4 = nn.Conv1d(
            in_channels=28, out_channels=56, kernel_size=3, padding=1
        )
        self.bn4 = nn.BatchNorm1d(56)
        self.relu4 = nn.LeakyReLU()
        self.dropout4 = nn.Dropout(dropout)

        self.conv5 = nn.Conv1d(
            in_channels=56, out_channels=112, kernel_size=3, padding=1
        )
        self.bn5 = nn.BatchNorm1d(112)
        self.relu5 = nn.LeakyReLU()
        self.dropout5 = nn.Dropout(dropout)

        self.conv6 = nn.Conv1d(
            in_channels=112, out_channels=112, kernel_size=3, padding=1
        )
        self.bn6 = nn.BatchNorm1d(112)
        self.relu6 = nn.LeakyReLU()
        self.dropout6 = nn.Dropout(dropout)

        self.conv7 = nn.Conv1d(
            in_channels=112, out_channels=output_dim, kernel_size=5, padding=1, stride=5
        )
        self.conv7.bias.data[1] = 1.0

    def forward(self, x):
        x = self.initial_dropout(
            self.initial_bn(self.initial_relu(self.initial_layer(x)))
        )
        x = x.view(x.size(0), 7, -1)

        #### reverse ordering of batch and relu
        out = self.dropout1(self.bn1(self.relu1(self.conv1(x))))
        out = self.dropout2(self.bn2(self.relu2(self.conv2(out))))

        identity = out
        out = self.dropout3(self.bn3(self.relu3(self.conv3(out))))
        out += identity  # Add skip connection

        out = self.dropout4(self.bn4(self.relu4(self.conv4(out))))
        out = self.dropout5(self.bn5(self.relu5(self.conv5(out))))

        identity = out
        out = self.dropout6(self.bn6(self.relu6(self.conv6(out))))
        out += identity  # Add skip connection

        out = self.conv7(out)
        means, var = out[:, 0], F.softplus(out[:, 1])

        output = torch.cat([means, var], dim=1)
        return output

    def forward_with_dropout(self, x):
        # Activating dropout during inference, usually not recommended
        for layer in [self.dropout1, self.dropout2, self.dropout3, self.dropout4]:
            layer.train(True)
        return self.forward(x)


class ResidualMultiTaskEncoderDecoder(nn.Module):
    """
    IAEmu Multi-Task Encoder-Decoder with residual connections, batch normalization, leaky ReLU, and dropout.
    """

    def __init__(self):
        super(ResidualMultiTaskEncoderDecoder, self).__init__()
        self.encoder = ResidualEncoder()
        self.decoder1 = ResidualDecoder()
        self.decoder2 = ResidualDecoder()
        self.decoder3 = ResidualDecoder()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded1 = self.decoder1(encoded)
        decoded2 = self.decoder2(encoded)
        decoded3 = self.decoder3(encoded)

        output = torch.stack([decoded1, decoded2, decoded3], dim=1)

        return output

    def forward_with_dropout(self, x):
        encoded = self.encoder.forward_with_dropout(x)
        decoded1 = self.decoder1.forward_with_dropout(encoded)
        decoded2 = self.decoder2.forward_with_dropout(encoded)
        decoded3 = self.decoder3.forward_with_dropout(encoded)

        output = torch.stack([decoded1, decoded2, decoded3], dim=1)
        return output


if __name__ == "__main__":
    model = ResidualMultiTaskEncoderDecoder()
    print(torchsummary.summary(model, (7,)))

    x = torch.randn(10, 7)
    print(model(x).shape)
