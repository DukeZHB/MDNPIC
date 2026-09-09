"""
Multi-scale texture branch of the MGCG network (Eq. 4 in the paper).

The whole pathology image is passed through three convolutional blocks with
different kernel scales (3x3, 5x5, 7x7). The 5x5 and 7x7 stages use residual
blocks, and a squeeze-and-excitation style channel attention re-weights the
multi-scale texture features before the final fully connected layer that
produces the texture guidance prior.
"""
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Pre-activation style residual block used inside the texture branch."""

    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = (nn.Conv2d(in_channels, out_channels, 1)
                         if in_channels != out_channels else nn.Identity())

    def forward(self, x):
        return nn.functional.relu(self.conv(x) + self.shortcut(x))


class TextureNetwork(nn.Module):
    """Multi-scale texture extractor producing the texture guidance prior."""

    def __init__(self, experiment_parameters):
        super(TextureNetwork, self).__init__()
        in_channels = experiment_parameters.get("in_channels", 3)
        num_classes = experiment_parameters["num_classes"]

        self.texture_extractor = nn.Sequential(
            # Block 1: 3x3 scale, 64 channels
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 5x5 scale, 128 channels (residual)
            ResidualBlock(64, 128, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 7x7 scale, 256 channels (residual)
            ResidualBlock(128, 256, kernel_size=7, padding=3),

            # channel attention (SE-style)
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(256, 256 // 16, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256 // 16, 256, 1),
                nn.Sigmoid(),
            ),

            # feature re-weighting, pooling and classification head
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x_original):
        """
        :param x_original: (N, C, H, W) pathology image batch.
        :return:           (N, num_classes) texture guidance prior.
        """
        return self.texture_extractor(x_original)
