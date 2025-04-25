import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(x, embedding_dim=32):
    frequencies = torch.exp(
        torch.linspace(
            math.log(1.0),
            math.log(1000.0),
            embedding_dim // 2,
            device=x.device,
        )
    )
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = torch.concat(
        [torch.sin(angular_speeds * x), torch.cos(angular_speeds * x)], dim=-1
    )
    return embeddings


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 1x1 conv to transform the input channels if needed
        self.channel_adaptation = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.block = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        if self.in_channels != self.out_channels:
            x = self.channel_adaptation(x)
        return self.block(x) + x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block_depth=2):
        super().__init__()
        self.block_depth = block_depth

        # First ResidualBlock adjusts in_channels to out_channels
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(in_channels, out_channels)
        ])
        # Other ResidualBlocks keep the same number of channels
        for _ in range(block_depth - 1):
            self.residual_blocks.append(ResidualBlock(out_channels, out_channels))
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        x, skips = x
        for block in self.residual_blocks:
            x = block(x)
            skips.append(x)
        x = self.pool(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block_depth=2):
        super().__init__()
        self.block_depth = block_depth
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        # First ResidualBlock needs to take the upsampled input concatenated with the skip connection
        # The skip connection from the corresponding down block has the same number of channels as the output
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(in_channels + out_channels, out_channels)
        ])
        # Other ResidualBlocks needs to take the output of the previous block concatenated with the skip connection
        for _ in range(block_depth - 1):
            self.residual_blocks.append(ResidualBlock(out_channels + out_channels, out_channels))

    def forward(self, x):
        x, skips = x
        x = self.upsample(x)

        for block in self.residual_blocks:
            skip = skips.pop()  # Last skip connection

            # Padding if the spatial dimensions do not match
            if x.shape[-2:] != skip.shape[-2:]:
                diff_y = skip.shape[-2] - x.shape[-2]
                diff_x = skip.shape[-1] - x.shape[-1]
                x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                              diff_y // 2, diff_y - diff_y // 2])

            x = torch.cat([x, skip], dim=1)  # Concatenate the skip connection
            x = block(x)

        return x


class UNet(nn.Module):
    def __init__(self, in_shape, out_shape, features=None, embedding_dim=64):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512, 1024]
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.features = features
        self.embedding_dim = embedding_dim

        self.embedding_upsample = nn.UpsamplingBilinear2d(size=self.in_shape[1:])
        self.in_conv = nn.Conv2d(self.in_shape[0], features[0], kernel_size=1)

        self.down_blocks = nn.ModuleList([
            # First Block is concatenated with the sinusoidal embedding
            DownBlock(features[0] + embedding_dim, features[0])
        ])
        for i in range(len(features) - 2):  # -2 because the first block is already added
            self.down_blocks.append(DownBlock(features[i], features[i + 1]))

        self.bottleneck = nn.Sequential(
            # First ResidualBlock adjusts the number of channels of the last down block
            ResidualBlock(features[-2], features[-1]),
            ResidualBlock(features[-1], features[-1]),
        )

        self.up_blocks = nn.ModuleList()
        for i in reversed(range(len(features) - 1)):
            # First up block will take the bottleneck output
            self.up_blocks.append(UpBlock(features[i + 1], features[i]))

        self.out_conv = nn.Conv2d(features[0], out_shape[0], kernel_size=1)

    def forward(self, x, t):
        # x are the noisy images
        # t are noise variances

        # Create the sinusoidal embedding
        embedding = sinusoidal_embedding(t, self.embedding_dim).to(x.device)
        embedding = embedding.view(-1, self.embedding_dim, 1, 1)
        embedding = self.embedding_upsample(embedding)

        # Bring the input to the right shape and concatenate the embedding
        x = self.in_conv(x)
        x = torch.cat([x, embedding], dim=1)

        # List to store the skip connections
        skips = []

        # Downward pass
        for block in self.down_blocks:
            x = block([x, skips])

        # Bottleneck
        x = self.bottleneck(x)

        # Upward pass
        for block in self.up_blocks:
            x = block([x, skips])

        # Bring the output to the image shape
        x = self.out_conv(x)

        return x
