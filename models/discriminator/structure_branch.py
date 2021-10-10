import torch
import torch.nn as nn

from utils.misc import weights_init, spectral_norm


class StructureBranch(nn.Module):

    def __init__(self, in_channels=4, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(StructureBranch, self).__init__()

        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm)
        )

        if init_weights:
            self.apply(weights_init())
        
    def forward(self, edge):

        edge_pred = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(edge)))))
        
        if self.use_sigmoid:
            edge_pred = torch.sigmoid(edge_pred)

        return edge_pred


class EdgeDetector(nn.Module):

    def __init__(self, in_channels=3, mid_channels=16, out_channels=1):
        super(EdgeDetector, self).__init__()

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.res_layer = nn.Sequential(
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels)
        )
        self.relu = nn.ReLU(inplace=True)
        self.out_layer = nn.Sequential(
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, image):
        
        image = self.projection(image)
        edge = self.res_layer(image)
        edge = self.relu(edge + image)
        edge = self.out_layer(edge)

        return edge
