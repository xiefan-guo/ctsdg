import torch
import torch.nn as nn

from models.discriminator.structure_branch import EdgeDetector, StructureBranch
from models.discriminator.texture_branch import TextureBranch


class Discriminator(nn.Module):

    def __init__(self, image_in_channels, edge_in_channels):
        super(Discriminator, self).__init__()

        self.texture_branch = TextureBranch(in_channels=image_in_channels)
        self.structure_branch = StructureBranch(in_channels=edge_in_channels)
        self.edge_detector = EdgeDetector()

    def forward(self, output, gray_image, real_edge, is_real):

        if is_real == True:

            texture_pred = self.texture_branch(output)
            fake_edge = self.edge_detector(output)
            structure_pred = self.structure_branch(torch.cat((real_edge, gray_image), dim=1))

        else:

            texture_pred = self.texture_branch(output)
            fake_edge = self.edge_detector(output)
            structure_pred = self.structure_branch(torch.cat((fake_edge, gray_image), dim=1))

        return torch.cat((texture_pred, structure_pred), dim=1), fake_edge
        

