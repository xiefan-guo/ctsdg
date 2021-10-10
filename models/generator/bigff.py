import torch
import torch.nn as nn


class BiGFF(nn.Module):
    '''Bi-directional Gated Feature Fusion.'''
    
    def __init__(self, in_channels, out_channels):
        super(BiGFF, self).__init__()

        self.structure_gate = nn.Sequential(
            nn.Conv2d(in_channels=in_channels + in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.texture_gate = nn.Sequential(
            nn.Conv2d(in_channels=in_channels + in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.structure_gamma = nn.Parameter(torch.zeros(1))
        self.texture_gamma = nn.Parameter(torch.zeros(1))

    def forward(self, texture_feature, structure_feature):

        energy = torch.cat((texture_feature, structure_feature), dim=1)

        gate_structure_to_texture = self.structure_gate(energy)
        gate_texture_to_structure = self.texture_gate(energy)

        texture_feature = texture_feature + self.texture_gamma * (gate_structure_to_texture * structure_feature)
        structure_feature = structure_feature + self.structure_gamma * (gate_texture_to_structure * texture_feature)

        return torch.cat((texture_feature, structure_feature), dim=1)