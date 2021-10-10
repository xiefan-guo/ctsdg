import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.misc import extract_patches


class RAL(nn.Module):
    '''Region affinity learning.'''

    def __init__(self, kernel_size=3, stride=1, rate=2, softmax_scale=10.):
        super(RAL, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.rate = rate
        self.softmax_scale = softmax_scale

    def forward(self, background, foreground):
        
        # accelerated calculation 
        foreground = F.interpolate(foreground, scale_factor=1. / self.rate, mode='bilinear', align_corners=True)

        foreground_size, background_size = list(foreground.size()), list(background.size())

        background_kernel_size = 2 * self.rate
        background_patches = extract_patches(background, kernel_size=background_kernel_size, stride=self.stride * self.rate)
        background_patches = background_patches.view(background_size[0], -1, 
            background_size[1], background_kernel_size, background_kernel_size)
        background_patches_list = torch.split(background_patches, 1, dim=0)

        foreground_list = torch.split(foreground, 1, dim=0)
        foreground_patches = extract_patches(foreground, kernel_size=self.kernel_size, stride=self.stride)
        foreground_patches = foreground_patches.view(foreground_size[0], -1,
            foreground_size[1], self.kernel_size, self.kernel_size)
        foreground_patches_list = torch.split(foreground_patches, 1, dim=0)

        output_list = []
        padding = 0 if self.kernel_size == 1 else 1
        escape_NaN = torch.FloatTensor([1e-4])
        if torch.cuda.is_available():
            escape_NaN = escape_NaN.cuda()

        for foreground_item, foreground_patches_item, background_patches_item in zip(
            foreground_list, foreground_patches_list, background_patches_list
        ):

            foreground_patches_item = foreground_patches_item[0]
            foreground_patches_item_normed = foreground_patches_item / torch.max(
                torch.sqrt((foreground_patches_item * foreground_patches_item).sum([1, 2, 3], keepdim=True)), escape_NaN)

            score_map = F.conv2d(foreground_item, foreground_patches_item_normed, stride=1, padding=padding)
            score_map = score_map.view(1, foreground_size[2] // self.stride * foreground_size[3] // self.stride,
                foreground_size[2], foreground_size[3])
            attention_map = F.softmax(score_map * self.softmax_scale, dim=1)
            attention_map = attention_map.clamp(min=1e-8)

            background_patches_item = background_patches_item[0]
            output_item = F.conv_transpose2d(attention_map, background_patches_item, stride=self.rate, padding=1) / 4.
            output_list.append(output_item)

        output = torch.cat(output_list, dim=0)
        output = output.view(background_size)
        return output


class MSFA(nn.Module):
    '''Multi-scale feature aggregation.'''

    def __init__(self, in_channels=64, out_channels=64, dilation_rate_list=[1, 2, 4, 8]):
        super(MSFA, self).__init__()

        self.dilation_rate_list = dilation_rate_list

        for _, dilation_rate in enumerate(dilation_rate_list):

            self.__setattr__('dilated_conv_{:d}'.format(_), nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilation_rate, padding=dilation_rate),
                nn.ReLU(inplace=True))
            )

        self.weight_calc = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, len(dilation_rate_list), 1),
            nn.ReLU(inplace=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):

        weight_map = self.weight_calc(x)

        x_feature_list =[]
        for _, dilation_rate in enumerate(self.dilation_rate_list):
            x_feature_list.append(
                self.__getattr__('dilated_conv_{:d}'.format(_))(x)
            )
        
        output = weight_map[:, 0:1, :, :] * x_feature_list[0] + \
                 weight_map[:, 1:2, :, :] * x_feature_list[1] + \
                 weight_map[:, 2:3, :, :] * x_feature_list[2] + \
                 weight_map[:, 3:4, :, :] * x_feature_list[3]

        return output


class CFA(nn.Module):
    '''Contextual Feature Aggregation.'''

    def __init__(self, 
        kernel_size=3, stride=1, rate=2, softmax_scale=10.,
        in_channels=64, out_channels=64, dilation_rate_list=[1, 2, 4, 8]):
        super(CFA, self).__init__()

        self.ral = RAL(kernel_size=kernel_size, stride=stride, rate=rate, softmax_scale=softmax_scale)
        self.msfa = MSFA(in_channels=in_channels, out_channels=out_channels, dilation_rate_list=dilation_rate_list)
        
    def forward(self, background, foreground):

        output = self.ral(background, foreground)
        output = self.msfa(output)
        
        return output