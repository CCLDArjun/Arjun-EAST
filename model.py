import math

import torch
from torch.nn.modules import padding
from torch.nn.modules.conv import Conv2d
import torchvision.models as models
from torch.nn import functional as F


class VggHeadless(torch.nn.Module):
    """Get the VGG16 model's feature extractor layers."""
    def __init__(self):
        super().__init__()
        vgg = models.vgg16_bn(pretrained=True)
        self.feature_extraction = torch.nn.Sequential(*list(vgg.features))
        self.avgpool = vgg.avgpool

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.avgpool(x)
        return x


class FeatureExtractor(torch.nn.Module):
    """wrapping vgg16 to get intermediate layer outputs"""
    def __init__(self):
        super().__init__()
        vgg = VggHeadless()
        self.features = vgg.feature_extraction
    
    def forward(self, x):
        """returns output of every MaxPool2d layer"""
        int_featuremap = []
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, torch.nn.MaxPool2d):
                int_featuremap.append(x)
        
        return int_featuremap[1:]


# noinspection PyAttributeOutsideInit
class EastOCR(torch.nn.Module):
    def __init__(self, feature_extractor: FeatureExtractor):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.create_feature_merging_layers()
        self.create_output_layers()

    def create_feature_merging_layers(self):
        # Block 1
        self.conv1_1 = torch.nn.Conv2d(1024, 128, (1, 1), bias=True)
        self.bn1_1 = torch.nn.BatchNorm2d(128)
        self.relu1_1 = torch.nn.ReLU()
        self.conv2_1 = torch.nn.Conv2d(128, 128, (3, 3), padding=1, bias=True)
        self.bn2_1 = torch.nn.BatchNorm2d(128)
        self.relu2_1 = torch.nn.ReLU()

        # Block 2 
        self.conv1_2 = torch.nn.Conv2d(384, 64, (1, 1), bias=True)
        self.bn1_2 = torch.nn.BatchNorm2d(64)
        self.relu1_2 = torch.nn.ReLU()
        self.conv2_2 = torch.nn.Conv2d(64, 64, (3, 3), padding=1, bias=True)
        self.bn2_2 = torch.nn.BatchNorm2d(64)
        self.relu2_2 = torch.nn.ReLU()

        # Block 3 
        self.conv1_3 = torch.nn.Conv2d(192, 32, (1, 1), bias=True)
        self.bn1_3 = torch.nn.BatchNorm2d(32)
        self.relu1_3 = torch.nn.ReLU()
        self.conv2_3 = torch.nn.Conv2d(32, 32, (3, 3), padding=1, bias=True)
        self.bn2_3 = torch.nn.BatchNorm2d(32)
        self.relu2_3 = torch.nn.ReLU()

        # Output of feature merging branch
        self.conv1_4 = torch.nn.Conv2d(32, 32, (3, 3), padding=1, bias=True)
        self.bn1_4 = torch.nn.BatchNorm2d(32)
        self.relu1_4 = torch.nn.ReLU()

    def forward(self, x):
        features = self.feature_extractor(x)
        
        # Block 1
        x = F.interpolate(features[3], scale_factor=2, mode="bilinear", align_corners=True)
        x = torch.cat((x, features[2]), dim=1)
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.relu1_1(x)
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.relu2_1(x)

        # Block 2
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = torch.cat((x, features[1]), dim=1)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu1_2(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.relu2_2(x)

        # Block 3
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = torch.cat((x, features[0]), dim=1)
        x = self.conv1_3(x)
        x = self.bn1_3(x)
        x = self.relu1_3(x)
        x = self.conv2_3(x)
        x = self.bn2_3(x)
        x = self.relu2_3(x)

        # Output of feature merging branch
        x = self.conv1_4(x)
        x = self.bn1_4(x)
        x = self.relu1_4(x)

        score_map = self.score_map_conv(x)
        score_map = self.score_map_sigmoid(score_map) * 512

        rbox = self.rbox_conv(x)
        rbox = self.rbox_sigmoid(rbox)
        angle = self.rot_angle_conv(x)
        angle = (self.rot_sigmoid(angle) - 0.5) * math.pi
        geo = torch.cat((rbox, angle), dim=1)
        return score_map, geo
    
    def create_output_layers(self):
        self.score_map_conv = torch.nn.Conv2d(32, 1, (1, 1), bias=True)
        self.score_map_sigmoid = torch.nn.Sigmoid()

        self.rbox_conv = torch.nn.Conv2d(32, 4, (1, 1), bias=True)
        self.rbox_sigmoid = torch.nn.Sigmoid()

        self.rot_angle_conv = torch.nn.Conv2d(32, 1, (1, 1), bias=True)
        self.rot_sigmoid = torch.nn.Sigmoid()



