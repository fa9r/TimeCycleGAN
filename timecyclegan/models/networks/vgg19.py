"""
VGG19 implementation to be used as perceptual loss in GAN training
Taken (and slightly adjusted) from https://github.com/NVIDIA/pix2pixHD/blob/master/models/networks.py
"""

from torch import nn
from torchvision import models


class VGGLoss(nn.Module):
    """Perceptual loss using VGG19"""
    def __init__(self):
        """"""
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i, feature in enumerate(x_vgg):
            loss += self.weights[i] * self.criterion(feature, y_vgg[i].detach())
        return loss


class Vgg19(nn.Module):
    """VGG19 network"""
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for i in range(2):
            self.slice1.add_module(str(i), vgg_pretrained_features[i])
        for i in range(2, 7):
            self.slice2.add_module(str(i), vgg_pretrained_features[i])
        for i in range(7, 12):
            self.slice3.add_module(str(i), vgg_pretrained_features[i])
        for i in range(12, 21):
            self.slice4.add_module(str(i), vgg_pretrained_features[i])
        for i in range(21, 30):
            self.slice5.add_module(str(i), vgg_pretrained_features[i])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        h_relu1 = self.slice1(input)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
