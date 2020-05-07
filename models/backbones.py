# --------------------------------------------------------
# Written by Hamadi Chihaoui at 11:54 AM 5/7/2020
# --------------------------------------------------------

from torch import nn
from torchvision import models
from torchsummary import summary


def vgg_extractor(pretrained= True):
    vgg = models.vgg16(pretrained=pretrained)
    return nn.Sequential(*list(nn.Sequential.children(vgg))[:1])

def resnet18_extractor(pretrained= True):
    resnet18 = models.resnet18(pretrained=pretrained)
    return nn.Sequential(*list(nn.Sequential.children(resnet18))[:-2])

if __name__ == '__main__':
    vgg = resnet18_extractor()
    summary(vgg, (3, 224,224))




