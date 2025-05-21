import torch
import torch.nn as nn
from torchsummary import summary
from torchvision import models


def make_backbone_squeezenet(input_channels=10):
    squeezenet = models.squeezenet1_0()
    squeezenet.features[0] = nn.Conv2d(input_channels, 96, kernel_size=7, stride=2)
    squeezenet_features = squeezenet.features
    squeezenet_features = nn.Sequential(*list(squeezenet_features.children()))
    fc_layer = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU()
    )
    squeezenet_modified = nn.Sequential(
        squeezenet_features,
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        fc_layer
    )
    return squeezenet_modified


if __name__ == "__main__":
    resnet50 = models.resnet50()
    mobile_net = models.mobilenet_v2()
    squeeze_net = make_backbone_squeezenet()

    input_shape = (10, 128, 128)
    # summary(resnet50.cuda(), input_shape)
    summary(squeeze_net.cuda(), input_shape)
    # input_tensor = torch.randn((1, 10, 128, 128))
    # input_tensor.cuda()
    # output = squeeze_net(input_tensor)
    # print(output.size())

