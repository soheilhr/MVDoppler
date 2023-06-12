import torch
import torch.nn as nn

class MyMobileNet(torch.nn.Module):
    def __init__(self, num_classes):
        super(MyMobileNet, self).__init__()
        self.MobileNet = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        self.MobileNet.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.MobileNet(x)
        return x
    
class MyResNet34(torch.nn.Module):
    def __init__(self, num_classes):
        super(MyResNet34, self).__init__()
        self.resnet34 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
        self.resnet34.fc = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        x = self.resnet34(x)
        return x

class MyEfficientNet(torch.nn.Module):
    def __init__(self, num_classes):
        super(MyEfficientNet, self).__init__()
        self.efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
        self.efficientnet.classifier.fc = nn.Linear(in_features=1280, out_features=num_classes)
        
    def forward(self, x):
        x = self.efficientnet(x)
        return x