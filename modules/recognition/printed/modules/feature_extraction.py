import torch
from modules.recognition.printed.modules.repvgg import create_RepVGG_A2


class ResNet_FeatureExtractor(torch.nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ResNet_FeatureExtractor, self).__init__()
        self.extractor = create_RepVGG_A2(deploy=False, in_channels=input_channels, out_channels=output_channels)

    def forward(self, input):
        return self.extractor(input)
