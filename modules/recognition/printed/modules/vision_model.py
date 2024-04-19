import torch
from modules.recognition.printed.modules.feature_extraction import ResNet_FeatureExtractor
from modules.recognition.printed.modules.vision_attention import PositionAttention


class VisionModel(torch.nn.Module):
    def __init__(self, opt):
        super(VisionModel, self).__init__()
        self.opt = opt
        self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channels, opt.output_channels)
        self.Attention = PositionAttention(opt.batch_max_length, opt.output_channels)
        self.Classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(in_features=opt.output_channels, out_features=len(opt.symbols))
        )
        
    def forward(self, x, pad_mask=None):
        feature = self.FeatureExtraction(x)
        output = self.Attention(feature, pad_mask)
        logit = self.Classifier(output)
        return output, logit
