import torch
from modules.hrnet import hrnet18



class Model(torch.nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.fpn = hrnet18(pretrained=True)
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=270, out_channels=270, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(num_features=270),
            torch.nn.GELU(),
            torch.nn.Conv2d(in_channels=270, out_channels=2 * len(opt.categories), kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        fused_feat = self.fpn(x)
        out_head = self.head(fused_feat)
        P, T = out_head[:, :len(self.opt.categories), :, :], out_head[:, len(self.opt.categories):, :, :]
        B = 1 / (1 + torch.exp(-self.opt.k * (P - T)))

        if self.training:
            return P, T, B
        else:
            return P, T

        

