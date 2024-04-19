import torch


class Alignment(torch.nn.Module):
    def __init__(self, opt):
        super(Alignment, self).__init__()
        self.w_att = torch.nn.Linear(opt.output_channels * 2, opt.output_channels)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(opt.output_channels, len(opt.symbols))
        )
        
    def forward(self, F_v, F_l):
        F = torch.cat([F_l, F_v], dim=-1)
        F_att = torch.sigmoid(self.w_att(F))
        output = F_att * F_v + (1 - F_att) * F_l
        logits = self.classifier(output)
        
        return logits
