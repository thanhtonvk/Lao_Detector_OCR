import torch
from modules.recognition.printed.modules.vision_model import VisionModel
from modules.recognition.printed.modules.language_model import LanguageModel
from modules.recognition.printed.modules.aligment import Alignment


class Model(torch.nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.vision_model = VisionModel(opt)
        self.language_model = LanguageModel(opt)
        self.aligment = Alignment(opt)

    def forward(self, input, input_pad_mask=None):
        F_v, logit_v = self.vision_model(input, input_pad_mask)

        prob_v = logit_v.softmax(dim=-1).detach()
        pred_v = prob_v.argmax(dim=-1)

        if self.training:
            # pseudo input for language model training
            pred_v.scatter_(
                dim=-1,
                index=torch.randint(low=0, high=self.opt.batch_max_length // 2, size=(pred_v.size(0), 2)).to(pred_v.device),
                src=torch.randint(low=1, high=len(self.opt.symbols), size=pred_v.size()).to(pred_v.device)
            )
        pred_v_pad_mask = pred_v == self.opt.pad_token
        F_l, logit_l = self.language_model(pred_v, pred_v_pad_mask)
        logit_a = self.aligment(F_v, F_l)
        return logit_v.transpose(1, 2), logit_l.transpose(1, 2), logit_a.transpose(1, 2)
