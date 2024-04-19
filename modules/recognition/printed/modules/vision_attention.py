import torch
from modules.recognition.printed.utils import PositionalEncoding


class PositionAttention(torch.nn.Module):
    def __init__(self, batch_max_length, in_channels):
        super(PositionAttention, self).__init__()
        self.batch_max_length = batch_max_length

        self.k_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(in_channels)
        )

        self.pos_encoder = PositionalEncoding(in_channels, max_len=batch_max_length)
        self.q_project = torch.nn.Linear(in_channels, in_channels)
        self.norm = torch.nn.LayerNorm(normalized_shape=in_channels)

    def forward(self, x, mask=None):
        b, c, _, _ = x.size()
        k, v = x, x  # (b, c, h, w)

        k = self.k_encoder(k)

        zeros = x.new_zeros((self.batch_max_length, b, c))  # (t, b, c)
        q = self.pos_encoder(zeros)  # (t, b, c)
        q = q.permute(1, 0, 2)  # (b, t, c)
        q = self.q_project(q)  # (b, t, c)

        # calculate attention
        attn_scores = torch.bmm(q, k.flatten(2))  # (b, t, h * w)
        attn_scores = attn_scores / (c ** 0.5)
        if mask is not None:
            mask = mask.flatten(2)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        attn_scores = attn_scores.softmax(dim=-1)

        v = v.permute(0, 2, 3, 1).reshape(b, -1, c)  # (b, h * w, c)
        attn_vecs = torch.bmm(attn_scores, v)  # (b, t, c)
        attn_vecs = self.norm(attn_vecs)
        return attn_vecs
