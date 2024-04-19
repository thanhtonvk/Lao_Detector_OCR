import torch
from torch.nn import MultiheadAttention, TransformerDecoder
from modules.recognition.printed.utils import PositionalEncoding


def _get_activation_fn(activation):
    if activation == "relu":
        return torch.nn.functional.relu
    elif activation == "gelu":
        return torch.nn.functional.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class TransformerDecoderLayer(torch.nn.Module):
    r"""
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to self attention, multihead
            attention and feedforward operations, respectivaly. Otherwise it's done after.
            Default: ``False`` (after).
    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    Alternatively, when ``batch_first`` is ``True``:
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> memory = torch.rand(32, 10, 512)
        >>> tgt = torch.rand(32, 20, 512)
        >>> out = decoder_layer(tgt, memory)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='gelu',
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.dropout3 = torch.nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = torch.nn.functional.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        r"""Pass the inputs (and mask) through the decoder layer.
        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """

        x = tgt
        if self.norm_first:
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            x = self.norm3(x + self._ff_block(x))

        return x

    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        x = self.multihead_attn(x, mem, mem, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class LanguageModel(torch.nn.Module):
    def __init__(self, opt):
        super(LanguageModel, self).__init__()
        self.T = opt.batch_max_length + 1
        self.mapping = torch.nn.Embedding(num_embeddings=len(opt.symbols), embedding_dim=opt.d_model)
        self.token_encoder = PositionalEncoding(opt.d_model, max_len=self.T)
        self.pos_encoder = PositionalEncoding(opt.d_model, max_len=self.T)
        self.q_mapping = torch.nn.Sequential(
            torch.nn.Linear(opt.d_model, opt.d_model, bias=False),
            torch.nn.LayerNorm(normalized_shape=opt.d_model)
        )
        self.model = TransformerDecoder(TransformerDecoderLayer(d_model=opt.d_model, nhead=opt.n_heads), num_layers=opt.n_layers)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(opt.d_model, len(opt.symbols))
        )
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, opt.d_model))


    def forward(self, x, pad_mask=None):
        embed = self.mapping(x) # (b, t, e)
        embed = torch.cat([
            self.cls_token.repeat(embed.size(0), 1, 1), embed
        ], dim=1)

        if pad_mask is not None:
            pad_mask = torch.cat([
                torch.empty(embed.size(0), 1).fill_(True).to(pad_mask.device), pad_mask],
            dim=1)

        embed = embed.transpose(0, 1) # (t + 1, b, e)
        embed = self.token_encoder(embed) # (t + 1, b, e)

        zeros = embed.new_zeros(*embed.size())
        query = self.pos_encoder(zeros) # (t + 1, b, e)
        query = self.q_mapping(query)

        output = self.model(query, embed, memory_mask=None, memory_key_padding_mask=pad_mask) # (t + 1, b, e)
        logit = self.classifier(output)

        return output.transpose(0, 1)[:, :-1, :], logit.transpose(0, 1)[:, :-1, :]