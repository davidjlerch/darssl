"""
Implementation of the transformer model.
Reference:
1. https://buomsoo-kim.github.io/attention/2020/04/21/Attention-mechanism-19.md/
2. http://nlp.seas.harvard.edu/annotated-transformer/#encoder-and-decoder-stacks
3. https://datascience.stackexchange.com/questions/93144/minimal-working-example-or-tutorial-showing-how-to-use-pytorchs-nn-transformerd
"""

import torch
import torch.nn as nn
from typing import Dict, Callable, Tuple, Any, Optional
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1, max_position_embeddings=512):
        super().__init__()
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor):
        """
        :param x: (B, T, C)
        :return: (B, T, C)
        """
        seq_length = x.size(1)
        position_ids = self.position_ids[:, :seq_length]
        x = x + self.position_embeddings(position_ids)
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, orig_dim, hidden_dim, depth=1, num_heads=4, mlp_ratio=4, norm_first=False, activation='gelu',
                 dropout=0.1, use_cls=True, layer_norm_eps=1e-6, max_position_embeddings=512,
                 autoregressive=True, model_type='tafar', embed_mask=0.1, num_classes=None):
        super().__init__()

        self.autoregressive = autoregressive
        self.hidden_dim = hidden_dim
        self.model_type = model_type

        # only one embedding layer, since both inputs for enc and dec are poses
        self.embd_layer = nn.Linear(orig_dim, hidden_dim)

        # cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim)) if use_cls else None
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.embed_mask_prob = embed_mask

        self._knc = dict()

        # positional encoding layers
        self.enc_pe = PositionalEncoding(hidden_dim, dropout, max_position_embeddings)

        mlp_ratio = int(mlp_ratio)
        # Transformer Encoder
        enc_layer = nn.TransformerEncoderLayer(
            hidden_dim, num_heads, hidden_dim * mlp_ratio, dropout,
            activation, layer_norm_eps, batch_first=True, norm_first=norm_first)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)

        if num_classes is not None:
            self.fc = nn.Linear(hidden_dim, num_classes)
        else:
            self.fc = None

        self.init_weights()  # initialization

    def set_knc(self, knc, key="auto"):
        self._knc[key] = knc

    def get_knc(self, key="auto"):
        return self._knc[key]

    def init_weights(self) -> None:
        # Based on Timm
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=.02)
            # nn.init.trunc_normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, module) -> None:
        """Based on Huggingface Bert but use truncated normal instead of normal.
        (Timm used trunc_normal in VisionTransformer)
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.trunc_normal_(module.weight, std=.02)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    @staticmethod
    def select_memory(memory: Tensor, memory_type: str, src_key_padding_mask: Tensor,
                      ) -> Tuple[Tensor, Tensor]:
        assert memory_type in ['cls_only', 'seq_only', 'cls_with_seq']
        if memory_type == 'cls_only':
            memory = memory[:, :1, :]
            memory_key_padding_mask = src_key_padding_mask[:, :1]
        elif memory_type == 'seq_only':
            memory = memory[:, 1:, :]
            memory_key_padding_mask = src_key_padding_mask
        else:
            memory = memory
            memory_key_padding_mask = torch.cat([src_key_padding_mask[:, :1],
                                                 src_key_padding_mask], dim=1)
        return memory, memory_key_padding_mask

    @staticmethod
    def generate_subsequent_mask(sz: int, sz1: Optional[int] = None) -> Tensor:
        """Generate a causal mask (not necessarily square) for the sequence.
        The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
        """
        sz1 = sz1 or sz
        return torch.triu(torch.full((sz, sz1), float('-inf')), diagonal=1)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, src_embd: Tensor,
                        src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        # prepend the cls token for source if needed
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(src_embd.size(0), -1, -1)
            src_embd = torch.cat((cls_token, src_embd), dim=1)

        mask, ids_restore = None, None
        if self.model_type in ['smae']:
            # masking: length -> length * mask_ratio
            src_embd, mask, ids_restore = self.random_masking(src_embd[:, 1:, :], 0.5)

            # add positional embeddings and encoder forwarding
            src_input = self.enc_pe(src_embd)
            # (B, T+1, C) with cls token
            memory = self.encoder(src_input)
        else:
            # add positional embeddings and encoder forwarding
            src_input = self.enc_pe(src_embd)
            # (B, T+1, C) with cls token
            memory = self.encoder(src_input)
        if self.fc is not None:
            memory = self.fc(memory)
        return memory, mask, ids_restore

    def forward(self, src: Tensor, total_frames: Tensor,
                memory_type: str = 'seq_only') -> Tuple[Tensor, Tensor]:
        """ For training

        Args:
            total_frames:
            src: a sequence of feature vector (B, T, C)
            memory_type: one of ['cls_only', 'seq_only', 'cls_with_seq']

        Returns
            a sequence of feature vector reconstructuring the original one
        """
        assert memory_type in ['cls_only', 'seq_only', 'cls_with_seq', 'seq_mean'], \
            f'Invalid memory type: {memory_type}'
        src_embd = self.embd_layer(src)
        memory, _, ids_restore = self.forward_encoder(src_embd)
        if memory_type == 'cls_only':
            memory = memory[:, 0]
        elif memory_type == 'seq_mean':
            mem_mean = torch.zeros(memory[:, 0].shape).to(memory)
            for i in range(len(memory)):
                mem_mean[i] = torch.mean(memory[i, 1:total_frames[i]], dim=0)
            memory = mem_mean.clone()
            del mem_mean
        memory = memory.expand(1, -1, -1)
        memory = memory.permute(1, 0, 2)
        return memory, memory

    def generate(self, src: Tensor, tgt: Tensor,
                 src_key_padding_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None,
                 memory_type: str = 'seq_only') -> Tuple[Tensor, Tensor]:
        """For Testing"""
        assert memory_type in ['cls_only', 'seq_only', 'cls_with_seq'], f'Invalid memory type: {memory_type}'

        max_len = src.size(1)
        if self.model_type in ['bert', 'smae', 'casmae']:
            src = tgt
        src_embd = self.embd_layer(src)
        memory = self.forward_encoder(src_embd, src_key_padding_mask)
        selected_memory, memory_key_padding_mask = self.select_memory(
            memory, memory_type, src_key_padding_mask)

        if self.model_type == 'bert':
            final_out = self.output_layer(memory)
            return memory, final_out
        if self.autoregressive:
            for i in range(max_len - 1):
                tgt_embd = self.embd_layer(tgt)
                dec_out = self.forward_decoder(
                    tgt_embd, selected_memory,
                    memory_key_padding_mask=memory_key_padding_mask,
                    autoregressive=False)
                out = self.output_layer(dec_out)
                tgt = torch.cat([tgt, out[:, -1:]], dim=1)
        else:
            if self.model_type == 'tafar':
                tgt_embd = tgt
            elif self.model_type == 'smae' or self.model_type == 'casmae':
                tgt_embd = memory
            tgt = tgt_embd
        return memory, tgt


if __name__ == '__main__':
    """Minimal example of using TransformerModel autoregressively or non-autoregressively"""
    B, T, C = 64, 75, 75
    hidden_dim = 128
    inputs = torch.randn(B, T, C)

    model_autoreg = TransformerModel(orig_dim=C, hidden_dim=hidden_dim, depth=2, num_heads=4, autoregressive=True)
    model_non = TransformerModel(orig_dim=C, hidden_dim=hidden_dim, depth=2, num_heads=4, autoregressive=False)

    # autoregressive
    src_autoreg = inputs
    start_pose = torch.zeros(B, 1, C)
    tgt_autoreg = torch.cat([start_pose, src_autoreg[:, :-1]], dim=1)
    out_autoreg = model_autoreg(src_autoreg, tgt_autoreg)

    # non-autoregressive
    src_non = inputs
    tgt_non = torch.zeros(B, T, hidden_dim)
    out_non = model_non(src_non, tgt_non)

    print(1)
