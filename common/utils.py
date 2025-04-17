import numpy
import torch
from torch import Tensor
from typing import List, Optional, Tuple, Union
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import logging
import joblib


joblib.parallel_backend('multiprocessing')


def pad_sequence(
        sequences: List[Tensor],
        padding_value: Optional[float] = 0.0,
        mask_padding_value: Optional[int] = 0,
        return_mask: Optional[bool] = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Pad a list of variable length Tensors with ``padding_value``

    Args:
        sequences: a list of tensors
        padding_value: value to be padded
        mask_padding_value: value for paddings in mask
        return_mask: if padding mask (0 for pad) should be returned

    Returns:
        Tensor with paddings ``(B, T, *)`` w/wo mask ``(B, T, *)``
    """
    num = len(sequences)
    max_len = max([s.shape[0] for s in sequences])
    out_dims = (num, max_len, *sequences[0].shape[1:])
    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)

    mask, mask_value = None, 1 - mask_padding_value
    if return_mask:
        mask = sequences[0].data.new(*out_dims[:2]).fill_(mask_padding_value)

    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        out_tensor[i, :length] = tensor

        if return_mask:
            mask[i, :length] = mask_value

    if return_mask:
        return out_tensor, mask

    return out_tensor


class CustomKNeighborsClassifier:
    def __init__(self, embedding, label, n_neighbors=1, metric='cosine'):
        self.knc = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, n_jobs=20)
        assert type(embedding) in [torch.Tensor, numpy.ndarray], f'Invalid embedding type: {type(embedding)}'
        if isinstance(embedding, torch.Tensor):
            embedding = np.asarray(embedding.cpu().detach().numpy())
        if isinstance(label, torch.Tensor):
            label = np.asarray(label.cpu().detach().numpy())
        # embedding = np.nan_to_num(embedding)
        # label = np.nan_to_num(label)
        self.knc.fit(embedding, label)

    def __call__(self, embedding):
        assert type(embedding) in [torch.Tensor, numpy.ndarray], f'Invalid embedding type: {type(embedding)}'
        if isinstance(embedding, torch.Tensor):
            embedding = np.asarray(embedding.cpu().detach().numpy())
        return self.knc.predict(embedding)


def cfg2dict(cfg):
    """Convert a cfg node to a dictionary recursively"""
    out = {}
    for key, val in cfg.items():
        if isinstance(val, dict):
            out[key] = cfg2dict(val)
        else:
            out[key] = val
    return out


def get_logger(fpath: Optional[str] = None, level: Optional[str] = 'info',
               ) -> logging.Logger:
    handlers = [logging.StreamHandler()]
    if fpath is not None:
        handlers.append(logging.FileHandler(f'{fpath}/logs.log'))

    level_dict = {'info': logging.INFO, 'debug': logging.DEBUG}
    logging.basicConfig(
            format='%(asctime)s %(levelname)s %(message)s',
            datefmt='%H:%M:%S',
            level=level_dict[level],
            handlers=handlers,
    )

    return logging.getLogger('Root')


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule
