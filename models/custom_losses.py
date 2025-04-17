import torch
import numpy as np
from torch import nn, Tensor
import torch.distributed as dist
import torch.nn.functional as F
from typing import Optional


class MSELossWithMask(nn.MSELoss):
    def forward(self, inp: Tensor, tgt: Tensor,
                keep_mask: Optional[Tensor] = None,
                use_exp: Optional[bool] = False) -> Tensor:
        """
        Args:
            inp: (B, T, 1)
            tgt: (B, T)
            keep_mask: (B, T) index of inputs to be keeped, other inputs are ignored
            use_exp: use exponential loss
        """
        assert self.reduction == 'none', f'No reduction should be applied when using {self.__str__()}'

        inp = inp.squeeze(-1)
        if use_exp:
            inp = torch.exp(inp)

        res = super().forward(inp, tgt)

        if keep_mask is None:
            return res.mean()

        res *= keep_mask
        res = res.sum() / keep_mask.sum()
        return res


class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_output = None
        self.async_batch_center = None

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_output, teacher_temp):
        self.apply_center_update()
        # teacher centering and sharpening
        return F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_iterations=3):
        teacher_output = teacher_output.float()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        Q = torch.exp(teacher_output / teacher_temp).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] * world_size  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q.t()

    def forward(self, student_output_list, teacher_out_softmaxed_centered_list):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        # TODO: Use cross_entropy_distribution here
        total_loss = 0
        for s in student_output_list:
            lsm = F.log_softmax(s / self.student_temp, dim=-1)
            for t in teacher_out_softmaxed_centered_list:
                loss = torch.sum(t * lsm, dim=-1)
                total_loss -= loss.mean()
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        self.reduce_center_update(teacher_output)

    @torch.no_grad()
    def reduce_center_update(self, teacher_output):
        self.updated = False
        self.len_teacher_output = len(teacher_output)
        self.async_batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if dist.is_initialized():
            self.reduce_handle = dist.all_reduce(self.async_batch_center, async_op=True)

    @torch.no_grad()
    def apply_center_update(self):
        if self.updated is False:
            world_size = dist.get_world_size() if dist.is_initialized() else 1

            if self.reduce_handle is not None:
                self.reduce_handle.wait()
            _t = self.async_batch_center / (self.len_teacher_output * world_size)

            self.center = self.center * self.center_momentum + _t * (1 - self.center_momentum)

            self.updated = True


class CustomDinoLoss(DINOLoss):
    def __init__(self, out_dim=128, ncrops=4, warmup_teacher_temp=0.03,
                 teacher_temp=0.03, warmup_teacher_temp_epochs=0, student_temp=0.1, 
                 center_momentum=0.9, nepochs=1) -> None:
        super().__init__(out_dim=out_dim,
                         ncrops=ncrops,
                         warmup_teacher_temp=warmup_teacher_temp,
                         teacher_temp=teacher_temp,
                         warmup_teacher_temp_epochs=warmup_teacher_temp_epochs,
                         student_temp=student_temp,
                         center_momentum=center_momentum,
                         nepochs=nepochs)
        self.cuda()
    
    def forward(self, inp: Tensor, tgt: Tensor):
        """
        Args:
            inp: (B, N, T, C)
            tgt: (B, N, T, C)
        """
        super().forward(inp, tgt)


class SmoothL1LossWithMask(nn.SmoothL1Loss):
    def forward(self, inp: Tensor, tgt: Tensor,
                keep_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            inp: (B, T, C)
            tgt: (B, T, C)
            keep_mask: (B, T) index of inputs to be keeped, other inputs are ignored
            use_exp: use exponential loss
        """
        assert self.reduction == 'none', f'No reduction should be applied when using {self.__str__()}'

        res = super().forward(inp, tgt)

        if keep_mask is None:
            return res.mean()

        if keep_mask.dim() < inp.dim():
            keep_mask = torch.broadcast_to(keep_mask.unsqueeze(dim=-1), inp.shape)

        assert keep_mask.dim() == inp.dim()
        res *= keep_mask
        res = res.sum() / keep_mask.sum()
        return res
