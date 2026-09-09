"""
Cross-Modal Semantic Alignment Regularization (CMASR).

CMASR aligns the multi-granularity feature streams (global, local, texture)
to a shared noise-related target in the latent space via cosine similarity,
constrained by a gating threshold (Eq. 20-21 in the paper):

    L_CMASR = sum_m lambda_m * E[ max(0, tau - cos(f_m, e)) ]

where f_m is the noise prediction of granularity stream m, e is the shared
Gaussian noise sample used during training, and the max(0, .) operation
activates the constraint only when the cosine similarity falls below tau.

Compared to Maximum Mean Discrepancy (MMD) regularization, the cosine
similarity based formulation yields faster convergence and a more stable
optimization process (see the CMASR efficiency analysis in the paper).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CMASR(nn.Module):
    def __init__(self, tau=0.2, lambda_global=0.5, lambda_local=0.5,
                 lambda_texture=0.5):
        super(CMASR, self).__init__()
        self.tau = tau
        self.lambda_global = lambda_global
        self.lambda_local = lambda_local
        self.lambda_texture = lambda_texture

    def _alignment_loss(self, pred, target):
        """cosine-similarity alignment of one granularity stream to the target."""
        cos = F.cosine_similarity(pred, target, dim=1)
        return F.relu(self.tau - cos).mean()

    def forward(self, pred_global, pred_local, pred_texture, target):
        """
        :param pred_global:   noise prediction of the global stream.
        :param pred_local:    noise prediction of the local stream.
        :param pred_texture:  noise prediction of the texture stream.
        :param target:        shared noise sample e used in q_sample.
        :return:              scalar total loss and per-granularity values.
        """
        loss_global = self._alignment_loss(pred_global, target)
        loss_local = self._alignment_loss(pred_local, target)
        loss_texture = self._alignment_loss(pred_texture, target)

        loss = self.lambda_global * loss_global + \
               self.lambda_local * loss_local + \
               self.lambda_texture * loss_texture
        details = {
            "cmasr_global": loss_global.detach(),
            "cmasr_local": loss_local.detach(),
            "cmasr_texture": loss_texture.detach(),
        }
        return loss, details
