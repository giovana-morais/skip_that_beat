import torch
import torch.nn.functional as F


def masked_binary_cross_entropy(pred, target):
    """
    Args:
        pred (torch.Tensor): Model predictions (after sigmoid), shape [batch_size, 1]
        target (torch.Tensor): Ground truth labels, shape [batch_size, 1].
                              `-1` indicates missing values.
    Returns:
        torch.Tensor: Masked loss (only computed where target != -1)
    """
    # 1 where valid, 0 otherwise
    mask = (target != -1).float()
    loss = F.binary_cross_entropy(pred * mask, target * mask)
    return loss
