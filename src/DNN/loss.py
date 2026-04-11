import torch
import torch.nn.functional as F
import numpy as np


def _fspecial_gaussian3d(size, channel, sigma):
    coords = torch.tensor([(x - (size - 1.0) / 2.0) for x in range(size)])
    coords = -(coords**2) / (2.0 * sigma**2)
    grid = coords.view(1, -1, 1) + coords.view(-1, 1, 1) + coords.view(1, 1, -1)
    grid = grid.view(1, -1)
    grid = grid.softmax(-1)
    kernel = grid.view(1, 1, size, size, size)
    kernel = kernel.expand(channel, 1, size, size, size).contiguous()
    return kernel


def _ssim3d(input, target, max_val, k1, k2, channel, kernel):
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    mu1 = F.conv3d(input, kernel, groups=channel)
    mu2 = F.conv3d(target, kernel, groups=channel)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(input * input, kernel, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(target * target, kernel, groups=channel) - mu2_sq
    sigma12 = F.conv3d(input * target, kernel, groups=channel) - mu1_mu2

    v1 = 2 * sigma12 + c2
    v2 = sigma1_sq + sigma2_sq + c2

    ssim = ((2 * mu1_mu2 + c1) * v1) / ((mu1_sq + mu2_sq + c1) * v2)
    return ssim, v1 / v2


def ms_ssim_loss3d(
    input,
    target,
    max_val,
    filter_size=11,
    k1=0.01,
    k2=0.03,
    sigma=1.5,
    kernel=None,
    weights=None,
    reduction="mean",
):

    if input.size() != target.size():
        raise ValueError(
            "Expected input size ({}) to match target size ({}).".format(
                input.size(0), target.size(0)
            )
        )

    dim = input.dim()
    if dim == 2:
        input = input.expand(1, 1, 1, input.dim(-2), input.dim(-1))
        target = target.expand(1, 1, 1, target.dim(-2), target.dim(-1))
    elif dim == 3:
        input = input.expand(1, 1, input.dim(-3), input.dim(-2), input.dim(-1))
        target = target.expand(1, 1, target.dim(-3), target.dim(-2), target.dim(-1))
    elif dim == 4:
        input = input.expand(
            1, input.dim(-4), input.dim(-3), input.dim(-2), input.dim(-1)
        )
        target = target.expand(
            1, target.dim(-4), target.dim(-3), target.dim(-2), target.dim(-1)
        )
    elif dim != 5:
        raise ValueError("Expected 2, 3, 4, or 5 dimensions (got {})".format(dim))

    _, channel, _, _, _ = input.size()

    if kernel is None:
        kernel = _fspecial_gaussian3d(filter_size, channel, sigma)
    kernel = kernel.to(device=input.device)

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights = torch.tensor(weights, device=input.device)
    weights = weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    levels = weights.size(0)
    mssim = []
    mcs = []
    for _ in range(levels):
        ssim, cs = _ssim3d(input, target, max_val, k1, k2, channel, kernel)
        ssim = ssim.mean((2, 3, 4))
        cs = cs.mean((2, 3, 4))
        mssim.append(ssim)
        mcs.append(cs)

        input = F.avg_pool3d(input, (2, 2, 2))
        target = F.avg_pool3d(target, (2, 2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)
    mssim = (mssim + 1) / 2
    mcs = (mcs + 1) / 2
    p1 = mcs**weights
    p2 = mssim**weights

    ret = torch.prod(p1[:-1], 0) * p2[-1]

    if reduction != "none":
        ret = torch.mean(ret) if reduction == "mean" else torch.sum(ret)
    return ret


class MultiScaleSSIMLoss3d(torch.nn.Module):
    __constants__ = ["filter_size", "k1", "k2", "sigma", "kernel", "reduction"]

    def __init__(
        self, channel=1, filter_size=7, k1=0.01, k2=0.03, sigma=1.5, reduction="mean"
    ):
        super(MultiScaleSSIMLoss3d, self).__init__()
        self.filter_size = filter_size
        self.k1 = k1
        self.k2 = k2
        self.sigma = sigma
        self.kernel = _fspecial_gaussian3d(filter_size, channel, sigma)
        self.reduction = reduction

    def forward(
        self,
        input,
        target,
        weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
        max_val=1.0,
    ):
        return ms_ssim_loss3d(
            input,
            target,
            max_val=max_val,
            k1=self.k1,
            k2=self.k2,
            sigma=self.sigma,
            kernel=self.kernel,
            weights=weights,
            filter_size=self.filter_size,
            reduction=self.reduction,
        )


class SSIM3DLoss(torch.nn.Module):
    def __init__(self):
        super(SSIM3DLoss, self).__init__()
        self.ssim = MultiScaleSSIMLoss3d(channel=1)
        self.name = "SSIM"

    def forward(self, output, target):
        loss = 1 - self.ssim(output, target)
        return loss

    def getLossName(self):
        return self.name
