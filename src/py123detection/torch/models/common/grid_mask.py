import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GridMask(nn.Module):
    """Grid masking for data augmentation (pure PyTorch, no NumPy/PIL)."""
    def __init__(
        self,
        use_h: bool,
        use_w: bool,
        rotate: int = 1,
        offset: bool = False,
        ratio: float = 0.5,
        mode: int = 0,
        prob: float = 1.
    ):
        """Initialize GridMask.

        Args:
            use_h: Whether to mask horizontally
            use_w: Whether to mask vertically
            rotate: Maximum rotation angle (degrees)
            offset: Whether to add offset
            ratio: Ratio of masked area
            mode: Mode of masking (0=mask grid, 1=keep grid)
            prob: Probability of masking
        """
        super(GridMask, self).__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob

    def set_prob(self, epoch: int, max_epoch: int):
        """Set the probability of masking.

        Args:
            epoch: Current epoch
            max_epoch: Maximum epoch
        """
        self.prob = self.st_prob * epoch / max_epoch

    def forward(self, x: Tensor):
        """Apply grid masking to image.

        Args:
            x: Image to mask (B*N_cams, C, H, W)

        Returns:
            Masked image
        """
        if torch.rand(1).item() > self.prob or not self.training:
            return x
        n, c, h, w = x.size()
        x = x.view(-1, h, w)

        hh = int(1.5 * h)
        ww = int(1.5 * w)

        d = int(torch.randint(2, h, (1,)))
        length = min(max(int(d * self.ratio + 0.5), 1), d - 1)

        # Build grid mask using pure PyTorch arange + modulo
        st_h = int(torch.randint(0, d, (1,)))
        st_w = int(torch.randint(0, d, (1,)))

        # Create 1D stripe patterns: 1 where visible, 0 where masked
        # Within each period of length d, the first `length` positions are
        # masked (0), the rest are visible (1).
        if self.use_h:
            h_coords = torch.arange(hh, device=x.device)
            h_stripe = ((h_coords + st_h) % d >= length).float()
        else:
            h_stripe = torch.ones(hh, device=x.device)

        if self.use_w:
            w_coords = torch.arange(ww, device=x.device)
            w_stripe = ((w_coords + st_w) % d >= length).float()
        else:
            w_stripe = torch.ones(ww, device=x.device)

        # Combine into 2D mask via outer product
        # mask[i, j] = 1 only if both h_stripe[i] and w_stripe[j] are 1
        # (i.e., the pixel is visible through both horizontal and vertical
        # stripes)
        mask = h_stripe.unsqueeze(1) * w_stripe.unsqueeze(0)  # (hh, ww)

        # Apply rotation using affine_grid + grid_sample
        r = int(torch.randint(0, max(self.rotate, 1), (1,)))
        if r != 0:
            angle_rad = r * math.pi / 180.0
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            # Rotation matrix (2x3 affine)
            theta = torch.tensor([
                [cos_a, -sin_a, 0.0],
                [sin_a,  cos_a, 0.0],
            ], dtype=torch.float32, device=x.device).unsqueeze(0)

            grid = F.affine_grid(
                theta, [1, 1, hh, ww], align_corners=False
            )
            mask_4d = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, hh, ww)
            mask = F.grid_sample(
                mask_4d, grid, mode='nearest', padding_mode='zeros',
                align_corners=False
            ).squeeze(0).squeeze(0)

        # Crop to original size (center crop from the larger mask)
        y_start = (hh - h) // 2
        x_start = (ww - w) // 2
        mask = mask[y_start:y_start + h, x_start:x_start + w]

        if self.mode == 1:
            mask = 1 - mask
        mask = mask.expand_as(x)
        if self.offset:
            offset_tensor = (
                2 * torch.rand(h, w, device=x.device) - 1
            )
            x = x * mask + offset_tensor * (1 - mask)
        else:
            x = x * mask

        return x.view(n, c, h, w)
