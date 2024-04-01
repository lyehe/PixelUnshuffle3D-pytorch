import torch
import torch.nn as nn


class PixelShuffle3d(nn.Module):
    """
    This class is a 3d version of pixelshuffle.
    """

    def __init__(self, scale: int = 2):
        """
        :param scale: upsample scale
        """
        super().__init__()
        self.scale = scale

    def forward(self, input: torch.Tensor):
        if input.dim() != 5:
            raise ValueError(f"Input tensor must be 5D , but got {input.dim()}")
        scale = self.scale
        batch, in_channels, z, x, y = input.shape
        out_channels = in_channels // (scale**3)
        if in_channels % (scale**3) != 0:
            raise ValueError(
                f"Input channels must be divisible by scale^3, but got {in_channels}"
            )
        out_z = z * scale
        out_x = x * scale
        out_y = y * scale
        input_view = input.contiguous().view(
            batch, out_channels, scale, scale, scale, z, x, y
        )
        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        return output.view(batch, out_channels, out_z, out_x, out_y)


class PixelUnshuffle3d(nn.Module):
    """
    This class is a 3d version of pixelunshuffle.
    """

    def __init__(self, scale: int = 2):
        """
        :param scale: downsample scale
        """
        super().__init__()
        self.scale = scale

    def forward(self, input: torch.Tensor):
        if input.dim() != 5:
            raise ValueError(f"Input tensor must be 5D , but got {input.dim()}")
        scale = self.scale
        batch, in_channels, z, x, y = input.shape
        out_channels = in_channels * (self.scale**3)
        out_z = z // scale
        out_x = x // scale
        out_y = y // scale
        if z % self.scale != 0 or x % self.scale != 0 or y % self.scale != 0:
            raise ValueError(f"Size must be divisible by scale, but got {z}, {x}, {y}")
        input_view = input.contiguous().view(
            batch, in_channels, z, scale, x, scale, y, scale
        )
        output = input_view.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        return output.view(batch, out_channels, out_z, out_x, out_y)
