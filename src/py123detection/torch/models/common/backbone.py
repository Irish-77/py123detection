"""The image backbone both detectors run over a camera rig.

A ResNet-50 or ResNet-101 with deformable convolutions in its last two stages, feeding PETR's CPFPN
rather than a standard FPN. This is the configuration the PETR paper reports and the thesis
trained, so it is kept as it is rather than generalised.

Three things here exist for memory and for distributed training rather than for accuracy. Gradient
checkpointing is applied by wrapping each ResNet block's ``forward``, which recomputes activations
instead of storing them. BatchNorm is frozen, as official PETR's ``requires_grad=False`` config
does, because a batch of two frames is too small to estimate statistics from. And early stages can
be frozen outright, which is also what keeps DDP from complaining about parameters that never
receive a gradient.
"""

from typing import List

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from torch import Tensor
from torch.nn import functional as F
from torchvision.models import resnet101
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import DeformConv2d


class DCNWrapper(nn.Module):
    """A deformable convolution that learns its own sampling offsets.

    An ordinary convolution reads a fixed grid of neighbours. This one predicts a per-pixel offset
    and a per-pixel weight for each tap first, so the kernel can follow an object rather than a
    square. PETR uses it in the backbone's last two stages, where the features are coarse enough
    that a rigid grid costs accuracy.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        padding: int,
        dilation: int,
        groups: int,
        bias: bool
    ):
        super(DCNWrapper, self).__init__()

        # One 2D offset and one weight per kernel tap, predicted per pixel.
        self.offset_conv = nn.Conv2d(
            in_channels,
            # 3 * kernel_size^2 for 2D offsets + mask
            3 * 9,
            kernel_size=3,
            stride=stride,
            padding=padding,
            bias=True
        )
        # Initialize offset to zero
        self.offset_conv.weight.data.zero_()
        self.offset_conv.bias.data.zero_()

        self.dcn = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

    def forward(self, x: Tensor) -> Tensor:
        """``(BN, C, H, W)`` in, ``(BN, C, H, W)`` out."""
        out = self.offset_conv(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return self.dcn(x, offset, mask)


class CPFPN(nn.Module):
    """PETR's feature pyramid, which is a standard FPN with most of the smoothing left out.

    A standard FPN runs a 3x3 convolution over every level after the top-down merge. This one does
    it for the highest-resolution level only; the rest are the 1x1 lateral outputs as they come.
    PETR ships it this way (``cp_fpn.py`` in its mmdet3d plugin) and the weights expect it, so the
    missing convolutions are not an omission.
    """
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_outs: int,
        start_level: int = 0,
        end_level: int = -1,
        add_extra_convs: bool = False,
    ):
        """
        :param in_channels: Channel count of each input scale.
        :param out_channels: Channel count every output scale is brought to.
        :param num_outs: How many scales to emit.
        :param start_level: First backbone level to read.
        :param end_level: Last backbone level to read; -1 means all of them.
        :param add_extra_convs: Extra downsampling levels on top. Default PETR uses none.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.start_level = start_level

        if end_level == -1:
            self.backbone_end_level = self.num_ins
        else:
            self.backbone_end_level = end_level

        self.add_extra_convs = add_extra_convs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = nn.Conv2d(in_channels[i], out_channels, kernel_size=1)
            self.lateral_convs.append(l_conv)

            # Official PETR's CPFPN smooths the first level only; the others pass the lateral
            # output through untouched.
            if i == 0:
                fpn_conv = nn.Conv2d(
                    out_channels, out_channels, kernel_size=3, padding=1
                )
                self.fpn_convs.append(fpn_conv)
            else:
                # Nothing to append: fpn_convs stays shorter than the level count, as upstream.
                pass

    def forward(self, inputs: List[Tensor]) -> List[Tensor]:
        """Backbone features in, one tensor per output scale."""
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # Top-down path: each level gains the one above it, upsampled.
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # Upsample top level and add to current level
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                laterals[i], size=prev_shape, mode='nearest'
            )

        # Build outputs
        # part 1: from original levels
        outs = []
        for i in range(used_backbone_levels):
            if i == 0:
                # Apply 3x3 conv to the highest resolution feature map
                outs.append(self.fpn_convs[0](laterals[i]))
            else:
                # For other levels, use the lateral output directly
                # (already fused top-down)
                outs.append(laterals[i])

        return outs


class FeatureExtractor(nn.Module):
    """Feature extractor for 3D object detection.

    The image backbone: a ResNet with deformable convolutions in its last two stages, feeding
    PETR's CPFPN. This is what the official PETR configuration builds.
    """
    def __init__(
        self, 
        out_channels: int = 256,
        return_layers: dict = None,
        backbone='resnet101',
        train_backbone: bool = True,
        # -1: train all, 0: freeze stem, 1: freeze layer1, etc.
        frozen_stages: int = -1,
        # Force BN to be in eval mode
        norm_eval: bool = True,
        with_cp: bool = True,
    ):
        """
        :param out_channels: Channel count of every FPN output.
        :param return_layers: Which ResNet stages to tap, as torchvision's
            ``IntermediateLayerGetter`` wants them. Defaults to C3, C4 and C5, which is what
            DETR3D and PETR use.
        :param backbone: ``resnet50`` or ``resnet101``.
        :param train_backbone: Let the backbone's weights receive gradients.
        :param frozen_stages: How deep to freeze: -1 trains everything, 0 freezes the stem, 1 also
            freezes ``layer1``.
        :param norm_eval: Keep BatchNorm in eval mode even during training.
        :param with_cp: Gradient-checkpoint the ResNet blocks, trading step time for memory.
        """
        super().__init__()

        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        # Using ResNet-101 or ResNet-50 with FPN and Deformable Convolutions
        if backbone == 'resnet101':
            from torchvision.models import ResNet101_Weights
            resnet_model = resnet101(weights=ResNet101_Weights.DEFAULT)
        elif backbone == 'resnet50':
            from torchvision.models import ResNet50_Weights, resnet50
            resnet_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.backbone = resnet_model

        # Apply DCN to layer3 and layer4
        for layer_name in ['layer3', 'layer4']:
            layer = getattr(self.backbone, layer_name)
            for block in layer:
                # Replace the 3x3 convolution (conv2) with a DCN equivalent
                original_conv = block.conv2
                dcn_wrapper = DCNWrapper(
                    original_conv.in_channels,
                    original_conv.out_channels,
                    original_conv.stride,
                    original_conv.padding,
                    original_conv.dilation,
                    original_conv.groups,
                    original_conv.bias is not None
                )

                # Copy original weights to the new DCN layer
                dcn_wrapper.dcn.weight.data.copy_(original_conv.weight.data)
                block.conv2 = dcn_wrapper

        # Apply gradient checkpointing to all ResNet blocks
        if with_cp:
            self._add_checkpoint_to_resnet_blocks()

        # Disable gradient tracking for all BatchNorm layers
        # (matching official PETR config)
        # norm_cfg=dict(type='BN2d', requires_grad=False)
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                for param in m.parameters():
                    param.requires_grad = False

        # Freeze stages logic
        if frozen_stages >= 0:
            # Freeze stem
            if frozen_stages >= 0:
                self.backbone.conv1.eval()
                for param in self.backbone.conv1.parameters():
                    param.requires_grad = False
                self.backbone.bn1.eval()
                for param in self.backbone.bn1.parameters():
                    param.requires_grad = False

            # Freeze layer1
            if frozen_stages >= 1:
                self.backbone.layer1.eval()
                for param in self.backbone.layer1.parameters():
                    param.requires_grad = False

            # Can extend to layer2 etc if needed

        if not train_backbone:
            for n, p in self.backbone.named_parameters():
                p.requires_grad = False

        if return_layers is None:
            # C3, C4, C5, as DETR3D and PETR use.
            return_layers = {'layer2': '0', 'layer3': '1', 'layer4': '2'}
            in_channels_list = [512, 1024, 2048]
        else:
            in_channels_list = []
            layer_dims = {
                'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048
            }
            # IntermediateLayerGetter yields its features in insertion order, so the channel list
            # has to be built in the same order the FPN will see them. Sorting the layer names
            # gives that regardless of how the caller spelled the dict.
            sorted_layers = sorted(return_layers.keys())
            for name in sorted_layers:
                if name in layer_dims:
                    in_channels_list.append(layer_dims[name])
                else:
                    raise ValueError(f"Unknown layer {name}")

        self.body = IntermediateLayerGetter(
            self.backbone, return_layers=return_layers
        )

        # PETR's CPFPN rather than torchvision's FPN; see the class above for the difference.
        self.fpn = CPFPN(
            in_channels=in_channels_list,
            out_channels=out_channels,
            num_outs=len(in_channels_list)
        )

        # To fix find unused parameters
        self.backbone.fc = nn.Identity()
        self.backbone.avgpool = nn.Identity()

    def _add_checkpoint_to_resnet_blocks(self):
        """Wrap ResNet Bottleneck blocks to use gradient checkpointing."""
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(self.backbone, layer_name)
            for block in layer:
                # Store original forward method
                original_forward = block.forward

                # Create checkpointed forward wrapper
                def make_checkpoint_forward(original_fn):
                    def checkpoint_forward(x):
                        if self.training and self.with_cp:
                            return cp.checkpoint(
                                original_fn, x, use_reentrant=False
                            )
                        else:
                            return original_fn(x)
                    return checkpoint_forward

                # Replace forward method
                block.forward = make_checkpoint_forward(original_forward)

    def forward(self, x: Tensor) -> List[Tensor]:
        """``(BN, C, H, W)`` in, one tensor per FPN level out."""
        # The dict is ordered by the stage order the constructor sorted, so this list matches the
        # channel counts the FPN was built with.
        x_dict = self.body(x)
        x_list = list(x_dict.values())

        features = self.fpn(x_list)
        return features

    def train(self, mode=True):
        """Put the module in training mode, keeping BatchNorm frozen if ``norm_eval`` is set."""
        super(FeatureExtractor, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # Only BatchNorm behaves differently between train and eval here.
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
