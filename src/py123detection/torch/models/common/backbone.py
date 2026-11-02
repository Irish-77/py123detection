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
    """Wrapper for Deformable Convolution.

    In a nutshell, deformable convolutions are a type of convolutional layer
    that allows the weights to be deformed by learning a set of offsets and
    masks. This can be used to learn a more complex and adaptive convolutional
    layer.
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
        """Initialize the DCNWrapper.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride of the convolution.
            padding (int): Padding of the convolution.
            dilation (int): Dilation of the convolution.
            groups (int): Number of groups for grouped convolution.
            bias (bool): Whether to use bias.
        """

        super(DCNWrapper, self).__init__()

        # Learnable offset and mask
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
        """Forward pass.

        Args:
            x (Tensor): Input tensor (BN, C, H, W)

        Returns:
            Tensor: Output tensor (BN, C, H, W)
        """
        out = self.offset_conv(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return self.dcn(x, offset, mask)


class CPFPN(nn.Module):
    """CPFPN: Feature Pyramid Network with Checkpointing support and specific
    convolution logic.

    Adapted from PETR official implementation
    (projects/mmdet3d_plugin/models/necks/cp_fpn.py).
    Differences from standard FPN:
    - Only applies 3x3 convolution to the highest resolution output
        (first level).
    - Lower resolution feature maps (from backbone) are passed through 1x1
        lateral convs but NOT 3x3 smoothed.
    - Supports gradient checkpointing (though we handle that in backbone/
        transformer wrappers).
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
        Args:
            in_channels (List[int]): Number of input channels per scale.
            out_channels (int): Number of output channels (used at each scale)
            num_outs (int): Number of output scales.
            start_level (int): Index of the start input backbone level.
            end_level (int): Index of the end input backbone level.
            add_extra_convs (bool): Whether to add extra conv layers
                (not used in default PETR).
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

            # Official PETR CPFPN only adds 3x3 conv for the FIRST level
            # (i==0 relative to start_level).
            # Other levels just use the lateral output directly
            if i == 0:
                fpn_conv = nn.Conv2d(
                    out_channels, out_channels, kernel_size=3, padding=1
                )
                self.fpn_convs.append(fpn_conv)
            else:
                # Placeholder to keep indexing simple
                # In official repo, they just don't append to fpn_convs for
                # i > 0
                pass

    def forward(self, inputs: List[Tensor]) -> List[Tensor]:
        """Forward pass.

        Args:
            inputs (List[Tensor]): List of backbone features.

        Returns:
            List[Tensor]: FPN outputs.
        """
        # Build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # Build top-down path
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

    Feature extractor based on ResNet-101/50 with FPN and Deformable
    Convolutions.
    Aligns with official PETR implementation using CPFPN logic.
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
        """Initialize the FeatureExtractor.

        Args:
            out_channels (int): Number of output channels.
            return_layers (dict, optional): Mapping of layer names to return
                keys.Defaults to {'layer2': '0', 'layer3': '1', 'layer4': '2'}.
            backbone (str): Backbone name ('resnet50' or 'resnet101').
            train_backbone (bool): Whether to train the backbone.
            frozen_stages (int): Stages to freeze.
            norm_eval (bool): Whether to force BN layers to eval mode.
            with_cp (bool): Whether to use gradient checkpointing on backbone.
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

        # Return features
        if return_layers is None:
            # Default used by DETR3D (C3, C4, C5) -> 512, 1024, 2048
            return_layers = {'layer2': '0', 'layer3': '1', 'layer4': '2'}
            in_channels_list = [512, 1024, 2048]
        else:
            # Infer in_channels_list based on layers
            in_channels_list = []
            layer_dims = {
                'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048
            }
            # return_layers keys must be sorted to ensure order matches FPN
            # expectation
            # IntermediateLayerGetter returns OrderedDict based on insertion
            # order (which is iteration order)
            # So we rely on the caller passing return_layers in correct order
            # or sorted here
            sorted_layers = sorted(return_layers.keys())
            for name in sorted_layers:
                if name in layer_dims:
                    in_channels_list.append(layer_dims[name])
                else:
                    raise ValueError(f"Unknown layer {name}")

        self.body = IntermediateLayerGetter(
            self.backbone, return_layers=return_layers
        )

        # Replace torchvision FPN with CPFPN (Checkpoint FPN) logic
        self.fpn = CPFPN(
            in_channels=in_channels_list,
            out_channels=out_channels,
            # Default to same number of outputs as inputs
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
        """Forward pass.

        Args:
            x (Tensor): Input tensor (BN, C, H, W)

        Returns:
            List[Tensor]: List of tensors (FPN outputs)
        """
        # IntermediateLayerGetter returns an OrderedDict
        x_dict = self.body(x)
        # Convert to list of tensors for CPFPN
        # Ensure correct order based on layer sorting if needed,
        # but OrderedDict preserves insertion
        # which depends on iteration order in IntermediateLayerGetter,
        # which depends on return_layers
        x_list = list(x_dict.values())

        features = self.fpn(x_list)
        # print("DONE EXTRACTING FEATURES")
        return features

    def train(self, mode=True):
        """Override train to ensure BN stays in eval mode if norm_eval is True.
        """
        super(FeatureExtractor, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
