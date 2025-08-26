import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Reduce


# Simple Lambda wrapper for arbitrary functions
class Lambda(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x)


def millify(n, bytes=False, return_float=False):
    n = float(n)
    if bytes:
        millnames = ["B", "KB", "MB", "GB", "TB", "PB"]
    else:
        millnames = ["", "K", "M", "B", "T"]
    millidx = max(
        0,
        min(
            len(millnames) - 1,
            int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3)),
        ),
    )
    if return_float:
        return n / 10 ** (3 * millidx)
    else:
        return f"{int(n / 10 ** (3 * millidx))}{millnames[millidx]}"


class Network(nn.Module):
    """A network that takes architectural modules and wraps them with a stem and a head."""

    def __init__(self, backbone, backbone_output_shape, output_shape, config):
        super(Network, self).__init__()
        self.config = config
        self.backbone = backbone
        if "einspace" in self.config["search_space"]:
            self.stem = nn.Sequential(
                # conv stem to even number of channels?
                # positional embedding?
                nn.Identity()
            )
        elif self.config["search_space"] == "hnasbench201":
            self.stem = nn.Sequential(
                nn.LazyConv2d(16, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(16),
            )

        if len(backbone_output_shape) == 2:
            self.head = nn.Sequential(
                nn.Linear(backbone_output_shape[1], output_shape),
            )
        elif len(backbone_output_shape) == 3:
            if config["dataset"] in ["darcyflow", "psicov", "cosmic"]:
                self.head = nn.Sequential(
                    # On the wide ResNet backbone, we add an adaptive averaging pooling operation to upsample the features back to their original dimensions before output.
                    nn.AdaptiveAvgPool2d(output_shape),
                    # add in another dimension to match the input
                    nn.Lambda(lambda x: x.unsqueeze(1)),
                )
            else:
                self.head = nn.Sequential(
                    Reduce("b s d -> b s", "mean"),
                    nn.Linear(backbone_output_shape[1], output_shape),
                )
        elif len(backbone_output_shape) == 4:
            if config["dataset"] in ["darcyflow", "psicov", "cosmic"]:
                self.head = nn.Sequential(
                    # On the wide ResNet backbone, we add an adaptive averaging pooling operation to upsample the features back to their original dimensions before output.
                    nn.AdaptiveAvgPool2d(output_shape),
                    # change the channels down to match the input
                    nn.Conv2d(backbone_output_shape[1], 1, kernel_size=1, stride=1, padding=0),
                )
            else:
                self.head = nn.Sequential(
                    Reduce("b c h w -> b c", "mean"),
                    nn.Linear(backbone_output_shape[1], output_shape),
                )
        self.backbone_output_shape = backbone_output_shape
        # print(f"Network")
        # print(self)

    def forward(self, x):
        # print("input to stem", x.shape)
        out = self.stem(x)
        # print("input to backbone", out.shape)
        out = self.backbone(out)
        # print("input to head", out.shape, self.backbone_output_shape)
        out = self.head(out)
        # print("output", out.shape)
        return out

    def forward_window(self, x, L=128, stride=-1):
        _, _, _, s_length = x.shape

        if stride == -1:  # Default to window size
            stride = L
            assert (s_length % L == 0)

        y = torch.zeros_like(x)[:, :1, :, :]
        # print("x, y", x.shape, y.shape)
        counts = torch.zeros_like(x)[:, :1, :, :]
        for i in range((((s_length - L) // stride)) + 1):
            ip = i * stride
            for j in range((((s_length - L) // stride)) + 1):
                jp = j * stride
                out = self.forward(x[:, :, ip:ip + L, jp:jp + L])
                # print("x slice, out", x[:, :, ip:ip + L, jp:jp + L].shape, out.shape)
                # out = out.permute(0, 3, 1, 2).contiguous()
                # print("y slice, out", y[:, :, ip:ip + L, jp:jp + L].shape, out.shape)
                y[:, :, ip:ip + L, jp:jp + L] += out
                counts[:, :, ip:ip + L, jp:jp + L] += torch.ones_like(out)
        return y / counts

    def numel(self):
        num_params = sum([p.numel() for p in self.parameters()])
        return num_params

    def num_parameters(self):
        return f"Num params: {millify(self.numel())}"


class UNetFromStackedCells(nn.Module):
    def __init__(self, input_shape, stacked_cells: nn.ModuleList, num_classes=1):
        """
        Args:
            stacked_cells: nn.ModuleList of k identical stacked cells
                           (each is itself an nn.Module).
            input_shape: tuple including batch dimension
            num_classes: number of output channels for final segmentation
        """
        super().__init__()
        self.k = len(stacked_cells)

        # Make encoder
        self.encoder = nn.ModuleList(stacked_cells)

        # Infer shapes at each encoder output
        self.out_channels = self._infer_out_shapes(input_shape, stacked_cells)

        # Make decoder (upsample layers)
        self.decoder = nn.ModuleList()
        self.post_concat_convs = nn.ModuleList()
        for enc_shape, dec_input_shape, dec_output_shape in zip(
            self.encoder, reversed(self.out_channels), reversed(self.out_channels[:-1])
        ):
            # Upsample module
            self.decoder.append(self._make_upsample_layer(dec_input_shape, dec_output_shape))

            # Post-concat conv to reduce channels back to decoder output channels
            C_in = dec_output_shape[0] + dec_output_shape[0]  # concat channels
            C_out = dec_output_shape[0]
            self.post_concat_convs.append(nn.Conv2d(C_in, C_out, kernel_size=3, padding=1))

        # Final 1x1 conv to get desired number of classes
        self.final = nn.Conv2d(self.out_channels[0][0], num_classes, kernel_size=1)

    def _infer_out_shapes(self, input_shape, stacked_cells):
        x = torch.randn(*input_shape)
        out_shapes = []
        for cell in stacked_cells:
            x = cell(x)
            out_shapes.append(x.shape[1:])  # ignore batch dimension
        return out_shapes

    def _make_upsample_layer(self, input_shape, output_shape):
        """
        Returns a module that reshapes and/or upsamples a tensor from input_shape to output_shape.
        Handles mismatched ranks by reshaping and adjusting channels.
        
        Args:
            input_shape: tuple, e.g., (C_in, H_in, W_in) or (C_in, D_in)
            output_shape: tuple, e.g., (C_out, H_out, W_out) or (C_out, D_out)
        
        Returns:
            nn.Module mapping tensor of shape (B, *input_shape) to (B, *output_shape)
        """
        
        C_in, *in_spatial = input_shape
        C_out, *out_spatial = output_shape
        
        layers = []

        # 1. Adjust channels if needed
        if C_in != C_out:
            if len(in_spatial) == 1:
                layers.append(nn.Conv1d(C_in, C_out, kernel_size=1))
            elif len(in_spatial) == 2:
                layers.append(nn.Conv2d(C_in, C_out, kernel_size=1))
            else:
                raise NotImplementedError("Unsupported spatial dimensions: {}".format(len(in_spatial)))

        # 2. Handle rank mismatch by reshaping
        if len(in_spatial) != len(out_spatial):
            def reshape_fn(x):
                B = x.shape[0]
                # flatten or unsqueeze spatial dims to match rank
                return x.view(B, C_out, *([1]*len(out_spatial)))
            layers.append(Lambda(reshape_fn))
            in_spatial = [1]*len(out_spatial)

        # 3. Upsample spatial dimensions if needed
        if in_spatial != out_spatial:
            mode = 'trilinear' if len(out_spatial) == 3 else 'bilinear'
            layers.append(nn.Upsample(size=out_spatial, mode=mode, align_corners=False))

        if len(layers) == 1:
            return layers[0]
        else:
            return nn.Sequential(*layers)

    def forward(self, x):
        skips = []
        # Encoder
        for i, enc in enumerate(self.encoder):
            # print(f"Layer {i}: Input shape : {x.shape}")
            x = enc(x)
            skips.append(x)
            # print(f"Layer {i}: Output shape: {x.shape}")

        # Decoder
        for i, (up, skip) in enumerate(zip(self.decoder, reversed(skips[:-1]))):
            # print(f"Layer {len(self.encoder) + i}: Input shape : {x.shape}")
            x = up(x)
            # Ensure spatial dimensions match (just in case)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = self.post_concat_convs[i](x)
            # print(f"Layer {len(self.encoder) + i}: Output shape : {x.shape}")

        # Final layer
        x = self.final(x)
        return x
