from abc import abstractmethod
import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ldm.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from ldm.modules.attention import SpatialTransformer
from ldm.util import exists


# dummy replace
def convert_module_to_f16(x):
    pass

def convert_module_to_f32(x):
    pass


## go
class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5)
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def __init__(self, *args):
        super(TimestepEmbedSequential, self).__init__(*args)
        self.adjust_channels = nn.Conv2d(in_channels=1, out_channels=11, kernel_size=1)
        self.layer = nn.Conv2d(in_channels=8, out_channels=192, kernel_size=3, padding=1)

    def forward(self, x, emb, context=None):

        print(f"Input shape before conv3d: {x.shape}")
        if x.shape[1] != 11:
            x = self.adjust_channels(x)
        # Ensure the input tensor has the correct dimensions
        if len(x.shape) == 3:
            # Add batch and channel dimensions if missing
            x = x.unsqueeze(0).unsqueeze(0)
        elif len(x.shape) == 4:
            # Add batch dimension if missing
            x = x.unsqueeze(0)
        # Adjust the number of channels if necessary
        if x.dim() == 2:
            x = x.repeat(1, 4)  # Adjust for 2D tensor
        elif x.dim() == 3:
            x = x.repeat(1, 4, 1)  # Adjust for 3D tensor
        elif x.dim() == 4:
            # Add batch dimension if missing
            x = x.unsqueeze(0)
        elif x.dim() == 5:
            x = x.repeat(1, 4, 1, 1, 1)  # Adjust for 5D tensor
        elif x.dim() == 6:
            #x = x.repeat(1, 4, 1, 1, 1, 1)  # Adjust for 6D tensor
            x = x.squeeze(1).squeeze(1)
        else:
            raise ValueError(f"Unsupported tensor dimension: {x.dim()}")
        
        x = x.unsqueeze(0)
        # Adjust the number of channels if necessary
        if x.shape[1] == 1:
            if x.dim() == 2:
                x = x.repeat(1, 4)  # Adjust for 2D tensor
            elif x.dim() == 3:
                x = x.repeat(1, 4, 1)  # Adjust for 3D tensor
            elif x.dim() == 4:
                # Add batch dimension if missing
                x = x.unsqueeze(0)
            elif x.dim() == 5:
                x = x.repeat(1, 4, 1, 1, 1)  # Adjust for 5D tensor
            elif x.dim() == 6:
                #x = x.repeat(1, 4, 1, 1, 1, 1)  # Adjust for 6D tensor
                x = x.squeeze(1).squeeze(1)
            else:
                raise ValueError(f"Unsupported tensor dimension: {x.dim()}")
            
        print(f"Input shape after adjustment: {x.shape}")

        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, emb, context)
            elif isinstance(layer, nn.AvgPool3d):
                # Adjust kernel size if input dimensions are smaller
                kernel_size = (min(2, x.shape[2]), min(2, x.shape[3]), min(2, x.shape[4]))
                if kernel_size == (1, 1, 1):
                    raise RuntimeError(f"Input image dimensions are too small for avg_pool3d: {x.shape[2:]}")
                x = F.avg_pool3d(x, kernel_size)
            else:
                # Assuming x is the input tensor with shape [1, 40, 11, 640, 400]
                # Remove the extra dimension
                # Check if the tensor has an extra dimension and remove it if necessary
                if len(x.shape) == 5:
                    x = x.squeeze(2)  # Remove the third dimension (index 2)
                # Print the shape of the input tensor before passing it through the layer
                print(f"Shape before layer: {x.shape}")

                # Calculate the total number of elements in the input tensor
                total_elements = x.numel()
                # Define the target shape
                #new_shape = (total_elements // (11 * 640 * 400), 11, 640, 400)
                #new_shape = (32, 10, 640, 400)
                new_shape = (10, 10, 640, -1)
                #x = x.view(new_shape)
                print(f"New tensor shape: {x.shape}")

                # Calculate the appropriate size for the last dimension
                desired_dims = (10, 10, 640)
                remaining_elements = x.numel()

                product = desired_dims[0] * desired_dims[1] * desired_dims[2]

                if remaining_elements % product != 0:
                    # Find the largest divisor for the last dimension
                    new_dim = remaining_elements // product
                    if new_dim == 0:
                        raise RuntimeError("Cannot reshape tensor with the given desired dimensions.")
                    new_shape = desired_dims + (new_dim,)
                else:
                    new_shape = desired_dims + (remaining_elements // product,)
                new_dim = remaining_elements // product
                new_shape = desired_dims + (new_dim,)

                print(f"remaining_elements: {remaining_elements}")
                print(f"new_shape: {new_shape}")
            
                # Ensure the new shape's product matches the total number of elements
                assert remaining_elements == new_shape[0] * new_shape[1] * new_shape[2] * new_shape[3], "Shape mismatch"

                x = x.view(new_shape)
                print(f"New tensor shape: {x.shape}")
                # print(f"Original shape: {x.shape}")
                # print(f"Total elements: {total_elements}")
                # print(f"New shape: {new_shape}")

                # product_new_shape = 1
                # for dim in new_shape:
                #     product_new_shape *= dim
                # print(f"Product of new_shape dimensions: {product_new_shape}")
                # if total_elements != product_new_shape:
                #     raise ValueError("Total elements do not match the product of new_shape dimensions.")



                # Calculate the expected number of elements for the target shape
                # expected_elements = 1
                # for dim in new_shape:
                #     if dim != -1:
                #         expected_elements *= dim

                # Check if the reshaping is possible
                # if total_elements != new_shape:
                #     raise ValueError(f"Cannot reshape tensor of total size {total_elements} to shape {new_shape}")

                # # Reshape the tensor
                # x = x.view(new_shape)

                layer = nn.Conv2d(in_channels=10, out_channels=192, kernel_size=3, stride=1, padding=1)
                layer = layer.cuda()
                x = layer(x)
                # Print the shape of the tensor after passing it through the layer
                print(f"Shape after layer: {x.shape}")

        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2]*2, x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class TransposedUpsample(nn.Module):
    'Learned 2x upsampling without padding'
    def __init__(self, channels, out_channels=None, ks=5):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.up = nn.ConvTranspose2d(self.channels,self.out_channels,kernel_size=ks,stride=2)

    def forward(self,x):
        return self.up(x)


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None,padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.kernel_size = (2,2)
        self.pool = th.nn.AvgPool3d(kernel_size=(2, 2, 2))
        #self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=self.kernel_size)
        #self.conv = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=2, stride=1, padding=0)
        self.conv = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=2, stride=1, padding=0)
        stride = 2 if dims != 3 else (2, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forwardOld(self, x):
        print(f"x.shape: {x.shape}")
        print(f"self.channels: {self.channels}")

        # Check input dimensions
        #if x.size(2) < 2 or x.size(3) < 2 or x.size(4) < 2:
        #    raise ValueError("Input dimensions are smaller than the kernel size.")
        
        if x.size(2) < self.kernel_size or x.size(3) < self.kernel_size:
            raise ValueError("Input dimensions are smaller than the kernel size.")
        
        if x.shape[1] != self.channels:
            print(f"Adjusting self.channels from {self.channels} to {x.shape[1]}")
            self.channels = x.shape[1]
        assert x.shape[1] == self.channels

        # Apply pooling
        x = self.pool(x)
        return self.op(x)
    
    def forward(self, x):
        kernel_size = (min(2, x.shape[2]), min(2, x.shape[3]), min(2, x.shape[4]))
        print(f"Input shape: {x.shape}, kernel_size: {kernel_size}")
        kernel_depth, kernel_height, kernel_width = kernel_size
        
        # Ensure kernel size is smaller than input dimensions
        if x.shape[2] <= kernel_depth:
            kernel_depth = x.shape[2] - 1
        if x.shape[3] <= kernel_height:
            kernel_height = x.shape[3] - 1
        if x.shape[4] <= kernel_width:
            kernel_width = x.shape[4] - 1
        
        kernel_size = (kernel_depth, kernel_height, kernel_width)
        print(f"Adjusted kernel_size: {kernel_size}")
        
        if x.shape[2] <= kernel_depth or x.shape[3] <= kernel_height or x.shape[4] <= kernel_width:
            raise ValueError("Input dimensions are smaller than the kernel size.")
        
        if x.dim() == 5:
            # Reshape the tensor to remove the extra dimension
            x = x.squeeze(2)  # Remove the third dimension (index 2)
        return self.conv(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.adjust_channels = nn.Conv3d(in_channels=128, out_channels=512, kernel_size=1)
        self.reduce_channels = nn.Conv3d(in_channels=512, out_channels=128, kernel_size=1)
        self.in_conv = nn.Conv3d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=1)
        #self.in_conv = nn.Conv3d(in_channels=1, out_channels=128, kernel_size=3, padding=1)
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """

        if x.shape[1] != 128:
            x = self.in_conv(x)
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        kernel_size = (min(2, x.shape[2]), min(2, x.shape[3]), min(2, x.shape[4]))
        print(f"Input shape: {x.shape}, kernel_size: {kernel_size}")
        kernel_depth, kernel_height, kernel_width = kernel_size

        # Ensure kernel size is smaller than input dimensions
        if x.shape[2] <= kernel_depth:
            kernel_depth = x.shape[2] - 1
        if x.shape[3] <= kernel_height:
            kernel_height = x.shape[3] - 1
        if x.shape[4] <= kernel_width:
            kernel_width = x.shape[4] - 1
        if x.shape[2] <= kernel_depth or x.shape[3] <= kernel_height or x.shape[4] <= kernel_width:
            raise ValueError(f"Input dimensions {x.shape[2:]}, are smaller than or equal to the kernel size {kernel_size}.")
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            # Assuming 'h' is the input tensor
            print(h.shape)  # Should print torch.Size([1, 1, 512, 639, 399])

            # Define a 1x1 convolution to increase the number of channels from 1 to 128
            conv1x1 = nn.Conv3d(1, 128, kernel_size=1)

            # Ensure `h` is on the same device as `conv1x1`
            device = next(conv1x1.parameters()).device
            h = h.to(device)

            # Apply the 1x1 convolution
            h = conv1x1(h)

            # Now 'h' should have the shape [1, 128, 512, 639, 399]
            print(h.shape)

            # Ensure the input tensor is on the same device as the weight tensor
            h = h.to(in_conv.weight.device)
            h = in_conv(h)
        else:
            print(f"Input shape before adjust_channels: {x.shape}")
            x = self.adjust_channels(x)
            print(f"Input shape before reduce_channels: {x.shape}")
            x = self.reduce_channels(x)
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        print(x.shape)
        return checkpoint(self._forward, (x,), self.parameters(), True)   # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
        #return pt_checkpoint(self._forward, x)  # pytorch

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None
    ):
        #super().__init__()
        super(UNetModel, self).__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        #self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")  # todo: convert to warning

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        print('using attention in resolution', ds, 'res block', nr)
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or i < num_attention_blocks[level]:
                        print('using attention in resolution', ds, 'res block', i)
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads_upsample,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa
                            )
                        )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
                # Move the model to the device
                self.to(self.device)

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
            normalization(ch),
            conv_nd(dims, model_channels, n_embed, 1),
            #nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps=None, context=None, y=None,**kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []

        # Ensure x is on the correct device
        x = x.to(self.device)

        # Ensure timesteps is not None before calling timestep_embedding
        if timesteps is None:
            timesteps = th.tensor([0])
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        t_emb = t_emb.to(self.device)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)

        # total_elements = h.numel()
        # #target_shape = (10, 10, 640, 400)
        # target_shape = (10, 10, 640, 440)
        # # Print the number of elements and the target shape
        # print(f"Number of elements in h: {total_elements}")
        # print(f"Target shape: {target_shape}")
        # print(f"Product of target shape dimensions: {np.prod(target_shape)}")

        # # # Ensure the target shape is valid
        # if total_elements != np.prod(target_shape):
        #     raise ValueError(f"Invalid target shape {target_shape} for input of size {total_elements}")
        
        new_shape = (10, 10, 640, -1)
        x = x.view(new_shape)
        print(f"New tensor shape: {x.shape}")

        # target_shapes = [
        #     (10, 10, 640, 400)
        # ]
        # # Check if any of the target shapes match the total elements
        # valid_shape = None
        # for shape in target_shapes:
        #     expected_elements = shape[0] * shape[1] * shape[2] * shape[3]
        #     if total_elements == expected_elements:
        #         valid_shape = shape
        #         break
        
        # if valid_shape is None:
        #     raise ValueError(f"Invalid target shape for input of size {total_elements}")

        # h = h.view(valid_shape)

        #input_tensor = input_tensor.view(batch_size, channels, height, width)
        
        # if total_elements != total_elements_target_shape:
        #     raise ValueError(f"Cannot reshape tensor of total size {total_elements} to shape {target_shape}")
        



        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        # print(h.shape)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)
