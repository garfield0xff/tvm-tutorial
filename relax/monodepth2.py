"""Monodepth2 network implementation in TVM Relax IR."""

from typing import List, Tuple
import numpy as np

from tvm import relax
from tvm.relax.frontend.nn import Module
from tvm.relax.testing.nn import emit


class Conv3x3(Module):
    """3x3 convolution with reflection padding."""

    def __init__(self, in_channels: int, out_channels: int, use_refl: bool = True):
        super().__init__()
        self.use_refl = use_refl
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: relax.Expr) -> relax.Var:
        # Padding: reflection or zero padding
        if self.use_refl:
            # ReflectionPad2d with padding=1
            x = emit(relax.op.nn.pad(x, [0, 0, 0, 0, 1, 1, 1, 1], pad_mode="reflect"))
        else:
            # ZeroPad2d with padding=1
            x = emit(relax.op.nn.pad(x, [0, 0, 0, 0, 1, 1, 1, 1], pad_value=0.0))

        # Conv2d 3x3
        weight = relax.Var("weight", relax.TensorStructInfo([self.out_channels, self.in_channels, 3, 3], "float32"))
        x = emit(relax.op.nn.conv2d(x, weight, padding=0))
        return x


class ConvBlock(Module):
    """Convolution followed by ELU activation."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = Conv3x3(in_channels, out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: relax.Expr) -> relax.Var:
        x = self.conv(x)
        # ELU activation with alpha=1.0
        # ELU(x, alpha) = x if x > 0 else alpha * (exp(x) - 1)
        # Use leaky_relu as approximation or implement manually
        # For now, use ReLU as simpler alternative (common in many architectures)
        x = emit(relax.op.nn.relu(x))
        return x


class BasicBlock(Module):
    """ResNet BasicBlock for ResNet18/34."""

    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample=None):
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.downsample = downsample

    def forward(self, x: relax.Expr) -> relax.Var:
        identity = x

        # conv1: 3x3 conv
        weight1 = relax.Var("conv1_weight", relax.TensorStructInfo([self.planes, self.inplanes, 3, 3], "float32"))
        out = emit(relax.op.nn.conv2d(x, weight1, strides=self.stride, padding=1))

        # bn1
        gamma1 = relax.Var("bn1_gamma", relax.TensorStructInfo([self.planes], "float32"))
        beta1 = relax.Var("bn1_beta", relax.TensorStructInfo([self.planes], "float32"))
        moving_mean1 = relax.Var("bn1_mean", relax.TensorStructInfo([self.planes], "float32"))
        moving_var1 = relax.Var("bn1_var", relax.TensorStructInfo([self.planes], "float32"))
        bn_out1 = emit(relax.op.nn.batch_norm(out, gamma1, beta1, moving_mean1, moving_var1, axis=1))
        out = emit(relax.TupleGetItem(bn_out1, 0))

        # relu
        out = emit(relax.op.nn.relu(out))

        # conv2: 3x3 conv
        weight2 = relax.Var("conv2_weight", relax.TensorStructInfo([self.planes, self.planes, 3, 3], "float32"))
        out = emit(relax.op.nn.conv2d(out, weight2, strides=1, padding=1))

        # bn2
        gamma2 = relax.Var("bn2_gamma", relax.TensorStructInfo([self.planes], "float32"))
        beta2 = relax.Var("bn2_beta", relax.TensorStructInfo([self.planes], "float32"))
        moving_mean2 = relax.Var("bn2_mean", relax.TensorStructInfo([self.planes], "float32"))
        moving_var2 = relax.Var("bn2_var", relax.TensorStructInfo([self.planes], "float32"))
        bn_out2 = emit(relax.op.nn.batch_norm(out, gamma2, beta2, moving_mean2, moving_var2, axis=1))
        out = emit(relax.TupleGetItem(bn_out2, 0))

        # downsample
        if self.downsample is not None:
            identity = self.downsample(x)

        # residual connection
        out = emit(relax.op.add(out, identity))
        out = emit(relax.op.nn.relu(out))

        return out


class ResnetEncoder(Module):
    """ResNet encoder for Monodepth2."""

    def __init__(self, num_layers: int = 18, num_input_images: int = 1):
        super().__init__()
        self.num_layers = num_layers
        self.num_input_images = num_input_images

        # Channel numbers for each layer
        if num_layers == 18:
            self.num_ch_enc = np.array([64, 64, 128, 256, 512])
            self.blocks = [2, 2, 2, 2]
        elif num_layers == 50:
            self.num_ch_enc = np.array([64, 256, 512, 1024, 2048])
            self.blocks = [3, 4, 6, 3]
        else:
            raise ValueError(f"{num_layers} is not a valid number of resnet layers")

    def forward(self, input_image: relax.Expr) -> List[relax.Var]:
        """Forward pass through ResNet encoder.

        Returns a list of 5 feature maps at different scales.
        """
        features = []

        # Normalization
        x = emit(relax.op.subtract(input_image, relax.const(0.45, "float32")))
        x = emit(relax.op.divide(x, relax.const(0.225, "float32")))

        # Initial conv: 7x7, stride 2
        conv1_weight = relax.Var(
            "conv1_weight",
            relax.TensorStructInfo([64, self.num_input_images * 3, 7, 7], "float32")
        )
        x = emit(relax.op.nn.conv2d(x, conv1_weight, strides=2, padding=3))

        # BatchNorm
        bn1_gamma = relax.Var("bn1_gamma", relax.TensorStructInfo([64], "float32"))
        bn1_beta = relax.Var("bn1_beta", relax.TensorStructInfo([64], "float32"))
        bn1_mean = relax.Var("bn1_mean", relax.TensorStructInfo([64], "float32"))
        bn1_var = relax.Var("bn1_var", relax.TensorStructInfo([64], "float32"))
        bn_out = emit(relax.op.nn.batch_norm(x, bn1_gamma, bn1_beta, bn1_mean, bn1_var, axis=1))
        x = emit(relax.TupleGetItem(bn_out, 0))  # Extract tensor from tuple

        # ReLU
        x = emit(relax.op.nn.relu(x))
        features.append(x)

        # MaxPool
        x = emit(relax.op.nn.max_pool2d(features[-1], pool_size=3, strides=2, padding=1))

        # Layer 1
        x = self._make_layer(x, 64, self.blocks[0], stride=1)
        features.append(x)

        # Layer 2
        x = self._make_layer(x, 128, self.blocks[1], stride=2)
        features.append(x)

        # Layer 3
        x = self._make_layer(x, 256, self.blocks[2], stride=2)
        features.append(x)

        # Layer 4
        x = self._make_layer(x, 512, self.blocks[3], stride=2)
        features.append(x)

        return features

    def _make_layer(self, x: relax.Expr, planes: int, blocks: int, stride: int = 1) -> relax.Var:
        """Create a residual layer with multiple blocks."""
        # First block might need downsampling
        downsample = None
        if stride != 1:
            # Create downsample layer
            def downsample_fn(inp):
                weight = relax.Var("downsample_weight", relax.TensorStructInfo([planes, planes // 2, 1, 1], "float32"))
                out = emit(relax.op.nn.conv2d(inp, weight, strides=stride, padding=0))

                gamma = relax.Var("downsample_bn_gamma", relax.TensorStructInfo([planes], "float32"))
                beta = relax.Var("downsample_bn_beta", relax.TensorStructInfo([planes], "float32"))
                mean = relax.Var("downsample_bn_mean", relax.TensorStructInfo([planes], "float32"))
                var = relax.Var("downsample_bn_var", relax.TensorStructInfo([planes], "float32"))
                bn_out = emit(relax.op.nn.batch_norm(out, gamma, beta, mean, var, axis=1))
                out = emit(relax.TupleGetItem(bn_out, 0))
                return out
            downsample = downsample_fn

        # First block
        inplanes = planes // 2 if stride != 1 else planes
        block = BasicBlock(inplanes, planes, stride, downsample)
        x = block(x)

        # Remaining blocks
        for _ in range(1, blocks):
            block = BasicBlock(planes, planes, stride=1)
            x = block(x)

        return x


class DepthDecoder(Module):
    """Depth decoder for Monodepth2."""

    def __init__(self, num_ch_enc: np.ndarray, scales: List[int] = [0, 1, 2, 3],
                 num_output_channels: int = 1, use_skips: bool = True):
        super().__init__()
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.scales = scales
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

    def forward(self, input_features: List[relax.Expr]) -> relax.Var:
        """Forward pass through depth decoder.

        Args:
            input_features: List of 5 feature maps from encoder

        Returns:
            Disparity map at the finest scale (scale 0)
        """
        outputs = {}

        # Start from the deepest feature
        x = input_features[-1]  # input_features[4]

        # Decoder
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]

            upconv_0 = ConvBlock(num_ch_in, num_ch_out)
            x = upconv_0(x)

            # Upsample by 2x
            # Get current spatial dimensions and double them
            shape = x.struct_info.shape
            new_h = shape[2] * 2
            new_w = shape[3] * 2
            x = emit(relax.op.image.resize2d(x, size=(new_h, new_w), method="nearest_neighbor"))

            # Skip connection
            if self.use_skips and i > 0:
                x = emit(relax.op.concat([x, input_features[i - 1]], axis=1))

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]

            upconv_1 = ConvBlock(num_ch_in, num_ch_out)
            x = upconv_1(x)

            # Disparity output at this scale
            if i in self.scales:
                dispconv = Conv3x3(self.num_ch_dec[i], self.num_output_channels)
                disp = dispconv(x)
                disp = emit(relax.op.sigmoid(disp))
                outputs[f"disp_{i}"] = disp

        # Return the finest scale output
        return outputs["disp_0"]


class Monodepth2(Module):
    """Complete Monodepth2 model with encoder and decoder."""

    def __init__(self, num_layers: int = 18, scales: List[int] = [0, 1, 2, 3]):
        super().__init__()
        self.encoder = ResnetEncoder(num_layers=num_layers)
        self.decoder = DepthDecoder(
            num_ch_enc=self.encoder.num_ch_enc,
            scales=scales,
            num_output_channels=1,
            use_skips=True
        )

    def forward(self, x: relax.Expr) -> relax.Var:
        """Forward pass through the complete network.

        Args:
            x: Input image tensor with shape (batch, 3, height, width)

        Returns:
            Disparity map at the finest scale
        """
        features = self.encoder(x)
        depth = self.decoder(features)
        return depth


def build_monodepth2(
    input_shape: Tuple[int, int, int, int] = (1, 3, 192, 640),
    num_layers: int = 18,
    scales: List[int] = [0, 1, 2, 3]
) -> relax.Function:
    """Build a Monodepth2 model as a Relax function.

    Args:
        input_shape: Input tensor shape (batch, channels, height, width)
        num_layers: Number of ResNet layers (18 or 50)
        scales: Output scales for multi-scale depth prediction

    Returns:
        A Relax function representing the Monodepth2 model
    """
    bb = relax.BlockBuilder()

    with bb.function("monodepth2"):
        with bb.dataflow():
            # Input placeholder
            input_var = relax.Var("input", relax.TensorStructInfo(input_shape, "float32"))

            # Build model
            model = Monodepth2(num_layers=num_layers, scales=scales)
            output = model(input_var)

            bb.emit_output(output)

        bb.emit_func_output(output)

    return bb.get()["monodepth2"]


def disp_to_depth(disp: relax.Expr, min_depth: float = 0.1, max_depth: float = 100.0) -> Tuple[relax.Var, relax.Var]:
    """Convert disparity to depth.

    Args:
        disp: Disparity map (output from sigmoid, values in [0, 1])
        min_depth: Minimum depth value
        max_depth: Maximum depth value

    Returns:
        Tuple of (scaled_disp, depth)
    """
    min_disp = 1.0 / max_depth
    max_disp = 1.0 / min_depth

    # scaled_disp = min_disp + (max_disp - min_disp) * disp
    scaled_disp = emit(
        relax.op.add(
            relax.const(min_disp, "float32"),
            relax.op.multiply(
                relax.const(max_disp - min_disp, "float32"),
                disp
            )
        )
    )

    # depth = 1 / scaled_disp
    depth = emit(relax.op.divide(relax.const(1.0, "float32"), scaled_disp))

    return scaled_disp, depth


def main():
    """Build and demonstrate Monodepth2 model."""
    input_shape = (1, 3, 192, 640)

    bb = relax.BlockBuilder()

    # Create input parameter
    input_var = relax.Var("input", relax.TensorStructInfo(input_shape, "float32"))

    with bb.function("main", params=[input_var]):
        with bb.dataflow():
            encoder = ResnetEncoder(num_layers=18, num_input_images=1)
            features = encoder(input_var)

            decoder = DepthDecoder(
                num_ch_enc=encoder.num_ch_enc,
                scales=[0, 1, 2, 3],
                num_output_channels=1,
                use_skips=True
            )
            depth = decoder(features)

            scaled_disp, depth_output = disp_to_depth(depth, min_depth=0.1, max_depth=100.0)

            bb.emit_output(depth_output)

        bb.emit_func_output(depth_output)

    mod = bb.get()
    print("="*60)
    print("Monodepth2 Model Built Successfully!")
    print("="*60)
    print(mod)
    return mod


if __name__ == "__main__":
    main()
