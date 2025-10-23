import os

import tvm
from tvm.script import tir as T
from tvm import relax

'''
Relax-level operator
'''
def create_relax_model():
    """Create a Relax model with convolution operations including ReLU."""
    bb = relax.BlockBuilder()

    # Input: (batch=1, channels=3, height=224, width=224)
    batch_size = 1
    in_channels = 3
    out_channels = 16
    height = 224
    width = 224
    kernel_size = 3

    with bb.function("conv_compute"):
        # Input tensor: NCHW format
        x = relax.Var("x", relax.TensorStructInfo((batch_size, in_channels, height, width), "float32"))
        # Convolution weight: (out_channels, in_channels, kernel_h, kernel_w)
        weight = relax.Var("weight", relax.TensorStructInfo((out_channels, in_channels, kernel_size, kernel_size), "float32"))
        # Bias -> set only type
        bias = relax.Var("bias", relax.TensorStructInfo((out_channels,), "float32"))

        with bb.dataflow():
            conv_out = bb.emit(relax.op.nn.conv2d(
                x,
                weight,
                strides=(1, 1),
                padding=(1, 1),  # same padding
                dilation=(1, 1),
                groups=1,
                data_layout="NCHW",
                kernel_layout="OIHW"
            ))

            # Add bias: reshape bias to (1, out_channels, 1, 1) for broadcasting -> set tensor type
            bias_reshaped = bb.emit(relax.op.reshape(bias, (1, out_channels, 1, 1)))
            conv_bias = bb.emit(relax.op.add(conv_out, bias_reshaped))

            # Apply ReLU
            result = bb.emit(relax.op.nn.relu(conv_bias))
            output = bb.emit_output(result)

        bb.emit_func_output(output, params=[x, weight, bias])

    return bb.get()


if __name__ == "__main__":
    relax_model = create_relax_model()
    