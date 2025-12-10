"""
MobileNetV1 TVM Relax Model Builder Example
"""

import numpy as np
import tvm
from tvm import relax


def create_mobilenetv1_model_with_weights(weights):
    """
    """
    bb = relax.BlockBuilder()
    batch_size = 1

    # =========================================================================
    # Helper Function 
    # =========================================================================

    def emit_conv_bn_relu(x, conv_key, bn_key, out_channels, stride=1, padding=1, groups=1):
        """
        Conv2d + BatchNorm + ReLU 블록

        PyTorch:
            nn.Conv2d(..., bias=False)
            nn.BatchNorm2d(...)
            nn.ReLU()

        Args:
            x: Input Tensor
            conv_key: conv weight key (e.g., "model.0.0")
            bn_key: bn weight key (e.g., "model.0.1")
            out_channels: output channel
            stride: convolution stride
            padding: convolution padding
            groups: groups (1=normal conv, in_channels=depthwise conv)
        """
        conv_weight = relax.const(weights[f'{conv_key}.weight'], "float32")

        # Convolution
        conv_out = bb.emit(relax.op.nn.conv2d(
            x, conv_weight,
            strides=(stride, stride),
            padding=(padding, padding),
            groups=groups,
            data_layout="NCHW",
            kernel_layout="OIHW"
        ))

        # BatchNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta
        bn_gamma = relax.const(weights[f'{bn_key}.weight'], "float32")
        bn_beta = relax.const(weights[f'{bn_key}.bias'], "float32")
        bn_mean = relax.const(weights[f'{bn_key}.running_mean'], "float32")
        bn_var = relax.const(weights[f'{bn_key}.running_var'], "float32")

        # reshape for broadcasting: (C,) -> (1, C, 1, 1)
        bn_gamma_r = bb.emit(relax.op.reshape(bn_gamma, (1, out_channels, 1, 1)))
        bn_beta_r = bb.emit(relax.op.reshape(bn_beta, (1, out_channels, 1, 1)))
        bn_mean_r = bb.emit(relax.op.reshape(bn_mean, (1, out_channels, 1, 1)))
        bn_var_r = bb.emit(relax.op.reshape(bn_var, (1, out_channels, 1, 1)))

        eps = relax.const(1e-5, "float32")

        # (x - mean) / sqrt(var + eps) * gamma + beta
        x_centered = bb.emit(relax.op.subtract(conv_out, bn_mean_r))
        var_eps = bb.emit(relax.op.add(bn_var_r, eps))
        std = bb.emit(relax.op.sqrt(var_eps))
        x_norm = bb.emit(relax.op.divide(x_centered, std))
        x_scaled = bb.emit(relax.op.multiply(x_norm, bn_gamma_r))
        bn_out = bb.emit(relax.op.add(x_scaled, bn_beta_r))

        # ReLU
        relu_out = bb.emit(relax.op.nn.relu(bn_out))

        return relu_out

    def emit_depthwise_separable_conv(x, layer_idx, in_channels, out_channels, stride=1):
        """
        Depthwise Separable Convolution 

        PyTorch:
            self.depthwise = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch)
            self.bn1 = nn.BatchNorm2d(in_ch)
            self.pointwise = nn.Conv2d(in_ch, out_ch, 1, 1, 0)
            self.bn2 = nn.BatchNorm2d(out_ch)
        """
        # ===== Depthwise Conv (groups=in_channels) =====
        dw_weight = relax.const(weights[f'model.{layer_idx}.depthwise.weight'], "float32")

        dw_out = bb.emit(relax.op.nn.conv2d(
            x, dw_weight,
            strides=(stride, stride),
            padding=(1, 1),
            groups=in_channels,  # Depthwise: groups = in_channels
            data_layout="NCHW",
            kernel_layout="OIHW"
        ))

        # Depthwise BatchNorm
        dw_bn_gamma = relax.const(weights[f'model.{layer_idx}.bn1.weight'], "float32")
        dw_bn_beta = relax.const(weights[f'model.{layer_idx}.bn1.bias'], "float32")
        dw_bn_mean = relax.const(weights[f'model.{layer_idx}.bn1.running_mean'], "float32")
        dw_bn_var = relax.const(weights[f'model.{layer_idx}.bn1.running_var'], "float32")

        dw_bn_gamma_r = bb.emit(relax.op.reshape(dw_bn_gamma, (1, in_channels, 1, 1)))
        dw_bn_beta_r = bb.emit(relax.op.reshape(dw_bn_beta, (1, in_channels, 1, 1)))
        dw_bn_mean_r = bb.emit(relax.op.reshape(dw_bn_mean, (1, in_channels, 1, 1)))
        dw_bn_var_r = bb.emit(relax.op.reshape(dw_bn_var, (1, in_channels, 1, 1)))

        eps = relax.const(1e-5, "float32")
        dw_centered = bb.emit(relax.op.subtract(dw_out, dw_bn_mean_r))
        dw_var_eps = bb.emit(relax.op.add(dw_bn_var_r, eps))
        dw_std = bb.emit(relax.op.sqrt(dw_var_eps))
        dw_norm = bb.emit(relax.op.divide(dw_centered, dw_std))
        dw_scaled = bb.emit(relax.op.multiply(dw_norm, dw_bn_gamma_r))
        dw_bn_out = bb.emit(relax.op.add(dw_scaled, dw_bn_beta_r))
        dw_relu = bb.emit(relax.op.nn.relu(dw_bn_out))

        # ===== Pointwise Conv (1x1 conv) =====
        pw_weight = relax.const(weights[f'model.{layer_idx}.pointwise.weight'], "float32")

        pw_out = bb.emit(relax.op.nn.conv2d(
            dw_relu, pw_weight,
            strides=(1, 1),
            padding=(0, 0),  # 1x1 conv는 padding 없음
            groups=1,
            data_layout="NCHW",
            kernel_layout="OIHW"
        ))

        # Pointwise BatchNorm
        pw_bn_gamma = relax.const(weights[f'model.{layer_idx}.bn2.weight'], "float32")
        pw_bn_beta = relax.const(weights[f'model.{layer_idx}.bn2.bias'], "float32")
        pw_bn_mean = relax.const(weights[f'model.{layer_idx}.bn2.running_mean'], "float32")
        pw_bn_var = relax.const(weights[f'model.{layer_idx}.bn2.running_var'], "float32")

        pw_bn_gamma_r = bb.emit(relax.op.reshape(pw_bn_gamma, (1, out_channels, 1, 1)))
        pw_bn_beta_r = bb.emit(relax.op.reshape(pw_bn_beta, (1, out_channels, 1, 1)))
        pw_bn_mean_r = bb.emit(relax.op.reshape(pw_bn_mean, (1, out_channels, 1, 1)))
        pw_bn_var_r = bb.emit(relax.op.reshape(pw_bn_var, (1, out_channels, 1, 1)))

        pw_centered = bb.emit(relax.op.subtract(pw_out, pw_bn_mean_r))
        pw_var_eps = bb.emit(relax.op.add(pw_bn_var_r, eps))
        pw_std = bb.emit(relax.op.sqrt(pw_var_eps))
        pw_norm = bb.emit(relax.op.divide(pw_centered, pw_std))
        pw_scaled = bb.emit(relax.op.multiply(pw_norm, pw_bn_gamma_r))
        pw_bn_out = bb.emit(relax.op.add(pw_scaled, pw_bn_beta_r))
        pw_relu = bb.emit(relax.op.nn.relu(pw_bn_out))

        return pw_relu

    # =========================================================================
    # Main Graph
    # =========================================================================

    with bb.function("main"):
        # Input: NCHW format (batch=1, channels=3, height=224, width=224)
        x = relax.Var("x", relax.TensorStructInfo((batch_size, 3, 224, 224), "float32"))

        with bb.dataflow():
            # ========== Initial Conv Block (model.0) ==========
            # Conv 3x3/2: 3x224x224 -> 32x112x112
            out = emit_conv_bn_relu(
                x,
                conv_key="model.0.0",
                bn_key="model.0.1",
                out_channels=32,
                stride=2,
                padding=1,
                groups=1
            )

            # ========== Depthwise Separable Conv Blocks ==========

            # Layer 1: 32x112x112 -> 64x112x112 (stride=1)
            out = emit_depthwise_separable_conv(out, layer_idx=1,
                                                 in_channels=32, out_channels=64, stride=1)

            # Layer 2: 64x112x112 -> 128x56x56 (stride=2)
            out = emit_depthwise_separable_conv(out, layer_idx=2,
                                                 in_channels=64, out_channels=128, stride=2)

            # Layer 3: 128x56x56 -> 128x56x56 (stride=1)
            out = emit_depthwise_separable_conv(out, layer_idx=3,
                                                 in_channels=128, out_channels=128, stride=1)

            # Layer 4: 128x56x56 -> 256x28x28 (stride=2)
            out = emit_depthwise_separable_conv(out, layer_idx=4,
                                                 in_channels=128, out_channels=256, stride=2)

            # Layer 5: 256x28x28 -> 256x28x28 (stride=1)
            out = emit_depthwise_separable_conv(out, layer_idx=5,
                                                 in_channels=256, out_channels=256, stride=1)

            # Layer 6: 256x28x28 -> 512x14x14 (stride=2)
            out = emit_depthwise_separable_conv(out, layer_idx=6,
                                                 in_channels=256, out_channels=512, stride=2)

            # Layers 7-11: 512x14x14 -> 512x14x14 (5x stride=1)
            for i in range(7, 12):
                out = emit_depthwise_separable_conv(out, layer_idx=i,
                                                     in_channels=512, out_channels=512, stride=1)

            # Layer 12: 512x14x14 -> 1024x7x7 (stride=2)
            out = emit_depthwise_separable_conv(out, layer_idx=12,
                                                 in_channels=512, out_channels=1024, stride=2)

            # Layer 13: 1024x7x7 -> 1024x7x7 (stride=1)
            out = emit_depthwise_separable_conv(out, layer_idx=13,
                                                 in_channels=1024, out_channels=1024, stride=1)

            # ========== Global Average Pooling ==========
            # 1024x7x7 -> 1024x1x1
            # TVM Relax에서는 adaptive_avg_pool2d 사용
            pooled = bb.emit(relax.op.nn.adaptive_avg_pool2d(out, output_size=(1, 1)))

            # ========== Flatten ==========
            # 1024x1x1 -> 1024
            flattened = bb.emit(relax.op.reshape(pooled, (batch_size, 1024)))

            # ========== FC Layer ==========
            fc_weight = relax.const(weights['fc.weight'], "float32")
            fc_bias = relax.const(weights['fc.bias'], "float32")

            # FC: (1, 1024) @ (1024, 1000) -> (1, 1000)
            fc_weight_t = bb.emit(relax.op.permute_dims(fc_weight))
            fc_out = bb.emit(relax.op.matmul(flattened, fc_weight_t))
            output = bb.emit(relax.op.add(fc_out, fc_bias))

            final_output = bb.emit_output(output)

        bb.emit_func_output(final_output, params=[x])

    return bb.get()


def main():
    print("=" * 70)
    print("MobileNetV1 TVM Relax Model Builder")
    print("=" * 70)

    weights_file = "mobilenetv1_weights.npz"
    print(f"\n1. Loading weights from {weights_file}...")

    try:
        data = np.load(weights_file)
        weights = {key: data[key] for key in data.files}
        print(f"   Loaded {len(weights)} weight tensors")
    except FileNotFoundError:
        print(f"   Error: {weights_file} not found!")
        print("   Run build_mobileNetv1.py first to generate weights.")
        return
 
    print("\n2. Creating TVM Relax model...")
    mod = create_mobilenetv1_model_with_weights(weights)

    print("\n3. TVM Relax IR (truncated):")
    print("-" * 70)
    ir_script = mod.script()
    print(ir_script[:3000] + "\n... (truncated)")
    print("-" * 70)

    print("\n4. Building model...")
    target = tvm.target.Target("llvm -mtriple=arm64-apple-macos")
    ex = relax.build(mod, target)
    print("   Build successful!")

    print("\n5. Testing inference...")
    vm = relax.VirtualMachine(ex, tvm.cpu())

    # 랜덤 입력
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    x_tvm = tvm.runtime.tensor(dummy_input)

    result = vm["main"](x_tvm)
    output = result.numpy()

    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min():.4f}, {output.max():.4f}]")
 
    top5_indices = np.argsort(output[0])[-5:][::-1]
    print(f"   Top-5 class indices: {top5_indices}")

    output_path = "mobilenetv1_model.so"
    ex.export_library(output_path)
    print(f"\n6. Model exported to: {output_path}")

if __name__ == "__main__":
    main()
