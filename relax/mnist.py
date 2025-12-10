import os
import tvm
from tvm import relax
import numpy as np

def create_relax_model_with_weights(weights):
    bb = relax.BlockBuilder()
    batch_size = 1

    with bb.function("main"):
        x = relax.Var("x", relax.TensorStructInfo((batch_size, 1, 28, 28), "float32"))
        y = relax.Var("y", relax.TensorStructInfo((batch_size, 1, 120, 120,), "float32"))

        with bb.dataflow():
            conv1_weight_const = relax.const(weights['conv1_weight'], "float32")
            conv1_bias_const = relax.const(weights['conv1_bias'], "float32")

            conv1_out = bb.emit(relax.op.nn.conv2d(
                x, conv1_weight_const,
                strides=(1, 1),
                padding=(2, 2),
                data_layout="NCHW",
                kernel_layout="OIHW",
            ))
            
            conv1_bias_reshaped = bb.emit(relax.op.reshape(conv1_bias_const, (1, 6, 1, 1)))
            conv1_bias_add = bb.emit(relax.op.add(conv1_out, conv1_bias_reshaped))
            conv1_relu = bb.emit(relax.op.nn.relu(conv1_bias_add))
            pool1 = bb.emit(relax.op.nn.max_pool2d(
                conv1_relu,
                pool_size=(2, 2),
                strides=(2, 2),
                padding=(0, 0),
                layout="NCHW"
            ))

            conv2_weight_const = relax.const(weights['conv2_weight'], "float32")
            conv2_bias_const = relax.const(weights['conv2_bias'], "float32")

            conv2_out = bb.emit(relax.op.nn.conv2d(
                pool1, conv2_weight_const,
                strides=(1, 1),
                padding=(0, 0),
                data_layout="NCHW",
                kernel_layout="OIHW"
            ))
            conv2_bias_reshaped = bb.emit(relax.op.reshape(conv2_bias_const, (1, 16, 1, 1)))
            conv2_add = bb.emit(relax.op.add(conv2_out, conv2_bias_reshaped))
            conv2_relu = bb.emit(relax.op.nn.relu(conv2_add))
            pool2 = bb.emit(relax.op.nn.max_pool2d(
                conv2_relu,
                pool_size=(2, 2),
                strides=(2, 2),
                padding=(0, 0),
                layout="NCHW"
            ))

            flatten = bb.emit(relax.op.reshape(pool2, (batch_size, 400)))

            
            fc1_weight_const = relax.const(weights['fc1_weight'], "float32")
            fc1_bias_const = relax.const(weights['fc1_bias'], "float32")

            fc1_weight_t = bb.emit(relax.op.permute_dims(fc1_weight_const))
            fc1_out = bb.emit(relax.op.matmul(flatten, fc1_weight_t))
            fc1_add = bb.emit(relax.op.add(fc1_out, fc1_bias_const))
            fc1_relu = bb.emit(relax.op.nn.relu(fc1_add))

            fc2_weight_const = relax.const(weights['fc2_weight'], "float32")
            fc2_bias_const = relax.const(weights['fc2_bias'], "float32")

            fc2_weight_t = bb.emit(relax.op.permute_dims(fc2_weight_const))
            fc2_out = bb.emit(relax.op.matmul(fc1_relu, fc2_weight_t))
            fc2_add = bb.emit(relax.op.add(fc2_out, fc2_bias_const))
            fc2_relu = bb.emit(relax.op.nn.relu(fc2_add))

            fc3_weight_const = relax.const(weights['fc3_weight'], "float32")
            fc3_bias_const = relax.const(weights['fc3_bias'], "float32")

            fc3_weight_t = bb.emit(relax.op.permute_dims(fc3_weight_const))
            fc3_out = bb.emit(relax.op.matmul(fc2_relu, fc3_weight_t))
            fc3_add = bb.emit(relax.op.add(fc3_out, fc3_bias_const))
            
            output = bb.emit_output(fc3_add)

        bb.emit_func_output(output, params=[x])

    return bb.get(); 


def build_and_export(weights):
    mod = create_relax_model_with_weights(weights)

    print(mod.script()[:2000] + "\n... (truncated)")

    target = tvm.target.Target("llvm -mtriple=arm64-apple-macos")
    ex = relax.build(mod, target)

    output_path = "mnist_model_copy.so"
    ex.export_library(output_path)
    print(f"Model exported to : {output_path}")

def main():
    weight_file = "mnist_weights.npz"
    
    if os.path.exists(weight_file):
        print(f"Loading pretrained weights from {weight_file}...")

    data = np.load(weight_file)
    weights = {key: data[key] for key in  data.files}
    ex = build_and_export(weights)





if __name__ == "__main__":
    main()
