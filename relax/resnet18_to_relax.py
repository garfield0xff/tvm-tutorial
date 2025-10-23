import os
import torch
from torch.export import export
from torchvision.models import ResNet18_Weights, resnet18

import tvm
import numpy as np

def create_resnet18_with_weights(pytorch_model, keep_params=False):
    import torch.fx as fx

    with torch.no_grad():
        traced_model = fx.symbolic_trace(pytorch_model)

    from tvm.relax.frontend.torch import from_fx
    input_info = [((1, 3, 224, 224), "float32")]

    with torch.no_grad():
        relax_mod = from_fx(
            traced_model,
            input_info,
            keep_params_as_input=keep_params
        )

    return relax_mod


if __name__ == "__main__":
    try:
        pytorch_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        pytorch_model.eval()
        relax_mod = create_resnet18_with_weights(pytorch_model)
        print(relax_mod)

    except Exception as e:
        print(f"\n Error Executed: {e}")
        import traceback
        traceback.print_exc()        

