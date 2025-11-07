import os
import torch

import time

from torch.export import export
from torchvision.models import ResNet18_Weights, resnet18
from torchvision import transforms
from tvm import relax

import tvm
import numpy as np
import multiprocessing
import urllib.request
import json
from PIL import Image

'''
Downlaod ImageNet 1000 class Dataset
'''
def load_imagenet_labels():
    
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"

    try:
        with urllib.request.urlopen(url) as response:
            labels = json.loads(response.read().decode())
        return {i: label for i, label in enumerate(labels)}
    except Exception as e:
        return {i: f"class_{i}" for i in range(1000)}
    
"""
load img & preprocess
"""
def load_and_preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        raise FileNotFoundError(f"Cannot find img: {image_path}\nError: {e}")

    # ImageNet Preprocess
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_tensor = preprocess(img)
    # add Batch dimension (3, 224, 224) -> (1, 3, 224, 224)
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor, img

'''
torch.fx : pytorch model -> fx graph
  # graph:
  #   %x : [num_users=1] = placeholder[target=x]
  #   %conv1 : [num_users=1] = call_module[target=conv1](args = (%x,))
  #   %relu : [num_users=1] = call_module[target=relu](args = (%conv1,))
  #   return relu

from_fx : fx graph -> tvm relax ir
'''
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

def classify_image(vm, device, image_tensor, labels, top_k=5):
    # PyTorch tensor → TVM
    img_np = image_tensor.numpy()
    img_tvm = tvm.runtime.tensor(img_np)

    # Warmup
    for _ in range(5):
        _ = vm["main"](img_tvm)

    # Inference Benchmark (100)
    num_iterations = 10
    print(f"  ⏱ {num_iterations}회 inference Benchmark...")

    inference_times = []

    for i in range(num_iterations):
        start_time = time.time()
        output = vm["main"](img_tvm)
        end_time = time.time()

        inference_times.append((end_time - start_time) * 1000)  

        if (i + 1) % 20 == 0:
            print(f"    process: {i+1}/{num_iterations} complete...")

    # statistic
    avg_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    median_time = np.median(inference_times)
    p95_time = np.percentile(inference_times, 95)
    p99_time = np.percentile(inference_times, 99)

    print(f"\n  Benchmark Statistic ({num_iterations}회):")
    print(f"    Mean:      {avg_time:.4f} ms")
    print(f"    Median:  {median_time:.4f} ms")
    print(f"    Std:   {std_time:.4f} ms")
    print(f"    Min:       {min_time:.4f} ms")
    print(f"    Max:       {max_time:.4f} ms")
    print(f"    95 percentile:    {p95_time:.4f} ms")
    print(f"    99 percentile:    {p99_time:.4f} ms")

    output_np = output.numpy()

    # Softmax
    exp_output = np.exp(output_np - np.max(output_np))
    probabilities = exp_output / np.sum(exp_output)

    # Top-K
    top_indices = np.argsort(probabilities[0])[::-1][:top_k]
    top_predictions = [
        (labels.get(idx, f"class_{idx}"), probabilities[0][idx] * 100)
        for idx in top_indices
    ]

    return avg_time, top_predictions

def compile_model(relax_mod, use_auto_tuning=True, num_trials=64, opt_level=0, max_workers=None):
    num_cores = multiprocessing.cpu_count()
    target = tvm.target.Target(f"llvm -num-cores {num_cores}")

    with target:
        relax_mod = relax.get_pipeline("zero")(relax_mod)

    if use_auto_tuning:
        import tvm.meta_schedule as ms
        from tvm.meta_schedule.builder import LocalBuilder
        from tvm.ir.transform import PassContext

        work_dir = "tuning_database"
        os.makedirs(work_dir, exist_ok=True)

        if max_workers is None:
            max_workers = num_cores

        print(f"🔧 Meta Schedule Tuning 시작 (max_workers={max_workers})")

        builder = LocalBuilder(max_workers=max_workers)

        # TIR 함수들을 추출하고 tuning
        with target:
            # MetaSchedule 내부 연산인 tune_tir 직접 실행 -> worker 숫자를 여기서 지정할 수 있음.
            ms.tune_tir(
                mod=relax_mod,
                target=target,
                work_dir=work_dir,
                max_trials_global=num_trials,
                builder=builder,
                num_tuning_cores=max_workers,  
            )

        with target, PassContext(opt_level=opt_level):
            application_pass = relax.transform.MetaScheduleApplyDatabase(work_dir)
            relax_mod = application_pass(relax_mod)

    # Compile
    ex = relax.build(relax_mod, target)

    # Run
    device = tvm.cpu()
    vm = relax.VirtualMachine(ex, device)

    return vm, device
    
    
if __name__ == "__main__":
    try:
        IMAGE_PATH = "../sample/img/dog.jpeg"

        USE_AUTO_TUNING = True
        NUM_TRIALS = 2
        OPT_LEVEL = 3
        MAX_WORKERS = 4  #  set worker num

        labels = load_imagenet_labels()
        image_tensor, original_image = load_and_preprocess_image(IMAGE_PATH)

        pytorch_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        pytorch_model.eval()
        relax_mod = create_resnet18_with_weights(pytorch_model)
        vm, device = compile_model(
            relax_mod,
            use_auto_tuning=USE_AUTO_TUNING,
            num_trials=NUM_TRIALS,
            opt_level=OPT_LEVEL,
            max_workers=MAX_WORKERS
        )

        tvm_time, tvm_predictions = classify_image(vm, device, image_tensor, labels, top_k=5)

        for i, (label, prob) in enumerate(tvm_predictions):
            print(f"  {i+1}. {label}: {prob:.2f}%")

    except Exception as e:
        print(f"\n Error Executed: {e}")
        import traceback
        traceback.print_exc()        
