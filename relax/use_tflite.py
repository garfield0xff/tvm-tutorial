import tempfile
import numpy as np


def create_sample_tflite_model():
    """sample network """
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), padding='same'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10),
    ])

    # TFLite로 변환
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    return tflite_model


def convert_tflite_to_onnx(tflite_model_path: str, onnx_model_path: str):
    """TFLite to ONNX

    Parameters
    ----------
    tflite_model_path : str
    onnx_model_path : str -> 출력 ONNX 모델 경로
    """
    import subprocess

    # tf2onnx CLI를 사용하여 변환
    cmd = [
        "python", "-m", "tf2onnx.convert",
        "--tflite", tflite_model_path,
        "--output", onnx_model_path,
        "--opset", "14"
    ]
    subprocess.run(cmd, check=True)
    print(f"ONNX 모델 저장됨: {onnx_model_path}")


def load_and_compile_onnx_to_relax(onnx_model_path: str, target: str = "llvm"):
    """ONNX model -> tvm realax

    Parameters
    ----------
    onnx_model_path : str -> ONNX 모델 경로
    target : str -> ("llvm", "cuda")

    Returns
    -------
    vm : relax.VirtualMachine -> 컴파일된 VirtualMachine
    params : dict 모델 파라미터
    """
    import onnx
    import tvm
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx

    onnx_model = onnx.load(onnx_model_path)
    print(f"ONNX 모델 로드됨: {onnx_model_path}")

    # keep_params_in_input=True: 파라미터를 입력으로 유지 (나중에 분리)
    relax_mod = from_onnx(onnx_model, opset=14, keep_params_in_input=True)
    print("Relax IR 변환 완료")

    # BatchNorm 등을 추론에 적합한 형태로 변환
    relax_mod = relax.transform.DecomposeOpsForInference()(relax_mod)

    # 4. Relax 연산자를 TensorIR로 변환 (legalization)
    relax_mod = relax.transform.LegalizeOps()(relax_mod)

    # 5. 모델과 파라미터 분리
    relax_mod, params = relax.frontend.detach_params(relax_mod)

    # 6. 컴파일
    with tvm.transform.PassContext(opt_level=3):
        ex = tvm.compile(relax_mod, target=target)

    # 7. VirtualMachine 생성
    if target == "cuda":
        device = tvm.cuda()
    else:
        device = tvm.cpu()

    vm = relax.VirtualMachine(ex, device)

    return vm, params, relax_mod


def run_inference(vm, params, relax_mod, input_data: np.ndarray):
    """ Inference to TVM Relax VirtualMachine

    Parameters
    ----------
    vm : relax.VirtualMachine -> 컴파일된 VM
    params : dict -> Model parameter
    relax_mod : IRModule -> Relax Module
    input_data : np.ndarray

    Returns
    -------
    output : np.ndarray -> 추론 결과
    """
    # 입력 준비: [input_data] + params
    input_list = [input_data]
    if params and "main" in params:
        input_list += params["main"]

    # 추론 실행
    vm.set_input("main", *input_list)
    vm.invoke_stateful("main")
    outputs = vm.get_outputs("main")

    # 결과 변환
    if isinstance(outputs, (list, tuple)):
        return [out.numpy() for out in outputs]
    return outputs.numpy()


def run_onnx_inference(onnx_model_path: str, input_data: np.ndarray):
    """inference onnx runtime (결과 비교용)

    Parameters
    ----------
    onnx_model_path : str -> ONNX 모델 경로
    input_data : np.ndarray -> 입력 데이터

    Returns
    -------
    output : np.ndarray -> 추론 결과
    """
    import onnx
    import onnxruntime

    onnx_model = onnx.load(onnx_model_path)

    # 입력 이름 추출
    input_name = onnx_model.graph.input[0].name

    # ONNX Runtime 세션 생성 및 실행
    session = onnxruntime.InferenceSession(
        onnx_model_path,
        providers=["CPUExecutionProvider"]
    )
    outputs = session.run(None, {input_name: input_data})

    return outputs[0]


def main():
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        tflite_path = os.path.join(tmpdir, "model.tflite")
        onnx_path = os.path.join(tmpdir, "model.onnx")

        # Crate Sample TFLite Model
        print("=" * 50)
        print("Create Tflite Sample Model")
        print("=" * 50)
        tflite_model = create_sample_tflite_model()
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print(f"saved to : {tflite_path}")

        # TFLite -> ONNX 
        print("\n" + "=" * 50)
        print("TFLite -> ONNX ")
        print("=" * 50)
        convert_tflite_to_onnx(tflite_path, onnx_path)

        # ONNX -> TVM Relax
        print("\n" + "=" * 50)
        print("Step 3: ONNX -> TVM Relax ")
        print("=" * 50)
        vm, params, relax_mod = load_and_compile_onnx_to_relax(onnx_path, target="llvm")

        # Inference
        print("\n" + "=" * 50)
        print("Inference")
        print("=" * 50)

        # test Dummy data
        input_data = np.random.randn(1, 224, 224, 3).astype(np.float32)
        print(f"input shape: {input_data.shape}")

        # TVM Relax inference
        tvm_output = run_inference(vm, params, relax_mod, input_data)
        print(f"TVM output shape: {tvm_output.shape}")

        # ONNX Runtime inference
        onnx_output = run_onnx_inference(onnx_path, input_data)
        print(f"ONNX output shape: {onnx_output.shape}")

        # Result test
        np.testing.assert_allclose(tvm_output, onnx_output, rtol=1e-5, atol=1e-5)
        print("\nTest Success.")


if __name__ == "__main__":
    main()
