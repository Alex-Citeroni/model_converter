import tensorflow as tf
import tf2onnx
import onnx
import onnxruntime as ort
import numpy as np


def convert_keras_to_onnx(
    h5_path: str,
    onnx_path: str,
    input_shape: tuple = (128, 128, 3),
    input_name: str = "input",
    opset: int = 13,
    validate: bool = True,
    run_dummy: bool = True,
):
    print(f"[INFO] Loading model from {h5_path}")
    model = tf.keras.models.load_model(h5_path)

    # Ensure input is batch-first
    input_sig = (tf.TensorSpec((None, *input_shape), tf.float32, name=input_name),)

    print("[INFO] Converting to ONNX...")
    onnx_model, _ = tf2onnx.convert.from_keras(
        model, input_signature=input_sig, opset=opset, output_path=onnx_path
    )

    if validate:
        print("[INFO] Checking ONNX model validity...")
        onnx.checker.check_model(onnx_path)
        print("✅ ONNX model is valid.")

    if run_dummy:
        print("[INFO] Running dummy inference using ONNX Runtime...")
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        dummy_input = np.zeros((1, *input_shape), dtype=np.float32)
        outputs = session.run(None, {input_name: dummy_input})
        print(f"✅ Dummy inference OK. Output shape: {outputs[0].shape}")
