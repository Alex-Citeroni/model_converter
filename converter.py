import os, tempfile, numpy as np, onnx
from typing import Tuple, Optional

# Caricamento / conversione
import tensorflow as tf, tf2onnx
import onnxruntime as ort
from onnx2pytorch import ConvertModel
import torch


def _keras_to_onnx(
    keras_model: tf.keras.Model,
    onnx_path: str,
    input_shape: Tuple[int, int, int],
    input_name: str,
    opset: int,
):
    sig = (tf.TensorSpec((None, *input_shape), tf.float32, name=input_name),)
    tf2onnx.convert.from_keras(
        keras_model, input_signature=sig, opset=opset, output_path=onnx_path
    )


def convert_model(
    input_path: str,
    onnx_path: Optional[str],
    torch_path: Optional[str],
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    input_name: str = "input",
    opset: int = 13,
    validate: bool = True,
    run_dummy: bool = True,
    save_pytorch: bool = False,
    save_weights: bool = False,
):
    ext = os.path.splitext(input_path)[1].lower()
    is_keras = ext in {".h5", ".keras"}
    is_onnx = ext == ".onnx"

    if not is_keras and not is_onnx:
        raise ValueError("Formato non supportato: usa .h5, .keras o .onnx")

    # ---------------------------------------------------- KERAS → ONNX
    if is_keras:
        print(f"[INFO] Loading Keras model: {input_path}")
        keras_model = tf.keras.models.load_model(input_path)

        if onnx_path or save_pytorch:
            onnx_path_final = (
                onnx_path
                or tempfile.NamedTemporaryFile(suffix=".onnx", delete=False).name
            )
            print(f"[INFO] Converting to ONNX → {onnx_path_final}")
            _keras_to_onnx(keras_model, onnx_path_final, input_shape, input_name, opset)
        else:
            onnx_path_final = None

    # ---------------------------------------------------- ONNX già pronto
    else:
        onnx_path_final = input_path
        if onnx_path and onnx_path != input_path:
            # Copia o rinomina se serve
            import shutil

            shutil.copy2(input_path, onnx_path)
            print(f"[INFO] Copied ONNX to {onnx_path}")

    # ---------------------------------------------------- Verifica / dummy
    if onnx_path_final and validate:
        print("[INFO] Verifica ONNX...")
        onnx.checker.check_model(onnx_path_final)
        print("✅ ONNX OK")

    if onnx_path_final and onnx_path and run_dummy:
        print("[INFO] Dummy inference...")
        sess = ort.InferenceSession(onnx_path_final, providers=["CPUExecutionProvider"])
        dummy = np.zeros((1, *input_shape), dtype=np.float32)
        out = sess.run(None, {input_name: dummy})
        print(f"✅ Shape output: {out[0].shape}")

    onnx_src = onnx_path_final
    # -------------- ONNX → PyTorch intero  (.pt) --------------
    if save_pytorch:
        if not onnx_src:
            raise RuntimeError("Serve un ONNX per salvare il modello PyTorch")
        print(f"[INFO] Converting ONNX → PyTorch (full) → {torch_path}")
        torch_model = ConvertModel(onnx.load(onnx_src)).eval()
        torch.save(torch_model, torch_path)
        print(f"✅ Salvato modello completo a {torch_path}")

    # -------------- ONNX → state_dict  (.pth) -----------------
    elif save_weights:
        if not onnx_src:
            raise RuntimeError("Serve un ONNX per salvare lo state_dict")
        print(f"[INFO] Converting ONNX → state_dict → {torch_path}")
        torch_model = ConvertModel(onnx.load(onnx_src))
        torch.save(torch_model.state_dict(), torch_path)
        print(f"✅ Salvati soli pesi a {torch_path}")
