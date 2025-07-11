import os
import sys
import tempfile
from typing import Tuple, Optional

import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf
import tf2onnx
import torch


# ──────────────────────────────────────────────────────────────────────────────
# UTIL ─ Keras → ONNX
# ──────────────────────────────────────────────────────────────────────────────
def _keras_to_onnx(
    keras_model: tf.keras.Model,
    onnx_path: str,
    input_shape: Tuple[int, int, int],
    input_name: str,
    opset: int,
):
    sig = (tf.TensorSpec((None, *input_shape), tf.float32, name=input_name),)
    tf2onnx.convert.from_keras(
        keras_model,
        input_signature=sig,
        opset=opset,
        output_path=onnx_path,
        large_model=False,
    )


# ──────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────────────────────────────────────
def convert_model(
    input_path: str,
    onnx_path: Optional[str],
    torch_path: Optional[str],
    *,
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    input_name: str = "input",
    opset: int = 13,
    validate: bool = True,
    run_dummy: bool = True,
    save_pytorch: bool = False,
    save_weights: bool = False,
    save_ts: bool = False,
):
    """
    Convert Keras (.h5 / .keras) or ONNX into:
      • ONNX (if --onnx)
      • PyTorch pickle .pt (if --save-pytorch)
      • PyTorch state_dict .pth (if --save-weights)
      • TorchScript .ts.pt (if --save-ts)
    """

    ext = os.path.splitext(input_path)[1].lower()
    is_keras = ext in {".h5", ".keras"}
    is_onnx = ext == ".onnx"
    if not (is_keras or is_onnx):
        raise ValueError("Formato non supportato: usa .h5, .keras o .onnx")

    # ─────────────────────────#
    # 1. Ottieni un file ONNX #
    # ─────────────────────────#
    if is_keras:
        print(f"[INFO] Loading Keras model: {input_path}")
        keras_model = tf.keras.models.load_model(input_path)

        need_onnx = bool(onnx_path or save_pytorch or save_weights or save_ts)
        if need_onnx:
            if onnx_path:
                onnx_path_final = onnx_path
            else:  # temporaneo
                tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
                onnx_path_final = tmp.name
            print(f"[INFO] Converting to ONNX → {onnx_path_final}")
            _keras_to_onnx(keras_model, onnx_path_final, input_shape, input_name, opset)
        else:
            onnx_path_final = None

    else:  # input già ONNX
        onnx_path_final = input_path
        if onnx_path and onnx_path != input_path:
            import shutil

            shutil.copy2(input_path, onnx_path)
            print(f"[INFO] Copied ONNX to {onnx_path}")

    # ─────────────────────────#
    # 2. Validazione / dummy  #
    # ─────────────────────────#
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

    # ─────────────────────────#
    # 3. Conversioni PyTorch  #
    # ─────────────────────────#
    if save_pytorch or save_weights or save_ts:
        if not onnx_path_final:
            raise RuntimeError("Serve un file ONNX per la conversione PyTorch")

        try:
            from onnx2pytorch import ConvertModel
        except ModuleNotFoundError:
            print("❌  onnx2pytorch non installato. `pip install onnx2pytorch`")
            sys.exit(1)

        torch_model = ConvertModel(onnx.load(onnx_path_final)).eval()

        # full pickle
        if save_pytorch:
            torch.save(torch_model, torch_path)
            print(f"✅ Modello completo salvato in {torch_path}")

        # state_dict
        if save_weights:
            torch.save(torch_model.state_dict(), torch_path)
            print(f"✅ state_dict salvato in {torch_path}")

        # TorchScript
        if save_ts:
            example = torch.randn((1,) + input_shape)
            scripted = torch.jit.trace(torch_model, example, strict=False)
            scripted.save(torch_path)
            print(f"✅ TorchScript salvato in {torch_path}")
