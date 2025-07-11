# 🧠 keras2onnx-converter

Converti facilmente modelli Keras (`.h5`) in formato ONNX, con validazione e test automatico tramite ONNX Runtime.

---

## 🚀 Caratteristiche

* ✅ Conversione `.h5` → `.onnx` tramite `tf2onnx`
* 🔍 Verifica automatica del file ONNX (`onnx.checker`)
* 🧪 Inference di prova con input dummy (`onnxruntime`)
* 🔧 Input shape, nome input e opset personalizzabili via CLI
* 📦 Pronto per l’integrazione in pipeline PyTorch o ONNX Runtime

---

## 📦 Requisiti

Installa le dipendenze:

```bash
pip install -r requirements.txt
```

---

## 🛠️ Utilizzo

Esegui la conversione:

```bash
python app.py \
  --h5 models/motion_blur_cnn_simple.h5 \
  --onnx models/motion_blur_cnn_simple.onnx \
  --input-shape 128 128 3
```

---

### 📌 Opzioni disponibili

| Flag            | Descrizione                             | Default         |
| --------------- | --------------------------------------- | --------------- |
| `--h5`          | Path del file `.h5` Keras da convertire | —               |
| `--onnx`        | Path di output del file `.onnx`         | —               |
| `--input-shape` | Shape dell'input (es. 128 128 3)        | `(128, 128, 3)` |
| `--input-name`  | Nome del tensore di input               | `input`         |
| `--opset`       | Versione ONNX opset da usare            | `13`            |
| `--no-check`    | Salta la verifica del file ONNX         | disabilitato    |
| `--no-dummy`    | Salta l’inferenza dummy con input zero  | disabilitato    |

---

## 📂 Struttura del progetto

```
keras2onnx_converter/
├── app.py                 # Script principale
├── converter.py           # Logica di conversione
├── requirements.txt       # Dipendenze Python
├── README.md              # Questo file
└── models/
    └── motion_blur_cnn_simple.h5
```

---

## ✅ Output atteso

```
[INFO] Loading model from models/motion_blur_cnn_simple.h5
[INFO] Converting to ONNX...
[INFO] Checking ONNX model validity...
✅ ONNX model is valid.
[INFO] Running dummy inference using ONNX Runtime...
✅ Dummy inference OK. Output shape: (1, 1)
```

---

## 🔧 Note tecniche

* Se usi **NumPy ≥ 2.0**, `tf2onnx` potrebbe mostrare warning su `np.cast`.
  Soluzione: `pip install "numpy<2.0"` oppure usa la branch aggiornata di `tf2onnx`.

---

## 📜 Licenza

MIT License © 2025 – Alex Citeroni
