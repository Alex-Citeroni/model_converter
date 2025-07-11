# 🧠 keras‑onnx‑torch converter

Convert **Keras** models (`.h5` / `.keras`) **and/or** existing **ONNX** files in pochi secondi:

* Keras → ONNX
  \* Keras / ONNX → PyTorch `.pt`
  \* Verifica ONNX + dummy‑inference opzionale
* Salvataggio automatico in una cartella di output

---

## 🚀 Funzionalità principali

| ✔                                                | Descrizione                                           |
| ------------------------------------------------ | ----------------------------------------------------- |
| 💾 **Input flessibile**                           | qualunque `.h5`, `.keras` **o** `.onnx`               |
| 🛠️ **Output multipli**                            | scegli ONNX (`--onnx`) e/o PyTorch (`--save‑pytorch`) |
| 🔍 **Check ONNX** (`--no-check` per saltare)      | `onnx.checker` con report immediato                   |
| 🧪 **Dummy inference** (`--no-dummy` per saltare) | test automatico con `onnxruntime`                     |
| 🔧 **Parametri personalizzabili**                 | `--input-shape`, `--input-name`, `--opset`            |
| 📂 **Cartella di output**                         | `--out-dir` (default `output/`)                       |

---

## 📦 Requisiti

```bash
pip install -r requirements.txt
```

> Dipendenze principali: `tensorflow`, `tf2onnx`, `onnx`, `onnxruntime`, `onnx2pytorch`, `torch`.

---

## 🛠️ CLI – esempi rapidi

### 1 ▸ Solo ONNX

```bash
python app.py \
  --input models/model.h5 \
  --onnx                 # verrà salvato in output/model.onnx
```

### 2 ▸ Solo PyTorch `.pt`

```bash
python app.py \
  --input models/model.h5 \
  --save-pytorch          # creerà output/model.pt (usa ONNX temp.)
```

### 3 ▸ Entrambi con percorsi custom

```bash
python app.py \
  --input   models/model.h5 \
  --onnx    export/my_model.onnx \
  --save-pytorch \
  --torch   export/my_model.pt \
  --out-dir export \
  --input-shape 128 128 3
```

Se l’input è già un **ONNX**, puoi saltare la conversione Keras:

```bash
python app.py --input models/already.onnx --save-pytorch
```

---

## 📌 Opzioni CLI

| Flag             | Descrizione                                                     | Default          |
| ---------------- | --------------------------------------------------------------- | ---------------- |
| `--input`        | Path al modello `.h5`, `.keras` **o** `.onnx`                   | **obbligatorio** |
| `--out-dir`      | Cartella di destinazione                                        | `output/`        |
| `--onnx`         | Path del file ONNX generato (se omesso = `out-dir/<name>.onnx`) | —                |
| `--save-pytorch` | Converte l’ONNX in `torch.nn.Module` e salva `.pt`              | —                |
| `--torch`        | Path per il file `.pt` (se omesso = `out-dir/<name>.pt`)        | —                |
| `--opset`        | Versione ONNX opset (solo Keras → ONNX)                         | `13`             |
| `--input-shape`  | Shape input H W C (solo Keras → ONNX)                           | `128 128 3`      |
| `--input-name`   | Nome tensore input                                              | `input`          |
| `--no-check`     | Salta la validazione ONNX                                       | disabilitato     |
| `--no-dummy`     | Salta dummy inference                                           | disabilitato     |

---

## 📂 Struttura del progetto

```
keras_onnx_torch_converter/
├── app.py            # Entry‑point CLI
├── converter.py      # Motore di conversione
├── requirements.txt  # Dipendenze
└── README.md         # Questo file
```

---

## ✅ Esempio output

```
[INFO] Loading Keras model: models/model.h5
[INFO] Converting to ONNX → output/model.onnx
[INFO] Verifica ONNX...
✅ ONNX OK
[INFO] Dummy inference...
✅ Shape output: (1, 1)
[INFO] Converting ONNX → PyTorch (output/model.pt)
✅ Salvato PyTorch a output/model.pt
```

---

## 🔧 Note tecniche

* Con **NumPy ≥ 2.0** potresti vedere un warning `np.cast` da `tf2onnx`; se accade usa `pip install "numpy<2.0"` o la branch nightly di tf2onnx.
* Se il modello Keras è salvato con canali `NHWC`, `tf2onnx` gestisce la trasposizione automaticamente.

---

## 📜 Licenza

MIT License © 2025 – Alex Citeroni
