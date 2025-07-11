# 🧠 keras‑onnx‑torch converter

Converti **modelli Keras** (`.h5` / `.keras`) **e/o** file **ONNX** già esistenti in pochi secondi:

* **Keras → ONNX**
* **Keras / ONNX → PyTorch** (`.pt` *pickle completo* **oppure** `.pth` *solo pesi*)
* Verifica automatica del modello ONNX + dummy‑inference opzionale
* Salvataggio in una cartella di output configurabile

---

## 🚀 Funzionalità principali

| ✔                                | Descrizione                                                                 |
| -------------------------------- | --------------------------------------------------------------------------- |
| 💾 **Input flessibile**           | qualunque `.h5`, `.keras` **o** `.onnx`                                     |
| 🛠️ **Output multipli**            | `--onnx`, `--save‑pytorch` **(pickle)**, `--save‑weights` **(state\_dict)** |
| 🔍 **Check ONNX**                 | `onnx.checker` (disattivabile con `--no-check`)                             |
| 🧪 **Dummy inference**            | test rapido con `onnxruntime` (disattivabile con `--no-dummy`)              |
| 🔧 **Parametri personalizzabili** | `--input-shape`, `--input-name`, `--opset`                                  |
| 📂 **Cartella di output**         | `--out-dir` (default `output/`)                                             |

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
  --onnx                 # salva output/model.onnx
```

### 2 ▸ Solo PyTorch **pickle** `.pt`

```bash
python app.py \
  --input models/model.h5 \
  --save-pytorch          # crea output/model.pt (usa ONNX temporaneo)
```

### 3 ▸ Solo **pesi** PyTorch `.pth`

```bash
python app.py \
  --input models/model.h5 \
  --save-weights          # crea output/model.pth
```

### 4 ▸ Entrambi con percorsi custom

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
python app.py --input models/already.onnx --save-weights
```

---

## 📌 Opzioni CLI

| Flag             | Descrizione                                   | Default               |        |
| ---------------- | --------------------------------------------- | --------------------- | ------ |
| `--input`        | Path al modello `.h5`, `.keras` **o** `.onnx` | **obbligatorio**      |        |
| `--out-dir`      | Cartella di destinazione                      | `output/`             |        |
| `--onnx`         | Path del file ONNX generato                   | `out-dir/<name>.onnx` |        |
| `--save-pytorch` | Salva modello completo pickled `.pt`          | —                     |        |
| `--save-weights` | Salva solo `state_dict` `.pth`                | —                     |        |
| `--torch`        | Path per `.pt` **o** `.pth`                   | \`out-dir/<name>.pt   | .pth\` |
| `--opset`        | Versione ONNX opset (solo Keras → ONNX)       | `13`                  |        |
| `--input-shape`  | Shape input H W C (solo Keras → ONNX)         | `128 128 3`           |        |
| `--input-name`   | Nome tensore input                            | `input`               |        |
| `--no-check`     | Salta validazione ONNX                        | disabilitato          |        |
| `--no-dummy`     | Salta dummy inference                         | disabilitato          |        |

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

## ✅ Esempio output

```
[INFO] Loading Keras model: models/model.h5
[INFO] Converting to ONNX → output/model.onnx
[INFO] Verifica ONNX...
✅ ONNX OK
[INFO] Dummy inference...
✅ Shape output: (1, 1)
[INFO] Converting ONNX → state_dict → output/model.pth
✅ Salvati soli pesi a output/model.pth
```

---

## 🔧 Note tecniche

* Con **NumPy ≥ 2.0** potresti vedere un warning `np.cast` da `tf2onnx`; se accade usa `pip install "numpy<2.0"` o la branch nightly di `tf2onnx`.
* Se salvi il modello pickled (`--save-pytorch`) e usi PyTorch ≥ 2.6, servono *safe\_globals* oppure carica con `weights_only=False`.

---

## 📜 Licenza

MIT License © 2025 – Alex Citeroni
