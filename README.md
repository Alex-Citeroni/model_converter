# 🧠 keras‑onnx‑torch converter

Converti **modelli Keras** (`.h5` / `.keras`) **e/o** file **ONNX** in quattro formati pratici — in un solo comando:

* **Keras → ONNX** (`.onnx`)
* **Keras / ONNX → PyTorch**

  * modello completo pickle (`.pt`)
  * soli pesi state\_dict (`.pth`)
  * modulo TorchScript portabile (`.ts.pt`)
* Verifica automatica ONNX + dummy‑inference opzionale
* Percorsi di output generati automaticamente (override opzionale)

---

## 🚀 Funzionalità

| ✔                      | Descrizione                                               |
| ---------------------- | --------------------------------------------------------- |
| **Input flessibile**   | accetta `.h5`, `.keras`, `.onnx`                          |
| **Output mirato**      | scegli uno o più fra `--onnx`, `--pt`, `--pth`, `--ts`    |
| **Check ONNX**         | validazione con `onnx.checker` (`--no-check` per saltare) |
| **Dummy inference**    | test con `onnxruntime` (`--no-dummy` per saltare)         |
| **Parametri custom**   | `--input-shape`, `--input-name`, `--opset`                |
| **Cartella di output** | `--out-dir` (default `output/`)                           |

---

## 📦 Installazione

```bash
pip install -r requirements.txt
```

Dipendenze principali: **tensorflow**, **tf2onnx**, **onnx**, **onnxruntime**, **onnx2pytorch**, **torch**.

---

## 🛠️ Utilizzo rapido

### 1 ▸ Solo ONNX

```bash
python app.py \
  --input models/model.h5 \
  --onnx                # salva output/model.onnx
```

### 2 ▸ Solo modello PyTorch (.pt)

```bash
python app.py \
  --input models/model.h5 \
  --pt                  # crea output/model.pt (ONNX temporaneo)
```

### 3 ▸ Solo pesi (.pth)

```bash
python app.py --input models/model.h5 --pth
```

### 4 ▸ TorchScript con percorso custom

```bash
python app.py \
  --input models/model.h5 \
  --ts export/my_model.ts.pt
```

### 5 ▸ ONNX + state\_dict

```bash
python app.py --input models/model.h5 --onnx --pth
```

L’input può già essere un **ONNX**:

```bash
python app.py --input pretrained.onnx --pt
```

---

## 📌 Opzioni principali

| Flag                  | Azione                     | Valore di default se omesso |
| --------------------- | -------------------------- | --------------------------- |
| `--onnx [FILE]`       | salva `.onnx`              | `output/<name>.onnx`        |
| `--pt   [FILE]`       | salva pickle `.pt`         | `output/<name>.pt`          |
| `--pth  [FILE]`       | salva pesi `.pth`          | `output/<name>.pth`         |
| `--ts   [FILE]`       | salva TorchScript `.ts.pt` | `output/<name>.ts.pt`       |
| `--out-dir DIR`       | cartella base output       | `output/`                   |
| `--opset N`           | opset ONNX (Keras→ONNX)    | `13`                        |
| `--input-shape H W C` | shape input Keras          | `128 128 3`                 |
| `--input-name NAME`   | tensor name Keras          | `input`                     |
| `--no-check`          | salta validazione ONNX     | —                           |
| `--no-dummy`          | salta dummy inference      | —                           |

*Se ometti `FILE` il programma genera automaticamente il nome
all’interno di `--out-dir`.*

---

## 📂 Struttura del progetto

```
keras_onnx_torch_converter/
├── app.py            # CLI
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
[INFO] Converting ONNX → TorchScript → output/model.ts.pt
✅ TorchScript salvato in output/model.ts.pt
```

---

## 🔧 Note tecniche

* Con **NumPy ≥ 2.0** `tf2onnx` emette un warning su `np.cast`. Se serve: `pip install "numpy<2.0"` o usa una build nightly di `tf2onnx`.
* I modelli `.pt` pickled includono la classe `onnx2pytorch.ConvertModel`; per caricarli senza `onnx2pytorch` usa `--ts` o `--pth`.

---

## 📜 Licenza

MIT License © 2025 – Alex Citeroni
