# app.py
import argparse, sys, pathlib
from converter import convert_model

# ───────────────────── argparse ─────────────────────
p = argparse.ArgumentParser(
    description="Convert Keras (.h5/.keras) or ONNX into ONNX/PT/PTH/TorchScript"
)

p.add_argument(
    "--input", required=True, help="Path to input model (.h5 | .keras | .onnx)"
)
p.add_argument(
    "--out-dir", default="output", help="Destination folder for auto-generated names"
)

# output formats: 0/1 arg  (flag   OR  flag <path>)
p.add_argument("--onnx", nargs="?", const=True, help="Save ONNX  (opt: custom path)")
p.add_argument("--pt", nargs="?", const=True, help="Save pickled model (.pt)")
p.add_argument("--pth", nargs="?", const=True, help="Save state_dict (.pth)")
p.add_argument("--ts", nargs="?", const=True, help="Save TorchScript (.ts.pt)")

# extra options (rarely touched → advanced group)
g = p.add_argument_group("advanced")
g.add_argument("--opset", type=int, default=13, help="ONNX opset (Keras→ONNX)")
g.add_argument(
    "--input-shape",
    nargs=3,
    type=int,
    default=[128, 128, 3],
    metavar=("H", "W", "C"),
    help="Input shape for Keras models",
)
g.add_argument("--input-name", default="input", help="Input tensor name (Keras)")
g.add_argument("--no-check", action="store_true", help="Skip ONNX checker")
g.add_argument("--no-dummy", action="store_true", help="Skip dummy inference")

args = p.parse_args()

# ───────────────────── validation ─────────────────────
if not any([args.onnx, args.pt, args.pth, args.ts]):
    p.error("specify at least one output: --onnx / --pt / --pth / --ts")

out_dir = pathlib.Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
stem = pathlib.Path(args.input).stem


def _resolve(flag, default_ext):
    if flag is True:  # flag without value
        return str(out_dir / f"{stem}{default_ext}")
    elif isinstance(flag, str):  # custom path
        return flag
    return None  # flag absent


onnx_path = _resolve(args.onnx, ".onnx")
pt_path = _resolve(args.pt, ".pt")
pth_path = _resolve(args.pth, ".pth")
ts_path = _resolve(args.ts, ".ts.pt")

# ───────────────────── call converter ─────────────────────
convert_model(
    input_path=args.input,
    onnx_path=onnx_path,
    torch_path=pt_path or pth_path or ts_path,
    input_shape=tuple(args.input_shape),
    input_name=args.input_name,
    opset=args.opset,
    validate=not args.no_check,
    run_dummy=not args.no_dummy,
    save_pytorch=bool(args.pt),
    save_weights=bool(args.pth),
    save_ts=bool(args.ts),
)
