import argparse, sys, pathlib
from converter import convert_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Keras (.h5/.keras) or ONNX to other formats"
    )

    parser.add_argument(
        "--input", required=True, help="Path to input model (.h5 / .keras / .onnx)"
    )
    parser.add_argument(
        "--out-dir", default="output", help="Base folder for converted models"
    )
    parser.add_argument("--onnx", help="Custom output name for ONNX file")
    parser.add_argument(
        "--save-pytorch", action="store_true", help="Also save as PyTorch .pt model"
    )
    parser.add_argument("--torch", help="Custom output name for PyTorch .pt file")
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version (when converting from Keras)",
    )
    parser.add_argument(
        "--input-shape",
        nargs=3,
        type=int,
        default=[128, 128, 3],
        help="Input shape H W C (needed for Keras→ONNX)",
    )
    parser.add_argument(
        "--input-name", default="input", help="Input tensor name (Keras→ONNX)"
    )
    parser.add_argument(
        "--no-check", action="store_true", help="Skip ONNX validity check"
    )
    parser.add_argument(
        "--no-dummy", action="store_true", help="Skip dummy inference on ONNX"
    )

    args = parser.parse_args()

    # almeno un formato in uscita
    if not args.onnx and not args.save_pytorch:
        print("❌  Devi specificare --onnx e/o --save-pytorch")
        sys.exit(1)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # path di default se non passati
    stem = pathlib.Path(args.input).stem
    onnx_path = (
        args.onnx or str(out_dir / f"{stem}.onnx")
        if args.onnx or not args.save_pytorch
        else None
    )
    torch_path = (
        args.torch or str(out_dir / f"{stem}.pt") if args.save_pytorch else None
    )

    convert_model(
        input_path=args.input,
        onnx_path=onnx_path,
        torch_path=torch_path,
        input_shape=tuple(args.input_shape),
        input_name=args.input_name,
        opset=args.opset,
        validate=not args.no_check,
        run_dummy=not args.no_dummy,
        save_pytorch=args.save_pytorch,
    )
