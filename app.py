import argparse
from converter import convert_keras_to_onnx

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Keras .h5 to ONNX")

    parser.add_argument("--h5", required=True, help="Path to Keras .h5 model")
    parser.add_argument("--onnx", required=True, help="Output path for ONNX model")
    parser.add_argument(
        "--input-shape",
        nargs=3,
        type=int,
        default=[128, 128, 3],
        help="Input shape (H W C)",
    )
    parser.add_argument("--input-name", default="input", help="Input tensor name")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version")
    parser.add_argument(
        "--no-check", action="store_true", help="Skip ONNX validity check"
    )
    parser.add_argument("--no-dummy", action="store_true", help="Skip dummy inference")

    args = parser.parse_args()

    convert_keras_to_onnx(
        h5_path=args.h5,
        onnx_path=args.onnx,
        input_shape=tuple(args.input_shape),
        input_name=args.input_name,
        opset=args.opset,
        validate=not args.no_check,
        run_dummy=not args.no_dummy,
    )
