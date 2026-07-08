import argparse
from pathlib import Path

from rknn.api import RKNN


def build_rknn(onnx_model: Path, dataset_txt: Path, output_rknn: Path, target: str = "rk3588") -> None:
    rknn = RKNN(verbose=False)
    ret = rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=target,
        quantized_algorithm="normal",
        quantized_method="channel",
    )
    if ret != 0:
        raise RuntimeError("RKNN config failed")

    ret = rknn.load_onnx(model=str(onnx_model))
    if ret != 0:
        raise RuntimeError("Load ONNX failed")

    ret = rknn.build(do_quantization=True, dataset=str(dataset_txt))
    if ret != 0:
        raise RuntimeError("RKNN build failed")

    ret = rknn.export_rknn(str(output_rknn))
    if ret != 0:
        raise RuntimeError("RKNN export failed")

    rknn.release()
    print(f"RKNN exported: {output_rknn}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ONNX to RKNN for RK3588.")
    parser.add_argument("--onnx", required=True, help="Path to ONNX model.")
    parser.add_argument("--dataset-txt", required=True, help="Calibration image list file.")
    parser.add_argument("--output", default="project/convert/model.rknn", help="Output RKNN path.")
    parser.add_argument("--target", default="rk3588", help="RK target platform.")
    args = parser.parse_args()

    build_rknn(Path(args.onnx), Path(args.dataset_txt), Path(args.output), args.target)


if __name__ == "__main__":
    main()
