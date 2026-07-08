import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from rknnlite.api import RKNNLite


def preprocess(img_path: Path, size: int):
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Failed to read {img_path}")
    img = cv2.resize(img, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inp = np.expand_dims(img, 0).astype(np.float32)
    return inp


def run_onnx(onnx_path: Path, inp: np.ndarray):
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    return sess.run(None, {input_name: inp})


def run_rknn(rknn_path: Path, inp: np.ndarray):
    rknn = RKNNLite()
    if rknn.load_rknn(str(rknn_path)) != 0:
        raise RuntimeError("load_rknn failed")
    if rknn.init_runtime() != 0:
        raise RuntimeError("init_runtime failed")
    out = rknn.inference(inputs=[inp])
    rknn.release()
    return out


def summarize_diff(onnx_out, rknn_out):
    n = min(len(onnx_out), len(rknn_out))
    for i in range(n):
        a = np.asarray(onnx_out[i]).astype(np.float32).ravel()
        b = np.asarray(rknn_out[i]).astype(np.float32).ravel()
        m = min(len(a), len(b))
        if m == 0:
            print(f"Output[{i}] empty.")
            continue
        diff = np.abs(a[:m] - b[:m])
        print(f"Output[{i}] mean_abs_diff={diff.mean():.6f} max_abs_diff={diff.max():.6f}")


def main():
    parser = argparse.ArgumentParser(description="Quick ONNX vs RKNN numeric diff.")
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--rknn", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--size", type=int, default=640)
    args = parser.parse_args()

    inp = preprocess(Path(args.image), args.size)
    onnx_out = run_onnx(Path(args.onnx), inp)
    rknn_out = run_rknn(Path(args.rknn), inp)
    summarize_diff(onnx_out, rknn_out)


if __name__ == "__main__":
    main()
