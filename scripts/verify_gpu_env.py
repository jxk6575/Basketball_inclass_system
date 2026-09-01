#!/usr/bin/env python3
"""Verify GPU environment and model files."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# onnxruntime-gpu 需在 import 前找到 conda 的 libcudnn
_conda = os.environ.get("CONDA_PREFIX")
if _conda:
    _lib = os.path.join(_conda, "lib")
    os.environ["LD_LIBRARY_PATH"] = f"{_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml


def check(name: str, ok: bool, detail: str = "") -> bool:
    mark = "OK" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> int:
    print("=== GPU Environment Verification ===\n")
    all_ok = True

    cfg_path = ROOT / "configs" / "models.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # PyTorch
    try:
        import torch
        cuda = torch.cuda.is_available()
        all_ok &= check("PyTorch", True, torch.__version__)
        all_ok &= check("CUDA", cuda, torch.cuda.get_device_name(0) if cuda else "unavailable")
    except ImportError:
        all_ok &= check("PyTorch", False, "not installed")

    # ONNX Runtime
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        has_cuda = "CUDAExecutionProvider" in providers
        all_ok &= check("ONNX Runtime", True, ort.__version__)
        all_ok &= check("ONNX CUDA EP", has_cuda, str(providers))
    except ImportError:
        all_ok &= check("ONNX Runtime", False)

    # rtmlib
    try:
        import rtmlib  # noqa: F401
        all_ok &= check("rtmlib", True)
    except ImportError:
        all_ok &= check("rtmlib", False)

    # InsightFace
    try:
        import insightface  # noqa: F401
        all_ok &= check("insightface", True)
    except ImportError:
        all_ok &= check("insightface", False)

    # Model files
    print("\n=== Model Files ===\n")
    for key in ("detector", "pose", "body_reid"):
        sub = cfg[key]
        p = ROOT / sub["path"]
        all_ok &= check(sub["name"], p.exists(), str(p))

    p3d = cfg.get("pose3d", {})
    if p3d.get("path"):
        p = ROOT / p3d["path"]
        ok = check(p3d["name"], p.exists(), str(p))
        if not ok:
            print("  [info] rtmw3d_x is optional — run: python scripts/download_models.py")

    # MotionBERT weights are optional / historical (v1 uses rule action)
    mb = cfg.get("motionbert") or {}
    if mb.get("pretrain"):
        mb_pre = ROOT / mb["pretrain"]
        if mb_pre.exists():
            check("MotionBERT pretrain (optional)", True, str(mb_pre))
        else:
            print(f"  [skip] MotionBERT optional — not required for v1")

    # Quick inference smoke test
    print("\n=== Smoke Test ===\n")
    try:
        from src.identity.backends import get_runtime_config
        from src.perception.rtmlib_backend import RTMLibPerception

        rt = get_runtime_config()
        backend = RTMLibPerception(
            det_model=str(ROOT / cfg["detector"]["path"]),
            pose_model=str(ROOT / cfg["pose"]["path"]),
            device="cuda" if rt["use_cuda"] else "cpu",
        )
        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = backend.detect_persons(frame)
        all_ok &= check("RTMLib detect (empty frame)", True, f"{len(dets)} detections")
    except Exception as e:
        all_ok &= check("RTMLib smoke test", False, str(e))

    try:
        from src.identity.embedders import create_face_embedder
        fe = create_face_embedder()
        all_ok &= check("Face embedder", True, type(fe).__name__)
    except Exception as e:
        all_ok &= check("Face embedder", False, str(e))

    print("\n" + ("All checks passed." if all_ok else "Some checks failed — see above."))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
