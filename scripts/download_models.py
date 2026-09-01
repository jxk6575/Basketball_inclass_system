#!/usr/bin/env python3
"""Download all model weights for GPU inference."""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODELS = ROOT / "models"
CONFIG_PATH = ROOT / "configs" / "models.yaml"


def load_models_cfg() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_url(url: str, dest: Path, desc: str = "") -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {desc or dest.name} already exists")
        return dest
    print(f"[download] {desc or url}")

    if "huggingface.co" in url:
        from huggingface_hub import hf_hub_download
        parts = url.split("/resolve/main/")
        if len(parts) == 2:
            repo = parts[0].split("huggingface.co/")[-1]
            filename = parts[1]
            path = hf_hub_download(repo_id=repo, filename=filename, local_dir=str(dest.parent))
            final = dest.parent / filename.split("/")[-1]
            if Path(path) != dest and Path(path).exists():
                shutil.copy2(path, dest)
            return dest

    import requests
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    tmp.rename(dest)
    print(f"  -> {dest}")
    return dest


def extract_onnx_from_zip(zip_path: Path, out_dir: Path) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    onnx_files = list(out_dir.rglob("*.onnx"))
    end2end = [p for p in onnx_files if "end2end" in p.name.lower()]
    chosen = end2end[0] if end2end else (onnx_files[0] if onnx_files else None)
    if chosen:
        target = out_dir / "end2end.onnx"
        if chosen.resolve() != target.resolve():
            shutil.copy2(chosen, target)
        return target
    return None


def download_zip_onnx(url: str, out_dir: Path, label: str) -> Path | None:
    target = out_dir / "end2end.onnx"
    if target.exists():
        print(f"[skip] {label} onnx ready")
        return target
    zip_path = out_dir / "model.zip"
    if not zip_path.exists():
        download_url(url, zip_path, label)
    return extract_onnx_from_zip(zip_path, out_dir)


def download_motionbert(cfg: dict) -> None:
    mb = cfg["motionbert"]
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[warn] huggingface_hub not installed, skip MotionBERT")
        return
    for rel in mb["files"]:
        dest = MODELS / "motionbert" / rel
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[skip] MotionBERT {rel}")
            continue
        print(f"[download] MotionBERT {rel}")
        hf_hub_download(
            repo_id=mb["repo_id"],
            filename=rel,
            local_dir=str(MODELS / "motionbert"),
        )


def download_insightface() -> None:
    try:
        from insightface.app import FaceAnalysis
        import onnxruntime as ort
        providers = ort.get_available_providers()
        use_cuda = "CUDAExecutionProvider" in providers
        prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
    except ImportError:
        print("[warn] insightface not installed, skip")
        return
    print(f"[download] InsightFace buffalo_l (providers={prov})")
    app = FaceAnalysis(name="buffalo_l", providers=prov)
    app.prepare(ctx_id=0 if use_cuda else -1, det_size=(640, 640))
    print("  -> InsightFace ready")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-face", action="store_true")
    parser.add_argument(
        "--with-motionbert",
        action="store_true",
        help="Also download MotionBERT weights (optional; v1 uses rule action)",
    )
    parser.add_argument("--skip-motionbert", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-pose3d", action="store_true")
    args = parser.parse_args()

    cfg = load_models_cfg()
    MODELS.mkdir(exist_ok=True)
    errors = []

    try:
        download_zip_onnx(cfg["detector"]["url_zip"], MODELS / "detection" / "yolox_m", "YOLOX-m")
    except Exception as e:
        errors.append(f"detector: {e}")

    try:
        download_zip_onnx(cfg["pose"]["url_zip"], MODELS / "pose" / "rtmw_l", "RTMW-l")
    except Exception as e:
        errors.append(f"pose: {e}")

    if not args.skip_pose3d:
        p3d = cfg.get("pose3d", {})
        if p3d.get("url"):
            try:
                dest = MODELS / "pose3d" / "rtmw3d_x" / "rtmw3d-x.onnx"
                download_url(p3d["url"], dest, "RTMW3D-x")
            except Exception as e:
                errors.append(f"pose3d (optional): {e}")

    try:
        reid = cfg["body_reid"]
        dest = MODELS / reid["path"]
        if dest.exists():
            print(f"[skip] {reid['name']}")
        else:
            from huggingface_hub import hf_hub_download
            print(f"[download] {reid['name']} from HF")
            src = hf_hub_download(
                repo_id=reid["repo_id"],
                filename=reid["filename"],
                local_dir=str(MODELS / "reid"),
            )
            src_path = Path(src)
            if not dest.exists():
                shutil.copy2(src_path, dest)
    except Exception as e:
        errors.append(f"reid: {e}")

    if args.with_motionbert and not args.skip_motionbert:
        try:
            download_motionbert(cfg)
        except Exception as e:
            errors.append(f"motionbert: {e}")
    else:
        print("[skip] MotionBERT (optional; pass --with-motionbert to download)")

    try:
        # Canonical YOLO11m-Pose person detector
        yp = MODELS / "detection" / "yolo_pose"
        yp.mkdir(parents=True, exist_ok=True)
        dest = yp / "yolo11m-pose.pt"
        if dest.exists() and dest.stat().st_size > 1_000_000:
            print("[skip] yolo11m-pose.pt")
        else:
            import urllib.request
            mirrors = [
                "https://hf-mirror.com/Ultralytics/YOLO11/resolve/main/yolo11m-pose.pt",
                "https://huggingface.co/Ultralytics/YOLO11/resolve/main/yolo11m-pose.pt",
            ]
            for u in mirrors:
                try:
                    print(f"[download] yolo11m-pose.pt <- {u}")
                    urllib.request.urlretrieve(u, dest)
                    if dest.stat().st_size > 1_000_000:
                        break
                except Exception as e:
                    print(f"  fail {e}")
                    if dest.exists():
                        dest.unlink()
            if not dest.exists() or dest.stat().st_size < 1_000_000:
                errors.append("yolo11m-pose: download failed")
    except Exception as e:
        errors.append(f"yolo11-pose: {e}")

    if not args.skip_face:
        try:
            download_insightface()
        except Exception as e:
            errors.append(f"insightface: {e}")

    print("\n[done] Models under:", MODELS)
    if errors:
        print("[warnings]")
        for e in errors:
            print(" ", e)
    print("Verify: python scripts/verify_gpu_env.py")


if __name__ == "__main__":
    main()
