"""Runtime backend configuration from configs/models.yaml."""

from __future__ import annotations

from src.config import load_models_config, model_path, ROOT


def get_runtime_config() -> dict:
    cfg = load_models_config()
    device_cfg = cfg.get("device", {})
    prefer_gpu = device_cfg.get("prefer_gpu", True)

    use_cuda = False
    providers = ["CPUExecutionProvider"]
    try:
        import torch
        use_cuda = prefer_gpu and torch.cuda.is_available()
    except ImportError:
        pass

    if use_cuda:
        try:
            import onnxruntime as ort
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers = device_cfg.get("onnx_providers", [
                    "CUDAExecutionProvider", "CPUExecutionProvider"
                ])
        except ImportError:
            use_cuda = False

    det_path = cfg.get("detector", {}).get("path")
    pose_path = cfg.get("pose", {}).get("path")
    reid_path = cfg.get("body_reid", {}).get("path")

    return {
        "use_cuda": use_cuda,
        "onnx_providers": providers,
        "insightface_ctx_id": device_cfg.get("insightface_ctx_id", 0) if use_cuda else -1,
        "detector_path": str(model_path(det_path)) if det_path else None,
        "pose_path": str(model_path(pose_path)) if pose_path else None,
        "reid_path": str(model_path(reid_path)) if reid_path else None,
        "pose_mode": cfg.get("pose", {}).get("mode", "balanced"),
        "models_ready": bool(
            det_path and pose_path
            and (ROOT / det_path).exists()
            and (ROOT / pose_path).exists()
        ),
    }
