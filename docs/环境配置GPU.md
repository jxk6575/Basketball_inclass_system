# GPU 环境配置指南

## 硬件要求

- NVIDIA GPU + 驱动（本机示例：RTX 4070，CUDA 13.2 驱动）
- 建议显存 ≥ 8GB（四路离线批处理单路推理约 2–4GB）

> 若 GPU 已被其他进程占满，可先结束占用进程再运行感知/下载脚本。

## 一键安装

```bash
cd /home/jiang/code/Basketball_inclass_system
chmod +x scripts/setup_gpu_env.sh
./scripts/setup_gpu_env.sh
```

将创建 conda 环境 `basketball_classroom`（Python 3.10 + PyTorch CUDA 12.4），并自动：

1. 安装 `requirements.txt` + `requirements-gpu.txt`
2. 下载模型到 `models/`
3. 运行 `scripts/verify_gpu_env.py`

## 手动分步（可选）

```bash
conda env create -f environment.yml -n basketball_classroom
conda activate basketball_classroom
conda install -y -c nvidia cudnn cuda-version=12   # onnxruntime-gpu 依赖 libcudnn
pip install -r requirements.txt -r requirements-gpu.txt
python scripts/download_models.py
python scripts/verify_gpu_env.py
```

## 下载的模型清单

| 组件 | 路径 | 用途 |
|------|------|------|
| **YOLO11m-Pose**（默认） | `models/detection/yolo_pose/yolo11m-pose.pt` · `person_detector: yolo_pose` | 人体检测 + COCO-17 粗姿态 |
| YOLOX-m（可选 fallback） | `models/detection/yolox_m/end2end.onnx` | 人体检测 |
| RTMW-l 133点 | `models/pose/rtmw_l/end2end.onnx` | 全身精细姿态 |
| RTMW3D-x | `models/pose3d/rtmw3d_x/rtmw3d-x.onnx` | 单目3D预览（可选） |
| OSNet | `models/reid/osnet_x1_0_msmt17.pth` | 身体 ReID（生产默认 + 衣服颜色） |
| InsightFace buffalo_l | `~/.insightface/models/buffalo_l/` | 人脸 ArcFace（`face_body*` 模式 / 注册可选） |
| Basketball_v1 | `models/detection/yolo_ball/Basketball_v1.pt` | cam_04 球/筐，`imgsz=1280` |
| MotionBERT（可选） | `models/motionbert/...` | **默认不下载**；`python scripts/download_models.py --with-motionbert` |

配置入口：[`configs/models.yaml`](../configs/models.yaml)

## 验证 GPU 是否生效

```bash
conda activate basketball_classroom
python scripts/verify_gpu_env.py
```

期望输出包含：

- `[OK] CUDA`
- `[OK] ONNX CUDA EP`
- `[OK] RTMLib smoke test`
- `[OK] Face embedder — InsightFaceEmbedder`

## CLIP-ReID 完整方案（可选进阶）

当前默认使用 **OSNet**（`torchreid`）作身体 ReID，并叠加 **衣服 + 足部（鞋子）颜色** 描述子（`src/identity/clothing_color.py`：torso/臂/腿/shoe，`match_mode: body_color`）。课堂球衣与鞋色区分明显时优于 Face-Body / BoT-SORT。

若需论文级 **CLIP-ReID**：

```bash
git clone https://github.com/Syliz517/CLIP-ReID.git third_party/CLIP-ReID
cd third_party/CLIP-ReID
# 按官方 README 安装依赖并下载 Market-1501 微调权重
```

后续可将 `src/identity/body_reid.py` 扩展为 `CLIPReIDEmbedder`。

## 常见问题

### onnxruntime-gpu 无 CUDAExecutionProvider

1. 安装 cuDNN：`conda install -y -c nvidia cudnn cuda-version=12`
2. **不要**单独 `pip install onnxruntime`（CPU 版会覆盖 GPU）
3. 确保 `conda activate` 后 `LD_LIBRARY_PATH` 包含 `$CONDA_PREFIX/lib`（`setup_gpu_env.sh` 会自动安装 activate hook）

```bash
pip uninstall onnxruntime onnxruntime-gpu -y
pip install onnxruntime-gpu
# 重新激活环境
conda deactivate && conda activate basketball_classroom
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

### InsightFace 下载慢

模型首次运行自动下载到 `~/.insightface/`；可提前执行：

```bash
python -c "from insightface.app import FaceAnalysis; FaceAnalysis('buffalo_l').prepare(ctx_id=0)"
```

### 显存不足

在 `configs/models.yaml` 将 `pose.mode` 改为 `balanced` 或 `lightweight`，或使用 `rtmw-dw-m` 较小模型（需改 `url_zip`）。

## 启动服务

```bash
conda activate basketball_classroom
uvicorn apps.teacher_ui.main:app --host 127.0.0.1 --port 8000
```
