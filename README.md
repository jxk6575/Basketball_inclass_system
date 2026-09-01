# Basketball In-Class System — v2.0

多机位篮球课堂辅助教学系统。

- 四机位独立感知（cam_01–03：**YOLO11m-Pose** + **RTMW-l**；cam_04：球/筐）
- **v2 注册**：group0 在 **cam_01** 顺序正面注册 → 多人 gallery；后续组复用
- 事件对齐估常量时间偏移（出手峰 / 篮筐球段）
- 规则动作切分：`pass` | `triple_threat` | `free_throw` | **`jump_shot`** | `layup`
- 身份：默认 **`face_body_color`**（人脸 + OSNet + 衣服色）；`realtime`/`full` 不改 match_mode，只改跳帧/球检分辨率/viz
- cam_04 进球：球心门控 + 筐沿橙色遮挡否决 + 轨迹 make/miss
- 批处理：`realtime`（精简）/ `full`（全量 + viz）

版本见根目录 [`VERSION`](./VERSION)（当前 **2.0.7**）。源码快照：`versions/v2.0.7/`（另有历史 `versions/v1/`、`versions/v2/`）。

## 快速开始

```bash
chmod +x scripts/setup_gpu_env.sh
./scripts/setup_gpu_env.sh
conda activate basketball_classroom
python scripts/download_models.py          # 人体/姿态/ReID；MotionBERT 默认跳过
python -c "from src.privacy.db import init_db; init_db()"
```

详见 [docs/环境配置GPU.md](docs/环境配置GPU.md)。

## 文档

| 文档 | 用途 |
|------|------|
| [docs/README.md](docs/README.md) | 文档索引 |
| [系统架构与 Pipeline](docs/系统架构与Pipeline.md) | **架构真相源** |
| [输出格式设计](docs/输出格式设计.md) | 导出 schema |
| [球场标定指南](docs/球场标定指南.md) | 控制点标定 |
| [环境配置 GPU](docs/环境配置GPU.md) | 安装与模型 |

## 主入口

| 场景 | 命令 |
|------|------|
| **v2 测试集**（先注册再罚篮） | `PYTHONPATH=. python scripts/run_v2_testset.py --groups 0,1 --mode realtime` |
| v2 仅注册 | `PYTHONPATH=. python scripts/run_v2_testset.py --groups 0 --enroll-only` |
| v2 全量 | `PYTHONPATH=. python scripts/run_v2_testset.py --groups all --mode full` |
| **v3 测试集**（默认 full + viz） | `PYTHONPATH=. python scripts/run_v3_testset.py --groups all` |
| v3 仅补渲 viz/dashboard | `PYTHONPATH=. python scripts/run_v3_testset.py --groups 1,2,3,4,5 --rerender-viz` |
| v1 测试集批处理 | `PYTHONPATH=. python scripts/run_v1_testset.py --groups all --mode full` |
| 正式 session | `PYTHONPATH=. python pipelines/run_session.py --session-id <uuid> --from-stage perception --init-db` |
| 事件对齐 | `PYTHONPATH=. python scripts/sync_cameras.py --session <uuid> --student stu_00` |
| 球场标定 | `PYTHONPATH=. python scripts/calibrate_court.py …` |
| Dashboard | `PYTHONPATH=. python scripts/build_group_dashboard.py --all-v1` |
| 教师端 | `python -m apps.teacher_ui.main` |
| 单测 | `PYTHONPATH=. python tests/test_pipeline.py && PYTHONPATH=. python tests/test_v2.py` |

输出目录：
- v3：`data/outputs/v3/group_0X/`（含 viz 视频 + dashboard；`--mode full` 默认渲 viz）
- v2：`data/outputs/v2/group_0X/` + `gallery_manifest.json`
- v1：`data/outputs/v1/group_0X/`

## 动作判别要点（v2）

- **投篮族**（`free_throw` / `jump_shot` / `layup`）：cam_04 球心在筐心上门控
  - 助跑近筐 → `layup`
  - 站定 + **纵跳** → `jump_shot`
  - 站定无明显跳起 → `free_throw`
- **传球**：双手胸前姿态为主；须画面 ≥2 人且顺序归因
- **三威胁**：肢体为主（蹲姿 / 重心下降 + 手部准备）；含突破

## 隐私

- 视频与 embedding 仅存本地 `data/`
- 知情同意后方可分析；支持撤回与级联删除
