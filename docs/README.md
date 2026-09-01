# 文档索引 — 系统 v2.0

> 真相源优先：代码与 `configs/` > 本目录「现行」文档 > 历史调研/计划稿

## 现行（请以此为准）

| 文档 | 内容 |
|------|------|
| [系统架构与 Pipeline](./系统架构与Pipeline.md) | **架构真相源**：注册 / ReID / 动作（含 jump_shot）/ 进球 |
| [v2 测试集动作真值](./v2测试集动作真值.md) | group0–6 真值标签（效果验证基准） |
| [v2 验证报告](./v2验证报告.md) | 批跑结果 vs 真值对照与迭代记录 |
| [输出格式设计](./输出格式设计.md) | SessionOutput schema |
| [球场标定指南](./球场标定指南.md) | cam_01–03 控制点标注与求解 |
| [环境配置 GPU](./环境配置GPU.md) | conda / 模型 / 验证 |

## 历史（勿当现行方案）

| 文档 | 说明 |
|------|------|
| [调研报告（三维骨架与 ReID）](./调研报告_三维骨架提取与ReID.md) | 早期选型；生产已改为规则切分 |
| [系统架构与 pipeline.plan](./系统架构与pipeline.plan.md) | 早期实施计划；已被 Pipeline.md 取代 |

## 常用命令

```bash
./scripts/setup_gpu_env.sh && conda activate basketball_classroom
python scripts/download_models.py

# v2：group0 注册 → group1 罚篮（推荐）
PYTHONPATH=. python scripts/run_v2_testset.py --groups 0,1 --mode realtime
PYTHONPATH=. python scripts/run_v2_testset.py --groups all --mode full

# v1 四机位批处理（保留）
PYTHONPATH=. python scripts/run_v1_testset.py --groups all --mode full

# 标定 / 对齐 / dashboard
PYTHONPATH=. python scripts/calibrate_court.py solve \
  --ann data/calibration/annotations.json --out data/calibration/v2_4cam_zoned
PYTHONPATH=. python scripts/sync_cameras.py --session <uuid> --student stu_00
PYTHONPATH=. python scripts/build_group_dashboard.py --all-v1

# 单测
PYTHONPATH=. python tests/test_pipeline.py
PYTHONPATH=. python tests/test_v2.py
```
