# test_data_v3

四机位课堂录像（`{group}-{cam}.mkv`，cam=`1..4` → `cam_01..04`）。

## 切分说明（2026-08-06）

原始 **group1** 整段约 80s：前 **12s** 为注册（全员面向 **cam_02**），之后为动作段。已拆为：

| 组 | 文件 | 内容 | 时长（约） |
|----|------|------|------------|
| **group0** | `0-1.mkv` … `0-4.mkv` | 每人面向 **cam_02** 正面注册 | ~12s |
| **group1** | `1-1.mkv` … `1-4.mkv` | 轮流投篮等（见真值文档） | ~68s |
| group2–7 | `2-*` … `7-*` | 见 [docs/v3测试集动作真值.md](../../docs/v3测试集动作真值.md) | — |

完整未切原片保留在 `_archive_full/`（可删，仅作备份）：
- `1-*-full.mkv`：group1 未拆前整段
- `3-*-full.mkv`：group3 未裁前（工作集已去掉前 **30s**）
- `7-*-full.mkv`：group7 未裁前（工作集已去掉前 **40s**）

## 使用约定

- **注册 gallery**：从 **group0 / cam_02** 做顺序正面注册，得到多名 `student_id`。
- **group1+ 分析**：复用 group0 gallery 做 ReID 与动作切分。
- **四机位时间同步（推荐先做）**：v3 无音轨，事件自动同步依赖感知且常不稳定。请用可视化工具人工对齐同一 group 的 4 路：

  ```bash
  PYTHONPATH=. python scripts/sync_group_gui.py --data-dir data/test_data_v3 --group 1
  ```

  保存后写入 `data/test_data_v3/sync/group_01.json`；跑测试集时会自动注入 session。也可事后：

  ```bash
  PYTHONPATH=. python scripts/sync_cameras.py --session <uuid> --from-group 1 --data-dir data/test_data_v3
  ```

- 批处理：后续 `scripts/run_v3_testset.py`（开发中）；现阶段可参考 v2 入口并改 `--data-dir` / 注册机位。

## 动作真值

详见 **[docs/v3测试集动作真值.md](../../docs/v3测试集动作真值.md)**。
