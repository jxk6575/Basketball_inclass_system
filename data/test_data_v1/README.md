# test_data_v1

四机位课堂录像（`{group}-{cam}.mkv`，cam=`1..4` → `cam_01..04`）。

## 切分 / 裁剪说明

### group5（2026-08-30）

工作集 `5-1.mkv` … `5-4.mkv` 已裁掉前 **24 秒**（四机位同等裁切，相对 sync offset 不变）。

| 项 | 说明 |
|----|------|
| 裁前时长（约） | 82.8–83.3s |
| 裁后时长（约） | 58.8–59.3s |
| 原片备份 | `_archive_full/5-*-full.mkv`（可删，仅作备份） |
| sync | `sync/group_05.json`：`duration_ms` 已刷新；`camera_time_offsets_ms` 未改 |
| 真值时间轴 | 见 `docs/v1测试集动作真值.md` Group 5（已按 −24s 重标） |

重新从备份裁切示例：

```bash
for c in 1 2 3 4; do
  ffmpeg -y -i data/test_data_v1/_archive_full/5-${c}-full.mkv -ss 24 \
    -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p -an \
    data/test_data_v1/5-${c}.mkv
done
```

## 使用约定

- 时间同步：`data/test_data_v1/sync/group_XX.json`（可用 `scripts/sync_group_gui.py`）
- 批处理：`scripts/run_v1_testset.py`
