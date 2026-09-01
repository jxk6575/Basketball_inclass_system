# 篮球课堂辅助教学系统 — 系统架构与完整 Pipeline

> **版本：v2.0** · 更新：2026-07-24  
> 关联：[GPU 环境](./环境配置GPU.md) · [练习区域](../configs/zones.yaml) · [输出格式](./输出格式设计.md) · [历史调研](./调研报告_三维骨架提取与ReID.md)

---

## 1. 设计前提

| 项 | 约定 |
|----|------|
| 机位 | **cam_01~04** 四路固定机位，职责分离（见 §1.2） |
| 人体检测 | **YOLO11m-Pose**（`configs/models.yaml` → `yolo_pose`）；精细姿态 **RTMW-l**，质量不过时回退 YOLO COCO-17→133 |
| 帧同步 | **事件对齐**：出手峰 / 篮筐球段估常量 \(\Delta t\)；每路独立 `frame_idx` + `timestamp_ms` |
| 融合 | 出手：多机位 pose 峰 × cam_04「球心在筐心上方」；进球：cam_04；3D 按 **锚时钟 + offset** 取最近帧 |
| 身份 | **v2 默认 `face_body_color`**（正面注册用人脸+身体+衣服色）；`realtime`/`full` **不改变** match_mode，仅改跳帧/球检分辨率/viz。`body_color` 仍可选（跳过人脸以加速） |
| 动作语义 | **规则切分**；规范标签 `pass` \| `triple_threat` \| `free_throw` \| **`jump_shot`** \| `layup` |
| 关节角 | 设计目标 **Pose2Sim 真 3D**（当前 demo 用伪 3D / 标定三角化）；见 `docs/球场标定指南.md` |
| 进球判定 | **cam_04**：球心门控 → clip 贪心对齐 → **筐沿遮挡否决 + 轨迹** make/miss |
| 批处理模式 | `realtime`（精简/近实时）与 `full`（全分辨率 + viz）；v2 见 `scripts/run_v2_testset.py` |
| 近实时 | 目标：动作结束后 **≤10s** 给出类型 + 命中（见 §9.1） |

### 1.2 四机位职责

| ID | 物理位置 | 覆盖区域 | 主责 |
|----|----------|----------|------|
| **cam_01** | 左侧边线 | 整个三分区（左半） | **v2 顺序正面注册**、多人跟踪、侧身 3D 辅助 |
| **cam_02** | 右侧边线 | 整个三分区（右半） | 同上（镜像） |
| **cam_03** | 底线 | 罚球点 | 动作切分主时钟、正面 3D |
| **cam_04** | 篮筐附近 | 筐与出手细节 | 跟随动作、**进球/罚篮是否命中** |

```
         [cam_04 篮筐]
              |
   cam_01 ●--+--● cam_02
  (左三分区) | (右三分区)
              |
         ● cam_03
        (底线罚球)
```

配置：[`configs/cameras.yaml`](../configs/cameras.yaml) · [`configs/zones.yaml`](../configs/zones.yaml)

### 1.3 练习区域 ↔ 主责机位

| 区域 | 主责机位 | 辅助机位 |
|------|----------|----------|
| 左侧三分练习 | cam_01 | cam_02, cam_04 |
| 右侧三分练习 | cam_02 | cam_01, cam_04 |
| 罚球线 | cam_03 | cam_01, cam_02, cam_04 |
| 进球判定 | cam_04 | — |

---

## 2. 系统架构（按机位分离）

```
┌──────────────────────────────────────────────────────────────────┐
│ L0 治理：Consent · Audit · Retention                              │
├──────────────────────────────────────────────────────────────────┤
│ L1 采集：4 路独立录制（允许不同帧数/时长）                             │
├──────────────────────────────────────────────────────────────────┤
│ L2 感知（×4 并行、隔离）                                            │
│     cam_01~03 ──► YOLO11m-Pose 检测 + RTMW-133（失败回退 YOLO COCO-17）│
│     cam_04    ──► YOLO 球/筐（不跑人体骨架）                         │
├──────────────────────────────────────────────────────────────────┤
│ L3 时间对齐：sync/alignment.json（非帧号对齐）                       │
├──────────────────────────────────────────────────────────────────┤
│ L4 几何：Pose2Sim 时间戳融合 → 3D .trc → 视角无关关节角               │
├──────────────────────────────────────────────────────────────────┤
│ L5 语义：规则动作切分（区域主责机位 cam_03 等）                        │
├──────────────────────────────────────────────────────────────────┤
│ L6 进球：cam_04 YOLO 球/筐 + 轨迹 make/miss                         │
├──────────────────────────────────────────────────────────────────┤
│ L7 评分：参考模板 × 3D 角 + 可选进球结果                             │
├──────────────────────────────────────────────────────────────────┤
│ L8 呈现：Report · Teacher UI                                      │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Pipeline Stage 0–9

| Stage | 名称 | 说明 |
|-------|------|------|
| 0 | 合规 | Session + consent |
| 1 | 标定 | 四机位 calibration |
| 2 | 注册 | **cam_01** 顺序正面 → 多人 gallery（v2）；v1 曾用 cam_03 |
| 3 | 录制 | 4 路独立 mp4 + `sync_meta.json` |
| 4 | **逐机位感知** | 每路独立 pipeline，输出本地 frame + `timestamp_ms` |
| 5 | **时间对齐** | `sync/alignment.json`，clip 跨机位映射 |
| 6 | **3D 融合** | 按时间戳采样多视 2D → 三角化 → `angles.json` |
| 7 | **动作切分** | 区域主责机位规则切 phase（含 `start_ms`/`end_ms`） |
| 8 | **进球判定** | YOLO 球/筐轨迹 → `shot_outcomes/outcomes.json` |
| 9 | **评分报告** | 3D 角模板评分 + HTML |

状态机：

```
CREATED → … → RECORDED → PERCEPTION_DONE → SYNC_DONE →
  POSE3D_DONE → ACTION_DONE → SHOT_OUTCOME_DONE → SCORED → REPORT_READY
```

---

## 4. 身份识别（ReID）

### 4.1 生产默认（v2：多人注册）

| 环节 | 模型 / 策略 | 说明 |
|------|-------------|------|
| 人体检测 | YOLO11m-Pose | 权重：`models/detection/yolo_pose/yolo11m-pose.pt` |
| **注册（v2）** | cam_01 顺序正面 + OSNet / 衣服色 / 可选人脸 | `src/identity/sequential_enroll.py`；group0 产出多人 gallery |
| 关联 | `FaceBodyTracker`：IoU + 外观 | **不用 BoT-SORT/ByteTrack 作主路径** |
| 匹配模式 | `match_mode: body_color` | 跟踪阶段默认偏重身体+颜色；注册帧仍写入 face 样本 |

可选模式：`face_body_color` / `face_body`（正脸时 α 抬高，侧脸偏重身体）。

关键配置（`configs/cameras.yaml` → `identity`）：

| 参数 | 当前值 | 含义 |
|------|--------|------|
| `match_mode` | `body_color` | 生产默认 |
| `gallery_match_cost_threshold` | 0.52 | gallery 匹配代价上限 |
| `clothing_color_weight` | 0.45 | 衣服颜色在代价中的权重 |
| `ambiguity_margin` | 0.05 | 与第二名代价差；过小则拒识 |
| `face_alpha_high` | 0.85 | 仅 `face_body*` 模式：正脸人脸权重 |
| `face_score_threshold` | 0.8 | 判定「正脸」的阈值 |

代码：`src/identity/sequential_enroll.py` · `tracker.py` · `clothing_color.py` · `perception.py` · `enrollment.py`

### 4.2 跨机位原则

- 每机位 `track_id` 独立，**跨机位只认 `student_id`**
- 融合与评分阶段按 `student_id` + `timestamp_ms` 关联，不用 `track_id`

### 4.3 动作上的学生编号

每个 `ActionClip` 写入：

| 字段 | 含义 |
|------|------|
| `student_id` | 主执行者（传球为传球者） |
| `participant_ids` | 参与者列表；**传球为 `[passer, receiver]` 两人** |

- **v2**：`scripts/run_v2_testset.py` 先跑 **group0** 顺序注册（`stu_00`…），再把 gallery 拷到 group1+ session。
- **v1**：从 cam_03 最多注册 2 人（`stu_gXX` / `stu_gXX_p2`）。
- 传球双方靠窗内早段/晚段球距归因（`src/action/participants.py`）。
## 5. 动作切分（规则）

主逻辑：`src/action/pipeline.py`（统一入口）· `detect.py` · `multicam_release.py` · `pose_only.py`

### 5.0 核心原则（必须遵守）

1. **动作类型始终由系统自行判别**，不得依赖组号、课表或人工预标注。
2. **cam_04 篮筐/球轨迹只是投篮类动作的辅助**（出手候选门控 + 进球判定），**不参与** free_throw vs layup 的标签决策。
3. 其它动作（传球、三威胁/突破）走 **pose-only** 检测器，不依赖 cam_04。
4. 规范标签（`src/action/registry.py`）：`pass` | `triple_threat` | `free_throw` | **`jump_shot`** | `layup`（`triple_threat` 含突破；无独立 `dribble`）。
5. 扩展新动作：在 `pipeline.DEFAULT_DETECTORS` 注册检测器 + 在 `registry.py` 声明是否需要 rim 辅助。

```
pose (cam_01/02/03)
    ├─ shooting detector → release 候选 × cam_04 球心在筐心上方
    │                         → classify_release_action
    │                              → layup | jump_shot | free_throw
    └─ pose_only detector → pass / triple_threat
            ↓
      ActionClip.action_type（auto_classify）
            ↓
      仅 shooting → cam_04 make/miss（贪心对齐）
```

### 5.1 出手检测（多机位融合，仅投篮族）

主逻辑：`src/action/multicam_release.py`

1. 在 **cam_01 / cam_02 / cam_03** 上分别找右手腕高度局部峰（须高于肩与鼻）
2. 按时间戳聚类（约 0.9s）：同一出手应在多机位同时出现
3. 用 **cam_04**「球心在筐心上方」事件与聚类做 **贪心对齐**；无球事件则 **不产出** 投篮 clip
4. **每条候选**调用 `classify_release_action`：
   - 人→筐距离缩短 + 助跑 → `layup`
   - 站定 + **纵跳**（pelvis/ankle 上升）→ `jump_shot`
   - 站定无明显跳起 → `free_throw`
   - cam_04 **不参与** 三类标签决策
5. 主时钟写回 **cam_03** 的 `ActionClip`（`metadata.action_classify` / `multicam`）

### 5.1b 非投篮（pose-only，cam_01–03）

主逻辑：`src/action/pose_only.py` → `classify_pose_only_window`

| 标签 | 主线索 |
|------|--------|
| `pass` | 双手抬至胸前为主；窗中段球飞过 / 离手为辅；须 ≥2 人 + 顺序归因 |
| `triple_threat` | **肢体为主**：蹲姿 / 重心下降 + 手部准备；持球仅为软辅助（球检召回不足时不硬门控） |

传球检测依赖 **cam_01–03** 同机位球轨（不可仅靠 cam_04）。骨架绘制前经 `src/pose/skeleton_quality.py` 过滤离谱框外手臂等。

### 5.2 阶段划分

投篮类以 release 为锚点（罚篮）：

| 阶段 | 帧范围 |
|------|--------|
| load | release 前 40 帧 → 中点 |
| set | 中点 → release-3 |
| release | release±2 |
| follow_through | release+2 → release 后 25 帧 |

三步上篮（`layup`）自动使用更长起步窗：approach → gather → takeoff → release → finish。

### 5.3 评分用关节角

检测仅用手腕；打分在 `src/pose/angles.py` 计算肘、膝、腕角及 `wrist_height_m`。模板定义见 `configs/actions/*.yaml`。

球轨几何公共模块：`src/shot/track_geometry.py`（hoop / ball samples / shot-like segments）。

---

## 6. 参考模板

### 6.1 库里罚篮三维模板

路径：`data/templates/curry_free_throw.json`

| 字段 | 说明 |
|------|------|
| `joint_format` | H36M-17（骨盆、髋膝踝、肩肘腕等） |
| `coordinate_system` | 伪 3D，骨盆居中、颈-骨盆距离归一化 |
| `phases` | load / set / release / follow_through 各含 `joints_3d` + `angles` |
| `source_video` | `Curry_shoot_1.mp4`，release 帧 159 |

加载：`src/pose/reference_template.py` → `load_reference_template("curry_free_throw")`

### 6.2 动作评分模板

YAML 规范：`configs/actions/free_throw.yaml`（罚篮）· `layup.yaml` · `triple_threat.yaml`

---

## 7. 帧不对齐：如何处理

### 7.1 原则

- 每路视频可有**不同帧数、起止时间、fps 抖动**
- Stage 4 **禁止**用 `frame_idx` 跨机位对齐
- 统一使用 `timestamp_ms = frame_idx / fps × 1000`
- **事件对齐（默认）**：以 cam_03 出手峰为主时钟，与各机位出手峰 / cam_04 球段配对，估常量偏移  
  `common_ms = local_ms - offset_ms`（见 `sync/alignment.json`）

### 7.2 融合流程

```
cam_X pose2d + cam_04 ball_track
        ↓
事件对齐 run_temporal_alignment / scripts/sync_cameras.py
        ↓
sync/alignment.json  (camera_time_offsets_ms)
        ↓
动作 clip 锚定在主责机位 [start_ms, end_ms]（锚时钟）
        ↓
align_clips_across_cameras() → local_ms = common + offset
        ↓
collect_student_kpts_at_time(ts) → 三角化
```

CLI：

```bash
PYTHONPATH=. python scripts/sync_cameras.py --session <uuid> --student stu_g01
```

---

## 8. 模块与代码路径

| 模块 | 路径 |
|------|------|
| 机位注册表 | `src/cameras/registry.py` |
| 时间对齐 | `src/cameras/temporal.py` · `src/cameras/event_sync.py` |
| 单机位感知 | `src/perception/camera_pipeline.py` · `yolo_pose_detector.py` |
| ReID | `src/identity/`（`clothing_color.py` · `tracker.py`） |
| 骨架质量过滤 | `src/pose/skeleton_quality.py` |
| 规则动作 | `src/action/pipeline.py` · `multicam_release.py` · `pose_only.py` · `registry.py` |
| 近实时 | `src/streaming/fast_path.py` |
| 3D 融合 | `src/pose/pose2sim_wrapper.py`（session stub）· `src/pose/triangulate.py`（标定 DLT / viewer） |
| 球场标定 | `src/calibration/` · `scripts/calibrate_court.py` |
| 参考模板 | `src/pose/reference_template.py` |
| 进球 | `src/shot/outcome.py` · `geometry.py` · `track_geometry.py` · `tracker.py` · `yolo_detector.py` |
| 评分 | `src/scoring/fusion.py` |
| 编排 | `src/orchestrator/session_pipeline.py` |
| **Session 导出格式** | `src/output/schema.py` · [`docs/输出格式设计.md`](./输出格式设计.md) |

### 8.1 Session 数据目录

```
data/sessions/{id}/
  raw/                    cam_01.mp4 … cam_04.mp4
  perception/cam_XX/      独立 detections + pose2d
  sync/                   alignment.json, clip_align_*.json
  pose3d/                 时间对齐 3D
  angles/                 视角无关关节角
  actions/                规则 clips（含 ms）
  shot_outcomes/          ball_track.json + outcomes.json
  reports/
```

### 8.2 正式 v1 测试集输出

见 §10.2；根目录 `data/outputs/v1/`，每组含 `motion.json`（SessionOutput）、`report.json`、`dashboard.html` 与 `viz/`。

---

## 9. cam_04 进球识别

模型：`models/detection/yolo_ball/Basketball_v1.pt`（cls0=球，cls1=筐；`imgsz=1280`）  
球选框：同帧取 **最高置信度球**（已弃用「近筐优先」启发式）。

主路径（clip 锚定，生产使用）：
1. `ensure_ball_track()` 写出 `shot_outcomes/ball_track.json`
2. shot-like 球段过滤：球 bbox 面积须 **小于** 筐面积（抑制定位误检）
3. `outcomes_from_clips_and_track`：投篮 clip ↔ 球段 **贪心对齐**
4. `evaluate_make_miss`（`src/shot/outcome.py`）：
   - **筐沿橙色遮挡否决**（`rim_occlusion_indicates_miss`：球下行时上沿橙色被挡住 → 硬 miss；进球不会挡住上沿）
   - 否则 **轨迹证据**（穿筐 / 筐柱体 / 反弹 / 侧向逃逸等）

在线 `ShotTracker` 与离线路径共用同一 `evaluate_make_miss`；离线可按 `video_path` 抽帧做遮挡检查。

输出：
- `shot_outcomes/ball_track.json` — 逐帧球/筐位置
- `shot_outcomes/outcomes.json` — `made: true/false`（`scoring: rim_orange+trajectory`）

### 9.1 近实时 Fast-path（≤10s 教师反馈）

目标架构：always-on **cam_03 pose + cam_04 ball** → 环形缓冲 → 事件窗口 finalize → 立即返回类型/命中；3D / viz / dashboard 异步。

| 组件 | 路径 | 说明 |
|------|------|------|
| Finalize / 延迟估算 | `src/streaming/fast_path.py` | 对已感知 session 做 finalize 验证 |
| 环形缓冲类型 | `TimestampRingBuffer` | 为 always-on 采集预留；尚未接直播 |
| 延迟校验脚本 | `scripts/validate_fastpath_latency.py` | 读各组 `summary.json`，可写出延迟报告 |

```bash
PYTHONPATH=. python scripts/validate_fastpath_latency.py --groups 1,2,3,4,5,6,7,8
```

---

## 10. 运行

### 10.0 球场标定（三角化前置）

操作见 [球场标定指南](./球场标定指南.md)。产物默认：`data/calibration/v2_4cam_zoned/`。

- **已接通**：`scripts/build_pose3d_viewer.py` / `build_camera_scene_viewer.py` 可读标定做 metric 三角化与机位可视化  
- **未接通**：`pipelines/run_session.py` 的 pose3d 阶段仍走 `pose2sim_wrapper` stub（伪 3D / 占位）

### 10.1 正式四机位 Session

```bash
conda activate basketball_classroom
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# from_stage：从此阶段开始跑到结束（跳过更早阶段；需上游产物已存在）
python pipelines/run_session.py --session-id <uuid> --from-stage perception
# 可选：sync | pose3d | action | shot | scoring
```

### 10.2 正式 v1 四机位测试集

输入：`data/test_data_v1/{g}-{c}.mkv`（8 组 × 4 机位；`c`→`cam_0c`）。无人工标签；动作与进球由规则自动判定。身份：cam_03 自动注册（双人组最多 2 人）。**cam_04 仅跑球/筐，不跑人体骨架**。

```bash
conda activate basketball_classroom
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 快速精简 / 近实时
PYTHONPATH=. python scripts/run_v1_testset.py --groups all --mode realtime

# 离线全量（含 viz）
PYTHONPATH=. python scripts/run_v1_testset.py --groups all --mode full

# realtime 仍渲 viz / 仅重渲
PYTHONPATH=. python scripts/run_v1_testset.py --groups 7,8 --mode realtime --with-viz
PYTHONPATH=. python scripts/run_v1_testset.py --groups all --rerender-viz
```

| Flag | 含义 |
|------|------|
| `--mode realtime` | 精简：assist 跳帧 + pose 机位降本球轨，默认跳过 viz |
| `--mode full` | 全量：球轨全分辨率，默认渲 viz |
| `--with-viz` | realtime 下仍渲 viz |
| `--rerender-viz` | 不重跑感知，仅重渲 viz |
| `--skip-viz` | 跳过 viz |
| `--stride` | 帧步长（默认 2） |

输出：

```
data/outputs/v1/
  manifest.json
  group_01/ … group_08/
    motion.json / report.json / summary.json / dashboard.html
    skeleton3d_triangulated.json / skeleton3d_court_viewer.html   # 可选
    viz/
      cam_01_annotated.mp4 … cam_03_annotated.mp4
      cam_04_ball.mp4
      phases.mp4
    keyframes/
```

入口：`scripts/run_v1_testset.py` · 导出：`src/output/export.py`。

### 10.3 动作片段三角化骨架

```bash
PYTHONPATH=. python scripts/extract_action_skeletons_3d.py --groups 1,2,3,4 --stride 3
PYTHONPATH=. python scripts/build_group_dashboard.py --all-v1
```

### 10.4 环境验证

```bash
python scripts/verify_gpu_env.py
PYTHONPATH=. python tests/test_pipeline.py
PYTHONPATH=. python scripts/validate_fastpath_latency.py --groups 1,2,3,4
```

---

## 11. 能力清单

### v2.0

| 能力 | 状态 |
|------|------|
| cam_01 顺序正面多人注册 | ✅ `sequential_enroll` + group0 |
| 共享 gallery → group1+ ReID | ✅ `run_v2_testset.py` |
| 动作五类含 **jump_shot** | ✅ 纵跳 vs 站定罚篮 |
| 其余（感知 / 进球 / 批处理） | ✅ 继承 v1 |

### v1.0（基线）

| 能力 | 状态 |
|------|------|
| 四机位配置 + 按机位独立感知 | ✅ |
| 人体检测 YOLO11m-Pose + RTMW-l | ✅ |
| 身份 body ReID + 衣服颜色 | ✅ 默认 `body_color` |
| 时间戳 / 事件对齐 | ✅ |
| 规则动作：pass / triple_threat / free_throw / layup | ✅ |
| 三威胁：肢体为主、持球为辅 | ✅ |
| cam_04 球心门控 + 筐沿遮挡 / 轨迹 make-miss | ✅ |
| realtime \| full 批处理 + viz / dashboard | ✅ |
| Fast-path finalize + ≤10s 校验 | ✅ |
| SessionOutput 导出 | ✅ |
| Always-on 直播环形缓冲接入 | ⏳ 类型已预留 |
| Pose2Sim 真三角化进 session | ⏳ stub；viewer 已可用标定 DLT |
| Face-Body 双模态 | ✅ 代码保留，非默认 |
