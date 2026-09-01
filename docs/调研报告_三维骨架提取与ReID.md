# 篮球课堂辅助教学系统 — 三维骨架提取与 ReID 技术调研

> 调研日期：2026-07-06  
> **状态：历史调研文档** — 生产架构已演进为「规则动作切分 + 事件对齐（event_anchor）+ YOLO 进球」。  
> 现行方案请读：[系统架构与 Pipeline](./系统架构与Pipeline.md) · [文档索引](./README.md)  
> 下文保留选型对比与模型链接，**其中 MotionBERT 分段 / 硬件帧同步等建议已非默认路径**。

> 目标场景（调研当时）：纯视觉方案，检测学生**罚篮、三步上篮、三威胁突破**等动作是否规范；需支持**手部/手腕**关节；需**多学生 ReID**。

---

## 1. 需求拆解与系统架构建议

### 1.1 核心模块

| 模块 | 功能 | 关键指标 |
|------|------|----------|
| 人体检测 | 定位多名学生 | 实时性、遮挡鲁棒 |
| 姿态估计 | 2D/3D 骨架 + 手部 | 133 点全身、手腕角度 |
| ReID / 跟踪 | 跨帧/跨镜身份一致 | 人脸+衣着双模态、ID Switch 低 |
| 动作分段 | 识别投篮/上篮/突破起止 | 时序建模 |
| 规范评估 | 与标准模板比对 | 关节角、相位、速度 |

### 1.2 推荐总体流水线

```
摄像头(单/多) → 检测(YOLO/RTMDet) → 姿态(RTMW3D / 多视三角化)
    → 人脸+身体融合跟踪(InsightFace + CLIP-ReID) → 动作分段(MotionBERT)
    → 关节角计算 → 与标准动作模板比对 → 反馈
```

**单摄像头 + 实时**：RTMW3D（133 点含手）+ BoT-SORT-ReID + 规则/模板评分  
**多摄像头 + 高精度**：Pose2Sim / EasyMocap 多视三角化 + OpenSim 关节角  
**离线精细分析**：SMPLer-X / Hand4Whole++ 输出 SMPL-X mesh，再算手腕欧拉角

---

## 2. 三维骨架提取方案

### 2.1 方案分类概览

| 类别 | 代表方法 | 手部支持 | 实时性 | 3D 精度 | 商用许可 |
|------|----------|----------|--------|---------|----------|
| 单目 3D 关键点 | RTMW3D | ✅ 21×2 手点 | ⭐⭐⭐⭐ | 中（相对深度） | Apache 2.0 |
| 单目 SMPL-X mesh | SMPLer-X, AiOS, Hand4Whole++ | ✅ MANO 手参 | ⭐⭐ | 高 | 研究/需 SMPL-X 许可 |
| 单目视频时序 | WHAM, MotionBERT | ⚠️ 17 体点为主 | ⭐⭐⭐ | 中高 | 研究 |
| 多视三角化 | Pose2Sim, EasyMocap | ✅ 可选 133 点 | ⭐⭐ | 高（米级） | 开源 |
| 2D 全身 + 后处理 | DWPose, MediaPipe | ✅ 2D 手点 | ⭐⭐⭐⭐⭐ | 需额外 3D 步骤 | 各异 |
| NVIDIA 新方案 | GEM-X (SOMA 77 点) | ✅ 手+脸 | ⭐⭐ | 高 | **Apache 2.0** |

> **手腕弯曲角度**：若只需肘-腕-指夹角，133 点 2D/3D 关键点足够；若需精细掌指姿态，建议 SMPL-X/MANO 参数化手模型。

---

### 2.2 首选推荐：RTMW3D + rtmlib（实时单目）

**适合**：课堂单/双摄像头、需实时反馈、要手部关键点。

| 项目 | 链接 |
|------|------|
| RTMPose3D 官方 | https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose3d |
| 论文 RTMW | https://arxiv.org/html/2407.08634v1 |
| 轻量部署 rtmlib | https://github.com/Tau-J/rtmlib |
| HuggingFace 模型 | https://huggingface.co/rbarac/rtmpose3d |

**关键点定义（133 点 COCO-WholeBody）**：
- 0–16：身体（含左右腕 index 9/10）
- 91–111：左手 21 点
- 112–132：右手 21 点

**性能参考**（RTX 3090）：
- RTMW3D-L：MPJPE ~0.045 m，约 30 FPS
- RTMW3D-X：更高精度，速度略慢

**优势**：
- 端到端单目 3D，无需标定
- rtmlib 仅依赖 ONNXRuntime，无 mmcv 重依赖
- 与 DWPose/RTMPose 生态统一，篮球论文已验证 RTMPose 可行性

**局限**：
- 单目 Z 轴为相对深度，绝对尺度/世界坐标需额外标定或多视
- 快速运动、严重遮挡时手部点易抖动

**快速上手（rtmlib）**：
```python
from rtmlib import PoseTracker, Wholebody3d
import cv2

tracker = PoseTracker(
    Wholebody3d,
    mode='balanced',
    backend='onnxruntime',
    device='cuda'
)
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    keypoints, scores = tracker(frame)  # keypoints: [N, 133, 3]
```

**篮球相关验证**：  
MDPI 2025 论文 *Feasibility and Accuracy of an RTMPose-Based Markerless Motion Capture System for Single-Player Tasks in 3x3 Basketball* 使用 RTMPose + 多视 DLT 做 3x3 篮球位移/速度分析，证明 RTMPose 系列在篮球场景可行（单视精度仍有限）。  
- 论文：https://www.mdpi.com/1424-8220/25/13/4003  
- PMC：https://pmc.ncbi.nlm.nih.gov/articles/PMC12252153/

---

### 2.3 多摄像头高精度：Pose2Sim / EasyMocap

**适合**：固定球场多机位、需要**米级 3D 坐标**和**OpenSim 级关节角**（罚篮肘角、膝角、腕角）。

#### Pose2Sim（强烈推荐体育场景）

| 项目 | 链接 |
|------|------|
| GitHub | https://github.com/perfanalytics/pose2sim |
| 文档 | https://perfanalytics.github.io/pose2sim/ |

**特点**：
- 专为**运动场/户外**设计，支持手机/GoPro/网络摄像头
- 内置 RTMPose，支持 `Whole_body`（133 点）和 **`Whole_body_wrist`**（体+脚+每手 2 腕点，更快）
- 完整流程：2D 姿态 → 多视关联 → 三角化 → 滤波 → **OpenSim 关节角**
- v0.7+ 支持多人；v0.10+ 集成 OpenSim

**手腕模式**：Config.toml 中 `pose_model = 'Whole_body_wrist'`，保留手部关键腕点，忽略面部/指节，兼顾速度与手腕运动。

#### EasyMocap

| 项目 | 链接 |
|------|------|
| GitHub | https://github.com/zju3dv/EasyMocap |
| 文档 | https://chingswy.github.io/easymocap-public-doc/ |

**特点**：
- 多视多人 SMPL/SMPL-X 拟合，ZJU-MoCap 级方案
- 支持 YOLO + HRNet/ViTPose/MediaPipe 2D 检测
- 适合已有标定、追求 mesh 级重建

**对比**：

| 维度 | Pose2Sim | EasyMocap |
|------|----------|-----------|
| 上手难度 | 中（文档完善） | 中高 |
| 关节角输出 | OpenSim 直接输出 | 需 SMPL 后处理 |
| 手部 | 133 点 / wrist 模式 | SMPL-X 手参 |
| 体育适配 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

---

### 2.4 单目 Mesh 级（精度优先，非实时）

#### SMPLer-X（NeurIPS 2023，当前 EHPS 标杆之一）

| 项目 | 链接 |
|------|------|
| GitHub | https://github.com/MotrixLab/SMPLer-X |
| 论文 | https://arxiv.org/html/2309.17448v3 |
| 模型 | https://huggingface.co/caizhongang/SMPLer-X |

- 输出 **SMPL-X**（体+MANO 手+FLAME 脸）
- ViT-H 约 17 FPS（V100），Smaller 模型可近实时
- 需注册下载 SMPL-X/MANO 模型文件
- 手腕角度：从 MANO `global_orient` + `hand_pose`（15×3 轴角）直接读取

#### Hand4Whole++（CVPR 2026，手部增强）

| 项目 | 链接 |
|------|------|
| GitHub | https://github.com/mks0601/Hand4Whole-plus-plus_RELEASE |
| 论文 | https://arxiv.org/abs/2603.14726 |

- 专注提升**全身估计中的手部精度**
- 与 SMPLer-X 组合使用，适合投篮「压腕」「跟随动作」细评

#### AiOS（CVPR 2024，单阶段多人）

| 项目 | 链接 |
|------|------|
| GitHub | https://github.com/MotrixLab/AiOS |
| 论文 | https://arxiv.org/html/2403.17934v1 |

- **无需单独检测器**，DETR 式一次输出多人 SMPL-X
- 拥挤课堂场景有优势；速度低于 RTMW3D

#### GEM-X（NVIDIA，2025，商用友好）

| 项目 | 链接 |
|------|------|
| GitHub | https://github.com/NVlabs/GEM-X |
| 模型 | https://huggingface.co/nvidia/GEM-X |

- **77 点 SOMA 骨架**（体+手+脸），**Apache 2.0**
- 单目相机/世界坐标运动恢复
- 自带 2D 检测器，适合长期产品化

---

### 2.5 视频时序 3D（动作识别友好）

#### WHAM（CVPR 2024）

| 项目 | 链接 |
|------|------|
| 主页 | https://wham.is.tue.mpg.de/ |
| 论文 | https://arxiv.org/html/2312.07531v1 |

- 单目视频 → **世界坐标系** SMPL 运动
- 适合三步上篮**位移轨迹**、起跳/落地分析
- 手部为 SMPL 体模型，精细手指不如 SMPL-X

#### MotionBERT（ICCV 2023）

| 项目 | 链接 |
|------|------|
| GitHub | https://github.com/Walter0807/MotionBERT |
| 主页 | https://motionbert.github.io/ |

- 2D 序列 → 3D 姿态 + **动作识别** + mesh 恢复
- 17 点 H36M 格式，**不含手指**；适合突破/上篮**阶段分类**
- 可与 RTMPose 2D 串联：`RTMPose 2D → MotionBERT 3D + Action Head`

---

### 2.6 2D 全身（高帧率备选 / 3D 前置）

#### DWPose（ICCV 2023 Workshop）

| 项目 | 链接 |
|------|------|
| GitHub | https://github.com/IDEA-Research/DWPose |
| 论文 | https://arxiv.org/abs/2307.15880 |

- COCO-WholeBody 133 点 2D，蒸馏自 RTMPose，**速度与精度平衡**
- ONNX 分支可脱离 mmcv：`/onnx` 或 `opencv_onnx`
- 可与多视 DLT / Pose2Sim 组合得 3D

#### MediaPipe Holistic / BlazePose

| 项目 | 链接 |
|------|------|
| 文档 | https://google.github.io/mediapipe/solutions/pose.html |

- **33 体点 + 21×2 手点**，CPU 实时极强
- 3D 为世界坐标但**相对深度**，绝对精度不如 RTMW3D
- 已有大量篮球投篮分析开源项目采用（FormFix、JNKS 等）

---

### 2.7 手腕/关节角计算方法

从 3D 关键点计算关节角的开源参考：

| 资源 | 链接 | 说明 |
|------|------|------|
| joint_angles_calculate | https://github.com/TemugeB/joint_angles_calculate | 3D 关键点 → 关节角 |
| 原理博客 | https://temugeb.github.io/python/motion_capture/2021/09/16/joint_rotations.html | T-pose 局部坐标系法 |
| UPose | https://github.com/digitalworlds/UPose | MediaPipe 坐标 → 关节角 |

**常用角度（篮球投篮）**：
- 肘角：肩–肘–腕
- 膝角：髋–膝–踝
- 腕角：肘–腕–中指 MCP（或掌指关节）
- 躯干倾角：肩中点–髋中点 与 竖直轴

**FormFix 参考 biomechanics**：罚球膝角约 122°、肘角约 79°（文献值，需按教学标准校准）。

---

## 3. Person ReID 与身份跟踪方案

> **场景前提（已更新）**：学生**不穿同款校服**，衣着差异大；身份跟踪应**对人脸 + 衣着均敏感**，而非依赖队服/部位 ReID。

### 3.1 课堂场景特殊挑战

- 学生**衣着各异** → 全身 ReID **区分度较高**（CLIP-ReID / TransReID 可直接利用服装颜色、版型）
- 投篮/上篮时**面部常侧向或不可见** → 不能单靠人脸；需 **人脸（高置信时）+ 身体 ReID（主）** 动态融合
- 肢体遮挡、多人交叉、短暂出画 → 需 **Long-term ReID + 跟踪 + 课前注册库**
- 多机位 → 主视角人脸注册，侧视角以 **身体 ReID + 3D 轨迹** 关联
- **合规**：人脸采集需告知同意，数据本地存储、不上云（见 §3.6）

---

### 3.2 推荐方案：Face-Body 融合身份模块（正式选型）

#### 3.2.1 组件选型

| 层级 | 组件 | GitHub / 链接 | 作用 |
|------|------|---------------|------|
| 人脸检测 | **SCRFD / RetinaFace**（InsightFace 内置） | https://github.com/deepinsight/insightface | 小脸、侧脸鲁棒 |
| 人脸特征 | **ArcFace**（InsightFace `buffalo_l` / `glintr100`） | 同上 | 512D 身份嵌入，跨帧/跨镜最稳 |
| 人体检测 | **RTMDet / YOLOv11** | MMPose / Ultralytics | 与 Pose2Sim 一致 |
| 身体 ReID | **CLIP-ReID** 或 **TransReID** | https://github.com/Syliz517/CLIP-ReID | 对**不同衣服**敏感；CLIP 泛化好 |
| 跟踪框架 | **StrongSORT** 或 **BoT-SORT-ReID** | https://github.com/dyhBUPT/StrongSORT | IoU + 双模态外观关联 |
| 融合参考 | **HumanRecognition** | https://github.com/oele-isis-vanderbilt/HumanRecognition | 人脸+身体 ReID 实时 pipeline 范例 |
| 模块化集成 | **TrackLab**（可选） | https://github.com/TrackingLaboratory/tracklab | 检测+ReID+跟踪 YAML 配置 |

#### 3.2.2 融合关联公式（核心逻辑）

对 tracklet 与 detection 计算综合代价：

```
cost = w_iou · C_iou + w_id · C_identity

C_identity = α · (1 - sim_face) + (1-α) · (1 - sim_body)

α = face_quality × face_visible    # 动态权重，非固定值
```

| 条件 | α 倾向 | 说明 |
|------|--------|------|
| 正面、清晰、人脸 score > 0.8 | **0.7–0.9** | 以 ArcFace 为主，身体辅助 |
| 侧脸 / 部分遮挡 | **0.3–0.5** | 身体 ReID 权重上升 |
| 投篮抬头看筐、背对相机 | **≈ 0** | 纯 **CLIP-ReID 身体** + IoU/3D 轨迹 |
| 重新入画 | 与**注册库**比对，不限当前 track | Long-term ReID |

**课前注册（Enrollment）**：每人采集 3–5 张正脸 + 8–12 张全身（多角度、含运服），存入 `{face_emb, body_emb}`  gallery，绑定 `student_id`。

#### 3.2.3 与 Pose2Sim 多视 pipeline 的衔接

```
各视角: 检测 → 人脸 ArcFace + 身体 CLIP-ReID → StrongSORT 得 local track_id
                    ↓
         与注册库匹配 → global student_id
                    ↓
Pose2Sim 多视 epipolar 匹配时，用 student_id 约束同一人跨镜三角化（替代纯几何匹配）
                    ↓
MotionBERT 按 student_id 拆 clip
```

主视角（正对队列/罚球线）负责人脸注册质量；侧视角在 face 不可见时，**body ReID + 3D 位置** 维持 ID。

---

### 3.3 其他 ReID 模型对比（备选 / 降级）

| 模型 | GitHub | 本场景优先级 | 说明 |
|------|--------|--------------|------|
| **CLIP-ReID** | https://github.com/Syliz517/CLIP-ReID | 🥇 身体主选 | 不同衣着区分强；小样本 fine-tune |
| **TransReID** | https://github.com/damo-cv/TransReID | 🥇 备选 | ViT，遮挡数据集表现好 |
| **InsightFace ArcFace** | https://github.com/deepinsight/insightface | 🥇 人脸主选 | 工业级；ONNX 可部署 |
| **FastReID** | https://github.com/JDAI-CV/fast-reid | 🥈 工程基座 | 训练/部署工具箱 |
| **BPBReID** | https://github.com/VlSomers/bpbreid | 🥈 遮挡补充 | 投篮抬手遮挡时，**身体**部位特征仍有帮助 |
| **KPR** | https://github.com/VlSomers/keypoint_promptable_reidentification | 🥉 可选 | 骨架引导 ReID，与 Pose2Sim 133 点可联合 |
| ~~**PRTreID**~~ | https://github.com/VlSomers/prtreid | ⬇️ 降级 | 专为**同款队服**设计，非本场景首选 |

**论文**：
- CLIP-ReID (AAAI 2023): https://arxiv.org/pdf/2211.13977.pdf  
- ArcFace (CVPR 2019): https://arxiv.org/abs/1801.07698  
- Face+Body 融合: https://doi.org/10.1109/imcec51613.2021.9482048  

---

### 3.4 跟踪 + ReID 集成（更新配置）

**正式推荐配置**：

```
检测:     RTMDet / YOLOv11
人脸:     InsightFace (det + ArcFace 512D)
身体 ReID: CLIP-ReID ViT-B（可选课堂 1 epoch fine-tune）
跟踪:     StrongSORT（dual embedding: face + body）
融合:     动态 α + 注册库 long-term 重关联
Pose2Sim: student_id 约束多视人物匹配
```

**TrackLab 配置思路**（若采用框架）：
- 检测 + RTMPose 用 TrackLab 内置 wrapper
- ReID 模块扩展为 **DualReID**：同时输出 `face_embedding` 与 `body_embedding` 列
- 跟踪器改用 StrongSORT 双特征版

**篮球动作下的 ID 策略**：

| 阶段 | 主要依据 |
|------|----------|
| 列队、准备 | 人脸 ArcFace（α 高） |
| 运球、突破 | 身体 CLIP-ReID + 轨迹 |
| 投篮出手 | 身体 ReID + Pose2Sim 3D 位置连续性 |
| 短暂遮挡后重现 | 注册库 body+face 联合检索 |

---

### 3.5 课堂落地步骤

1. **课前 5 分钟注册**：主摄像机前逐人采集 → 写入本地 gallery（face + body emb）  
2. **CLIP-ReID 可选微调**：用当天衣着 crop 做 1–3 epoch（非必须，零样本通常已够用）  
3. **运行 StrongSORT**：每帧更新 track，低置信时与 gallery 做 top-1 检索绑定 `student_id`  
4. **Pose2Sim 阶段**：各视角 2D 带 `student_id` 标签 → 多视三角化（减少 ID swap）  
5. **遮挡降级**：人脸不可见时自动切 body-only；BPBReID 可作为 body 分支增强（可选）

---

### 3.6 合规与隐私

- 人脸属于敏感生物特征：需**明示同意**、限定用途（仅课堂动作分析）  
- 建议：**边缘端本地推理**，注册库加密存储，课后可按策略删除  
- 若学校不允许存人脸：可退化为 **纯 CLIP-ReID 身体** + 课前只注册全身照（精度下降，仍可行）

---

### 3.7 旧版说明（同款校服场景，已不适用）

原调研中 PRTreID / 队服 ReID / 「同款服装区分度低」等假设**不再成立**。若未来有统一运服比赛场景，可再启用 PRTreID。

---

## 4. 篮球动作分析与已有开源项目

### 4.1 可直接参考的项目

| 项目 | 链接 | 技术栈 | 可借鉴点 |
|------|------|--------|----------|
| **FormFix** | https://github.com/adityasingh2400/formfix | MediaPipe Holistic | 投篮阶段分割、手腕/跟随动作、生物力学阈值 |
| **JNKS** | https://github.com/nathsmith-cs/JNKS | YOLOv8 + MediaPipe | 与球星模板对比、四阶段评分 |
| **Basketball Layup Test System** | https://github.com/jxk6575/Basketball_Layup_Test_System | YOLOv8 + 姿态 | **上篮测试状态机**、计时、违规检测 |
| **basketball-vision-analytics** | https://github.com/vinod-polinati/basketball-vision-analytics | YOLO + Norfair + MediaPipe Hands |  shooter 识别、传球/投篮事件 |
| **AI-Basketball-Analysis** | https://github.com/AlbertCQY/AI-Basketball-Analysis | OpenPose | 肘/膝角评估投篮 |
| **Skeleton Action Recognition** | https://github.com/Abdirayimov/skeleton-action-recognition | YOLOv8 + RTMPose + ST-GCN | 骨架动作识别 pipeline 模板 |

### 4.2 动作质量评估（AQA）方法

| 方法 | 链接 | 适用 |
|------|------|------|
| **ST-GCN** | https://github.com/yysijie/st-gcn | 骨架序列动作分类 |
| **ST-GCN++ / PYSKL** | https://github.com/kennymckormick/pyskl | 更强骨架动作识别 |
| **MotionBERT Action Head** | https://github.com/Walter0807/MotionBERT | 预训练 motion 表示 + 微调 |
| **GCN-PSN (AQA)** | https://arxiv.org/html/2511.01194v1 | **单帧/序列姿态相似度**评分 |
| 花滑 AQA (ST-GCN) | https://doi.org/10.1109/itme53901.2021.00048 | 骨架→评分回归范式 |

**三威胁/突破/上篮**：公开篮球 AQA 数据集较少，建议：
1. 先用 ST-GCN / MotionBERT 做**动作阶段识别**  
2. 再用**规则引擎 + 标准模板关节角**做规范判定（可解释、易调参）  
3. 积累标注后训练 **GCN-PSN 式相似度** 或回归评分头

### 4.3 三类目标动作的关键关节

| 动作 | 建议监控关节/相位 | 备注 |
|------|-------------------|------|
| **罚篮** | 膝角、肘角、腕角、出手高度、跟随动作 | FormFix 五阶段：Load→Set→Rise→Release→Follow-through |
| **三步上篮** | 起跳步序（0-1-2 步）、起跳膝角、出手肘腕、平衡 | 参考 Layup Test System 状态机 |
| **三威胁** | 膝屈、球位（手髋相对位置）、重心、刺步/放球时序 | 需自定义「威胁姿态」模板 + 突破触发检测 |

---

## 5. 综合推荐方案

### 5.1 方案 A：快速原型（1–2 周）

```
YOLOv8/RTMDet 检测
  → rtmlib RTMW3D（133 点 3D）
  → Ultralytics BoT-SORT-ReID（通用 OSNet）
  → 规则关节角 + FormFix 式阶段检测
  → 教师可调阈值评分
```

- **优点**：全 ONNX/TensorRT 可部署，依赖少  
- **缺点**：单目 3D 尺度漂移；身份需单独做 Face-Body 融合跟踪

### 5.2 方案 B：课堂生产（**多机位正式路径，见 5.4**）

```
2–4 机位标定（Pose2Sim Calibration）
  → RTMPose Whole_body 2D
  → Pose2Sim 三角化 + OpenSim 关节角
  → InsightFace + CLIP-ReID 融合跟踪
  → MotionBERT 动作分段（不用其 3D 分支）
  → 标准动作库比对 + 可视化反馈
```

- **优点**：米级 3D、可靠腕角、可输出 OpenSim 报告  
- **缺点**：需标定与同步，工程量大

### 5.4 正式敲定：多机位 Pose2Sim + MotionBERT 技术路径

> **结论**：两者**可以且应当结合**，但必须**严格分工**——Pose2Sim 负责「几何真值 + 规范评分」，MotionBERT 负责「时序语义 + 动作分段」。**不要用 MotionBERT 的 3D Pose 分支替代 Pose2Sim 三角化**（多视下 Pose2Sim 精度更高、且有绝对尺度）。

#### 5.4.1 分工矩阵

| 能力 | Pose2Sim | MotionBERT | 说明 |
|------|----------|------------|------|
| 多视 3D 坐标（米） | ✅ 主责 | ❌ 不用 3D 头 | MotionBERT 3D 为单目 lifting，与多视冗余且更差 |
| 手腕/手部角度 | ✅ Whole_body 133 点 + OpenSim | ❌ 仅 17 体点 | 罚篮腕角、跟随动作必须走 Pose2Sim |
| 动作类型识别 | ❌ | ✅ Action Head | 罚篮 / 上篮 / 三威胁 分类 |
| 动作阶段切分 | ⚠️ 需规则 | ✅ 时序 encoder | 如投篮 Load→Release→Follow-through |
| 规范是否标准 | ✅ 关节角模板比对 | ❌ 不直接评分 | MotionBERT 输出「是什么」，不输出「好不好」 |
| 多人 ID | ✅ 多视关联 + **student_id** | ⚠️ 按 student_id 拆 clip | **InsightFace + CLIP-ReID** 融合跟踪 |

#### 5.4.2 系统架构（四层）

```
┌─────────────────────────────────────────────────────────────────┐
│ L0 采集层：2–4 路同步摄像头 + Pose2Sim 标定（Checkerboard/场景尺寸） │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ L1 感知层（Pose2Sim 核心）                                        │
│  RTMPose Whole_body(133) 逐帧 2D → 多视人物匹配 → 三角化 → 滤波   │
│  输出：每人每帧 3D keypoints (.trc) + 主视角 2D 序列               │
└───────────────┬─────────────────────────────┬───────────────────┘
                ▼                             ▼
┌───────────────────────────┐   ┌─────────────────────────────────┐
│ L2a 生物力学层（Pose2Sim）  │   │ L2b 时序语义层（MotionBERT）       │
│  OpenSim → 关节角时序        │   │  主视角 2D → H36M-17 转换         │
│  肘/膝/腕/躯干角            │   │  预训练 encoder + 篮球动作微调     │
│  与标准模板比对 → 分项得分   │   │  输出：动作类型 + 阶段起止帧       │
└───────────────┬─────────────┘   └──────────────┬──────────────────┘
                └──────────────┬───────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ L3 融合决策层                                                     │
│  仅在 MotionBERT 判定的「有效动作片段」内，读取 Pose2Sim 关节角打分  │
│  生成：总分 + 阶段反馈 + 3D 回放                                  │
└─────────────────────────────────────────────────────────────────┘
         ▲
         │ L1 侧路：InsightFace(ArcFace) + CLIP-ReID 融合 → student_id
         │          StrongSORT 跨帧；注册库 long-term；约束 Pose2Sim 多视匹配
```

#### 5.4.3 数据流与格式衔接

1. **Pose2Sim 配置**
   - `pose_model = 'Whole_body'`（罚篮需手腕+手指时用；若仅腕部可用 `Whole_body_wrist` 提速）
   - `multi_person = true`
   - 输出：`Pose3D/*.trc`（3D）、各视角 2D JSON

2. **送入 MotionBERT 的数据**
   - 来源：**主视角**（正对篮筐/侧前方）的 2D 序列，或 3D 投影回主视角
   - 格式：Halpe26 / COCO17 → **H36M 17 点**（复用 `halpe2h36m()`，见 MotionBERT `dataset_wild.py`）
   - 每人独立 clip，长度 ≤ 243 帧（MotionBERT 上限）；更长动作滑窗处理
   - **禁止**把 Pose2Sim 3D 再喂给 MotionBERT 3D 头（冗余）

3. **评分融合逻辑（示例：罚篮）**

   | 阶段 | MotionBERT | Pose2Sim + 规则 |
   |------|------------|-----------------|
   | 检测「正在罚球」 | 动作分类 conf > θ | 人站在罚球线区域 |
   | 蓄力段 | 输出 start/end 帧 | 膝角 ∈ [110°, 135°] |
   | 出手段 | Release 帧 | 肘角、腕角、出手高度 |
   | 跟随 | Follow-through 帧 | 腕延伸、肘未过早下落 |

#### 5.4.4 必须额外补齐的模块

| 模块 | 选型 | 原因 |
|------|------|------|
| 身份跟踪 | **InsightFace + CLIP-ReID + StrongSORT** | 人脸+不同衣着双模态；投篮时自动降 α 靠身体 ReID |
| 课前注册 | 正脸 + 全身 gallery | 绑定 `student_id`，跨镜/重入画检索 |
| 相机同步 | 硬件同步 或 Pose2Sim `synchronizeCams` | 三角化前提 |
| MotionBERT 微调 | 自采篮球 clip + 动作/阶段标签 | 预训练是 NTU 通用动作，零样本不适配篮球 |
| 手腕精细角 | Pose2Sim 133 点向量角 或 OpenSim | MotionBERT 无手点 |
| 可选增强 | BPBReID 作 body 分支 | 投篮抬手遮挡时部位特征辅助 |

#### 5.4.5 优劣势与风险

**优势**
- 几何与语义解耦：3D 评分可解释、可对标体育教材阈值
- Pose2Sim 已在运动场景验证，OpenSim 关节角可直接给教师看
- MotionBERT 预训练 motion 表示强，微调后阶段切分比纯规则 Robust

**风险与对策**

| 风险 | 对策 |
|------|------|
| MotionBERT 只有 17 点，丢失手指 | 规范评分**不依赖** MotionBERT；手/腕角只用 Pose2Sim |
| 多学生 MotionBERT 需逐人 clip | L1 关联后按 `person_id` 拆序列 |
| Pose2Sim 非实时（批处理） | 课堂可「练完一组 → 30s 内出报告」；实时预览用主视角 2D |
| 三威胁无公开数据集 | 先规则 + 少量标注微调 MotionBERT |

#### 5.4.6 实施阶段

| 阶段 | 目标 | 周期（估） |
|------|------|-----------|
| P0 | 2 机位标定 + Pose2Sim 单人 3D + 肘膝腕角可视化 | 2–3 周 |
| P1 | 多人 **Face-Body 融合跟踪** + Pose2Sim 多视；罚篮 OpenSim 模板评分 | 3–4 周 |
| P2 | MotionBERT 罚篮阶段微调；L3 融合出分 | 3–4 周 |
| P3 | 扩展上篮、三威胁；UI 与报告 | 4–6 周 |

#### 5.4.7 明确不采用的替代

- ❌ MotionBERT 3D Pose 替代 Pose2Sim 三角化  
- ❌ 单目 RTMW3D 作为生产 3D 源（可作 P0 调试预览）  
- ❌ 仅用 MotionBERT 端到端评「动作好坏」（黑盒、无法解释、缺手腕）  
- ⚠️ ST-GCN++ 可作为 MotionBERT 备选（若希望直接在 **3D 133 点**上做动作识别，手信息保留更好）；当前路径仍推荐 MotionBERT 因其预训练 motion 表示更强、微调成本更低

### 5.3 方案 C：精细生物力学分析（研究/纠错）

```
SMPLer-X 或 Hand4Whole++（SMPL-X mesh）
  → MANO 手腕轴角 + 全身 kinematics
  → WHAM 世界轨迹（上篮位移）
  → 离线批处理，非实时
```

---

## 6. 模型选型速查表

### 6.1 三维骨架（按优先级）

| 优先级 | 模型 | 何时选 |
|--------|------|--------|
| 🥇 | **RTMW3D + rtmlib** | 实时单目、要手、快速集成 |
| 🥇 | **Pose2Sim + RTMPose** | 多视、要精确关节角、体育场景 |
| 🥈 | **SMPLer-X / Hand4Whole++** | 要 MANO 级手腕、可接受离线 |
| 🥈 | **GEM-X** | 商用许可、77 点全身 |
| 🥉 | **WHAM + MotionBERT** | 时序动作/位移分析 |
| 备选 | **MediaPipe / DWPose 2D** | 极致轻量或作 2D 前端 |

### 6.2 身份跟踪（按优先级）

| 优先级 | 模型 | 何时选 |
|--------|------|--------|
| 🥇 | **InsightFace ArcFace + CLIP-ReID + StrongSORT** | **正式路径**：不同衣着 + 人脸敏感 |
| 🥇 | **课前注册 gallery** | 绑定 student_id，跨镜/重入画 |
| 🥈 | **TransReID** | CLIP-ReID 备选 body 编码器 |
| 🥈 | **BPBReID** | 投篮遮挡时 body 部位增强（可选） |
| 🥉 | **KPR** | 骨架引导 ReID，与 Pose2Sim 联合（可选） |
| 基线 | **FastReID / BoT-SORT** | 工程 fallback |
| 不适用 | ~~PRTreID~~ | 同款队服场景专用 |

---

## 7. 依赖与许可注意事项

| 组件 | 许可/限制 |
|------|-----------|
| RTMPose / RTMW3D / rtmlib | Apache 2.0，商用友好 |
| SMPL-X / MANO / FLAME | 需注册 https://smpl-x.is.tue.mpg.de/  academic license |
| MediaPipe | Apache 2.0 |
| CLIP-ReID | MIT |
| InsightFace | Apache 2.0（模型需查各 bundle 说明） |
| FastReID | Apache 2.0 |
| GEM-X | Apache 2.0（NVIDIA 自有数据训练） |

---

## 8. 建议的下一步

1. **确定摄像头方案**：多机位 Pose2Sim 为主路径  
2. **搭建 InsightFace + CLIP-ReID + StrongSORT**：课前注册 + 融合跟踪 demo  
3. **2 机位 Pose2Sim P0**：验证 3D 手腕角与 student_id 绑定  
4. **定义标准动作 JSON 模板**（关节角范围 + 阶段时序），参考 FormFix  
5. **罚篮 MVP**：MotionBERT 阶段切分 + Pose2Sim 关节角评分  

---

## 9. 参考文献与链接汇总

### 姿态估计
- RTMW: https://arxiv.org/html/2407.08634v1  
- DWPose: https://arxiv.org/abs/2307.15880  
- SMPLer-X: https://arxiv.org/html/2309.17448v3  
- Hand4Whole++: https://arxiv.org/abs/2603.14726  
- AiOS: https://arxiv.org/html/2403.17934v1  
- WHAM: https://arxiv.org/html/2312.07531v1  
- MotionBERT: https://motionbert.github.io/  
- GEM-X: https://github.com/NVlabs/GEM-X  

### 身份跟踪（Face-Body 融合）
- InsightFace: https://github.com/deepinsight/insightface  
- CLIP-ReID: https://github.com/Syliz517/CLIP-ReID  
- HumanRecognition（融合范例）: https://github.com/oele-isis-vanderbilt/HumanRecognition  
- StrongSORT: https://github.com/dyhBUPT/StrongSORT  
- BoT-SORT: https://github.com/NirAharon/BOT-SORT  
- TrackLab: https://github.com/TrackingLaboratory/tracklab  
- BPBReID（可选遮挡增强）: https://github.com/VlSomers/bpbreid  

### 多视 / 体育
- Pose2Sim: https://github.com/perfanalytics/pose2sim  
- EasyMocap: https://github.com/zju3dv/EasyMocap  
- 篮球 RTMPose 论文: https://www.mdpi.com/1424-8220/25/13/4003  

### 篮球应用
- FormFix: https://github.com/adityasingh2400/formfix  
- Layup System: https://github.com/jxk6575/Basketball_Layup_Test_System  
- JNKS: https://github.com/nathsmith-cs/JNKS  

### 关节角
- joint_angles_calculate: https://github.com/TemugeB/joint_angles_calculate  
- AQA GCN-PSN: https://arxiv.org/html/2511.01194v1  

---

*文档由技术调研自动生成，实施前建议在目标硬件与真实课堂环境中做小规模 benchmark。*
