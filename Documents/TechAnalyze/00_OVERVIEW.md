# SAM3 技术分析文档 — 总览与导航

> 基于 ComfyUI-SAM3-main 源码、官方 facebookresearch/sam3 仓库、SAM3 论文三源交叉分析

---

## 📚 文档体系导航

| 文档 | 内容 | 阅读时间 | 难度 | 适合谁 |
|------|------|----------|------|--------|
| [00_OVERVIEW.md](00_OVERVIEW.md) | 本文档：总览、导航、核心概念、三方对比 | 10 min | ★☆☆ | 所有人 |
| [01_WORKFLOW_ANALYSIS.md](01_WORKFLOW_ANALYSIS.md) | SAM.json 工作流逐节点分析 | 15 min | ★★☆ | 工作流使用者 |
| [02_NODE_REFERENCE.md](02_NODE_REFERENCE.md) | 全部节点参数/输入/输出速查 | 20 min | ★★☆ | 开发者 |
| [03_DATA_FORMATS.md](03_DATA_FORMATS.md) | 数据格式、坐标体系、转换链 | 15 min | ★★★ | **必读** — Bug 高发区 |
| [04_SOURCE_DEEP_DIVE.md](04_SOURCE_DEEP_DIVE.md) | 源码逐文件深度分析（含官方 vs ComfyUI 对比） | 40 min | ★★★ | 架构师/复刻者 |
| [05_REPLICATION_GUIDE.md](05_REPLICATION_GUIDE.md) | 独立工具复刻实现指南 | 30 min | ★★★ | 复刻开发者 |
| [06_QUICK_REFERENCE.md](06_QUICK_REFERENCE.md) | 快速参考卡、速查表 | 5 min | ★☆☆ | 所有人 |

---

## 🎯 SAM3 是什么

**SAM3 (Segment Anything Model 3)** 是 Meta 推出的第三代图像分割模型，核心论文概念：

- **PCS 任务 (Promptable Concept Segmentation)**：通过文本/示例/两者结合，检测、分割并追踪概念的所有实例
- **解耦架构**：DETR 检测器 + SAM2 追踪器共享视觉骨干网络 (Perception Encoder)
- **Presence Token**：将"是什么"(presence) 与"在哪里"(object queries) 解耦，最终分数 = presence_score × object_score
- **SA-Co 数据引擎**：4 阶段，4M 概念，52M 掩码，467K 视频 masklets
- **848M 总参数**：ViT-L/14@1008 骨干 + DETR 检测头 + SAM2 追踪头

### 三种分割模式

| 模式 | 输入 | 输出 | 官方 API 入口 |
|------|------|------|---------------|
| **文本定位 (Text Grounding)** | "dog" → 所有狗 | 掩码 + 框 + 分数 | `Sam3Processor.set_text_prompt()` |
| **交互式分割 (Interactive)** | 点击/框选 → 精确区域 | 掩码 + 分数 | `Sam3Image.predict_inst()` |
| **视频追踪 (Video Tracking)** | 标注一帧 → 全视频 | 视频掩码 | `Sam3BasePredictor.handle_request()` |

### SAM 3.1 更新 (2026-03-27)

SAM 3.1 引入 **Object Multiplex** 机制，在 128 对象场景下实现约 **7× 加速**：
- `Sam3MultiplexVideoPredictor`：替代 `Sam3VideoPredictor`
- `max_num_objects=16`，`multiplex_count=16`
- 通过 `build_sam3_predictor(version="sam3.1")` 统一入口
- 新 checkpoint 在 huggingface.co/facebook/sam3.1

---

## 🏗️ 系统架构全景图

### 三层架构（官方 vs ComfyUI 对比）

```
┌─────────────────────────────────────────────────────────────────────┐
│                   官方 facebookresearch/sam3 架构                    │
│                                                                     │
│  build_sam3_image_model() → Sam3Processor                          │
│      set_image() → set_text_prompt() → add_geometric_prompt()      │
│      confidence_threshold=0.5, 无 NMS                               │
│                                                                     │
│  build_sam3_image_model() → Sam3Image.predict_inst()               │
│      box=[x0,y0,x1,y1] 像素坐标                                    │
│                                                                     │
│  build_sam3_predictor(version=) → Sam3BasePredictor                │
│      handle_request(dict) → 请求式调度                              │
│      add_prompt(boxes_xywh=[xmin,ymin,w,h] 归一化)                 │
│      propagate_in_video()                                           │
│                                                                     │
│  build_sam3_multiplex_video_predictor() → SAM 3.1 Multiplex        │
│      handle_request() / handle_stream_request()                     │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │ ComfyUI 封装层
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     ComfyUI 工作流层                                  │
│  LoadImage ──→ LoadSAM3Model ──→ SAM3MultiRegionCollector           │
│                                       │                              │
│                                       ▼                              │
│                              SAM3MultipromptSegmentation             │
│                                       │                              │
│                                       ▼                              │
│                                 PreviewImage                         │
└────────────────────────────┬────────────────────────────────────────┘
                             │ SAM3_MODEL_CONFIG (dict)
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     节点层 (nodes/)                                  │
│                                                                     │
│  ┌──────────────┐  ┌──────────────────┐  ┌─────────────────────┐   │
│  │ LoadSAM3Model│  │ segmentation.py  │  │ sam3_interactive.py │   │
│  │  → CONFIG    │  │ Grounding/Seg/   │  │ Point/BBox/         │   │
│  │              │  │ Multiprompt      │  │ MultiRegion/        │   │
│  └──────────────┘  └────────┬─────────┘  │ Interactive         │   │
│                             │            └──────────┬──────────┘   │
│                             ▼                       ▼               │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              _model_cache.py (模块级单例)                     │  │
│  │   get_or_build_model(config) → SAM3UnifiedModel             │  │
│  └───────────────────────┬──────────────────────────────────────┘  │
│                          │                                          │
│                          ▼                                          │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │          sam3_model_patcher.py                               │  │
│  │   SAM3UnifiedModel(ModelPatcher)                             │  │
│  │   ├── video_predictor: Sam3VideoPredictor                    │  │
│  │   └── processor: Sam3Processor (ComfyUI 修改版)             │  │
│  └───────────────────────┬──────────────────────────────────────┘  │
└──────────────────────────┼──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   SAM3 核心库 (sam3/) — ComfyUI vendored 精简版      │
│                                                                     │
│  ┌──────────────────┐  ┌─────────────────┐  ┌────────────────────┐ │
│  │ Sam3VideoPredictor│  │ Sam3Processor   │  │ model.py           │ │
│  │ (session管理/     │  │ (Grounding路径)  │  │ Sam3Image          │ │
│  │  add_prompt/      │  │ confidence=0.2  │  │ predict_inst →     │ │
│  │  propagate)       │  │ 有 NMS          │  │ inst_interactive   │ │
│  │                  │  │ 额外方法:        │  │ _predictor         │ │
│  │                  │  │  add_point_prompt│  │                    │ │
│  │                  │  │  add_mask_prompt │  │                    │ │
│  │                  │  │  add_multiple_   │  │                    │ │
│  │                  │  │  box_prompts     │  │                    │ │
│  └──────────────────┘  └─────────────────┘  └────────────────────┘ │
│                                                                     │
│  ┌──────────────┐  ┌────────────────┐  ┌────────────────────┐      │
│  │ attention.py │  │ text_encoder.py│  │ __init__.py        │      │
│  │ SplitMHA     │  │ VETextEncoder  │  │ build_sam3_video   │      │
│  │ RoPEAttn     │  │                │  │ _model → model     │      │
│  └──────────────┘  └────────────────┘  └────────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## ⚠️ 关键差异：官方 vs ComfyUI（必读）

> 以下差异是复刻和调试中最容易踩坑的地方

### 差异 1：Box 坐标格式不一致 🔴

| API | 框格式 | 坐标系 | 示例 |
|-----|--------|--------|------|
| 官方 `Sam3Processor.add_geometric_prompt()` | `[cx, cy, w, h]` | 归一化 0-1 | `[0.5, 0.3, 0.2, 0.4]` |
| 官方 `SAM3InteractiveImagePredictor.predict()` | `[x0, y0, x1, y1]` | 像素 | `[100, 200, 300, 400]` |
| 官方 `Sam3VideoInference.add_prompt(boxes_xywh=)` | `[xmin, ymin, w, h]` | 归一化 0-1 | `[0.1, 0.2, 0.3, 0.4]` |
| ComfyUI `add_multiple_box_prompts()` | `[cx, cy, w, h]` | 归一化 0-1 | `[0.5, 0.3, 0.2, 0.4]` |
| ComfyUI `predict_inst(box=)` | `[x1, y1, x2, y2]` | 像素 | `[100, 200, 300, 400]` |

**三种不同的框格式！** Grounding 用 center 归一化，Interactive 用 corner 像素，Video 用 origin 归一化。

### 差异 2：Point Prompts 在 Grounding 路径被忽略 🔴

官方 `Sam3Processor._forward_grounding()` 中：
```python
print("Warning: Point prompts are ignored in PCS.")
```

但 ComfyUI 的 `add_point_prompt()` 方法调用了 `_forward_grounding()`，这意味着**通过 add_point_prompt 添加的点实际上被静默忽略了**。这可能是一个 Bug。

### 差异 3：NMS 差异 🔴

| 版本 | `_forward_grounding` 后处理 |
|------|---------------------------|
| 官方 | 无 NMS |
| ComfyUI | `nms_masks(iou_threshold=0.5)` |

ComfyUI 额外添加了 NMS 后处理，可能影响检测结果。

### 差异 4：confidence_threshold 默认值 🔴

| 版本 | 默认值 |
|------|--------|
| 官方 | 0.5 |
| ComfyUI | 0.2 |

2.5 倍差异！ComfyUI 会检测到更多（但可能更嘈杂的）结果。

### 差异 5：ComfyUI 额外方法 🟡

ComfyUI 的 `Sam3Processor` 添加了官方没有的方法：

| 方法 | 官方 | ComfyUI | 说明 |
|------|------|---------|------|
| `set_image()` | ✅ | ✅ | 相同 |
| `set_text_prompt()` | ✅ | ✅ | 相同 |
| `add_geometric_prompt()` | ✅ | ❌ | 官方有，ComfyUI 未暴露 |
| `add_multiple_box_prompts()` | ❌ | ✅ | ComfyUI 扩展 |
| `add_point_prompt()` | ❌ | ✅ | ComfyUI 扩展（⚠️ 调用 _forward_grounding 会忽略点） |
| `add_mask_prompt()` | ❌ | ✅ | ComfyUI 扩展 |
| `reset_all_prompts()` | ✅ | ❌ | 官方有，ComfyUI 未暴露 |
| `set_confidence_threshold()` | ✅ | ✅ | 相同 |
| `_forward_grounding()` | ✅ (无NMS) | ✅ (有NMS) | 实现不同 |

### 差异 6：Exemplar 提示机制 🟡

官方支持 **示例提示 (Exemplar Prompt)**：通过在特定帧上标注正/负框来迭代优化概念检测。
- 内部使用 `TEXT_ID_FOR_VISUAL=1` / `text_str="visual"` 路径
- `Sam3Processor.add_geometric_prompt()` 支持 `text_str="visual"` 参数
- 论文称之为 "exemplar prompts"，但代码中与文本路径共享

---

## 🔑 核心概念速览

### 1. 三条推理路径

| | Grounding 路径 | Interactive 路径 | Video 路径 |
|---|---|---|---|
| **用途** | 文本定位 | 点击/框选分割 | 视频追踪 |
| **入口** | `processor.set_text_prompt()` | `model.predict_inst()` | `predictor.add_prompt()` |
| **输出** | `state` dict | numpy arrays | video masks |
| **节点** | SAM3Grounding | SAM3Segmentation | SAM3VideoSegmentation |
| **框格式** | [cx,cy,w,h] 归一化 | [x0,y0,x1,y2] 像素 | [xmin,ymin,w,h] 归一化 |
| **前提** | 无 | `enable_inst_interactivity=True` | 需要 session |

### 2. 坐标归一化三段链

```
前端像素坐标          收集器归一化          分割节点反归一化        predict_inst内部
(px, py)         →  (nx, ny)         →  (px, py)          →  (nnx, nny)
x: 0~img_w           x: 0~1               x: 0~img_w           x: 0~1
y: 0~img_h           y: 0~1               y: 0~img_h           y: 0~1
                  ÷ img_w              × img_w              ÷ img_w (normalize_coords=True)
```

> ⚠️ 这是 Bug 最高发的区域！详见 [03_DATA_FORMATS.md](03_DATA_FORMATS.md)

### 3. 模型加载的 "Config 而非 Model" 模式

`LoadSAM3Model` **不**返回模型对象，而是返回一个 JSON-safe 的配置字典：

```python
SAM3_MODEL_CONFIG = {
    "checkpoint_path": "models/sam3/sam3.safetensors",
    "bpe_path": "nodes/sam3/bpe_simple_vocab_16e6.txt.gz",
    "precision": "auto",   # "auto" | "bf16" | "fp16" | "fp32"
    "dtype": "bf16",       # 解析后的实际 dtype 字符串
    "compile": False
}
```

官方 API 使用不同的入口：
```python
# 官方推荐
from sam3 import build_sam3_predictor
predictor = build_sam3_predictor(version="sam3")  # 或 "sam3.1"

# ComfyUI 使用
from sam3 import build_sam3_video_model
predictor = build_sam3_video_model(config)
```

### 4. Meta Device 构建（零 RAM 开销）

模型构建流程：
1. `torch.device("meta")` 上构建模型骨架 → 零内存
2. `load_state_dict(assign=True)` → 直接映射权重文件，不复制
3. 修复 meta device 上的残留 buffer → 重建因果注意力 mask 等
4. 选择性 weight casting → backbone + inst_interactive_predictor 转 bf16/fp16

### 5. ComfyUI ModelPatcher 集成

`SAM3UnifiedModel` 继承 `ModelPatcher`，让 ComfyUI 负责：
- GPU/CPU 自动 offload（lowvram / novram 模式）
- `load_models_gpu()` 自动调度
- 设备同步（`_sync_model_device`, `_sync_processor_device`）

### 6. Presence Token 机制（论文核心创新）

传统 DETR 的 object query 同时编码"是什么"和"在哪里"，当概念与位置耦合时效果差。
SAM3 引入 Presence Token：
- **Presence Token**：独立预测"图中是否存在该概念"（what）
- **Object Queries**：仅预测位置和形状（where）
- **最终分数** = `presence_score × object_score`
- 解耦使得模型能更好地处理多实例和遮挡

### 7. 跨架构特征注入

SAM3 的骨干网络 (Perception Encoder) 同时为检测器和追踪器提供特征：
- **检测器路径**：FPN 特征 → DETR 检测头
- **追踪器路径**：通过 `conv_s0` / `conv_s1` 适配层 → SAM2 的 `sam_mask_decoder`
- 这种设计让检测器和追踪器共享同一视觉特征，但各自有专用解码器

---

## 🔬 研究问题清单

通过三方交叉分析（论文 × 官方代码 × ComfyUI 代码）发现的问题：

| # | 严重性 | 问题 | 影响 |
|---|--------|------|------|
| 1 | 🔴 | Box 格式跨 API 不一致（3 种格式） | 复刻时极易混淆 |
| 2 | 🔴 | Point prompts 在 Grounding 路径被静默忽略 | `add_point_prompt()` 可能是 Bug |
| 3 | 🔴 | NMS 差异（官方无 vs ComfyUI 有） | 结果不可复现 |
| 4 | 🔴 | confidence_threshold 默认值 0.5 vs 0.2 | 检测数量差异大 |
| 5 | 🟡 | Exemplar 机制使用 "visual" 文本路径 | 文档缺失，不易发现 |
| 6 | 🟡 | Prompt 类支持 mask_embeddings | 论文未提及 mask prompts |
| 7 | 🟡 | 跨架构特征注入 (conv_s0/conv_s1) | 复刻时需注意适配层 |
| 8 | 🟡 | 视频 add_prompt() 每次调用 reset_state() | 检测器重置追踪器 |
| 9 | 🟢 | 848M 参数分解未公开 | 无法估算各组件资源 |
| 10 | 🟢 | SA-Co 数据引擎复现细节缺失 | 训练不可复现 |
| 11 | 🟢 | build_sam3_predictor 版本差异不明确 | sam3 vs sam3.1 细节 |

---

## 📖 推荐阅读路径

### 路径 A：工作流使用者
```
00 → 01 → 02 → 06
```
了解工作流如何运转、每个节点怎么配、速查表备用。

### 路径 B：集成开发者
```
00 → 03 → 02 → 06
```
先搞懂数据格式和坐标体系（避免踩坑），再查节点参数。

### 路径 C：复刻开发者
```
00 → 03 → 04 → 05 → 06
```
深入源码理解每个环节，然后按复刻指南实现。**强烈建议先读官方 API 再看 ComfyUI 封装**。

### 路径 D：Bug 排查
```
00（差异表） → 03（坐标/格式） → 04（推理路径） → 06（错误排查表）
```

### 路径 E：官方 API 研究
```
00 → 04（官方架构部分） → 05（官方 API 示例） → 06
```
直接使用官方 facebookresearch/sam3 仓库的场景。

---

## 📦 项目文件结构

### ComfyUI-SAM3-main

```
ComfyUI-SAM3-main/
├── __init__.py                    # 插件入口
├── install.py                     # 依赖安装
├── prestartup_script.py           # 预启动脚本
├── pyproject.toml                 # 项目配置
├── requirements.txt               # Python 依赖
├── nodes/
│   ├── __init__.py                # 节点注册汇总
│   ├── load_model.py              # LoadSAM3Model → SAM3_MODEL_CONFIG
│   ├── segmentation.py            # SAM3Grounding / SAM3Segmentation / SAM3MultipromptSegmentation / 辅助节点
│   ├── sam3_interactive.py        # SAM3PointCollector / SAM3BBoxCollector / SAM3MultiRegionCollector / SAM3InteractiveCollector
│   ├── sam3_video_nodes.py        # SAM3VideoSegmentation / SAM3Propagate / SAM3VideoOutput
│   ├── _model_cache.py            # 模型单例缓存（核心：get_or_build_model）
│   ├── sam3_model_patcher.py      # SAM3UnifiedModel(ModelPatcher)
│   ├── utils.py                   # 图像/掩码转换、可视化
│   ├── image_utils.py             # 规范化格式转换函数
│   ├── video_state.py             # 不可变视频状态数据结构
│   ├── inference_reconstructor.py # 推理状态按需重建
│   └── sam3/                      # SAM3 核心库（vendored 精简版）
│       ├── __init__.py            # 构建器：build_sam3_video_model、权重转换
│       ├── model.py               # 模型定义：Sam3Image、SAM3InteractiveImagePredictor 等
│       ├── predictor.py           # Sam3VideoPredictor（session/add_prompt/propagate）
│       ├── attention.py           # SplitMultiheadAttention、RoPEAttention、sam3_attention
│       ├── text_encoder.py        # VETextEncoder、LayerScale
│       ├── tokenizer.py           # SimpleTokenizer (BPE)
│       ├── perflib.py             # 性能辅助（mask_iou 等）
│       └── utils.py               # 工具函数（box转换、FindStage、Sam3Processor 等）
└── web/                           # 前端 JS 交互组件
    ├── sam3_points_widget.js      # 点选交互
    ├── sam3_bbox_widget.js        # 框选交互
    ├── sam3_multiregion_widget.js # 多区域交互
    ├── sam3_interactive_widget.js # 实时分割预览
    └── sam3_video_dynamic.js      # 视频帧交互
```

### 官方 facebookresearch/sam3（关键文件）

```
sam3/                              # pip install sam3
├── __init__.py                    # 导出: build_sam3_image_model, build_sam3_predictor
├── model_builder.py               # 5 个构建函数（见下方）
├── model/
│   ├── sam3_image.py              # Sam3Image, predict_inst, forward_grounding
│   ├── sam3_image_processor.py    # Sam3Processor（官方版，confidence=0.5，无NMS）
│   ├── sam3_base_predictor.py     # Sam3BasePredictor: handle_request 调度
│   ├── sam3_video_predictor.py    # Sam3VideoPredictor
│   ├── sam3_multiplex_video_predictor.py  # SAM 3.1 Multiplex
│   ├── sam3_video_inference.py    # Sam3VideoInferenceWithInstanceInteractivity
│   ├── sam1_task_predictor.py     # SAM3InteractiveImagePredictor
│   ├── geometry_encoders.py       # Prompt 类（支持 mask_embeddings）
│   ├── vl_combiner.py             # SAM3VLBackbone, SAM3VLBackboneTri
│   ├── sam3_tracker_base.py       # Sam3TrackerBase, _forward_sam_heads
│   ├── necks.py                   # Sam3DualViTDetNeck, Sam3TriViTDetNeck
│   └── sam/
│       └── prompt_encoder.py      # PromptEncoder（SAM2-style）
└── agent/
    └── client_sam3.py             # sam3_inference helper
```

**官方 5 个构建函数**：
1. `build_sam3_image_model()` → 图像 Grounding/Interactive
2. `build_sam3_video_model()` → 视频模型
3. `build_sam3_video_predictor()` → 视频预测器
4. `build_sam3_multiplex_video_predictor()` → SAM 3.1 Multiplex
5. `build_sam3_predictor(version=)` → 统一入口（"sam3" 或 "sam3.1"）

---

## 📊 Transform Pipeline

官方 SAM3 的图像预处理链（ComfyUI 使用相同配置）：

```
ToDtype(uint8) → Resize(1008, 1008) → ToDtype(float32) → Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
```

- **输入**：任意尺寸 RGB 图像
- **Resize**：等比缩放到 1008×1008（72 × 14 patch）
- **Normalize**：使用 [0.5, 0.5, 0.5] 均值和标准差（不是 ImageNet 的 [0.485, 0.456, 0.406]）
