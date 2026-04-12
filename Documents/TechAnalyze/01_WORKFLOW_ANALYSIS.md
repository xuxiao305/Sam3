# SAM3 工作流分析

> 基于 SAM.json 工作流文件（5 节点版）及官方 API 对比分析

---

## 目录

1. [工作流全景](#1-工作流全景)
2. [节点详解](#2-节点详解)
3. [连接线解析](#3-连接线解析)
4. [执行顺序](#4-执行顺序)
5. [参数配置分析](#5-参数配置分析)
6. [用户交互场景还原](#6-用户交互场景还原)
7. [官方 API 工作流对比](#7-官方-api-工作流对比)

---

## 1. 工作流全景

### 数据流图

```
┌──────────────┐
│  LoadImage   │ (771)
│   IMAGE ─────┼──────────────────────────────────┐
│              │                                  │
└──────────────┘                                  │
                                                  ▼
┌──────────────┐     ┌───────────────────────┐   ┌─────────────────────────┐
│LoadSAM3Model │     │SAM3MultiRegionCollector│   │SAM3MultipromptSegmentation│
│   (772)      │     │        (775)          │   │         (774)            │
│CONFIG────────┼─┐   │                       │   │                          │
└──────────────┘ │   │  image ←─────────────┼───┤  image ←────────────────┘
                 │   │                       │   │                          │
                 │   │  multi_prompts ───────┼──→│  multi_prompts           │
                 │   └───────────────────────┘   │                          │
                 │                                │  sam3_model_config ←────┘
                 └───────────────────────────────→│                          │
                                                  │  masks (MASK) → (未连接) │
                                                  │  visualization ──────────┼──→ ┌─────────────┐
                                                  └──────────────────────────┘   │ PreviewImage │
                                                                                 │    (773)     │
                                                                                 └─────────────┘
```

### 关键特征

- **5 个节点**，5 条连接线
- 输入：一张图片 + 用户交互标注
- 输出：多区域分割掩码 + 可视化预览
- 核心推理在 `SAM3MultipromptSegmentation`（节点 774）

---

## 2. 节点详解

### 2.1 LoadImage (节点 771)

| 属性 | 值 |
|------|-----|
| **类型** | `LoadImage` (ComfyUI 内置) |
| **执行顺序** | 0 (最先执行) |
| **输入** | 无（上传图片） |
| **输出** | `IMAGE` [B,H,W,C] float32 → 连接到 774(image) 和 775(image) |
| | `MASK` → 未连接 |

**工作流配置**：
```json
"widgets_values": ["segment_test_bot.png", "image"]
```

> ⚠️ 注意：同一张 IMAGE 同时连到 Collector 和 Segmentation，这是必要的——Collector 需要图片尺寸做坐标归一化，Segmentation 需要图片做推理。

---

### 2.2 LoadSAM3Model (节点 772)

| 属性 | 值 |
|------|-----|
| **类型** | `LoadSAM3Model` |
| **执行顺序** | 1 |
| **输入** | precision (widget), compile (widget) |
| **输出** | `SAM3_MODEL_CONFIG` → 连接到 774(sam3_model_config) |

**工作流配置**：
```json
"widgets_values": ["fp32", "flash_attn"]
```

> ⚠️ 工作流 JSON 中有 `flash_attn` 作为第二个 widget 值，这是旧版参数（当前版本 LoadSAM3Model 只有 `precision` 和 `compile` 两个输入，`attention` 参数已被移除）。ComfyUI 会忽略多余的 widget 值。

**输出格式** (SAM3_MODEL_CONFIG)：
```json
{
    "checkpoint_path": "models/sam3/sam3.safetensors",
    "bpe_path": "nodes/sam3/bpe_simple_vocab_16e6.txt.gz",
    "precision": "fp32",
    "dtype": "fp32",
    "compile": false
}
```

**与官方 API 的对比**：
- ComfyUI 使用 `build_sam3_video_model(config)` 构建
- 官方推荐使用 `build_sam3_predictor(version="sam3")` 统一入口
- 官方支持 SAM 3.1 版本通过 `version="sam3.1"` 选择

---

### 2.3 SAM3MultiRegionCollector (节点 775)

| 属性 | 值 |
|------|-----|
| **类型** | `SAM3MultiRegionCollector` |
| **执行顺序** | 2 |
| **输入** | `image` ← 从 771(IMAGE)，`multi_prompts_store` (隐藏 widget) |
| **输出** | `multi_prompts` (SAM3_MULTI_PROMPTS) → 连接到 774(multi_prompts) |

**工作流配置**：
```json
"widgets_values": [
    "[{\"positive_points\":[{\"x\":276.78,\"y\":418.49}], ... }]",
    ""
]
```

这是**用户交互核心节点**。用户在画布上点击/框选产生像素坐标，节点内部将像素坐标归一化为 0-1 范围。

**当前标注数据**（从 widgets_values 解析）：
- **区域 0**：1 个正点 + 2 个负点
- **区域 1**：3 个正点（无负点/框）
- **区域 2**：4 个正点（无负点/框）
- **区域 3**：8 个正点 + 10 个负点（复杂区域）

> 详见 [03_DATA_FORMATS.md](03_DATA_FORMATS.md) 了解坐标归一化细节。

---

### 2.4 SAM3MultipromptSegmentation (节点 774)

| 属性 | 值 |
|------|-----|
| **类型** | `SAM3MultipromptSegmentation` |
| **执行顺序** | 3 |
| **输入** | `sam3_model_config` ← 772, `image` ← 771, `multi_prompts` ← 775 |
| **输出** | `masks` (MASK) → 未连接，`visualization` (IMAGE) → 连接到 773 |

**工作流配置**：
```json
"widgets_values": [10, false]
```

参数解析：
| 参数 | 值 | 含义 |
|------|-----|------|
| `refinement_iterations` | 10 | 每个区域迭代优化 10 次（将掩码反馈回模型） |
| `use_multimask` | false | 每区域只生成 1 个掩码（而非 3 个候选） |

> ⚠️ 工作流中有一个未连接的 `sam3_model` (SAM3_MODEL) 输入——这是旧版接口，当前版本已移除。不影响运行。

**内部执行流程**：
1. `get_or_build_model(config)` → 获取/构建模型
2. `load_models_gpu([sam3_model])` → 确保模型在 GPU
3. `processor.set_image(pil_image)` → 提取图像特征
4. 对每个 prompt 区域循环：
   - 反归一化坐标：`(nx, ny) → (nx * img_w, ny * img_h)` 像素坐标
   - 反归一化框：center 格式 `[cx,cy,w,h]` → corner 格式 `[x1,y1,x2,y2]` 像素坐标
   - `model.predict_inst(state, ...)` → 获取掩码
   - refinement 迭代：将 best mask 反馈为 `mask_input`
5. 堆叠所有掩码，生成可视化

> ⚠️ **重要**：ComfyUI 的 `predict_inst` 接受 `[x1,y1,x2,y2]` 像素坐标的框，而官方 `SAM3InteractiveImagePredictor.predict()` 也接受 `[x0,y0,x1,y1]` 像素坐标——两者一致。

---

### 2.5 PreviewImage (节点 773)

| 属性 | 值 |
|------|-----|
| **类型** | `PreviewImage` (ComfyUI 内置) |
| **执行顺序** | 4 (最后执行) |
| **输入** | `images` ← 从 774(visualization) |
| **输出** | 无（显示在 UI） |

简单地将分割可视化结果显示在 ComfyUI 界面。

---

## 3. 连接线解析

工作流 JSON 中 `links` 数组格式：`[link_id, from_node, from_slot, to_node, to_slot, type]`

| Link ID | 源节点 | 源输出 | 目标节点 | 目标输入 | 数据类型 | 含义 |
|---------|--------|--------|----------|----------|----------|------|
| 1747 | 774 | 1 (visualization) | 773 | 0 (images) | IMAGE | 分割结果可视化 → 预览 |
| 1749 | 771 | 0 (IMAGE) | 774 | 1 (image) | IMAGE | 原图 → 分割节点 |
| 1750 | 775 | 0 (multi_prompts) | 774 | 2 (multi_prompts) | SAM3_MULTI_PROMPTS | 标注数据 → 分割节点 |
| 1751 | 771 | 0 (IMAGE) | 775 | 0 (image) | IMAGE | 原图 → 收集器 |
| 1917 | 772 | 0 (sam3_model_config) | 774 | 0 (sam3_model_config) | SAM3_MODEL_CONFIG | 模型配置 → 分割节点 |

**数据流拓扑**：
```
IMAGE (771) ──┬──→ 775:image        → 775:multi_prompts ──→ 774:multi_prompts
              └──→ 774:image        ←                        ↑
CONFIG (772) ────→ 774:sam3_model_config                    │
774:visualization ──→ 773:images
```

---

## 4. 执行顺序

ComfyUI 的 `order` 字段决定了节点执行顺序：

| 顺序 | 节点 ID | 类型 | 说明 |
|------|---------|------|------|
| 0 | 771 | LoadImage | 无依赖，最先执行 |
| 1 | 772 | LoadSAM3Model | 无依赖，可与 771 并行 |
| 2 | 775 | SAM3MultiRegionCollector | 依赖 771 (image) |
| 3 | 774 | SAM3MultipromptSegmentation | 依赖 772 + 771 + 775 |
| 4 | 773 | PreviewImage | 依赖 774 (visualization) |

**关键路径**：771 → 775 → 774 → 773

> 772 (LoadSAM3Model) 虽然独立于图像处理链，但 774 需要等待它完成。实际执行中 ComfyUI 会并行运行 771 和 772。

---

## 5. 参数配置分析

### 当前配置总结

| 参数 | 值 | 说明 |
|------|-----|------|
| 模型精度 | fp32 | 未使用半精度，兼容性最佳但 VRAM 占用大 |
| 编译优化 | 否 (flash_attn 旧参数被忽略) | 未启用 torch.compile |
| 迭代优化 | 10 次 | 较高，每区域分割精度优先 |
| 多掩码模式 | 关闭 | 每区域单掩码，确定性输出 |
| 置信度阈值 | 0.2 (ComfyUI 默认) | ⚠️ 官方默认为 0.5，2.5 倍差异 |
| NMS | 开启 (iou=0.5) | ⚠️ 官方无 NMS |

### 配置建议

| 场景 | precision | refinement | use_multimask | confidence | 说明 |
|------|-----------|------------|---------------|------------|------|
| 快速预览 | bf16 | 0 | false | 0.5 | 最快，官方默认阈值 |
| 精细分割 | bf16 | 5-10 | true | 0.3 | 3 候选选最优 |
| 多点标注 | bf16 | 3-5 | false | 0.3 | 多点已提供强约束 |
| 单点模糊 | bf16 | 5-10 | true | 0.2 | 单点歧义大，需多掩码候选 |

---

## 6. 用户交互场景还原

### 场景：分割图中的 4 个不同区域

基于 SAM.json 中的 `widgets_values`，用户的操作过程如下：

**Step 1：上传图片**
- 用户在 LoadImage 节点上传 `segment_test_bot.png`
- 图片显示在 SAM3MultiRegionCollector 的画布上

**Step 2：标注区域 0（简单目标）**
- 左键点击目标位置添加 1 个正点 → 坐标 (276.78, 418.49)
- Shift+左键在排除区域添加 2 个负点 → (230.16, 533.23) × 2
- 正点告诉模型"这里是我要的"，负点告诉模型"这里不是"

**Step 3：标注区域 1（多点标注）**
- 在目标不同部位添加 3 个正点 → (609.91, 391.98), (657.81, 556.23), (666.94, 695.38)
- 多正点帮助模型理解目标的完整范围

**Step 4：标注区域 2（细长目标）**
- 4 个正点沿目标分布 → 覆盖目标的各个部分

**Step 5：标注区域 3（复杂目标）**
- 8 个正点 + 10 个负点
- 正点覆盖目标的上下左右
- 负点排除相邻干扰区域
- 这是最复杂的标注，需要精细区分目标与背景

**Step 6：运行分割**
- 点击 Queue Prompt 执行工作流
- SAM3MultipromptSegmentation 依次处理 4 个区域
- 每个区域迭代优化 10 次
- 最终结果在 PreviewImage 中显示

### 坐标流经路径

```
用户点击 (276.78, 418.49) 像素
  ↓ SAM3MultiRegionCollector.execute()
  ↓ normalized_x = 276.78 / img_width
  ↓ normalized_y = 418.49 / img_height
  ↓ 输出: SAM3_MULTI_PROMPTS (归一化 0-1)
  ↓ SAM3MultipromptSegmentation.execute()
  ↓ px = normalized_x * img_w  (反归一化回像素)
  ↓ py = normalized_y * img_h
  ↓ model.predict_inst(state, point_coords=np.array([[px, py]]), normalize_coords=True)
  ↓ predict_inst 内部: coords / img_size (再次归一化)
  ↓ 模型处理
```

> 完整坐标转换链路详见 [03_DATA_FORMATS.md](03_DATA_FORMATS.md)

---

## 7. 官方 API 工作流对比

### 图像 Grounding 工作流

**官方 API**：
```python
from sam3 import build_sam3_image_model

model = build_sam3_image_model(checkpoint="sam3.safetensors")
processor = model.processor  # Sam3Processor

# Step 1: 设置图像
processor.set_image(image)

# Step 2: 文本提示
state = processor.set_text_prompt(text="cat", state=None)

# Step 3: 几何提示（可选）— 官方用 add_geometric_prompt
state = processor.add_geometric_prompt(
    boxes=torch.tensor([[cx, cy, w, h]]),  # [cx,cy,w,h] 归一化
    labels=torch.tensor([True]),
    text_str="cat",  # 可选，与文本提示关联
    state=state,
)

# 结果
masks = state["masks"]
scores = state["iou_scores"]
```

**ComfyUI 节点** (SAM3Grounding)：
```python
processor.set_confidence_threshold(0.2)  # ⚠️ 官方默认 0.5
state = processor.set_image(pil_image)
state = processor.set_text_prompt(text, state)
state = processor.add_multiple_box_prompts(boxes, labels, state)  # ⚠️ 官方用 add_geometric_prompt
# NMS 后处理（官方没有）
```

### 图像 Interactive 工作流

**官方 API**：
```python
from sam3 import build_sam3_image_model

model = build_sam3_image_model(
    checkpoint="sam3.safetensors",
    enable_inst_interactivity=True,  # 必须！
)
predictor = model.inst_interactive_predictor

# Step 1: 设置图像
model.processor.set_image(image)
state = model.processor.get_state()

# Step 2: 预测
masks, scores, low_res = predictor.predict(
    state,
    point_coords=np.array([[px, py]]),  # 像素坐标
    point_labels=np.array([1]),
    box=np.array([x0, y0, x1, y1]),     # 像素坐标！
    multimask_output=False,
)
```

**ComfyUI 节点** (SAM3Segmentation)：
```python
# 完全相同的 API
masks, scores, low_res = model.predict_inst(
    state,
    point_coords=np.array([[px, py]]),
    point_labels=np.array([1]),
    box=np.array([x1, y1, x2, y2]),
    normalize_coords=True,  # ComfyUI 额外参数
    multimask_output=False,
)
```

### 视频 PCS 工作流

**官方 API**：
```python
from sam3 import build_sam3_predictor

predictor = build_sam3_predictor(version="sam3")  # 或 "sam3.1"

# 请求式调度
result = predictor.handle_request({
    "type": "add_prompt",
    "frame_idx": 0,
    "boxes_xywh": [[xmin, ymin, w, h]],  # ⚠️ 归一化 xywh（非 cxcywh！）
    "text": "cat",
    "labels": [1],
})

# 传播
result = predictor.handle_request({
    "type": "propagate_in_video",
})
```

**ComfyUI 节点** (SAM3VideoSegmentation)：
```python
# 使用不同的 API — 直接方法调用而非 handle_request
predictor.start_session(session_id, images[0])
masks, scores = predictor.add_prompt(session_id, points=..., boxes=...)
video_masks = predictor.propagate_in_video(session_id, frames=images)
```

> ⚠️ 视频 API 的框格式不同于图像：
> - 官方视频: `boxes_xywh=[xmin, ymin, w, h]` 归一化
> - 官方图像 Grounding: `boxes=[cx, cy, w, h]` 归一化
> - 官方图像 Interactive: `box=[x0, y0, x1, y1]` 像素

### SAM 3.1 Multiplex 工作流（官方独有）

```python
from sam3 import build_sam3_predictor

predictor = build_sam3_predictor(version="sam3.1")

# 流式请求（适用于大量对象）
for request in requests:
    result = predictor.handle_stream_request(request)

# 或批量请求
result = predictor.handle_request({
    "type": "add_prompt",
    ...
})
```

> ComfyUI-SAM3-main 当前**不支持** SAM 3.1 Multiplex 模式。
