# SAM3 数据格式与坐标系统

> 三条推理路径的坐标链、数据流转、格式规范与常见陷阱

---

## 目录

1. [坐标系统总览](#1-坐标系统总览)
2. [点坐标格式](#2-点坐标格式)
3. [框坐标格式](#3-框坐标格式)
4. [掩码格式](#4-掩码格式)
5. [文本提示格式](#5-文本提示格式)
6. [完整坐标转换链](#6-完整坐标转换链)
7. [三条路径的格式差异](#7-三条路径的格式差异)
8. [官方 vs ComfyUI 格式对照](#8-官方-vs-comfyui-格式对照)
9. [常见陷阱与调试](#9-常见陷阱与调试)

---

## 1. 坐标系统总览

SAM3 存在**三种不同的框坐标格式**，分别用于三条推理路径：

| 路径 | 框格式 | 坐标系 | 归一化 |
|------|--------|--------|--------|
| **Grounding / PCS** | `[cx, cy, w, h]` | 中心+宽高 | ✅ 0-1 归一化 |
| **Interactive** | `[x0, y0, x1, y1]` | 左上+右下角点 | ❌ 像素坐标 |
| **Video** | `[xmin, ymin, w, h]` | 左上角+宽高 | ✅ 0-1 归一化 |

> ⚠️ 这是跨路径工作时最易出错的点！三种格式不可混用。

---

## 2. 点坐标格式

### 归一化点 (0-1)

用于 `SAM3MultiRegionCollector` 输出和 ComfyUI 交互节点：

```json
{
    "x": 0.3,
    "y": 0.5
}
```

**归一化公式**：
```
normalized_x = pixel_x / image_width
normalized_y = pixel_y / image_height
```

### 像素点

用于 `predict_inst()` / `predictor.predict()` 调用：

```python
point_coords = np.array([[px, py]])  # 像素坐标
point_labels = np.array([1])          # 1=正, 0=负
```

### 转换流程

```
用户点击 (276.78, 418.49) 像素
  ↓ SAM3MultiRegionCollector.execute()
  ↓ nx = 276.78 / img_w
  ↓ ny = 418.49 / img_h
  ↓ 输出: {"x": nx, "y": ny}
  ↓
  ↓ SAM3MultipromptSegmentation.execute()
  ↓ px = nx * img_w   (反归一化回像素)
  ↓ py = ny * img_h
  ↓ point_coords = np.array([[px, py]])
  ↓ predict_inst(state, point_coords=point_coords, normalize_coords=True)
  ↓ 内部再次归一化: coords / img_size
```

> ⚠️ 双重归一化：ComfyUI 反归一化到像素后，`predict_inst` 的 `normalize_coords=True` 又归一化一次。这是设计如此——`predict_inst` 内部接受像素坐标，自行归一化到模型需要的格式。

---

## 3. 框坐标格式

### 3.1 Grounding 路径: `[cx, cy, w, h]` 归一化

**用途**：`SAM3Grounding`、`add_geometric_prompt()`、`add_multiple_box_prompts()`

**格式**：
```python
# 中心点 x, 中心点 y, 宽度, 高度 — 全部归一化 0-1
boxes = torch.tensor([[0.5, 0.5, 0.3, 0.4]])  # 中心在图中间，宽30%高40%
```

**与官方一致**：
- 官方 `add_geometric_prompt(boxes=...)` 使用 `[cx, cy, w, h]` 归一化 ✅
- ComfyUI `add_multiple_box_prompts(boxes=...)` 使用 `[cx, cy, w, h]` 归一化 ✅

### 3.2 Interactive 路径: `[x0, y0, x1, y1]` 像素

**用途**：`SAM3Segmentation`、`SAM3MultipromptSegmentation`、`predict_inst()`

**格式**：
```python
# 左上角 x, 左上角 y, 右下角 x, 右下角 y — 像素坐标
box = np.array([100, 200, 300, 400])  # 像素
```

**转换公式**（从 cxcywh 归一化）：
```python
# [cx, cy, w, h] 归一化 → [x0, y0, x1, y1] 像素
x0 = (cx - w/2) * img_w
y0 = (cy - h/2) * img_h
x1 = (cx + w/2) * img_w
y1 = (cy + h/2) * img_h
```

**ComfyUI 中的转换代码**（`segmentation.py`）：
```python
# 从 multi_prompts 中反归一化框
cx, cy, w, h = region["bbox"]["x"], region["bbox"]["y"], region["bbox"]["w"], region["bbox"]["h"]
x1 = int((cx - w / 2) * img_w)
y1 = int((cy - h / 2) * img_h)
x2 = int((cx + w / 2) * img_w)
y2 = int((cy + h / 2) * img_h)
```

**与官方一致**：
- 官方 `predictor.predict(box=...)` 使用 `[x0, y0, x1, y1]` 像素坐标 ✅

### 3.3 Video 路径: `[xmin, ymin, w, h]` 归一化

**用途**：官方 `handle_request({"boxes_xywh": ...})`

**格式**：
```python
# 左上角 x, 左上角 y, 宽度, 高度 — 归一化 0-1
boxes_xywh = [[0.1, 0.2, 0.3, 0.4]]  # 从 10%位置开始，宽30%高40%
```

**⚠️ 与 Grounding 格式的区别**：
| 格式 | 参考点 | 示例 [0.5, 0.5, 0.3, 0.4] 含义 |
|------|--------|--------------------------------|
| Grounding `[cx,cy,w,h]` | **中心** | 中心在 (50%,50%)，宽30%高40% |
| Video `[xmin,ymin,w,h]` | **左上角** | 从 (50%,50%) 开始，宽30%高40% |

**转换公式**：
```python
# Grounding [cx,cy,w,h] → Video [xmin,ymin,w,h]
xmin = cx - w/2
ymin = cy - h/2
# w, h 不变

# Video [xmin,ymin,w,h] → Grounding [cx,cy,w,h]
cx = xmin + w/2
cy = ymin + h/2
# w, h 不变

# Grounding [cx,cy,w,h] 归一化 → Interactive [x0,y0,x1,y1] 像素
x0 = (cx - w/2) * img_w
y0 = (cy - h/2) * img_h
x1 = (cx + w/2) * img_w
y1 = (cy + h/2) * img_h

# Video [xmin,ymin,w,h] 归一化 → Interactive [x0,y0,x1,y1] 像素
x0 = xmin * img_w
y0 = ymin * img_h
x1 = (xmin + w) * img_w
y1 = (ymin + h) * img_h
```

---

## 4. 掩码格式

### ComfyUI MASK 格式

```python
# 形状: [B, H, W] 或 [N, H, W]
# 类型: float32
# 值域: 0.0 ~ 1.0 (0=背景, 1=前景)
```

### 模型输出 low_res_masks

```python
# 形状: [1, 1, 256, 256]
# 类型: float32
# 值域: 任意浮点数（logits）
# 需要阈值化: mask = (low_res > 0).float()
```

### 多区域掩码堆叠

```python
# SAM3MultipromptSegmentation 输出
# masks: [N, H, W] — N = 区域数量
# 每个区域一个二值掩码
```

### 可视化掩码

```python
# visualization: [1, H, W, C] 或 [N, H, W, C]
# 类型: float32, 值域 0-1
# 不同区域用不同颜色编码
# 背景为原图半透明 + 掩码着色
```

---

## 5. 文本提示格式

### Grounding 文本

```python
# 单个文本描述
text = "cat"                    # 简单文本
text = "red car on the street"  # 描述性文本
```

### 官方 API 处理

```python
# 内部通过 text_encoder + tokenizer 处理
# tokenizer: BPE tokenizer (bpe_simple_vocab_16e6.txt.gz)
# text_encoder: Transformer-based encoder
# 输出: text_embeddings [1, num_tokens, embed_dim]
```

### Exemplar 提示

```python
# 官方支持示例图提示 — 使用特殊标记
TEXT_ID_FOR_VISUAL = 1
text_str = "visual"  # 特殊标记，触发示例图路径

# handle_request 中的 exemplar
{
    "type": "add_prompt",
    "text": "visual",
    "exemplar_image": ...,  # 示例图像
}
```

> ComfyUI 当前**不支持** Exemplar 提示。

---

## 6. 完整坐标转换链

### 6.1 交互式路径（点）

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌───────────────┐
│  用户点击     │     │  Collector 节点   │     │  Segmentation 节点│     │  predict_inst │
│  (px, py)像素 │ ──→ │  nx = px / W     │ ──→ │  px = nx * W     │ ──→ │  coords / sz  │ ──→ 模型
│              │     │  ny = py / H     │     │  py = ny * H     │     │  (再次归一化)  │
└──────────────┘     └──────────────────┘     └──────────────────┘     └───────────────┘
                       归一化 0-1               反归一化像素              模型归一化
```

### 6.2 交互式路径（框）

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────────────┐     ┌───────────────┐
│  用户拖框     │     │  Collector 节点   │     │  Segmentation 节点       │     │  predict_inst │
│  像素坐标     │ ──→ │  cx = (x1+x2)/2W │ ──→ │  x1 = (cx-w/2)*W       │ ──→ │  box / sz     │ ──→ 模型
│              │     │  cy = (y1+y2)/2H │     │  y1 = (cy-h/2)*H       │     │  (再次归一化)  │
│              │     │  w = (x2-x1)/W   │     │  x2 = (cx+w/2)*W       │     │               │
│              │     │  h = (y2-y1)/H   │     │  y2 = (cy+h/2)*H       │     │               │
└──────────────┘     └──────────────────┘     └─────────────────────────┘     └───────────────┘
                       cxcywh 归一化             corner 像素坐标               模型归一化
```

### 6.3 Grounding 路径（框）

```
┌──────────────┐     ┌──────────────────┐
│  用户输入     │     │  SAM3Grounding    │
│  [cx,cy,w,h] │ ──→ │  直接传给          │ ──→ add_multiple_box_prompts() ──→ _forward_grounding()
│  归一化       │     │  add_geometric_   │     (内部处理归一化坐标)
│              │     │  prompt()         │
└──────────────┘     └──────────────────┘
```

### 6.4 视频 Grounding 路径（框）

```
┌──────────────────┐     ┌──────────────────┐
│  官方 API         │     │  handle_request() │
│  boxes_xywh      │ ──→ │  解析 [xmin,ymin, │ ──→ _forward_grounding() 或 tracker
│  [xmin,ymin,w,h] │     │  w,h] 归一化      │
│  归一化           │     │                   │
└──────────────────┘     └──────────────────┘
```

---

## 7. 三条路径的格式差异

### 完整对比表

| 维度 | Grounding/PCS | Interactive | Video |
|------|---------------|-------------|-------|
| **点格式** | ⚠️ 被忽略 | `[px, py]` 像素 | `[px, py]` 像素 |
| **框格式** | `[cx,cy,w,h]` 归一化 | `[x0,y0,x1,y1]` 像素 | `[xmin,ymin,w,h]` 归一化 |
| **文本** | ✅ 支持 | ❌ 不支持 | ✅ 支持 |
| **掩码输入** | ✅ (Prompt 类) | ✅ (mask_input) | ❌ |
| **Exemplar** | ✅ (官方) | ❌ | ✅ (官方) |
| **归一化** | 外部归一化 | `normalize_coords=True` | 外部归一化 |
| **NMS** | ComfyUI 加，官方无 | 无 | 无 |
| **confidence** | 0.2 (ComfyUI) / 0.5 (官方) | 0.2 (ComfyUI) | 0.5 (官方) |

### 🔴 点提示在 Grounding 路径被静默忽略

**这是最重要的格式陷阱！**

```python
# ComfyUI 的 SAM3Grounding 节点
state = processor.add_point_prompt(points, labels, state)
# ↓ 内部调用 _forward_grounding()
# ↓ _forward_grounding() 代码:
#   print("Warning: Point prompts are ignored in PCS.")
#   # 点被完全忽略！只有文本和框参与推理
```

**影响**：如果用户在 Grounding 路径同时提供了点和文本/框，点不会生效，但**不会报错**。

**解决方案**：
- 需要点提示 → 使用 Interactive 路径（`SAM3Segmentation` / `SAM3MultipromptSegmentation`）
- 需要文本提示 → 使用 Grounding 路径（`SAM3Grounding`）
- 需要两者 → 分别运行，取掩码交集

---

## 8. 官方 vs ComfyUI 格式对照

### 框格式对照

| 场景 | 官方 API | ComfyUI | 一致性 |
|------|----------|---------|--------|
| Grounding 框输入 | `add_geometric_prompt(boxes=[cx,cy,w,h])` 归一化 | `add_multiple_box_prompts(boxes=[cx,cy,w,h])` 归一化 | ✅ 格式一致 |
| Interactive 框输入 | `predictor.predict(box=[x0,y0,x1,y1])` 像素 | `predict_inst(box=[x1,y1,x2,y2])` 像素 | ✅ 格式一致 |
| Video 框输入 | `handle_request({boxes_xywh=[xmin,ymin,w,h]})` 归一化 | `add_prompt(boxes=...)` 像素 | ❌ 格式不同 |
| 文本 Exemplar | `text="visual"`, exemplar_image | ❌ 不支持 | — |

### 参数对照

| 参数 | 官方默认 | ComfyUI 默认 | 差异 |
|------|----------|-------------|------|
| `confidence_threshold` | 0.5 | 0.2 | 2.5 倍 |
| NMS iou_threshold | 无 NMS | 0.5 | 行为差异 |
| `normalize_coords` | N/A (各路径自行处理) | True (predict_inst) | ComfyUI 额外参数 |
| `multimask_output` | False | False | ✅ 一致 |
| `refinement_iterations` | 0 (无内置) | 0-10 (手动实现) | ComfyUI 扩展 |

---

## 9. 常见陷阱与调试

### 陷阱 1：框格式混淆 🔴

```python
# ❌ 错误：在 Interactive 路径使用 cxcywh 归一化
box = np.array([0.5, 0.5, 0.3, 0.4])  # 这是 cxcywh 归一化
masks = model.predict_inst(state, box=box)  # 会被当作像素坐标！

# ✅ 正确：转换后再传
x0 = int((0.5 - 0.3/2) * img_w)  # = 0.35 * W
y0 = int((0.5 - 0.4/2) * img_h)  # = 0.3 * H
x1 = int((0.5 + 0.3/2) * img_w)  # = 0.65 * W
y1 = int((0.5 + 0.4/2) * img_h)  # = 0.7 * H
box = np.array([x0, y0, x1, y1])  # 像素坐标
masks = model.predict_inst(state, box=box, normalize_coords=True)
```

### 陷阱 2：Video 框格式与 Grounding 不同 🔴

```python
# ❌ 错误：在 Video 路径使用 cxcywh
handle_request({"boxes_xywh": [[0.5, 0.5, 0.3, 0.4]]})
# 这会被解读为：从 (50%, 50%) 开始，宽 30%，高 40%
# 而不是：中心在 (50%, 50%)

# ✅ 正确：使用 xywh 格式
handle_request({"boxes_xywh": [[0.35, 0.3, 0.3, 0.4]]})
# 从 (35%, 30%) 开始，宽 30%，高 40%
```

### 陷阱 3：点在 Grounding 路径被忽略 🔴

```python
# ❌ 错误：在 Grounding 路径同时用点和文本
processor.set_text_prompt("cat", state)
processor.add_point_prompt(points, labels, state)  # 被静默忽略！

# ✅ 正确：选择正确的路径
# 需要点 → 用 Interactive 路径
masks = model.predict_inst(state, point_coords=points, point_labels=labels)
```

### 陷阱 4：confidence_threshold 过低 🟡

```python
# ComfyUI 默认 0.2 可能产生低质量掩码
# 官方默认 0.5 更严格但更可靠
processor.set_confidence_threshold(0.5)  # 推荐使用官方默认值
```

### 陷阱 5：NMS 改变输出行为 🟡

```python
# ComfyUI 对 Grounding 结果添加 NMS
# 官方不添加 NMS
# 如果需要复现官方行为，应跳过 NMS
```

### 陷阱 6：图像尺寸对坐标的影响 🟡

```python
# ComfyUI IMAGE 格式: [B, H, W, C] float32
# 需要注意 H, W 的提取
img_h = image.shape[1]  # 不是 shape[0]！
img_w = image.shape[2]

# PIL Image 尺寸: (width, height)
# 注意 PIL 和 tensor 的 H/W 顺序相反
```

### 调试清单

遇到坐标相关问题时，按此顺序检查：

1. ☐ 确认当前使用哪条推理路径（Grounding / Interactive / Video）
2. ☐ 确认框格式匹配路径要求
3. ☐ 检查是否意外在 Grounding 路径使用点提示
4. ☐ 验证归一化/反归一化是否正确（0-1 范围 vs 像素）
5. ☐ 检查图像尺寸提取是否正确（tensor [B,H,W,C] vs PIL [W,H]）
6. ☐ 对比 confidence_threshold 设置（0.2 vs 0.5）
7. ☐ 检查是否有 NMS 影响 Grounding 结果
