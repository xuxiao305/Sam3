# SAM3 快速参考卡

> API 速查 · 坐标公式 · 参数推荐 · 问题速查

---

## 构建 & 加载

```python
# 图像模型
from sam3 import build_sam3_image_model
model = build_sam3_image_model(
    checkpoint="sam3.safetensors",
    device="cuda",
    dtype=torch.bfloat16,
    enable_inst_interactivity=True,   # ⚠️ 交互模式必须
)
processor = model.processor
predictor = model.inst_interactive_predictor

# 视频预测器
from sam3 import build_sam3_predictor
predictor = build_sam3_predictor(
    version="sam3",       # "sam3" 或 "sam3.1"
    checkpoint="sam3.safetensors",
)
```

---

## 三条推理路径

| 路径 | 用途 | 入口 | 框格式 |
|------|------|------|--------|
| **Grounding** | 文本/框定位 | `processor.set_text_prompt()` + `add_geometric_prompt()` | `[cx,cy,w,h]` 归一化 |
| **Interactive** | 点/框交互 | `predictor.predict()` | `[x0,y0,x1,y1]` 像素 |
| **Video** | 视频分割 | `predictor.handle_request({})` | `[xmin,ymin,w,h]` 归一化 |

---

## 坐标格式速查

### 框格式转换

```
Grounding [cx,cy,w,h] 归一化
    ↕ 转换
Interactive [x0,y0,x1,y1] 像素
    ↕ 转换
Video [xmin,ymin,w,h] 归一化
```

### 转换公式

```python
# Grounding → Interactive
x0 = (cx - w/2) * W
y0 = (cy - h/2) * H
x1 = (cx + w/2) * W
y1 = (cy + h/2) * H

# Grounding → Video
xmin = cx - w/2
ymin = cy - h/2
# w, h 不变

# Video → Interactive
x0 = xmin * W
y0 = ymin * H
x1 = (xmin + w) * W
y1 = (ymin + h) * H

# Video → Grounding
cx = xmin + w/2
cy = ymin + h/2
# w, h 不变
```

### 点坐标

```python
# 像素 → 归一化
nx = px / W
ny = py / H

# 归一化 → 像素
px = nx * W
py = ny * H
```

---

## API 速查

### 图像 Grounding

```python
processor.set_image(image)
state = processor.set_text_prompt("cat", state)
state = processor.add_geometric_prompt(
    boxes=torch.tensor([[cx, cy, w, h]]),  # 归一化 cxcywh
    labels=torch.tensor([True]),
    text_str="cat",      # 可选
    state=state,
)
masks = state["masks"]
scores = state["iou_scores"]
```

### 图像 Interactive

```python
processor.set_image(image)
state = processor.get_state()
masks, scores, low_res = predictor.predict(
    state,
    point_coords=np.array([[px, py]]),  # 像素
    point_labels=np.array([1]),          # 1=正, 0=负
    box=np.array([x0, y0, x1, y1]),     # 像素（可选）
    mask_input=None,                      # [1,1,256,256]（可选）
    multimask_output=False,
)
```

### 视频

```python
# 添加提示
predictor.handle_request({
    "type": "add_prompt",
    "frame_idx": 0,
    "boxes_xywh": [[xmin, ymin, w, h]],  # 归一化 xywh
    "text": "person",
    "labels": [1],
})

# 传播
result = predictor.handle_request({"type": "propagate_in_video"})
masks = result["masks"]
```

### Refinement 迭代

```python
for _ in range(iterations):
    best = scores.argmax()
    mask_input = low_res[best:best+1]
    masks, scores, low_res = predictor.predict(
        state, mask_input=mask_input, multimask_output=False
    )
```

---

## 参数推荐

### 置信度阈值

| 场景 | 推荐值 | 说明 |
|------|--------|------|
| 通用 | 0.5 | 官方默认，可靠 |
| 宽松（多检） | 0.3 | 允许更多低质量 |
| 严格（高精） | 0.7 | 只保留高质量 |
| ComfyUI 默认 | 0.2 | ⚠️ 过低，可能产生噪声 |

### 精度

| 场景 | 推荐 | VRAM |
|------|------|------|
| 生产 | `bfloat16` | ~6 GB |
| 调试 | `float32` | ~12 GB |
| ❌ 避免 | `float16` | 数值不稳定 |

### Refinement 迭代

| 场景 | 推荐 | 说明 |
|------|------|------|
| 快速预览 | 0 | 最快 |
| 一般 | 3-5 | 平衡 |
| 精细 | 5-10 | 边界更精确 |
| >10 | 不推荐 | 收益递减 |

### Multimask

| 场景 | 推荐 | 说明 |
|------|------|------|
| 单点标注 | True | 歧义大，3 候选取最优 |
| 多点标注 | False | 约束充分，单掩码足够 |
| 框标注 | False | 框已提供强约束 |

---

## 官方 vs ComfyUI 差异

| 项目 | 官方 | ComfyUI | 影响 |
|------|------|---------|------|
| confidence_threshold | 0.5 | 0.2 | 🔴 掩码数量差异大 |
| NMS | 无 | iou=0.5 | 🔴 去重行为不同 |
| Grounding 框方法 | `add_geometric_prompt()` | `add_multiple_box_prompts()` | 🟡 方法名不同 |
| 视频框格式 | `[xmin,ymin,w,h]` 归一化 | 像素坐标 | 🔴 格式不同 |
| 视频调度 | `handle_request(dict)` | 直接方法调用 | 🟡 API 模式不同 |
| 点在 Grounding | ❌ 不支持（官方 API 无此方法） | 有 `add_point_prompt()` 但被忽略 | 🔴 静默失败 |
| SAM 3.1 | ✅ 支持 | ❌ | 🟡 功能缺失 |
| Exemplar | ✅ 支持 | ❌ | 🟡 功能缺失 |
| 模型缓存 | 无 | LRU(2) | 🟢 |

---

## 研究问题速查

### 🔴 严重

| # | 问题 | 摘要 |
|---|------|------|
| 1 | 框格式不一致 | Grounding `cxcywh` vs Interactive `x0y0x1y1` vs Video `xywh` |
| 2 | 点被静默忽略 | Grounding 路径的 `add_point_prompt()` 打印警告但不报错 |
| 3 | NMS 差异 | ComfyUI 加 NMS，官方不加，结果不可复现 |
| 4 | confidence 差异 | ComfyUI 0.2 vs 官方 0.5，2.5 倍差距 |

### 🟡 中等

| # | 问题 | 摘要 |
|---|------|------|
| 5 | Exemplar 机制 | 用 `text="visual"` 路径触发，ComfyUI 不支持 |
| 6 | mask_embeddings | Prompt 类支持，无节点暴露 |
| 7 | conv_s0/conv_s1 | 跨架构适配层，参数量/训练方式未明 |
| 8 | add_prompt 重置 | 视频路径每次添加提示都 reset_state |

### 🟢 低

| # | 问题 | 摘要 |
|---|------|------|
| 9 | 848M 参数分解 | 各组件参数量未详细说明 |
| 10 | SA-Co 数据引擎 | 训练数据生成细节缺失 |
| 11 | SAM 3.1 版本差异 | 除 Multiplex 外的差异未明确 |

---

## 错误排查

| 症状 | 可能原因 | 解决方案 |
|------|----------|----------|
| 掩码全空 | confidence 太高 | 降低到 0.3 |
| 掩码太多/质量差 | confidence 太低 | 提高到 0.5 |
| 框位置偏移 | 格式混淆 | 检查当前路径的框格式 |
| 点不生效 | 用了 Grounding 路径 | 改用 Interactive 路径 |
| `enable_inst_interactivity` 报错 | 未启用 | 加载时设 `True` |
| VRAM OOM | 精度太高/图太大 | 用 `bfloat16`，缩小图片 |
| 视频帧间不连贯 | 帧数太少/分辨率低 | 增加帧数，提高分辨率 |
| 多区域掩码不独立 | state 被覆盖 | 确保共享图像特征，不共享掩码 |
| NMS 改变了结果 | ComfyUI 特有 | 需复现官方时去掉 NMS |

---

## 版本信息

| 项目 | 值 |
|------|-----|
| SAM3 参数量 | 848M |
| 骨干网络 | Perception Encoder (ViT-H/14) |
| 检测器 | DETR-based (Grounding) |
| 追踪器 | SAM2-based (Interactive/Video) |
| SAM 3.1 发布 | 2026-03-27 |
| SAM 3.1 新特性 | Object Multiplex (~7x 加速 @128 对象) |
| 官方 Python | 3.12+ |
| 官方 PyTorch | 2.7+ |
| 官方 CUDA | 12.6+ |
| 输入分辨率 | 1008 × 1008 (内部) |
| 低分辨率掩码 | 256 × 256 |
| HuggingFace | `facebook/sam3`, `facebook/sam3.1` |
| GitHub | `facebookresearch/sam3` |
