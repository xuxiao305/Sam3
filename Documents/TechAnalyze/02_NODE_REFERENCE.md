# SAM3 节点参考手册

> ComfyUI-SAM3-main 全部节点 + 官方 API 对照

---

## 目录

1. [节点总览](#1-节点总览)
2. [模型加载节点](#2-模型加载节点)
3. [图像分割节点](#3-图像分割节点)
4. [视频分割节点](#4-视频分割节点)
5. [交互与标注节点](#5-交互与标注节点)
6. [辅助工具节点](#6-辅助工具节点)
7. [兼容性矩阵](#7-兼容性矩阵)
8. [官方 API 对照表](#8-官方-api-对照表)

---

## 1. 节点总览

### 分类

| 类别 | 节点 | 源文件 | 状态 |
|------|------|--------|------|
| **模型加载** | LoadSAM3Model | `load_model.py` | ✅ 稳定 |
| **图像-交互式** | SAM3Segmentation | `segmentation.py` | ✅ 稳定 |
| **图像-Grounding** | SAM3Grounding | `segmentation.py` | ✅ 稳定 |
| **图像-多区域** | SAM3MultipromptSegmentation | `segmentation.py` | ✅ 稳定 |
| **图像-点标注** | SAM3PointsSegmentation | `segmentation.py` | ✅ 稳定 |
| **图像-框标注** | SAM3BboxSegmentation | `segmentation.py` | ✅ 稳定 |
| **图像-文本** | SAM3TextSegmentation | `segmentation.py` | ✅ 稳定 |
| **视频-基础** | SAM3VideoSegmentation | `sam3_video_nodes.py` | ✅ 稳定 |
| **视频-点提示** | SAM3VideoPointPrompt | `sam3_video_nodes.py` | ✅ 稳定 |
| **视频-交互** | SAM3VideoInteractive | `sam3_video_nodes.py` | ✅ 稳定 |
| **交互-多区域收集** | SAM3MultiRegionCollector | `sam3_interactive.py` | ✅ 稳定 |
| **交互-点标注** | SAM3PointAnnotator | `sam3_interactive.py` | ✅ 稳定 |
| **交互-框标注** | SAM3BboxAnnotator | `sam3_interactive.py` | ✅ 稳定 |
| **模型补丁** | SAM3ModelPatcher | `sam3_model_patcher.py` | ⚠️ 实验性 |

### 继承关系

```
节点基类 (ComfyUI)
  ├── LoadSAM3Model          — 独立节点
  ├── SAM3Segmentation       — 图像交互式（单区域）
  ├── SAM3Grounding          — 图像 Grounding（文本+框）
  ├── SAM3MultipromptSegmentation — 图像多区域
  ├── SAM3PointsSegmentation — 图像纯点标注
  ├── SAM3BboxSegmentation   — 图像纯框标注
  ├── SAM3TextSegmentation   — 图像纯文本
  ├── SAM3VideoSegmentation  — 视频基础分割
  ├── SAM3VideoPointPrompt   — 视频点提示
  ├── SAM3VideoInteractive   — 视频交互式
  ├── SAM3MultiRegionCollector — 多区域标注收集器
  ├── SAM3PointAnnotator     — 点标注交互
  ├── SAM3BboxAnnotator      — 框标注交互
  └── SAM3ModelPatcher       — 模型热补丁
```

---

## 2. 模型加载节点

### LoadSAM3Model

**文件**：`nodes/load_model.py`  
**ComfyUI 节点名**：`LoadSAM3Model`  
**功能**：加载 SAM3 模型权重，返回模型配置对象

#### 输入

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `precision` | COMBO | `"fp32"` | 精度选项：`fp32`, `bf16`, `fp16` |
| `compile` | COMBO | `"disable"` | 是否启用 `torch.compile`：`enable`, `disable` |

#### 输出

| 输出 | 类型 | 说明 |
|------|------|------|
| `sam3_model_config` | SAM3_MODEL_CONFIG | 模型配置字典，包含路径和参数 |

#### 输出格式

```python
{
    "checkpoint_path": "models/sam3/sam3.safetensors",
    "bpe_path": "nodes/sam3/bpe_simple_vocab_16e6.txt.gz",
    "precision": "fp32",       # 用户选择的精度
    "dtype": "fp32",           # 映射后的 torch dtype 字符串
    "compile": False,          # 是否 torch.compile
}
```

#### 官方 API 对照

| 方面 | ComfyUI | 官方 API |
|------|---------|----------|
| 入口函数 | `build_sam3_video_model(config)` | `build_sam3_image_model()` 或 `build_sam3_predictor(version=)` |
| 模型缓存 | `MODEL_CACHE` (LRU, max 2) | 无（用户自行管理） |
| 精度控制 | `precision` 参数 | 传递 `device` + `dtype` 参数 |
| 版本选择 | 无（始终加载 sam3） | `version="sam3"` / `"sam3.1"` |
| 编译 | `torch.compile` 选项 | 无内置编译选项 |

#### ⚠️ 差异

1. **ComfyUI 不支持 SAM 3.1 版本选择** — 官方支持 `version="sam3.1"` 启用 Multiplex 模式
2. **ComfyUI 使用 `build_sam3_video_model`** 加载，但实际用于图像和视频——这是正确的，因为官方视频模型包含图像能力
3. **模型缓存策略不同** — ComfyUI 用 LRU 缓存最近 2 个模型；官方 API 不提供缓存

---

## 3. 图像分割节点

### 3.1 SAM3Segmentation (图像交互式)

**文件**：`nodes/segmentation.py`  
**功能**：单区域交互式分割（点+框）

#### 输入

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sam3_model_config` | SAM3_MODEL_CONFIG | — | 模型配置 |
| `image` | IMAGE | — | 输入图像 [B,H,W,C] |
| `positive_points` | STRING | `""` | 正点 JSON |
| `negative_points` | STRING | `""` | 负点 JSON |
| `bbox` | STRING | `""` | 框坐标 JSON |
| `refinement_iterations` | INT | 0 | 掩码迭代优化次数 |
| `use_multimask` | BOOLEAN | False | 是否生成 3 个候选掩码 |
| `confidence_threshold` | FLOAT | 0.2 | 置信度阈值 |

#### 输出

| 输出 | 类型 | 说明 |
|------|------|------|
| `masks` | MASK | 分割掩码 [B,H,W] |
| `visualization` | IMAGE | 可视化图像 [B,H,W,C] |

#### 官方 API 对照

```python
# ComfyUI 内部调用
masks, scores, low_res = model.predict_inst(
    state,
    point_coords=np.array([[px, py]]),
    point_labels=np.array([1]),
    box=np.array([x1, y1, x2, y2]),  # 像素坐标 [x0,y0,x1,y1]
    multimask_output=False,
)

# 官方 API
masks, scores, low_res = predictor.predict(
    state,
    point_coords=np.array([[px, py]]),
    point_labels=np.array([1]),
    box=np.array([x0, y0, x1, y1]),  # 像素坐标 [x0,y0,x1,y1]
    multimask_output=False,
)
```

> ✅ 图像交互式 API 基本一致

---

### 3.2 SAM3Grounding (文本+框 Grounding)

**文件**：`nodes/segmentation.py`  
**功能**：基于文本描述和/或框提示的 Grounding 分割

#### 输入

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sam3_model_config` | SAM3_MODEL_CONFIG | — | 模型配置 |
| `image` | IMAGE | — | 输入图像 |
| `text` | STRING | `""` | 目标文本描述 |
| `boxes` | STRING | `""` | 框坐标 JSON [cx,cy,w,h] 归一化 |
| `labels` | STRING | `""` | 框标签 JSON |
| `refinement_iterations` | INT | 0 | 掩码迭代优化次数 |
| `confidence_threshold` | FLOAT | 0.2 | 置信度阈值 |

#### 输出

| 输出 | 类型 | 说明 |
|------|------|------|
| `masks` | MASK | 分割掩码 |
| `visualization` | IMAGE | 可视化图像 |

#### 官方 API 对照

```python
# ComfyUI 调用
processor.set_confidence_threshold(0.2)       # ⚠️ 官方默认 0.5
state = processor.set_image(pil_image)
state = processor.set_text_prompt(text, state)
state = processor.add_multiple_box_prompts(    # ⚠️ 官方用 add_geometric_prompt
    boxes, labels, state
)
# + NMS 后处理 (iou=0.5)                       # ⚠️ 官方没有 NMS

# 官方 API
state = processor.set_image(image)
state = processor.set_text_prompt(text="cat", state)
state = processor.add_geometric_prompt(
    boxes=torch.tensor([[cx, cy, w, h]]),     # 归一化 cxcywh
    labels=torch.tensor([True]),
    text_str="cat",
    state=state,
)
# 无 NMS，无 confidence_threshold 设置
```

#### ⚠️ 关键差异

| 差异 | ComfyUI | 官方 | 影响 |
|------|---------|------|------|
| 方法名 | `add_multiple_box_prompts()` | `add_geometric_prompt()` | ComfyUI 扩展方法 |
| 置信度阈值 | 0.2 | 0.5 | ComfyUI 产生更多低质量掩码 |
| NMS | 有 (iou=0.5) | 无 | 可能改变输出行为 |
| 框格式 | [cx,cy,w,h] 归一化 | [cx,cy,w,h] 归一化 | ✅ 一致 |

---

### 3.3 SAM3MultipromptSegmentation (多区域分割)

**文件**：`nodes/segmentation.py`  
**功能**：多区域交互式分割，支持多个独立标注区域

#### 输入

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sam3_model_config` | SAM3_MODEL_CONFIG | — | 模型配置 |
| `image` | IMAGE | — | 输入图像 |
| `multi_prompts` | SAM3_MULTI_PROMPTS | — | 多区域标注数据 |
| `refinement_iterations` | INT | 10 | 每区域迭代优化次数 |
| `use_multimask` | BOOLEAN | False | 每区域是否生成 3 候选 |

#### 输出

| 输出 | 类型 | 说明 |
|------|------|------|
| `masks` | MASK | 所有区域掩码堆叠 [N,H,W] |
| `visualization` | IMAGE | 所有区域可视化（不同颜色） |

#### 内部执行流程

```
对每个 region in multi_prompts:
  1. 反归一化点: (nx, ny) → (nx*w, ny*h) 像素
  2. 反归一化框: [cx,cy,w,h] → [x1,y1,x2,y2] 像素
  3. model.predict_inst(state, points, box, ...)
  4. refinement: 将 best mask 反馈为 mask_input，重复 N 次
  5. 收集掩码
```

> 官方 API 没有等价的"多区域"封装——用户需自行循环调用 `predictor.predict()`。

---

### 3.4 SAM3PointsSegmentation

**文件**：`nodes/segmentation.py`  
**功能**：纯点标注分割

#### 输入

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sam3_model_config` | SAM3_MODEL_CONFIG | — | 模型配置 |
| `image` | IMAGE | — | 输入图像 |
| `positive_points` | STRING | `""` | 正点 JSON |
| `negative_points` | STRING | `""` | 负点 JSON |
| `refinement_iterations` | INT | 0 | 迭代优化次数 |
| `use_multimask` | BOOLEAN | False | 多掩码候选 |
| `confidence_threshold` | FLOAT | 0.2 | 置信度阈值 |

#### 输出

同 SAM3Segmentation

> 这是 SAM3Segmentation 的简化版本，省略了 `bbox` 输入。内部调用 `predict_inst` 时不传 box。

---

### 3.5 SAM3BboxSegmentation

**文件**：`nodes/segmentation.py`  
**功能**：纯框标注分割

#### 输入

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sam3_model_config` | SAM3_MODEL_CONFIG | — | 模型配置 |
| `image` | IMAGE | — | 输入图像 |
| `bbox` | STRING | `""` | 框坐标 JSON |
| `refinement_iterations` | INT | 0 | 迭代优化次数 |
| `use_multimask` | BOOLEAN | False | 多掩码候选 |
| `confidence_threshold` | FLOAT | 0.2 | 置信度阈值 |

#### 输出

同 SAM3Segmentation

> 这是 SAM3Segmentation 的简化版本，省略了点输入。

---

### 3.6 SAM3TextSegmentation

**文件**：`nodes/segmentation.py`  
**功能**：纯文本 Grounding 分割

#### 输入

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sam3_model_config` | SAM3_MODEL_CONFIG | — | 模型配置 |
| `image` | IMAGE | — | 输入图像 |
| `text` | STRING | `""` | 文本描述 |
| `refinement_iterations` | INT | 0 | 迭代优化次数 |
| `confidence_threshold` | FLOAT | 0.2 | 置信度阈值 |

#### 输出

同 SAM3Grounding

> SAM3Grounding 的简化版，省略了框输入。仅使用文本提示。

---

## 4. 视频分割节点

### 4.1 SAM3VideoSegmentation

**文件**：`nodes/sam3_video_nodes.py`  
**功能**：视频基础分割（帧序列输入）

#### 输入

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sam3_model_config` | SAM3_MODEL_CONFIG | — | 模型配置 |
| `images` | IMAGE | — | 帧序列 [B,H,W,C] |
| `points` | STRING | `""` | 点标注 JSON |
| `labels` | STRING | `""` | 点标签 JSON |
| `boxes` | STRING | `""` | 框坐标 JSON |
| `frame_idx` | INT | 0 | 标注所在帧索引 |

#### 输出

| 输出 | 类型 | 说明 |
|------|------|------|
| `masks` | MASK | 视频掩码 [N,T,H,W] |
| `visualization` | IMAGE | 可视化帧序列 |

#### 官方 API 对照

```python
# ComfyUI 调用
predictor.start_session(session_id, images[0])
masks, scores = predictor.add_prompt(session_id, points=..., boxes=...)
video_masks = predictor.propagate_in_video(session_id, frames=images)

# 官方 API (handle_request 模式)
result = predictor.handle_request({
    "type": "add_prompt",
    "frame_idx": 0,
    "points": [[x, y]],
    "labels": [1],
    "boxes_xywh": [[xmin, ymin, w, h]],  # ⚠️ 归一化 xywh
})

result = predictor.handle_request({
    "type": "propagate_in_video",
})
```

#### ⚠️ 关键差异

| 差异 | ComfyUI | 官方 |
|------|---------|------|
| API 模式 | 直接方法调用 | `handle_request(dict)` 请求式调度 |
| 框格式 | 像素坐标 [x0,y0,x1,y1] | 归一化 [xmin,ymin,w,h] |
| 会话管理 | session_id 字符串 | 内部状态 |
| SAM 3.1 | 不支持 | `Sam3MultiplexVideoPredictor` |

---

### 4.2 SAM3VideoPointPrompt

**文件**：`nodes/sam3_video_nodes.py`  
**功能**：视频点提示（配合视频交互使用）

#### 输入

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sam3_model_config` | SAM3_MODEL_CONFIG | — | 模型配置 |
| `images` | IMAGE | — | 帧序列 |
| `points` | STRING | `""` | 点标注 |
| `labels` | STRING | `""` | 标签 |
| `frame_idx` | INT | 0 | 标注帧 |

#### 输出

| 输出 | 类型 | 说明 |
|------|------|------|
| `masks` | MASK | 视频掩码 |
| `visualization` | IMAGE | 可视化 |

---

### 4.3 SAM3VideoInteractive

**文件**：`nodes/sam3_video_nodes.py`  
**功能**：视频交互式分割（支持动态帧加载）

#### 特殊功能
- 支持前端 JS 交互 `sam3_video_dynamic.js`
- 动态帧导航和标注
- 实时分割反馈

---

## 5. 交互与标注节点

### 5.1 SAM3MultiRegionCollector

**文件**：`nodes/sam3_interactive.py`  
**前端**：`web/sam3_multiregion_widget.js`  
**功能**：多区域标注收集器，支持在画布上标注多个独立区域

#### 输入

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `image` | IMAGE | — | 输入图像（用于显示和坐标归一化） |
| `multi_prompts_store` | STRING (hidden) | `""` | 标注数据 JSON（前端写入） |

#### 输出

| 输出 | 类型 | 说明 |
|------|------|------|
| `multi_prompts` | SAM3_MULTI_PROMPTS | 多区域标注数据 |

#### 输出格式

```python
{
    "regions": [
        {
            "positive_points": [{"x": 0.3, "y": 0.5}, ...],   # 归一化坐标
            "negative_points": [{"x": 0.1, "y": 0.2}, ...],   # 归一化坐标
            "bbox": {"x": 0.2, "y": 0.3, "w": 0.4, "h": 0.5}, # 归一化 cxcywh
        },
        ...
    ]
}
```

#### 用户交互

| 操作 | 效果 |
|------|------|
| 左键点击 | 添加正点 |
| Shift+左键 | 添加负点 |
| 拖拽 | 绘制框 |
| 区域选择器 | 切换标注区域 |

> 无官方等价——这是 ComfyUI 特有的交互层封装。

---

### 5.2 SAM3PointAnnotator

**文件**：`nodes/sam3_interactive.py`  
**前端**：`web/sam3_points_widget.js`  
**功能**：单区域点标注交互

#### 输入

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `image` | IMAGE | — | 输入图像 |
| `points_store` | STRING (hidden) | `""` | 点标注 JSON |

#### 输出

| 输出 | 类型 | 说明 |
|------|------|------|
| `positive_points` | STRING | 正点 JSON |
| `negative_points` | STRING | 负点 JSON |

---

### 5.3 SAM3BboxAnnotator

**文件**：`nodes/sam3_interactive.py`  
**前端**：`web/sam3_bbox_widget.js`  
**功能**：框标注交互

#### 输入

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `image` | IMAGE | — | 输入图像 |
| `bbox_store` | STRING (hidden) | `""` | 框标注 JSON |

#### 输出

| 输出 | 类型 | 说明 |
|------|------|------|
| `bbox` | STRING | 框坐标 JSON |

---

## 6. 辅助工具节点

### SAM3ModelPatcher

**文件**：`nodes/sam3_model_patcher.py`  
**功能**：对已加载模型进行热补丁（如修改注意力机制）

#### ⚠️ 实验性

此节点为实验性功能，可能不稳定。用于高级用户调试和优化模型行为。

---

## 7. 兼容性矩阵

### 输入格式兼容性

| 节点 | 点标注 | 框标注 | 文本标注 | 掩码标注 | 多区域 |
|------|--------|--------|----------|----------|--------|
| SAM3Segmentation | ✅ | ✅ | ❌ | ❌ | ❌ |
| SAM3Grounding | ❌ | ✅ | ✅ | ❌ | ❌ |
| SAM3MultipromptSegmentation | ✅ | ✅ | ❌ | ❌ | ✅ |
| SAM3PointsSegmentation | ✅ | ❌ | ❌ | ❌ | ❌ |
| SAM3BboxSegmentation | ❌ | ✅ | ❌ | ❌ | ❌ |
| SAM3TextSegmentation | ❌ | ❌ | ✅ | ❌ | ❌ |
| SAM3VideoSegmentation | ✅ | ✅ | ❌ | ❌ | ❌ |
| SAM3VideoPointPrompt | ✅ | ❌ | ❌ | ❌ | ❌ |

### 官方 API 兼容性

| 官方能力 | ComfyUI 对应 | 兼容程度 |
|----------|-------------|----------|
| `set_text_prompt()` | SAM3Grounding / SAM3TextSegmentation | ✅ 完全 |
| `add_geometric_prompt()` | SAM3Grounding (用 `add_multiple_box_prompts` 替代) | ⚠️ 方法名不同 |
| `predictor.predict()` (Interactive) | SAM3Segmentation / SAM3MultipromptSegmentation | ✅ 完全 |
| `handle_request()` (Video) | SAM3VideoSegmentation (用直接方法调用) | ⚠️ API 模式不同 |
| Exemplar Prompt | 无 | ❌ 未实现 |
| `add_mask_prompt()` | 无独立节点 | ⚠️ Prompt 类支持但无节点 |
| SAM 3.1 Multiplex | 无 | ❌ 未实现 |
| `build_sam3_predictor(version=)` | 无版本选择 | ❌ 仅 sam3 |

---

## 8. 官方 API 对照表

### 构建/加载

| 操作 | 官方 API | ComfyUI 等价 |
|------|----------|-------------|
| 图像模型 | `build_sam3_image_model(checkpoint, enable_inst_interactivity=)` | `LoadSAM3Model` + `get_or_build_model(config)` |
| 视频预测器 | `build_sam3_predictor(version=)` | `build_sam3_video_model(config)` |
| SAM 3.1 | `build_sam3_predictor(version="sam3.1")` | ❌ 不支持 |
| Multiplex | `Sam3MultiplexVideoPredictor` | ❌ 不支持 |

### 图像处理

| 操作 | 官方 API | ComfyUI 等价 |
|------|----------|-------------|
| 设置图像 | `processor.set_image(image)` | `processor.set_image(pil_image)` ✅ |
| 文本提示 | `processor.set_text_prompt(text, state)` | `processor.set_text_prompt(text, state)` ✅ |
| 几何提示 | `processor.add_geometric_prompt(boxes, labels, text_str, state)` | `processor.add_multiple_box_prompts(boxes, labels, state)` ⚠️ |
| 交互预测 | `predictor.predict(state, point_coords, point_labels, box, multimask_output)` | `model.predict_inst(state, ...)` ✅ |
| 置信度阈值 | 默认 0.5 | 默认 0.2 ⚠️ |
| NMS | 无 | `nms_masks(iou=0.5)` ⚠️ |

### 视频处理

| 操作 | 官方 API | ComfyUI 等价 |
|------|----------|-------------|
| 添加提示 | `handle_request({"type": "add_prompt", ...})` | `predictor.add_prompt(session_id, ...)` ⚠️ |
| 传播 | `handle_request({"type": "propagate_in_video"})` | `predictor.propagate_in_video(session_id, frames)` ⚠️ |
| 流式处理 | `handle_stream_request(request)` | ❌ 不支持 |
| 框格式 | `boxes_xywh=[xmin,ymin,w,h]` 归一化 | 像素坐标 ⚠️ |

### 独有扩展

| 扩展 | 提供方 | 说明 |
|------|--------|------|
| `add_multiple_box_prompts()` | ComfyUI | 批量框提示（官方只有单框 `add_geometric_prompt`） |
| `add_point_prompt()` | ComfyUI | 点提示通过 Grounding 路径（⚠️ 官方会忽略点） |
| `add_mask_prompt()` | ComfyUI | 掩码提示（Prompt 类支持，无独立节点） |
| 多区域收集器 | ComfyUI | 交互式多区域标注 |
| 可视化生成 | ComfyUI | 自动颜色编码可视化 |
| refinement 迭代 | ComfyUI | 将 best mask 反馈为 mask_input |
| 模型缓存 | ComfyUI | LRU 缓存，最多 2 个模型 |
