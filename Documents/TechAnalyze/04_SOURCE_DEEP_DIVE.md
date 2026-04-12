# SAM3 源码深度解析

> ComfyUI-SAM3-main 内部架构 + 官方 facebookresearch/sam3 架构对比

---

## 目录

1. [文件结构总览](#1-文件结构总览)
2. [模型加载与缓存](#2-模型加载与缓存)
3. [图像 Grounding 推理路径](#3-图像-grounding-推理路径)
4. [图像 Interactive 推理路径](#4-图像-interactive-推理路径)
5. [视频推理路径](#5-视频推理路径)
6. [核心模型架构](#6-核心模型架构)
7. [官方架构深度对比](#7-官方架构深度对比)
8. [跨架构特征注入](#8-跨架构特征注入)
9. [设计模式与缓存策略](#9-设计模式与缓存策略)
10. [研究问题标记](#10-研究问题标记)

---

## 1. 文件结构总览

### ComfyUI-SAM3-main

```
nodes/
├── __init__.py              # 节点注册入口
├── _model_cache.py          # LRU 模型缓存
├── image_utils.py           # 图像转换工具
├── inference_reconstructor.py  # 推理重建器
├── load_model.py            # 模型加载节点
├── sam3_interactive.py      # 交互式标注节点
├── sam3_model_patcher.py    # 模型热补丁
├── sam3_video_nodes.py      # 视频分割节点
├── segmentation.py          # 图像分割核心
├── utils.py                 # 通用工具
├── video_state.py           # 视频会话状态
└── sam3/                    # 内嵌简化版 sam3 库
    ├── __init__.py          # 导出 build_sam3_video_model 等
    ├── attention.py         # 注意力机制
    ├── model.py             # 模型定义
    ├── perflib.py           # 性能优化
    ├── predictor.py         # 预测器
    ├── text_encoder.py      # 文本编码器
    ├── tokenizer.py         # BPE tokenizer
    └── utils.py             # 工具函数
```

### 官方 facebookresearch/sam3（关键文件）

```
sam3/
├── __init__.py              # 导出 5 个 builder 函数
├── build_sam3.py            # model_builder — 5 个 builder 函数
├── modeling/
│   ├── sam3_base_predictor.py    # Sam3BasePredictor — handle_request 分发
│   ├── sam3_image_model.py       # Sam3ImageModel + Sam3Processor
│   ├── sam3_video_predictor.py   # Sam3VideoPredictor
│   ├── sam3_multiplex_video_predictor.py  # SAM 3.1 Multiplex
│   ├── sam3_det_model.py         # DETR 检测器
│   ├── sam3_prompt.py            # Prompt 数据类（含 mask 支持）
│   ├── sam3_tracker_model.py     # SAM2 追踪器
│   └── sam3_common.py            # 共享组件
└── utils/
    ├── transforms.py        # Transform Pipeline
    └── ...
```

---

## 2. 模型加载与缓存

### 2.1 ComfyUI 加载流程

```python
# load_model.py — LoadSAM3Model
class LoadSAM3Model:
    def execute(self, precision="fp32", compile="disable"):
        config = {
            "checkpoint_path": folder_paths.get_full_path("sam3", ...),
            "bpe_path": os.path.join(os.path.dirname(__file__), "sam3", ...),
            "precision": precision,
            "dtype": {"fp32": "fp32", "bf16": "bfloat16", "fp16": "float16"}[precision],
            "compile": compile == "enable",
        }
        return (config,)
```

```python
# segmentation.py — get_or_build_model()
def get_or_build_model(config):
    key = config["checkpoint_path"]
    model = MODEL_CACHE.get(key)  # LRU 缓存查找
    if model is None:
        model = build_sam3_video_model(config)  # 构建模型
        MODEL_CACHE.put(key, model)  # 放入缓存
    return model
```

### 2.2 模型缓存策略

```python
# _model_cache.py — LRUCache
class LRUCache:
    def __init__(self, max_size=2):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)  # 访问时移到末尾
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)  # 淘汰最久未用
            self.cache[key] = value
```

**缓存行为**：
- 最多缓存 2 个模型
- LRU 策略：最近使用的保留
- 切换 checkpoint 时，旧模型自动淘汰
- 被淘汰的模型需要重新加载

### 2.3 官方加载对比

```python
# 官方 API — 5 个 builder 函数
from sam3 import (
    build_sam3_image_model,       # 图像模型（含交互预测器）
    build_sam3_predictor,         # 视频预测器（version="sam3"/"sam3.1"）
    sam3_model_registry,          # 模型注册表
    sam3_hf_download,             # HuggingFace 下载
    Sam3Processor,                # 处理器类
)

# 图像模型加载
model = build_sam3_image_model(
    checkpoint="sam3.safetensors",
    device="cuda",
    dtype=torch.bfloat16,
    enable_inst_interactivity=True,  # 启用交互预测
)

# 视频预测器加载
predictor = build_sam3_predictor(
    version="sam3",       # 或 "sam3.1"
    checkpoint="sam3.safetensors",
    device="cuda",
)
```

**差异**：
| 方面 | ComfyUI | 官方 |
|------|---------|------|
| 入口 | `build_sam3_video_model(config)` | `build_sam3_image_model()` / `build_sam3_predictor()` |
| 缓存 | LRU(2) | 无 |
| 版本 | 仅 sam3 | sam3 / sam3.1 |
| 交互 | 始终启用 | 需 `enable_inst_interactivity=True` |
| 下载 | 本地路径 | 支持 HuggingFace 自动下载 |

---

## 3. 图像 Grounding 推理路径

### 3.1 调用链

```
SAM3Grounding.execute()
  ↓ get_or_build_model(config)
  ↓ load_models_gpu([sam3_model])
  ↓ processor.set_confidence_threshold(0.2)  ← ⚠️ 官方默认 0.5
  ↓ processor.set_image(pil_image)
  ↓ state = processor.set_text_prompt(text, state)
  ↓ state = processor.add_multiple_box_prompts(boxes, labels, state)
  ↓ masks = state["masks"]
  ↓ nms_masks(masks, iou_threshold=0.5)       ← ⚠️ 官方无 NMS
  ↓ visualization = create_visualization(masks, image)
```

### 3.2 Sam3Processor 内部

```python
# sam3/predictor.py (ComfyUI 内嵌版)
class Sam3Processor:
    def set_image(self, image):
        """提取图像特征"""
        # image: PIL Image
        # 内部: resize → normalize → backbone → FPN features
        self.image_embedding = self.backbone(preprocessed_image)
        self.image_size = image.size  # (W, H)

    def set_text_prompt(self, text, state=None):
        """编码文本提示"""
        tokens = self.tokenizer(text)
        text_embeddings = self.text_encoder(tokens)
        state = state or {}
        state["text_embeddings"] = text_embeddings
        return state

    def add_multiple_box_prompts(self, boxes, labels, state):
        """ComfyUI 扩展：批量框提示"""
        # boxes: [[cx,cy,w,h], ...] 归一化
        # 内部调用 _forward_grounding()
        return self._forward_grounding(boxes, labels, state)

    def add_geometric_prompt(self, boxes, labels, text_str=None, state=None):
        """官方方法：几何提示（框+可选文本关联）"""
        # boxes: [cx,cy,w,h] 归一化
        # text_str: 可选，与已有文本提示关联
        return self._forward_grounding(boxes, labels, state, text_str=text_str)

    def add_point_prompt(self, points, labels, state):
        """⚠️ ComfyUI 扩展：点提示（但在 Grounding 路径被忽略！）"""
        print("Warning: Point prompts are ignored in PCS.")
        return self._forward_grounding(None, labels, state, points=points)
```

### 3.3 _forward_grounding 核心

```python
def _forward_grounding(self, boxes, labels, state, text_str=None, points=None):
    """
    Grounding 核心推理：
    1. 文本/框编码 → Prompt Embedding
    2. DETR 检测器生成候选
    3. SAM2 掩码解码器生成掩码
    """
    # Step 1: 准备 prompt
    prompt = self._prepare_prompt(boxes, labels, text_str, points)

    # Step 2: DETR 检测器
    detections = self.det_model(
        image_features=self.image_embedding,
        prompt=prompt,
    )

    # Step 3: 掩码解码（SAM2 decoder）
    masks, scores = self.sam_mask_decoder(
        image_features=self.image_embedding,
        detections=detections,
    )

    # Step 4: 过滤
    if self.confidence_threshold > 0:
        keep = scores > self.confidence_threshold
        masks = masks[keep]
        scores = scores[keep]

    state["masks"] = masks
    state["iou_scores"] = scores
    return state
```

> **关键发现**：Grounding 路径中，点提示被打印警告后忽略。`_forward_grounding` 不使用点坐标。

---

## 4. 图像 Interactive 推理路径

### 4.1 调用链

```
SAM3MultipromptSegmentation.execute()
  ↓ get_or_build_model(config)
  ↓ load_models_gpu([sam3_model])
  ↓ processor.set_image(pil_image)
  ↓ state = processor.get_state()
  ↓
  ↓ 对每个 region:
  │   ↓ 反归一化坐标
  │   ↓ model.predict_inst(state, point_coords, point_labels, box, ...)
  │   ↓ refinement: 将 best mask 反馈为 mask_input
  │   ↓ 收集 masks
  ↓
  ↓ 堆叠所有区域 masks
  ↓ visualization = create_visualization(masks, image)
```

### 4.2 predict_inst 委托链

```python
# ComfyUI 内部
def predict_inst(self, state, point_coords=None, point_labels=None,
                 box=None, mask_input=None, multimask_output=False,
                 normalize_coords=True):
    """交互式预测 — 委托给 inst_interactive_predictor"""
    if self.inst_interactive_predictor is None:
        raise RuntimeError("需要 enable_inst_interactivity=True")

    return self.inst_interactive_predictor.predict(
        state=state,
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        mask_input=mask_input,
        multimask_output=multimask_output,
        normalize_coords=normalize_coords,
    )
```

```python
# Sam3InteractiveImagePredictor.predict()
def predict(self, state, point_coords=None, point_labels=None,
            box=None, mask_input=None, multimask_output=False,
            normalize_coords=True):
    """
    交互式预测核心：
    1. 坐标归一化（如果需要）
    2. Prompt 编码
    3. SAM2 掩码解码
    """
    # Step 1: 坐标处理
    if normalize_coords and point_coords is not None:
        point_coords = point_coords / max(state["image_size"])
    if normalize_coords and box is not None:
        box = box / max(state["image_size"])

    # Step 2: Prompt 编码
    sparse_embeddings, dense_embeddings = self.prompt_encoder(
        points=(point_coords, point_labels) if point_coords is not None else None,
        boxes=box,
        mask_input=mask_input,
    )

    # Step 3: SAM2 掩码解码
    low_res_masks, iou_scores = self.sam_mask_decoder(
        image_embeddings=state["image_embedding"],
        image_pe=state["image_pe"],
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=multimask_output,
    )

    # Step 4: 后处理
    masks = self.postprocess(low_res_masks, state["image_size"])

    return masks, iou_scores, low_res_masks
```

### 4.3 Refinement 迭代

```python
# segmentation.py 中的 refinement 循环
for iteration in range(refinement_iterations):
    # 将上一次的最佳掩码作为 mask_input 反馈
    best_idx = scores.argmax()
    mask_input = low_res_masks[best_idx:best_idx+1]  # [1,1,256,256]

    masks, scores, low_res_masks = model.predict_inst(
        state,
        mask_input=mask_input,  # 反馈掩码
        multimask_output=False,  # 单掩码模式
    )
```

> **原理**：将低分辨率的预测掩码反馈为 `mask_input`，相当于告诉模型"这是我上次的猜测，请优化"。每次迭代都基于上一次的结果精修边界。

---

## 5. 视频推理路径

### 5.1 ComfyUI 视频流程

```python
# sam3_video_nodes.py — SAM3VideoSegmentation
class SAM3VideoSegmentation:
    def execute(self, sam3_model_config, images, points, labels, boxes, frame_idx):
        # Step 1: 获取/构建模型
        predictor = get_or_build_video_predictor(config)

        # Step 2: 启动会话
        session_id = str(uuid.uuid4())
        predictor.start_session(session_id, images[0])

        # Step 3: 添加提示
        masks, scores = predictor.add_prompt(
            session_id,
            points=point_coords,
            point_labels=point_labels,
            boxes=box_coords,
            frame_idx=frame_idx,
        )

        # Step 4: 视频传播
        video_masks = predictor.propagate_in_video(
            session_id,
            frames=images,
        )

        # Step 5: 清理会话
        predictor.end_session(session_id)

        return (video_masks, visualization)
```

### 5.2 官方视频流程（handle_request 模式）

```python
# 官方 API — 请求式调度
from sam3 import build_sam3_predictor

predictor = build_sam3_predictor(version="sam3")

# Step 1: 添加提示
result = predictor.handle_request({
    "type": "add_prompt",
    "frame_idx": 0,
    "points": [[x1, y1], [x2, y2]],
    "labels": [1, 0],          # 正/负
    "boxes_xywh": [[xmin, ymin, w, h]],  # 归一化 xywh
    "text": "cat",
})

# Step 2: 传播
result = predictor.handle_request({
    "type": "propagate_in_video",
})

# Step 3: 获取结果
video_masks = result["masks"]
```

### 5.3 handle_request 分发机制

```python
# Sam3BasePredictor.handle_request()
class Sam3BasePredictor:
    def handle_request(self, request: dict):
        """统一请求式调度"""
        req_type = request.get("type")

        if req_type == "add_prompt":
            return self._handle_add_prompt(request)
        elif req_type == "propagate_in_video":
            return self._handle_propagate(request)
        elif req_type == "reset_state":
            return self._handle_reset(request)
        else:
            raise ValueError(f"Unknown request type: {req_type}")

    def _handle_add_prompt(self, request):
        """添加提示 — 内部调用 add_prompt()"""
        # 每次调用 add_prompt 都会 reset_state！
        self.reset_state()
        return self.add_prompt(...)
```

> ⚠️ **重要**：`handle_request("add_prompt")` 每次都会调用 `reset_state()`，意味着每次添加提示都会重置追踪器状态。这是设计如此——检测器为每个提示重新运行。

### 5.4 SAM 3.1 Multiplex（官方独有）

```python
# Sam3MultiplexVideoPredictor
class Sam3MultiplexVideoPredictor(Sam3VideoPredictor):
    """
    SAM 3.1 Object Multiplex 模式
    - 将大量对象分组处理（每组 max_num_objects=16）
    - multiplex_count=16 → 一次处理 16×16=256 个对象
    - 约 7 倍加速（相比逐个处理）
    """

    def handle_stream_request(self, request):
        """流式请求处理"""
        ...

    def handle_request(self, request):
        """批量请求处理"""
        ...
```

> ComfyUI 当前**不支持** Multiplex 模式。

---

## 6. 核心模型架构

### 6.1 SAM3 整体架构

```
                    ┌─────────────────────┐
                    │   Input Image       │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │  Perception Encoder  │  ← 共享视觉骨干
                    │  (ViT-H/14)         │     848M 参数
                    │  FPN features       │
                    └───┬─────────────┬───┘
                        │             │
            ┌───────────▼───┐   ┌─────▼──────────┐
            │  DETR Detector │   │  SAM2 Tracker   │
            │  (Grounding)   │   │  (Interactive)  │
            │                │   │                  │
            │  - Presence    │   │  - Prompt        │
            │    Token       │   │    Encoder       │
            │  - Text        │   │  - Mask          │
            │    Encoder     │   │    Decoder       │
            │  - Cross Attn  │   │  - Memory        │
            └───────┬────────┘   │    Attention     │
                    │            └────────┬─────────┘
                    │                     │
                    └──────┬──────────────┘
                           │
                    ┌──────▼──────┐
                    │   Masks     │
                    └─────────────┘
```

### 6.2 DETR 检测器 (Grounding 路径)

```python
# sam3_det_model.py (官方)
class Sam3DetModel(nn.Module):
    """
    DETR-based 目标检测器
    - 输入：图像 FPN 特征 + 文本/框/Exemplar 提示
    - 输出：检测框 + 对象嵌入
    - 关键：Presence Token 解耦 "what" 和 "where"
    """
    def __init__(self, ...):
        # Presence Token: 学习型嵌入，表示"此处有目标"
        self.presence_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Cross-attention: 与 FPN 特征交互
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads)

        # FFN
        self.ffn = FFN(embed_dim, ffn_dim)
```

**Presence Token 机制**：
- 传统 DETR：目标查询同时编码"是什么"和"在哪里"
- SAM3：Presence Token 只编码"有目标存在"（where），文本编码"是什么"（what）
- 好处：解耦后，文本和位置可以独立变化，更灵活

### 6.3 SAM2 追踪器 (Interactive 路径)

```python
# sam3_tracker_model.py (官方) / sam3/model.py (ComfyUI)
class Sam2Tracker(nn.Module):
    """
    SAM2-based 掩码追踪器
    - Prompt Encoder: 编码点/框/掩码提示
    - Mask Decoder: 生成掩码
    - Memory Attention: 视频帧间记忆（视频模式）
    """
    def __init__(self, ...):
        self.prompt_encoder = PromptEncoder(...)
        self.mask_decoder = MaskDecoder(...)
        self.memory_attention = MemoryAttention(...)  # 视频专用
```

### 6.4 跨架构连接

```
Perception Encoder
        │
        ├── FPN Level 0 ──→ conv_s0 ──→ SAM2 Mask Decoder (高分辨率特征)
        ├── FPN Level 1 ──→ conv_s1 ──→ SAM2 Mask Decoder (低分辨率特征)
        │
        └── FPN Features ──→ DETR Detector
```

> `conv_s0` 和 `conv_s1` 是适配层，将 SAM3 FPN 特征映射到 SAM2 的输入空间。

---

## 7. 官方架构深度对比

### 7.1 Prompt 类（官方 vs ComfyUI）

```python
# 官方 sam3_prompt.py
@dataclass
class Prompt:
    points: Optional[torch.Tensor] = None       # [N, 2] 像素
    point_labels: Optional[torch.Tensor] = None  # [N] 1/0
    boxes: Optional[torch.Tensor] = None         # [M, 4] 格式取决于路径
    box_labels: Optional[torch.Tensor] = None    # [M] True/False
    text: Optional[str] = None
    exemplar_image: Optional[torch.Tensor] = None  # ⚠️ 示例图
    mask_embeddings: Optional[torch.Tensor] = None  # ⚠️ 掩码嵌入
```

**ComfyUI 对比**：
- ✅ points, point_labels: 完全支持
- ✅ boxes, box_labels: 支持（格式差异见 03_DATA_FORMATS.md）
- ✅ text: 支持
- ❌ exemplar_image: 不支持
- ⚠️ mask_embeddings: Prompt 类支持，但无独立节点暴露

### 7.2 Sam3Processor 方法对照

| 方法 | 官方 | ComfyUI | 差异 |
|------|------|---------|------|
| `set_image()` | ✅ | ✅ | 一致 |
| `set_text_prompt()` | ✅ | ✅ | 一致 |
| `add_geometric_prompt()` | ✅ | ❌ (用 `add_multiple_box_prompts` 替代) | 方法名不同 |
| `add_multiple_box_prompts()` | ❌ | ✅ (ComfyUI 扩展) | 批量框 |
| `add_point_prompt()` | ❌ | ✅ (ComfyUI 扩展) | ⚠️ Grounding 路径被忽略 |
| `add_mask_prompt()` | ❌ | ✅ (ComfyUI 扩展) | Prompt 类支持 |

### 7.3 视频预测器对照

| 方面 | 官方 | ComfyUI |
|------|------|---------|
| 基类 | `Sam3BasePredictor` | `Sam3VideoPredictor` (内嵌版) |
| API 模式 | `handle_request(dict)` | 直接方法调用 |
| 版本 | sam3 / sam3.1 | 仅 sam3 |
| Multiplex | `Sam3MultiplexVideoPredictor` | ❌ |
| 流式 | `handle_stream_request()` | ❌ |

---

## 8. 跨架构特征注入

### 8.1 问题背景

SAM3 使用 **Perception Encoder** (ViT-H) 作为共享骨干，但 DETR 检测器和 SAM2 追踪器需要不同格式的特征。需要适配层桥接。

### 8.2 FPN → SAM2 注入

```python
# 官方代码中的适配层
class Sam3ImageModel(nn.Module):
    def __init__(self, ...):
        # 从 Perception Encoder FPN 到 SAM2 的适配
        self.conv_s0 = nn.Conv2d(fpn_dim_s0, sam2_dim_s0, 1)  # 1x1 卷积
        self.conv_s1 = nn.Conv2d(fpn_dim_s1, sam2_dim_s1, 1)  # 1x1 卷积

    def forward(self, image):
        # Perception Encoder 提取 FPN 特征
        fpn_features = self.perception_encoder(image)

        # 注入到 SAM2 的 backbone 特征
        sam2_features = {
            "s0": self.conv_s0(fpn_features[0]),  # 高分辨率
            "s1": self.conv_s1(fpn_features[1]),  # 低分辨率
        }

        return sam2_features
```

### 8.3 特征流图

```
Input Image
    │
    ▼
Perception Encoder (ViT-H)
    │
    ├── FPN Level 0 (高分辨率) ──→ conv_s0 ──→ SAM2 backbone.s0
    ├── FPN Level 1 (低分辨率) ──→ conv_s1 ──→ SAM2 backbone.s1
    │
    └── FPN All Levels ──→ DETR Detector
                              │
                              ├── Presence Token
                              ├── Text Embeddings
                              └── Object Queries
                                  │
                                  ▼
                              Detections → SAM2 Mask Decoder
```

### 8.4 为什么需要适配层

1. **维度不同**：Perception Encoder FPN 输出维度 ≠ SAM2 backbone 输出维度
2. **语义对齐**：1×1 卷积学习从 PE 特征空间到 SAM2 特征空间的映射
3. **独立优化**：适配层参数独立于骨干和追踪器，便于微调

---

## 9. 设计模式与缓存策略

### 9.1 模型缓存模式

```python
# 全局 LRU 缓存
MODEL_CACHE = LRUCache(max_size=2)

def get_or_build_model(config):
    """获取或构建模型 — 缓存命中则直接返回"""
    key = config["checkpoint_path"]
    model = MODEL_CACHE.get(key)
    if model is None:
        model = build_sam3_video_model(config)
        MODEL_CACHE.put(key, model)
    return model
```

**优点**：
- 同一 checkpoint 不重复加载
- LRU 自动管理显存

**缺点**：
- `max_size=2` 可能不够（多模型场景）
- 淘汰模型时没有显式清理 GPU 内存

### 9.2 State 模式

```python
# 推理状态通过 dict 传递
state = processor.get_state()
# state 包含：
# - image_embedding: 图像特征
# - image_pe: 位置编码
# - image_size: 图像尺寸
# - text_embeddings: 文本嵌入（Grounding 路径）
# - masks: 预测掩码
# - iou_scores: 置信度分数
```

**设计优点**：
- 无状态对象，线程安全
- 可序列化（理论上可跨进程传递）
- 方便链式调用：`state = method1(state); state = method2(state)`

### 9.3 视频会话模式

```python
# video_state.py
class VideoSessionManager:
    """管理多个视频分割会话"""
    sessions = {}  # session_id → session_state

    def create_session(self, session_id, first_frame):
        self.sessions[session_id] = {
            "predictor": predictor,
            "frames": [first_frame],
            "masks": {},
        }

    def get_session(self, session_id):
        return self.sessions[session_id]

    def end_session(self, session_id):
        del self.sessions[session_id]
```

### 9.4 官方 handle_request 分发模式

```python
# Sam3BasePredictor
class Sam3BasePredictor:
    def handle_request(self, request: dict):
        """统一请求式调度 — 官方推荐 API"""
        handlers = {
            "add_prompt": self._handle_add_prompt,
            "propagate_in_video": self._handle_propagate,
            "reset_state": self._handle_reset,
        }
        handler = handlers.get(request["type"])
        if handler is None:
            raise ValueError(f"Unknown: {request['type']}")
        return handler(request)
```

**设计优点**：
- 统一接口，易于序列化
- 便于前端/后端通信
- 支持流式扩展（`handle_stream_request`）

---

## 10. 研究问题标记

以下是源码分析中发现的待研究问题，按严重程度标记：

### 🔴 严重

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| 1 | **框格式不一致** | Grounding `[cx,cy,w,h]` vs Interactive `[x0,y0,x1,y1]` vs Video `[xmin,ymin,w,h]` | 跨路径复用坐标会出错 |
| 2 | **点在 Grounding 被忽略** | `add_point_prompt()` → `_forward_grounding()` → `print("Warning...")` | 用户以为点生效了实际没有 |
| 3 | **NMS 差异** | ComfyUI 添加 `nms_masks(iou=0.5)`, 官方无 | 结果不可复现 |
| 4 | **confidence_threshold 差异** | ComfyUI 默认 0.2, 官方默认 0.5 | 掩码质量差异大 |

### 🟡 中等

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| 5 | **Exemplar 机制** | 官方用 `text="visual"` 路径, ComfyUI 不支持 | 无法使用示例图提示 |
| 6 | **mask_embeddings** | Prompt 类支持, 无节点暴露 | 掩码提示能力不可用 |
| 7 | **conv_s0/conv_s1 适配层** | 跨架构特征注入 | 参数量/训练方式未明确 |
| 8 | **add_prompt 调用 reset_state** | 视频路径每次提示重置 | 无法增量添加多对象提示 |

### 🟢 低

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| 9 | **848M 参数分解** | 论文未详细说明 | 理解不完整 |
| 10 | **SA-Co 数据引擎** | 训练数据生成细节 | 复现困难 |
| 11 | **SAM 3.1 版本差异** | Multiplex 外的差异未明确 | 升级评估困难 |
