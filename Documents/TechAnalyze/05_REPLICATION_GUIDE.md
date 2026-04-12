# SAM3 复刻指南

> 从零构建 SAM3 独立工具的完整指南（基于 ComfyUI 源码 + 官方 API）

---

## 目录

1. [环境准备](#1-环境准备)
2. [最小可运行示例](#2-最小可运行示例)
3. [图像 Grounding 路径](#3-图像-grounding-路径)
4. [图像 Interactive 路径](#4-图像-interactive-路径)
5. [视频分割路径](#5-视频分割路径)
6. [高级功能](#6-高级功能)
7. [API 合约参考](#7-api-合约参考)
8. [性能优化指南](#8-性能优化指南)
9. [常见问题与陷阱](#9-常见问题与陷阱)
10. [迁移清单](#10-迁移清单)

---

## 1. 环境准备

### 1.1 系统要求

| 依赖 | 最低版本 | 推荐版本 | 说明 |
|------|----------|----------|------|
| Python | 3.10+ | 3.12+ | 官方推荐 3.12+ |
| PyTorch | 2.1+ | 2.7+ | 官方推荐 2.7+ |
| CUDA | 12.1+ | 12.6+ | 官方推荐 12.6+ |
| VRAM | 8 GB | 16+ GB | 图像 8GB 够，视频建议 16GB+ |

### 1.2 安装步骤

#### 方案 A：使用官方 PyPI 包（推荐）

```bash
pip install sam3
```

#### 方案 B：从源码安装

```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
```

#### 方案 C：使用 ComfyUI 内嵌版

```bash
# ComfyUI-SAM3-main/nodes/sam3/ 目录包含简化版
# 需要手动安装依赖
pip install torch torchvision
```

### 1.3 模型下载

```python
# 方案 1：HuggingFace 自动下载（官方推荐）
from sam3 import sam3_hf_download
checkpoint = sam3_hf_download("facebook/sam3")

# 方案 2：手动下载
# https://huggingface.co/facebook/sam3 → sam3.safetensors
# https://huggingface.co/facebook/sam3.1 → sam3.1.safetensors (SAM 3.1)
```

---

## 2. 最小可运行示例

### 2.1 图像交互式分割（5 行代码）

```python
from sam3 import build_sam3_image_model
import numpy as np
from PIL import Image

# 1. 加载模型
model = build_sam3_image_model(
    checkpoint="sam3.safetensors",
    enable_inst_interactivity=True,
)
processor = model.processor

# 2. 设置图像
image = Image.open("test.jpg")
processor.set_image(image)
state = processor.get_state()

# 3. 预测
masks, scores, low_res = model.inst_interactive_predictor.predict(
    state,
    point_coords=np.array([[500, 300]]),  # 像素坐标
    point_labels=np.array([1]),            # 1=正
)

# 4. 结果
print(f"掩码形状: {masks.shape}")   # [1, 1, H, W]
print(f"置信度: {scores}")          # [1]
```

### 2.2 图像 Grounding 分割（5 行代码）

```python
from sam3 import build_sam3_image_model

model = build_sam3_image_model(checkpoint="sam3.safetensors")
processor = model.processor

# 1. 设置图像
processor.set_image(image)
state = None

# 2. 文本 + 框提示
state = processor.set_text_prompt("cat", state)
state = processor.add_geometric_prompt(
    boxes=torch.tensor([[0.5, 0.5, 0.3, 0.4]]),  # [cx,cy,w,h] 归一化
    labels=torch.tensor([True]),
    text_str="cat",
    state=state,
)

# 3. 结果
masks = state["masks"]        # [N, H, W]
scores = state["iou_scores"]  # [N]
```

### 2.3 视频分割（5 行代码）

```python
from sam3 import build_sam3_predictor

predictor = build_sam3_predictor(version="sam3")

# 1. 添加提示
result = predictor.handle_request({
    "type": "add_prompt",
    "frame_idx": 0,
    "boxes_xywh": [[0.1, 0.2, 0.3, 0.4]],  # [xmin,ymin,w,h] 归一化
    "text": "person",
    "labels": [1],
})

# 2. 传播到所有帧
result = predictor.handle_request({
    "type": "propagate_in_video",
})

video_masks = result["masks"]
```

---

## 3. 图像 Grounding 路径

### 3.1 完整代码模板

```python
import torch
import numpy as np
from PIL import Image
from sam3 import build_sam3_image_model

class SAM3GroundingPipeline:
    """图像 Grounding 分割管线"""

    def __init__(self, checkpoint="sam3.safetensors", device="cuda", dtype=torch.bfloat16):
        self.model = build_sam3_image_model(
            checkpoint=checkpoint,
            device=device,
            dtype=dtype,
        )
        self.processor = self.model.processor

    def segment(self, image_path, text, boxes=None, confidence_threshold=0.5):
        """
        Grounding 分割

        Args:
            image_path: 图像路径
            text: 目标文本描述
            boxes: 可选框提示 [[cx,cy,w,h], ...] 归一化
            confidence_threshold: 置信度阈值（官方默认 0.5）

        Returns:
            masks: 分割掩码
            scores: 置信度分数
        """
        image = Image.open(image_path).convert("RGB")

        # 设置图像
        self.processor.set_image(image)
        state = None

        # 文本提示
        state = self.processor.set_text_prompt(text, state)

        # 框提示（可选）
        if boxes is not None:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor([True] * len(boxes))
            state = self.processor.add_geometric_prompt(
                boxes=boxes_tensor,
                labels=labels_tensor,
                text_str=text,
                state=state,
            )

        # 获取结果
        masks = state["masks"]
        scores = state["iou_scores"]

        # 置信度过滤（使用官方默认 0.5）
        if confidence_threshold > 0:
            keep = scores > confidence_threshold
            masks = masks[keep]
            scores = scores[keep]

        return masks, scores


# 使用示例
pipeline = SAM3GroundingPipeline()
masks, scores = pipeline.segment(
    "test.jpg",
    text="cat",
    boxes=[[0.5, 0.5, 0.3, 0.4]],  # 可选
)
```

### 3.2 注意事项

1. **框格式**：Grounding 路径使用 `[cx, cy, w, h]` 归一化
2. **不要用点**：Grounding 路径会静默忽略点提示
3. **confidence_threshold**：官方默认 0.5，ComfyUI 用 0.2
4. **NMS**：官方不加 NMS，如需去重请自行添加
5. **Exemplar**：官方支持示例图提示（`text="visual"`），ComfyUI 不支持

---

## 4. 图像 Interactive 路径

### 4.1 完整代码模板

```python
import torch
import numpy as np
from PIL import Image
from sam3 import build_sam3_image_model

class SAM3InteractivePipeline:
    """图像交互式分割管线"""

    def __init__(self, checkpoint="sam3.safetensors", device="cuda", dtype=torch.bfloat16):
        self.model = build_sam3_image_model(
            checkpoint=checkpoint,
            device=device,
            dtype=dtype,
            enable_inst_interactivity=True,  # ⚠️ 必须启用！
        )
        self.processor = self.model.processor
        self.predictor = self.model.inst_interactive_predictor

    def segment_single(self, image_path, points=None, point_labels=None,
                       box=None, mask_input=None, refinement=0):
        """
        单区域交互式分割

        Args:
            image_path: 图像路径
            points: 点坐标 [[x,y], ...] 像素
            point_labels: 点标签 [1, 0, ...] (1=正, 0=负)
            box: 框坐标 [x0,y0,x1,y1] 像素
            mask_input: 掩码输入 [1,1,256,256] (refinement 用)
            refinement: 迭代优化次数

        Returns:
            masks: [1, 1, H, W]
            scores: [1]
            low_res: [1, 1, 256, 256]
        """
        image = Image.open(image_path).convert("RGB")
        self.processor.set_image(image)
        state = self.processor.get_state()

        # 准备输入
        point_coords = np.array(points) if points else None
        point_labels = np.array(point_labels) if point_labels else None
        box_np = np.array(box) if box else None

        # 预测
        masks, scores, low_res = self.predictor.predict(
            state,
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_np,
            mask_input=mask_input,
            multimask_output=False,
        )

        # Refinement 迭代
        for _ in range(refinement):
            best_idx = scores.argmax()
            mask_input = low_res[best_idx:best_idx+1]
            masks, scores, low_res = self.predictor.predict(
                state,
                mask_input=mask_input,
                multimask_output=False,
            )

        return masks, scores, low_res

    def segment_multi_region(self, image_path, regions):
        """
        多区域交互式分割

        Args:
            image_path: 图像路径
            regions: 列表，每个元素为 dict:
                {
                    "points": [[x,y], ...],       # 像素坐标
                    "point_labels": [1, 0, ...],   # 正/负标签
                    "box": [x0,y0,x1,y1],          # 像素坐标 (可选)
                    "refinement": 5,                # 迭代次数
                }

        Returns:
            all_masks: [N, H, W] 所有区域掩码
            all_scores: [N] 所有区域置信度
        """
        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size
        self.processor.set_image(image)
        state = self.processor.get_state()

        all_masks = []
        all_scores = []

        for region in regions:
            # 坐标转换（如果输入是归一化的）
            points = region.get("points")
            point_labels = region.get("point_labels")
            box = region.get("box")
            refinement = region.get("refinement", 0)

            # 如果是归一化坐标，反归一化
            if region.get("normalized", False):
                if points:
                    points = [[x * img_w, y * img_h] for x, y in points]
                if box:
                    # cxcywh 归一化 → x0y0x1y1 像素
                    cx, cy, w, h = box
                    box = [
                        (cx - w/2) * img_w,
                        (cy - h/2) * img_h,
                        (cx + w/2) * img_w,
                        (cy + h/2) * img_h,
                    ]

            # 预测
            point_coords = np.array(points) if points else None
            point_labels_np = np.array(point_labels) if point_labels else None
            box_np = np.array(box) if box else None

            masks, scores, low_res = self.predictor.predict(
                state,
                point_coords=point_coords,
                point_labels=point_labels_np,
                box=box_np,
                multimask_output=False,
            )

            # Refinement
            for _ in range(refinement):
                best_idx = scores.argmax()
                mask_input = low_res[best_idx:best_idx+1]
                masks, scores, low_res = self.predictor.predict(
                    state,
                    mask_input=mask_input,
                    multimask_output=False,
                )

            all_masks.append(masks[0, 0])  # [H, W]
            all_scores.append(scores[0].item())

        return torch.stack(all_masks), torch.tensor(all_scores)


# 使用示例
pipeline = SAM3InteractivePipeline()

# 单区域
masks, scores, low_res = pipeline.segment_single(
    "test.jpg",
    points=[[500, 300], [400, 350]],
    point_labels=[1, 0],
    box=[200, 150, 600, 500],  # [x0,y0,x1,y1] 像素
    refinement=5,
)

# 多区域
masks, scores = pipeline.segment_multi_region(
    "test.jpg",
    regions=[
        {"points": [[500, 300]], "point_labels": [1], "refinement": 5},
        {"points": [[100, 200], [150, 250]], "point_labels": [1, 0], "refinement": 3},
    ],
)
```

### 4.2 注意事项

1. **必须启用 `enable_inst_interactivity=True`**
2. **框格式**：Interactive 路径使用 `[x0, y0, x1, y1]` 像素坐标
3. **normalize_coords**：`predict()` 默认 `normalize_coords=True`，会将像素坐标除以图像尺寸
4. **refinement**：将低分辨率掩码反馈为 `mask_input`，迭代优化边界

---

## 5. 视频分割路径

### 5.1 完整代码模板

```python
import torch
import numpy as np
from PIL import Image
from sam3 import build_sam3_predictor

class SAM3VideoPipeline:
    """视频分割管线"""

    def __init__(self, version="sam3", checkpoint="sam3.safetensors",
                 device="cuda", dtype=torch.bfloat16):
        self.predictor = build_sam3_predictor(
            version=version,
            checkpoint=checkpoint,
            device=device,
        )

    def segment_video(self, frames, prompt_frame_idx=0, text=None,
                      points=None, point_labels=None, boxes_xywh=None):
        """
        视频分割

        Args:
            frames: 帧列表 [PIL Image, ...] 或 tensor [T,H,W,C]
            prompt_frame_idx: 标注所在帧索引
            text: 文本描述 (可选)
            points: 点坐标 [[x,y], ...] (可选)
            point_labels: 点标签 [1, 0, ...] (可选)
            boxes_xywh: 框坐标 [[xmin,ymin,w,h], ...] 归一化 (可选)

        Returns:
            video_masks: [T, H, W] 视频掩码
        """
        # 添加提示
        request = {
            "type": "add_prompt",
            "frame_idx": prompt_frame_idx,
        }
        if text:
            request["text"] = text
        if points:
            request["points"] = points
            request["labels"] = point_labels
        if boxes_xywh:
            request["boxes_xywh"] = boxes_xywh  # ⚠️ 归一化 xywh

        result = self.predictor.handle_request(request)

        # 传播
        result = self.predictor.handle_request({
            "type": "propagate_in_video",
        })

        return result["masks"]


# 使用示例
pipeline = SAM3VideoPipeline(version="sam3")
video_masks = pipeline.segment_video(
    frames=frames,
    boxes_xywh=[[0.1, 0.2, 0.3, 0.4]],  # [xmin,ymin,w,h] 归一化
    text="person",
)
```

### 5.2 SAM 3.1 Multiplex 模式

```python
from sam3 import build_sam3_predictor

# 使用 SAM 3.1
predictor = build_sam3_predictor(version="sam3.1")

# 大量对象分割（Multiplex 模式）
# 最多 128 对象，每批 16 对象，16 批 → ~7x 加速
for request in object_requests:  # 最多 128 个请求
    result = predictor.handle_request(request)  # 或 handle_stream_request

result = predictor.handle_request({"type": "propagate_in_video"})
```

---

## 6. 高级功能

### 6.1 Exemplar 提示（官方独有）

```python
# 使用示例图作为提示
state = processor.set_text_prompt("visual", state)  # 特殊标记
state = processor.add_geometric_prompt(
    boxes=torch.tensor([[cx, cy, w, h]]),
    labels=torch.tensor([True]),
    text_str="visual",
    state=state,
    exemplar_image=example_image,  # 示例图像
)
```

> ⚠️ Exemplar 机制通过 `TEXT_ID_FOR_VISUAL=1` / `text_str="visual"` 路径触发，不是独立的提示类型。

### 6.2 掩码提示

```python
# 使用已有掩码作为提示（Interactive 路径）
masks, scores, low_res = predictor.predict(
    state,
    mask_input=prev_low_res_mask,  # [1,1,256,256]
    multimask_output=False,
)
```

### 6.3 多掩码候选

```python
# 生成 3 个候选掩码，选择最佳
masks, scores, low_res = predictor.predict(
    state,
    point_coords=np.array([[500, 300]]),
    point_labels=np.array([1]),
    multimask_output=True,  # 3 个候选
)

# 选择最佳
best_idx = scores.argmax()
best_mask = masks[0, best_idx]  # [H, W]
```

### 6.4 Transform Pipeline

```python
# 官方预处理管线
from sam3.utils.transforms import ToDtype, Resize, Normalize

transform = Compose([
    ToDtype(torch.uint8),           # 转 uint8
    Resize(1008, 1008),             # 统一缩放到 1008x1008
    ToDtype(torch.float32),         # 转 float32
    Normalize(mean=[0.5]*3, std=[0.5]*3),  # [-1, 1] 归一化
])
```

---

## 7. API 合约参考

### 7.1 build_sam3_image_model

```python
def build_sam3_image_model(
    checkpoint: str,                      # 模型路径
    device: str = "cuda",                 # 设备
    dtype: torch.dtype = torch.bfloat16,  # 精度
    enable_inst_interactivity: bool = False,  # 启用交互预测器
) -> Sam3ImageModel:
    """
    构建图像模型

    Returns:
        Sam3ImageModel:
            .processor: Sam3Processor 实例
            .inst_interactive_predictor: Sam3InteractiveImagePredictor (仅当 enable_inst_interactivity=True)
    """
```

### 7.2 build_sam3_predictor

```python
def build_sam3_predictor(
    version: str = "sam3",       # "sam3" 或 "sam3.1"
    checkpoint: str = None,      # 模型路径 (可选，自动下载)
    device: str = "cuda",        # 设备
) -> Sam3BasePredictor:
    """
    构建视频预测器

    Returns:
        Sam3BasePredictor:
            .handle_request(request: dict) → dict
            .handle_stream_request(request: dict) → dict  (仅 SAM 3.1)
    """
```

### 7.3 Sam3Processor

```python
class Sam3Processor:
    def set_image(self, image: PIL.Image) -> None:
        """设置图像（提取特征）"""

    def get_state(self) -> dict:
        """获取当前状态"""

    def set_text_prompt(self, text: str, state: dict = None) -> dict:
        """设置文本提示"""

    def add_geometric_prompt(self, boxes: torch.Tensor, labels: torch.Tensor,
                            text_str: str = None, state: dict = None) -> dict:
        """添加几何提示（框）

        Args:
            boxes: [N, 4] [cx,cy,w,h] 归一化
            labels: [N] True/False
            text_str: 可选文本关联
            state: 推理状态
        """
```

### 7.4 Sam3InteractiveImagePredictor

```python
class Sam3InteractiveImagePredictor:
    def predict(self, state: dict,
                point_coords: np.ndarray = None,   # [N, 2] 像素
                point_labels: np.ndarray = None,    # [N] 1/0
                box: np.ndarray = None,             # [4] [x0,y0,x1,y1] 像素
                mask_input: torch.Tensor = None,    # [1,1,256,256]
                multimask_output: bool = False,
                normalize_coords: bool = True) -> Tuple:
        """
        交互式预测

        Returns:
            masks: [1, num_masks, H, W]
            scores: [num_masks]
            low_res_masks: [1, num_masks, 256, 256]
        """
```

---

## 8. 性能优化指南

### 8.1 精度选择

| 精度 | VRAM | 速度 | 质量 | 推荐场景 |
|------|------|------|------|----------|
| fp32 | ~12 GB | 慢 | 最佳 | 调试/验证 |
| bf16 | ~6 GB | 快 | 几乎无损 | ✅ 通用推荐 |
| fp16 | ~6 GB | 快 | 可能有数值问题 | 不推荐 |

### 8.2 模型缓存

```python
# 避免重复加载
class ModelManager:
    _instance = None
    _model = None

    @classmethod
    def get_model(cls, checkpoint="sam3.safetensors"):
        if cls._model is None or cls._checkpoint != checkpoint:
            cls._model = build_sam3_image_model(checkpoint=checkpoint)
            cls._checkpoint = checkpoint
        return cls._model
```

### 8.3 图像预处理

```python
# 官方 Transform Pipeline 优化
# 1. 预缩放到 1008x1008 减少实时计算
# 2. 缓存图像特征（同一图像多次预测）
processor.set_image(image)  # 只调用一次
state = processor.get_state()  # 每次预测复用 state
```

### 8.4 批量处理

```python
# 多区域：共享图像特征
processor.set_image(image)
state = processor.get_state()

for region in regions:
    masks, scores, _ = predictor.predict(state, ...)
    # state 不变，图像特征被复用
```

### 8.5 视频 Multiplex

```python
# SAM 3.1: 大量对象时使用 Multiplex
# max_num_objects=16, multiplex_count=16
# 128 对象 → 8 批 → 约 7x 加速
predictor = build_sam3_predictor(version="sam3.1")
```

---

## 9. 常见问题与陷阱

### Q1: 图像特征提取很慢？

**A**: `set_image()` 只需调用一次，后续预测复用特征。不要在循环中重复调用。

### Q2: Grounding 结果太多/质量差？

**A**: 检查 `confidence_threshold` 设置。官方默认 0.5，ComfyUI 用 0.2（产生更多低质量掩码）。建议从 0.5 开始调试。

### Q3: 视频分割时框的位置偏移？

**A**: 检查框格式。视频路径使用 `[xmin, ymin, w, h]` 归一化，不是 Grounding 的 `[cx, cy, w, h]`。两者中心点计算不同。

### Q4: 点提示不生效？

**A**: 确认使用的是 Interactive 路径（`predictor.predict()`），不是 Grounding 路径。Grounding 路径会静默忽略点。

### Q5: 多区域结果不独立？

**A**: 确保每个区域使用同一个 `state`（共享图像特征），但掩码互不影响。每次 `predict()` 返回独立掩码。

### Q6: VRAM 不够？

**A**:
1. 使用 `dtype=torch.bfloat16` 半精度
2. 减小输入图像分辨率
3. 视频：减少帧数或降低分辨率
4. 使用 `torch.cuda.empty_cache()` 在不需要时清理

---

## 10. 迁移清单

从 ComfyUI-SAM3-main 迁移到独立工具时，注意以下差异：

### 必须修改

| # | 项目 | ComfyUI | 独立工具 |
|---|------|---------|----------|
| 1 | 模型加载 | `build_sam3_video_model(config)` | `build_sam3_image_model()` 或 `build_sam3_predictor()` |
| 2 | 置信度阈值 | 0.2 | 0.5（官方默认） |
| 3 | NMS | 有 (iou=0.5) | 无（按需自行添加） |
| 4 | 视频框格式 | 像素坐标 | `[xmin,ymin,w,h]` 归一化 |
| 5 | 框提示方法 | `add_multiple_box_prompts()` | `add_geometric_prompt()` |

### 建议修改

| # | 项目 | ComfyUI | 独立工具 |
|---|------|---------|----------|
| 6 | 版本选择 | 仅 sam3 | `version="sam3.1"` 可选 |
| 7 | 交互预测器 | 始终启用 | `enable_inst_interactivity=True` |
| 8 | 坐标归一化 | `normalize_coords=True` | 同（默认行为） |
| 9 | 模型缓存 | LRU(2) | 自行实现或每次新建 |
| 10 | 图像格式 | ComfyUI tensor [B,H,W,C] | PIL Image |

### 可选增强

| # | 项目 | ComfyUI | 独立工具 |
|---|------|---------|----------|
| 11 | Exemplar | ❌ | `text="visual"` + exemplar_image |
| 12 | SAM 3.1 | ❌ | `version="sam3.1"` |
| 13 | Multiplex | ❌ | `Sam3MultiplexVideoPredictor` |
| 14 | 流式处理 | ❌ | `handle_stream_request()` |
| 15 | 掩码提示 | Prompt 类支持，无节点 | `mask_input` 参数 |
