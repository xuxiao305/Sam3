# SAM3 Segment 开发文档

## 概述

SAM3 Segment 是一个独立的 PyQt6 应用，实现了 SAM3（Segment Anything Model 3）的交互式和基于文本的图像分割功能。本文档记录了核心接入经验和接口约定，适用于后续集成或复用此项目。

---

## 1. 环境配置

### 1.1 Python 环境要求

**强制要求**：必须使用 `.\.venv\python.exe`，而非系统 Python。

```bash
# ❌ 错误
python sam3_app/main.py

# ✅ 正确
.\.venv\python.exe sam3_app/main.py
```

**原因**：
- `.venv` 是指向 ComfyUI 的 `python_embeded` 的符号链接
- 系统 Python 3.9.x 缺少 PyTorch 2.8.0, CUDA 12.8 支持
- `Activate.ps1` 不存在（venv 是符号链接），需要直接调用可执行文件

### 1.2 必需的环境变量

```python
# 如果 ComfyUI-SAM3 源不在标准位置，设置
export SAM3_SOURCE_PATH="path/to/ComfyUI-SAM3-main"
```

### 1.3 模型路径

SAM3 模型文件位置：
```
d:\AI\ComfyUI-Easy-Install\ComfyUI\models\sam3\sam3.safetensors
```

hardcode 在 `backend.py` 中，如需更改请修改：
```python
# sam3_app/backend.py line ~80
model_path = os.path.join(
    os.environ.get("COMFYUI_ROOT", r"d:\AI\ComfyUI-Easy-Install\ComfyUI"),
    "models", "sam3", "sam3.safetensors"
)
```

---

## 2. 核心踩坑总结

### 2.1 ConvTranspose2d 类型桩问题 ⚠️ 关键

**问题**：`gelu(): argument 'input' must be Tensor, not NoneType`

**根本原因**：
```python
# PyTorch 源码差异
nn.Conv2d._conv_forward      # 有实现（返回实际结果）
nn.ConvTranspose2d._conv_forward  # 类型桩（返回 ...）
```

SAM3 的 decoder 中 FPN_Neck 使用了 ConvTranspose2d。我们的 `ManualCastConvTranspose2d.forward()` 调用 `self._conv_forward()` 得到 `Ellipsis`（Python 中 `...` 的值），后续 GELU 接收到 None 崩溃。

**解决方案** (`comfy_shim.py`):
```python
class ManualCastConvTranspose2d(nn.ConvTranspose2d):
    def forward(self, x, output_size=None):
        # 直接实现 forward，镜像真实的 nn.ConvTranspose2d.forward()
        # 而非调用 _conv_forward（类型桩）
        weight = cast_to_input(self.weight, x)
        bias = cast_to_input(self.bias, x) if self.bias is not None else None
        
        output_padding = self._output_padding(
            output_size, x, self.stride, self.padding, self.kernel_size,
            self.groups, self.dilation,
        )
        
        return F.conv_transpose2d(
            x, weight, bias,
            self.stride, self.padding,
            output_padding, self.groups, self.dilation,
        )
```

**避免方案**：不要假设 PyTorch 源码类中所有辅助方法都有真实实现。

---

### 2.2 交叉注意力中 Mask 维度不匹配 ⚠️ 关键

**问题**：`RuntimeError: output with shape [16, 32, 32] doesn't match the broadcast shape [1, 16, 32, 32]`

**根本原因**：SAM3 文本编码器中 `SplitMultiheadAttention._prepare_mask()` 返回 4D mask `[1, B*H, L, L]`，但经过 `skip_reshape=True` 后 q/k/v 被扁平化为 `[B*H, L, D]`，传给 `F.scaled_dot_product_attention` 时批次维度不匹配。

**解决方案** (`comfy_shim.py`):
```python
def attention_pytorch(q, k, v, heads, mask=None, ...):
    # ... reshape logic ...
    
    # 预处理 mask 维度
    sdpa_mask = mask
    if sdpa_mask is not None:
        if sdpa_mask.dim() == 4:
            # [1, B*H, L, L] -> [B*H, L, L]
            if sdpa_mask.shape[0] == 1:
                sdpa_mask = sdpa_mask.squeeze(0)
            # [B*H, 1, L, L] -> [B*H, L, L]
            elif sdpa_mask.shape[1] == 1:
                sdpa_mask = sdpa_mask.squeeze(1)
        elif sdpa_mask.dim() == 3 and sdpa_mask.shape[0] == 1:
            # [1, L, L] -> [L, L]
            sdpa_mask = sdpa_mask.squeeze(0)
    
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=sdpa_mask, scale=scale)
```

**通用原则**：处理 mask 时需要确保其与 q/k/v 的批次维度兼容。

---

### 2.3 Meta 设备参数问题 ⚠️ 中等

**问题**：模型加载后某些权重留在 meta device，导致后续计算失败

**根本原因**：`load_state_dict(assign=True)` 后某些参数仍是 meta 张量（只有形状无数据）

**解决方案** (`backend.py`):
```python
def load_model(self):
    # ... load checkpoint ...
    
    # Meta device 参数检查和修复
    for name, param in self.model.model.named_parameters():
        if param.device.type == 'meta':
            log.warning(f"Meta device param: {name}, shape={param.shape}")
            # 替换为 CPU 上的 zero-filled 张量
            parent, attr = name.rsplit(".", 1)
            setattr(eval(f"self.model.model.{parent}"), attr, 
                   torch.zeros_like(param, device=self.device, dtype=param.dtype))
```

**预防方案**：始终验证临界权重已正确加载到目标设备。

---

### 2.4 导出掩码全黑问题 🎨 UI/Export

**问题**：导出的 PNG 掩码全黑，无法看清

**根本原因**：使用了索引值 1, 2, 3... 作为像素值，在 0-255 灰度范围内几乎不可见

**解决方案** (`export.py`):
```python
def export_mask_png(masks, output_path, img_w, img_h):
    combined = np.zeros((img_h, img_w), dtype=np.uint8)
    n = len(masks)
    for i, mask in enumerate(masks):
        # ... squeeze and resize ...
        # ✅ 使用可见的灰度值
        if n == 1:
            combined[mask_resized] = 255  # 单掩码 -> 白色
        else:
            # 多掩码 -> 均匀分布 (e.g., 2个 -> 127, 255)
            combined[mask_resized] = int(255 * (i + 1) / n)
    Image.fromarray(combined, mode='L').save(output_path)
```

---

### 2.5 Mask 维度不一致问题 🔧 数据处理

**问题**：`segment_text()` 返回的 mask 可能是 3D `(1, H, W)` 或 `(H, W, 1)`，导致 cv2.resize 或索引操作崩溃

**解决方案**：所有 mask 处理前必须 squeeze 到 2D
```python
# Universally applied in app.py, export.py, backend.py
if mask.ndim > 2:
    mask = mask.squeeze()
```

**通用原则**：SAM3 不同推理路径（交互 vs 文本）返回的 mask 格式可能不同，需要统一处理。

---

## 3. 核心接口

### 3.1 Backend 接口

```python
from sam3_app.backend import SAM3Backend

backend = SAM3Backend()

# 加载模型（~1750MB）
backend.load_model()

# 设置图像并提取特征
from PIL import Image
img = Image.open("test.jpg")
backend.set_image(img)

# 交互式分割（点/框提示）
prompts = [
    {
        'id': 1,
        'positive_points': {'points': [[0.5, 0.5]], 'labels': [1]},  # 归一化坐标
        'negative_points': {'points': [], 'labels': []},
        'positive_boxes': {'boxes': [], 'labels': []},
        'negative_boxes': {'boxes': [], 'labels': []},
    }
]
masks, scores, vis_image = backend.segment_interactive(
    multi_prompts=prompts,
    img_w=512, img_h=512,
    refinement_iterations=0,
    use_multimask=True,
)

# 文本分割
masks, scores, boxes, vis_image = backend.segment_text(
    text_prompt="red square",
    img_w=512, img_h=512,
    confidence_threshold=0.1,
)
```

### 3.2 返回格式

**交互式分割**：
```python
# masks: List[np.ndarray] - 每个 (H, W) boolean 或 float32 [0-1]
# scores: List[float] - 置信度 [0-1]
# vis_image: np.ndarray - (H, W, 3) RGB uint8 可视化
```

**文本分割**：
```python
# masks: List[np.ndarray] - 同上
# scores: List[float] - 同上
# boxes: List[...] - 检测框（可选）
# vis_image: np.ndarray - 同上
```

### 3.3 坐标系统

**重要**：所有坐标为 **归一化坐标 [0, 1]**（不是像素绝对坐标）

```python
# 将像素坐标转为归一化坐标
norm_x = pixel_x / img_width
norm_y = pixel_y / img_height

# 在 backend 内部会转换回像素坐标
pixel_x = norm_x * img_width
```

---

## 4. ComfyUI Shim 接口

当其他项目需要独立运行 SAM3 时（不在 ComfyUI 环境中），必须先安装 shim：

```python
import sys
sys.path.insert(0, "/path/to/SAM3_Segment")

from sam3_app.comfy_shim import install_shims
install_shims()

# 之后才能 import SAM3 源码
from sam3 import ...
```

### 4.1 Shim 提供的接口

| 模块 | 替代物 | 用途 |
|------|-------|------|
| `comfy.ops.manual_cast` | `_ManualCastOps` | 自动精度转换（bf16 ↔ fp32） |
| `comfy.model_management` | 自定义实现 | 显存管理、设备处理 |
| `comfy.utils` | 自定义实现 | 通用工具函数 |
| `comfy.ldm.modules.attention` | `optimized_attention_for_device()` | 注意力机制（SDPA） |
| `comfy.model_patcher` | `ModelPatcher` | 模型权重管理 |

### 4.2 关键实现细节

**Manual Cast 机制**：
```python
# 自动将权重转为输入张量的 dtype/device
def cast_to_input(weight, input_tensor):
    if weight is None:
        return None
    if weight.device.type == 'meta':
        # Meta 参数直接初始化
        weight = torch.nn.init.xavier_uniform_(torch.zeros_like(weight, device=input_tensor.device))
    return weight.to(dtype=input_tensor.dtype, device=input_tensor.device)
```

---

## 5. GUI 应用指南

### 5.1 启动

```bash
.\.venv\python.exe sam3_app/main.py
```

### 5.2 工作流

1. **上传图像** → 自动提取 backbone 特征
2. **选择模式**：
   - 📍 点/框模式：在画布标记提示点或框
   - 📝 文本模式：输入文本描述
   - 🎬 视频模式：（未实现）
3. **点击"开始切割"** → 后台推理
4. **导出结果**：
   - 💾 导出掩码（PNG）
   - 📋 导出提示（JSON）
   - 🖼️ 导出可视化

### 5.3 UI 坐标系

Canvas 中的鼠标坐标自动转换为归一化坐标后发送给 backend。

---

## 6. 最佳实践

### 6.1 集成到其他项目

```python
# 推荐方式
import sys
sys.path.insert(0, "/path/to/SAM3_Segment/sam3_app")

# Step 1: 安装 shim
from comfy_shim import install_shims
install_shims()

# Step 2: 添加 SAM3 源
sam3_path = "/path/to/ComfyUI-SAM3/nodes"
if sam3_path not in sys.path:
    sys.path.insert(0, sam3_path)

# Step 3: 使用 backend
from backend import SAM3Backend
backend = SAM3Backend()
backend.load_model()
```

### 6.2 内存管理

```python
# 模型会自动管理显存
# 手动卸载（可选）
if backend.model is not None:
    del backend.model
    backend.model = None
```

### 6.3 错误处理

```python
try:
    masks, scores, vis = backend.segment_interactive(...)
except RuntimeError as e:
    if "CUDA" in str(e):
        print("GPU 内存不足，请减少图像尺寸或清空其他 GPU 应用")
    else:
        raise
```

---

## 7. 性能特性

| 操作 | 耗时 | 显存 |
|------|-----|-----|
| 模型加载 | ~8s | 1.7GB |
| 特征提取（512×512） | ~1s | +100MB |
| 单次交互推理 | ~0.2-0.5s | +200MB |
| 文本编码 + 推理 | ~1-2s | +300MB |

**优化建议**：
- 预加载模型以加快首次推理
- 对大图像使用 resize（512 以上性能下降）
- 使用 bf16 精度（已默认启用）

---

## 8. 已知限制

1. **视频分割**未实现
2. **Batch 处理**当前只支持单图像
3. **真实图像**上性能优于合成图像（对自然特征敏感）
4. **文本支持**仅限英文（基于 DINO 编码器）

---

## 9. 故障排查

| 症状 | 原因 | 解决 |
|------|------|------|
| `ModuleNotFoundError: comfy` | shim 未安装 | 调用 `install_shims()` |
| CUDA OOM | 显存不足 | 减少图像尺寸或清空其他应用 |
| 掩码全黑 | 导出 bug 已修复 | 更新到最新版本 |
| 文字模式 crash | mask 维度问题 已修复 | 更新到最新版本 |
| 按钮灰色 | 状态管理 bug 已修复 | 更新到最新版本 |

---

## 10. 版本历史

### v1.0 (Apr 13, 2026)

**新功能**：
- ✅ 交互式分割（点/框提示）
- ✅ 文本基础定位
- ✅ PyQt6 GUI 应用
- ✅ 结果导出（PNG/JSON）

**关键修复**：
- ✅ ConvTranspose2d 类型桩问题
- ✅ SDPA mask 维度不匹配
- ✅ 导出掩码可见化
- ✅ 按钮状态管理
- ✅ 异常错误处理

**已知问题**：
- ⏳ 视频分割未实现
- ⏳ 仅支持英文文本提示

---

## 11. 联系方式

项目地址：https://github.com/xuxiao305/Sam3.git

有问题或建议请提交 Issue 或 Pull Request。
