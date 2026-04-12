# SAM3 论文深度对比 — 原论文 vs 现有文档的新发现

> 完整阅读 SAM3 论文全文（含附录 A-G），与 TechAnalyze2 系列文档交叉比对，整理出所有**先前未记录**的重要发现

---

## 📋 新发现总览

| # | 严重度 | 类别 | 新发现 | 论文位置 | 影响范围 |
|---|--------|------|--------|----------|----------|
| 1 | 🔴 Critical | 任务定义 | PCS 任务仅支持 ≤30 秒短视频 | §2 | 视频分割 |
| 2 | 🔴 Critical | 约束条件 | Prompt 一致性要求：所有 prompt 必须一致定义类别，否则行为未定义 | §2 | 所有模式 |
| 3 | 🔴 Critical | 输入限制 | 文本 prompt 仅支持"简单名词短语"(NP)，不支持复杂指代表达 | §2, §B | 文本定位 |
| 4 | 🔴 Critical | 行为差异 | Exemplar 检测所有同类实例 vs PVS 仅追踪单个实例 | §3 | 示例交互 |
| 5 | 🟡 Important | 架构细节 | 语义分割头（Semantic Segmentation Head）— 第二个输出头 | §C.2 | 架构理解 |
| 6 | 🟡 Important | 架构细节 | Presence Token 的精确公式：p(query_i) = p(match\|present) × p(present) | §C.2 | 推理理解 |
| 7 | 🟡 Important | 架构细节 | 歧义处理模块：K=2 专家混合，Winner-Takes-All | §C.2 | 交互式分割 |
| 8 | 🟡 Important | 架构细节 | 4 个几何查询（geometric queries）用于 PVS 任务 | §C.2 | 架构理解 |
| 9 | 🟡 Important | 视频架构 | 视频管线正式公式：propagate → detect → match_and_update | §3 | 视频追踪 |
| 10 | 🟡 Important | 视频架构 | Masklet Detection Score (MDS) 和 Track Confirmation Delay (T=15) | §C.3 | 视频推理 |
| 11 | 🟡 Important | 视频架构 | Periodic Re-Prompting：每 N=16 帧用高置信检测重置追踪器 | §C.3 | 视频追踪 |
| 12 | 🟡 Important | 视频架构 | Detection-Guided Re-Prompting：bbox IoU<0.85 时重条件化 | §C.3 | 视频追踪 |
| 13 | 🟡 Important | 视频架构 | 重复/未确认 Masklet 移除策略 | §C.3 | 视频追踪 |
| 14 | 🟡 Important | 训练细节 | 四阶段训练流水线完整细节 | §C.4.1 | 复刻/理解 |
| 15 | 🟡 Important | 训练细节 | Hard Negatives 将 IL_MCC 从 0.44 提升到 0.68 | §A.2 | 数据策略 |
| 16 | 🟡 Important | 训练细节 | DAC-DETR + Align Loss 双重监督 | §C.2, §C.4.1 | 损失函数 |
| 17 | 🟡 Important | 训练细节 | 安全训练：随机采样不安全概念作为负样本 | §C.4.2 | 安全策略 |
| 18 | 🟡 Important | 评估指标 | cgF1 = 100 × pmF1 × IL_MCC 完整定义 | §E.3 | 评估理解 |
| 19 | 🟡 Important | 评估指标 | Oracle 评估协议：3 组独立标注取最优配对 | §E.4 | 评估理解 |
| 20 | 🟡 Important | Agent | SAM 3 Agent 四工具 + 上下文工程 | §G | Agent 设计 |
| 21 | 🟡 Important | 数据引擎 | AI Verifier (MV + EV) 细节与性能 | §D.4 | 数据引擎 |
| 22 | 🟡 Important | 数据引擎 | 领域自适应：合成数据可无人工标注提升新领域 | §A.3 | 应用策略 |
| 23 | 🟢 Info | 架构细节 | Fusion Encoder 6 层 + Decoder 6 层 | §C.2 | 架构参数 |
| 24 | 🟢 Info | 架构细节 | 200 个 Object Queries（默认） | §C.2 | 架构参数 |
| 25 | 🟢 Info | 架构细节 | Box-to-pixel relative position bias（非 deformable attention） | §C.2 | 架构选择 |
| 26 | 🟢 Info | 架构细节 | Vision Encoder: 窗口注意力 24×24，全局注意力仅 4/32 层 | §C.2 | 效率理解 |
| 27 | 🟢 Info | 架构细节 | Text Encoder: 因果，最大上下文长度 32 | §C.2 | 输入限制 |
| 28 | 🟢 Info | 架构细节 | 输入分辨率 1008×1008，不保留宽高比 | §C.4.2 | 预处理 |
| 29 | 🟢 Info | 架构细节 | SimpleFPN 提供多尺度特征 | §C.2 | 分割头 |
| 30 | 🟢 Info | 架构细节 | IoM (Intersection-over-Minimum) 用于嵌套实例检测 | §C.2 | 后处理 |
| 31 | 🟢 Info | 训练细节 | Mosaic 数据增强（最大 3×3 网格）| §C.4.2 | 数据增强 |
| 32 | 🟢 Info | 训练细节 | SA-Co Ontology: 22.4M Wikidata 节点, 17 顶级类别 | §D.2 | 数据规模 |
| 33 | 🟢 Info | 局限性 | 推理成本随追踪对象数线性增长 | §B | 部署规划 |
| 34 | 🟢 Info | 局限性 | 多对象追踪无共享上下文信息 | §B | 架构局限 |
| 35 | 🟢 Info | 局限性 | 概念/实例模式硬切换 | §B | 交互设计 |
| 36 | 🟢 Info | Agent | Agent 最多可进行 60 步试错 | §G.1 | Agent 行为 |
| 37 | 🟢 Info | 数据 | SA-Co/SYN: 39M images, 1.7B image-NPs, 1.4B masks | §E.1 | 数据规模 |

---

## 🔴 Critical 新发现详解

### 1. PCS 任务仅支持 ≤30 秒短视频

**论文原文**：
> "given an image or short video (≤30 secs)"

**现有文档状态**：未记录此限制

**影响**：
- 视频分割节点 `sam3_video_nodes.py` 需要添加视频时长校验
- 用户输入长视频时应有警告或自动分段
- ComfyUI 工作流可能需要预处理步骤来裁剪长视频

**代码对照**：
- `sam3_video_nodes.py` 的 `SAM3VideoSegmentation` 节点未检查视频时长
- `video_state.py` 无帧数/时长限制逻辑

---

### 2. Prompt 一致性要求

**论文原文**：
> "All prompts must be consistent in their category definition, or the model's behavior is undefined"

**现有文档状态**：未记录

**影响**：
- 这是 PCS 任务的根本约束 — 如果先用 "dog" 文本检测，又用 bbox 指向一只猫，行为未定义
- 混合 prompt（text + exemplar）时尤其关键：exemplar 必须与 text 描述同一概念
- ComfyUI 的 `SAM3MultiRegionCollector` 允许混合不同 prompt，但未校验一致性

**对 ComfyUI 的影响**：
```python
# sam3_interactive.py 中 MultiRegion 允许不同 prompt 类型混合
# 但未验证一致性，可能产生未定义行为
```

---

### 3. 文本 Prompt 仅支持简单名词短语

**论文原文**：
> "simple noun phrases (NPs) consisting of a noun and optional modifiers"
> "SAM 3 is constrained to simple noun phrase prompts and does not support multi-attribute queries beyond one or two attributes or longer phrases including referring expressions"

**现有文档状态**：未明确记录限制

**影响**：
- 复杂查询如 "the dog sitting on the left of the red car" 不被支持
- 指代表达（referring expressions）不被支持
- 需要配合 SAM 3 Agent（MLLM）来处理复杂查询
- 这解释了为什么 Agent 存在 — 它将复杂查询分解为简单 NP

**与现有发现的关系**：
- 我们的 `02_NODE_REFERENCE.md` 记录了 `text_prompt` 参数，但未说明 NP 限制
- 这也是 `SAM3MultipromptSegmentation` 存在的意义 — 将复杂概念拆成多个 NP

---

### 4. Exemplar 检测所有同类实例

**论文原文**：
> "given a positive bounding box on a dog, the model will detect ALL dogs in the image"

**现有文档状态**：部分记录了 Exemplar 机制，但未明确强调"检测所有实例"这一行为差异

**行为对比**：

| 模式 | 输入 | 输出行为 |
|------|------|----------|
| **PVS (Interactive)** | 点击一只狗 | 仅分割那一只狗 |
| **PCS (Exemplar)** | 框选一只狗 | 检测图中**所有狗** |
| **PCS (Text)** | "dog" | 检测图中**所有狗** |

**关键洞察**：Exemplar 的行为等同于文本 prompt — 它是"概念级"而非"实例级"的。这与 PVS 的实例级行为根本不同。

**对 ComfyUI 的影响**：
- `SAM3BBoxSegmentation` 节点中 bbox prompt 的行为取决于模式（PCS vs PVS）
- 用户可能期望 bbox 只分割框内对象（PVS 行为），但实际可能触发全图检测（PCS 行为）

---

## 🟡 Important 新发现详解

### 5. 语义分割头

**论文原文**：
> "we also have a semantic segmentation head, which predicts a binary label for every pixel in the image"
> "Semantic segmentation and instance segmentation share the same segmentation head"

**架构细节**：
- 与实例分割共享分割头
- 语义分割使用 fusion encoder 的 conditioned features
- 实例分割额外使用 decoder 的 object queries
- SimpleFPN 提供多尺度特征（因为 ViT 是单尺度）
- Presence score 被复用于语义分割

**代码对照**：
- ComfyUI `sam3/model.py` 中未发现独立的语义分割头
- 官方 `sam3_det_model.py` 中可能包含但被简化

---

### 6. Presence Token 精确公式

**论文原文**：
> p(query_i matches NP) = p(query_i matches NP | NP appears in image) × p(NP appears in image)

**关键细节**：
- Presence token 与 object queries 一起参与 decoder 所有操作
- 但 **被排除在 DAC (Divide-And-Conquer) 之外**
- 推理时使用乘积作为总分数
- 监督策略：仅当概念存在时计算 object query 分类损失（Setting (a) 最佳）
- 计数任务中可设 p(present) = 1，跳过存在性判断

**代码对照**：
- ComfyUI 的 `sam3/model.py` 中 `conv_s0`/`conv_s1` 适配层可能与此相关
- 官方代码中 presence token 应在 `sam3_det_model.py` 中实现

---

### 7. 歧义处理模块 (Ambiguity Head)

**论文原文**：
> "we add an ambiguity head to our model... a mixture of experts, where we train in parallel K experts, and only supervise the expert that gets the lowest loss (winner-takes-all)"

**关键细节**：
- **K=2 专家**（K>3 导致模式坍塌）
- **Winner-Takes-All**：仅最低 loss 的专家接收梯度
- 混合损失公式：L_WTA = L_{k*}，其中 k* = argmin_k L_k
- 训练分类头预测应使用哪个专家
- 歧义头**仅调整分类 logits**，不改变 masks、boxes、presence scores
- 在冻结的 SAM3 模型之上单独训练
- IoM (Intersection-over-Minimum) 用于检测重叠实例，比 IoU 更适合嵌套场景
- **效果**：15% 减少重叠实例

**示例**：
- "large circular shape" → Expert 1 检测大型圆形物体，Expert 2 检测圆形区域
- 这与 SAM1/SAM2 的多 mask 输出类似，但更系统地处理

**对现有文档的影响**：
- `04_SOURCE_DEEP_DIVE.md` 记录了 `predict_inst()` 的 4-mask 输出，但未解释歧义处理原理
- 现在我们知道这是 paper 的 ambiguity head 机制

---

### 8. 4 个几何查询 (Geometric Queries)

**论文原文**：
> "We also learn 4 geometric queries. Their function is similar to the 4 geometric queries in SAM 1 and 2 (where they were called 'output tokens') and are used to perform the PVS on individual image or video frames"

**关键细节**：
- 仅用于 PVS 任务（交互式分割），不用于 PCS
- 在 Stage 2/3 训练时激活
- Presence score 在 PVS 模式下设为 1（因为目标已知存在）
- 对应 SAM1/SAM2 的 4 个 output tokens

**代码对照**：
- ComfyUI `sam3/model.py` 中可能有这 4 个查询的实现
- 官方代码中应在 `sam3_det_model.py` 中

---

### 9. 视频管线正式公式

**论文原文**：
```
M̂t = propagate(Mt-1)     # 追踪器传播上一帧掩码
Ot = detect(It, P)         # 检测器在当前帧检测
Mt = match_and_update(M̂t, Ot)  # 匹配并更新
```

**现有文档状态**：`04_SOURCE_DEEP_DIVE.md` 描述了 `handle_request` 分发逻辑，但未给出正式数学公式

**关键洞察**：
- **双路径并行**：追踪器（传播）和检测器（检测）同时运行
- **匹配融合**：通过 IoU 阈值匹配检测和追踪结果
- 这解释了为什么视频推理成本随对象数线性增长（每对象独立追踪）

---

### 10. Masklet Detection Score (MDS) 与 Track Confirmation Delay

**论文原文**：

**MDS 定义**：
```
∆i(τ) = +1 if ∃d ∈ Dτ s.t. IoU(d, M̂iτ) > iou_threshold
         -1 otherwise

Si(t, t') = Σ_{τ=t}^{t'} ∆i(τ)   # 在时间窗口内的累积匹配分数
```

**Track Confirmation Delay (T=15)**：
- 输出延迟 T=15 帧（约 0.5 秒 @30fps）
- 在确认窗口内验证候选 masklet
- 未确认的 masklet（MDS < 0）被移除
- 确认阈值 V=0：masklet 必须在至少一半帧中被检测匹配

**代码对照**：
- `video_state.py` 中可能包含 MDS 计算
- `sam3_video_predictor.py` 中应有确认延迟逻辑

---

### 11. Periodic Re-Prompting

**论文原文**：
> "on every N-th frame τ, we compare each detection d ∈ Dτ with the tracker's current predictions M̂τ. If IoU ≥ 0.8 and both detection score and masklet score > 0.8, we re-initialize the tracker for that object"

**关键参数**：
- N = 16（每 16 帧重提示一次）
- IoU 阈值 = 0.8
- 置信度阈值 = 0.8
- 仅在对象未被遮挡且完全可见时最有效

**目的**：解决追踪器在遮挡或相似干扰物场景下的漂移问题

---

### 12. Detection-Guided Re-Prompting

**论文原文**：
> "If the highest-matching detection d has a low bounding box IoU (i.e., IoUbbox(d, M̂iτ) < 0.85) with the corresponding tracker prediction, we recondition the tracker for that object using the latest detector output"

**关键参数**：
- bbox IoU 阈值 = 0.85
- 每帧执行（非周期性）
- 解决追踪器预测漂移（leaky masks）

**与 Periodic Re-Prompting 的区别**：
- Periodic: 周期性（N=16），用高置信检测重置
- Detection-Guided: 每帧，当检测与追踪不一致时重条件化

---

### 13. 重复/未确认 Masklet 移除

**论文原文**：

**未确认移除**：
- 条件：Si(t, t+T) < V (=0) 且 t_i^first ≥ t
- 效果：移除假阳性检测

**重复移除**：
- 条件：两个 masklet 与同一检测 IoU ≥ 0.1 超过 ⌈T/2⌉ 帧
- 效果：移除后出现的 masklet

**Masklet Suppression**：
- 条件：Si(t_i^first, τ) < 0 在任意帧 τ
- 效果：置零 mask 但保留 tracker 状态（可能后续恢复）
- 主要处理边界进入场景的歧义对象

---

### 14. 四阶段训练流水线

| Stage | 目标 | 数据 | 关键设置 |
|-------|------|------|----------|
| **1** | PE 预训练 | 5.4B 图文对 | Vision 450M + Text 300M 参数 |
| **2** | 检测器预训练 | SA-Co/SYN + SA-Co/EXT + SA-1B 等 | 95k iter, bs=896, NP→visual query p=0.2, +bbox aug p=0.2, PVS 4步交互 |
| **3** | 高质量微调 + 交互性 | SA-Co/HQ（仅最高质量）| 5k iter, bs=512, 引入 presence token, PCS 5轮交互, PVS 7步交互, lr×0.025 |
| **4** | 视频追踪 | SA-Co/VIDEO + SA-V 等 | 190k iter, bs=512, backbone frozen, cosine schedule, peak lr=5e-4, 后续 16/32帧微调 60k iter |

**Stage 2 损失权重**：
- Box: L1 × 5 + GIoU × 2
- Classification: × 100
- Focal + Dice: × 200 + × 10
- Dropout: 0.1

**Stage 4 损失权重**：
- Focal + Dice for mask: 20:1
- MAE for IoU: 1
- CE for occlusion: 1

**数据构成（Tab 18）**：
- SA-Co/HQ 在 Stage 2,3 使用
- SA-Co/SYN 仅在 Stage 2 使用（质量不够高）
- SA-Co/EXT 在 Stage 2,3 使用
- SA-Co/VIDEO 在 Stage 4 使用

---

### 15. Hard Negatives 的巨大影响

**论文数据**：
- 无 hard negatives: IL_MCC = 0.44
- 有 hard negatives: IL_MCC = 0.68
- **提升 54%**

**Hard Negatives 定义**：
- 图中不存在、但 SAM3 之前版本会预测出 mask 的名词短语
- 即对当前模型的对抗性负样本

**生成策略**：
1. SA-Co Ontology 导航：兄弟/堂兄弟/叔伯节点（如 "Siamese cat" → "tabby cat" 兄弟, "dog" 叔伯, "Chihuahua" 堂兄弟）
2. MLLM (Llama 4) 提出视觉相似负样本
3. 对抗性检查：SAM3 预测非空且与正标注重叠则保留

**SA-Co/HQ 中 88.5% 为负样本**，hard negatives 是关键质量因素

---

### 16. DAC-DETR + Align Loss 双重监督

**DAC-DETR**：
- Divide-And-Conquer DETR（Hu et al., 2023）
- Presence token 被排除在 DAC 之外
- 使用 iterative box refinement + look-forward-twice + hybrid matching

**Align Loss**：
- 替代 SAM2 的 IoU prediction loss
- 共用 classification head 作为 object queries
- 在 Stage 2 的 PVS 任务中使用

---

### 17. 安全训练

**论文原文**：
> "To prevent the model from randomly making predictions for unsafe concepts, we randomly sample some of them at train time and add them as negatives. These concepts mainly include slurs of all kinds."

**还阻止**：
- 主观/非视觉形容词用于人（包括褒义如 "a smart person" 和贬义如 "a dull person"）

**数据增强中的语义增强**：
- Wikidata 映射：同义词扩展、负样本采样、层级闭包保证
- 例如："canoe" 和 "boat" 共存时，所有 canoe 也必须标注为 boat

---

### 18. cgF1 评估指标完整定义

**公式**：
```
cgF1 = 100 × pmF1 × IL_MCC
```

其中：

**pmF1 (positive micro F1)**：
- 仅在正样本（NP 确实存在）上计算
- 多 IoU 阈值平均 (0.5 到 0.95, 步长 0.05, 共 10 个)
- 使用最优二部匹配计算 TP/FP/FN

**IL_MCC (Image-Level MCC)**：
- 二元分类指标：NP 是否存在于图中
- 不关心 mask 质量，只看"是否预测了任何 mask"
- 置信度阈值 0.5
- MCC 优于 F1 的原因：对正负样本不平衡更鲁棒

**为什么不用 AP**：
- 开放词汇下类别数万级，AP 主导噪声
- 计算完整 precision-recall 曲线不可行
- AP 不考虑模型校准

---

### 19. Oracle 评估协议

**SA-Co/Gold 有 3 组独立人工标注**：
- Oracle: 取最优配对 → 人类 cgF1 = 76.2
- Random Pair: 随机配对 → 人类 cgF1 = 55.5
- 差距 20.7 点说明 PCS 任务**内在歧义性很高**

**视频评估**：
- SA-Co/VEval 默认 1 组标注
- 额外收集 2 组在 YT-Temporal-1B 和 SmartGlasses 上
- 视频同样存在显著 Oracle-Random 差距

---

### 20. SAM 3 Agent 设计

**四工具架构**：

| 工具 | 类型 | 功能 |
|------|------|------|
| `segment_phrase` | 中间工具 | 用简单 NP 调用 SAM3，生成 mask 并渲染 |
| `examine_each_mask` | 中间工具 | 逐个检查 mask，过滤错误 |
| `select_masks_and_return` | 返回工具 | 选择最终 mask 子集并返回 |
| `report_no_mask` | 返回工具 | 报告无匹配对象 |

**上下文工程**：
- `segment_phrase` 每次调用会**删除所有之前生成的 mask**
- Set-of-Marks 渲染：mask 编号 1-N（按置信度降序）
- 激进的上下文裁剪：仅保留初始查询 + 最近 segment_phrase 调用
- 维护已使用 NP prompt 列表避免重复

**支持的 MLLM 后端**：
- Qwen2.5-VL 7B/72B
- Qwen3-VL 8B/235B Thinking
- Llama4 Maverick
- Gemini2.5 Pro

**性能**：
- ReasonSeg 上 zero-shot gIoU 77.0（Gemini2.5 Pro）
- RefCOCO+ 和 RefCOCOg 超越之前 zero-shot SOTA
- 最多可达 60 步试错

---

### 21. AI Verifier 详情

**Mask Verification (MV)**：
- 输入：(image, phrase, mask) 三元组
- 输出：5 类判断 — Accept / Accept as text / Flag label / Whole image / Reject
- 基于微调 Llama 3.2
- AI 准确率 77.7% vs 人类 75.0%（AI 略优）

**Exhaustivity Verification (EV)**：
- 输入：(image, phrase, masks) 三元组
- 输出：6 类判断 — Accept / Reject / Reject but unseparated / Flag label / Whole image / Ungroundable
- Presence score 定义：1 - Prob(Accept | no boxes as input)
- AI 准确率 81.1% vs 人类 81.8%（基本持平）

**训练数据**：
- 预训练：200M+ image-text pairs
- 微调：~10M 高质量人类标注

**端到端效果**：
- SAM3 alone: cgF1 = 54.0
- SAM3 + EV: cgF1 = 61.2 (+7.2)
- SAM3 + EV + MV: cgF1 = 62.3 (+1.1)
- 关闭近一半与人类的差距

---

### 22. 领域自适应

**核心发现**：合成数据（SA-Co/SYN）可在**无人工标注**的情况下提升新领域性能

**三层数据质量**：
1. **PL-Food**：SAM3 伪标注（最低质量）
2. **SA-Co/SYN-Food**：PL + AI verifier 清洗（中等质量，零人工）
3. **SA-Co/HQ-Food**：PL + 人类验证（最高质量）

**关键结论**：
- SYN-Food 和 HQ-Food 扩展行为相似，SYN-Food 最终追上
- **必须混合高质量通用域数据**进行微调（1:1 比例最佳）
- 不混合通用数据时，SYN-Food 与 HQ-Food 差距显著

---

## 🟢 Info 新发现详解

### 23-29. 架构参数

| 参数 | 值 | 说明 |
|------|-----|------|
| Fusion Encoder | 6 层 transformer | Self-attention + cross-attention to prompt tokens + MLP |
| Decoder | 6 层 transformer | Object queries self-attend + cross-attend to prompts + conditioned features |
| Object Queries | Q=200 (默认) | DETR 式学习查询 |
| Position Bias | Box-to-pixel relative | 非 deformable attention，用相对位置偏置 |
| Vision Encoder 窗口 | 24×24 tokens | 全局注意力仅 4/32 层 |
| Text Encoder 上下文 | 最大 32 tokens | 因果 transformer |
| 输入分辨率 | 1008×1008 | 不保留宽高比，训练时随机 padding 分布 |
| 下采样 | 2×2 → 1296 tokens | Vision encoder 输出后降采样 |
| SimpleFPN | 多尺度特征 | 因为 ViT 是单尺度 |

### 30. IoM vs IoU

**Intersection-over-Minimum** 用于：
1. 嵌套实例检测（歧义头中，15% 减少重叠）
2. 计数后处理（NMS 中替代 IoU，阈值 0.5）
3. 检测 "whole vs part" 情况

**为什么 IoM 优于 IoU**：
- 小 mask 完全被大 mask 覆盖时，IoU 可能很低（分母是并集）
- 但 IoM 会很高（分母是较小 mask 面积），正确识别嵌套关系

### 31. Mosaic 数据增强

- 最大 3×3 网格，允许不规则配置
- 仅用于低遗漏标注风险的数据集
- 开放词汇设置下的 mosaic 需要特殊处理：只 mosaic 有穷尽标注的数据

### 32. SA-Co Ontology

- 基于 Wikidata 构建
- 22.4M 节点
- 17 顶级类别（动物、建筑、人类、食物...）
- 72 子类别
- NP → Ontology 节点映射：Sentence-BERT 检索 + Llama 3.2 判决

### 33-35. 局限性

**推理成本线性增长**：
- 每对象独立追踪，成本 = O(对象数)
- 实时 30FPS 需要：2×H200 支持 10 对象，4×H200 支持 28 对象，8×H200 支持 64 对象

**多对象无共享上下文**：
- 当前架构中各对象追踪独立
- 未来可通过共享全局记忆改善

**概念/实例模式硬切换**：
- 要修改单个实例不影响其他同概念实例，需强制从 concept 模式切到 instance 模式
- 未来可能更无缝地交织两种 prompt

---

## 🔬 与现有文档的关键差异汇总

### 02_NODE_REFERENCE.md 补充

| 节点 | 现有描述 | 需要补充 |
|------|----------|----------|
| `SAM3TextGrounding` | text_prompt 参数 | **NP 限制**：仅支持简单名词短语，不支持指代表达 |
| `SAM3BBoxSegmentation` | bbox prompt | **行为差异**：PCS 模式下 bbox 触发全图同类检测 |
| `SAM3VideoSegmentation` | 视频追踪 | **时长限制**：≤30 秒；**Confirmation Delay**：T=15 帧 |

### 03_DATA_FORMATS.md 补充

| 格式 | 现有描述 | 需要补充 |
|------|----------|----------|
| 置信度阈值 | 0.2 vs 0.5 差异 | **论文评估用 0.5**；presence score × object score = 总分 |
| IoU 阈值 | NMS IoU | **视频用 IoM** (Intersection-over-Minimum) 替代 IoU |

### 04_SOURCE_DEEP_DIVE.md 补充

| 模块 | 现有描述 | 需要补充 |
|------|----------|----------|
| `predict_inst()` 4-mask 输出 | 已记录 | **歧义头机制**：K=2 专家 WTA，仅调分类 logits |
| `handle_request` 分发 | 已记录 | **正式公式**：M̂t=propagate, Ot=detect, Mt=match_and_update |
| `conv_s0`/`conv_s1` 适配层 | 已记录 | **可能关联** presence token 或 geometric queries |
| 视频追踪 | 基本记录 | **5 项时间消歧策略**：Confirmation Delay, Unconfirmed Removal, Duplicate Removal, Masklet Suppression, Re-Prompting |

### 05_REPLICATION_GUIDE.md 补充

| 方面 | 现有描述 | 需要补充 |
|------|----------|----------|
| 训练策略 | 未涉及 | **四阶段训练**详细参数 |
| 数据策略 | 基本记录 | **Hard Negatives 至关重要**：IL_MCC +54% |
| 后处理 | NMS 记录 | **IoM-NMS** 用于计数和嵌套场景 |
| 安全 | 未涉及 | **安全训练**：不安全概念作为负样本 |

---

## 📊 论文关键数据点

### 模型规模
- 总参数：~850M
- Vision Encoder：~450M (PE-L+)
- Text Encoder：~300M
- Detector + Tracker：~100M

### SA-Co 数据集规模

| 子集 | 图像 | 图像-NP 对 | NP 数 | 掩码数 | 负样本比例 |
|------|------|-----------|-------|--------|-----------|
| SA-Co/HQ | 5.2M | 146.1M | 4.0M | 52.3M | 88.5% |
| SA-Co/SYN | 39.4M | 1.7B | 38.0M | 1.4B | 74.0% |
| SA-Co/EXT | 9.3M | 136.6M | 497.4K | 70.5M | 71.8% |
| SA-Co/VIDEO | 52.5K 视频 | 134.3K | 24.8K | 467.1K masklets | 26.7% |

### 评估基准

| 基准 | NP 数 | 图像 | 3×标注 | 零样本 NP |
|------|-------|------|--------|-----------|
| SA-Co/Gold | 51.8K | 15.8K | ✓ | 6.98% |
| SA-Co/Silver | 54.6K | 66.1K | ✗ | 8.00% |
| SA-Co/Bronze | 105.3K | 32.5K | ✗ | 57.25% |
| SA-Co/VEval | 5.2K | 1.7K 视频 | 部分有 | 6.37% |

### 性能对比

| 模型 | SA-Co/Gold cgF1 | 人类 Oracle |
|------|------------------|-------------|
| OWLv2* | 24.6 | - |
| DINO-X | 21.3 | - |
| SAM3 | **54.1** | 72.8 |
| SAM3 + EV + MV | **62.3** | 72.8 |

---

## 🎯 对复刻/应用的行动建议

### 立即需要做的（Critical）

1. **视频时长校验**：在 `sam3_video_nodes.py` 中添加 ≤30 秒 / ≤900 帧 @30fps 检查
2. **Prompt 一致性提示**：在混合 prompt 节点中添加一致性校验或警告
3. **文本 prompt 限制文档**：明确 NP 限制，引导用户使用 Agent 处理复杂查询

### 短期应该做的（Important）

4. **实现 IoM-NMS**：替代 IoU-NMS 用于嵌套实例和计数场景
5. **视频消歧策略**：在视频分割中实现 MDS + Confirmation Delay
6. **Presence Score 暴露**：将 presence score 作为可访问输出，允许用户做计数等任务
7. **语义分割模式**：探索是否可激活语义分割头

### 长期值得做的（Info）

8. **Hard Negatives 数据管道**：构建对抗性负样本生成流程
9. **领域自适应流程**：基于 AI Verifier 的合成数据生成
10. **SAM 3 Agent 集成**：将 MLLM + SAM3 Agent 架构作为高级模式
11. **安全过滤**：添加不安全概念过滤

---

*文档生成时间：基于 SAM3 论文完整阅读（含附录 A-G），与 TechAnalyze2 系列 7 篇文档交叉对比*
