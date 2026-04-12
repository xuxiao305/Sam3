"""
SAM3 Segment Tool - Backend

Wraps SAM3 model loading and inference, adapting ComfyUI-SAM3's
_model_cache + Sam3Processor for standalone use without ComfyUI runtime.

Uses the same model files and inference code as the ComfyUI plugin,
but bypasses ComfyUI's ModelPatcher / model_management system.
"""

import gc
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image

log = logging.getLogger("sam3_app")

# ─── Path Setup ──────────────────────────────────────────────────────────────
# ComfyUI-SAM3 source is accessible via the workspace symlink
_COMFYUI_SAM3_DIR = Path(r"d:\AI\ComfyUI-Easy-Install\ComfyUI\custom_nodes\ComfyUI-SAM3-main")
_SAM3_NODES_DIR = _COMFYUI_SAM3_DIR / "nodes"

# Model checkpoint location (same as ComfyUI)
_MODEL_DIR = Path(r"d:\AI\ComfyUI-Easy-Install\ComfyUI\models\sam3")
_MODEL_PATH = _MODEL_DIR / "sam3.safetensors"


class SAM3Backend:
    """
    Standalone SAM3 inference backend.

    Loads the SAM3 model directly using PyTorch (no ComfyUI runtime dependency)
    and provides simple methods for image segmentation.
    """

    def __init__(self):
        self.model = None           # Sam3VideoPredictor
        self.processor = None       # Sam3Processor
        self.detector = None        # model.detector
        self.device = None
        self.dtype = None
        self._loaded = False
        self._state = None          # Current image backbone state
        self._current_image = None  # Current PIL image

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_size_mb(self) -> float:
        if self.model is None:
            return 0
        # self.model is Sam3VideoPredictor (not nn.Module); self.model.model is the actual model
        target = self.model.model if hasattr(self.model, 'model') else self.model
        total = sum(p.numel() * p.element_size() for p in target.parameters())
        return total / 1024 / 1024

    def load_model(self, precision: str = "auto", compile_model: bool = False,
                   progress_cb=None) -> bool:
        """
        Load SAM3 model into GPU memory.

        Args:
            precision: "auto", "bf16", "fp16", "fp32"
            compile_model: Enable torch.compile
            progress_cb: Optional callback(status_text) for UI updates

        Returns:
            True if successful
        """
        if self._loaded:
            log.info("Model already loaded, skipping")
            return True

        try:
            # Add SAM3 source to Python path
            sam3_src = str(_SAM3_NODES_DIR)
            if sam3_src not in sys.path:
                sys.path.insert(0, sam3_src)

            # Resolve device
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
                log.warning("CUDA not available, using CPU (will be slow)")

            # Resolve dtype
            if precision == "auto":
                if self.device.type == "cuda":
                    cap = torch.cuda.get_device_capability()
                    if cap[0] >= 8:  # Ampere+
                        self.dtype = torch.bfloat16
                    elif cap[0] >= 7:  # Volta/Turing
                        self.dtype = torch.float16
                    else:
                        self.dtype = torch.float32
                else:
                    self.dtype = torch.float32
            else:
                self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

            log.info(f"Device: {self.device}, dtype: {self.dtype}")

            if progress_cb:
                progress_cb("构建模型结构...")

            # Build model on meta device (zero memory)
            from sam3 import build_sam3_video_model, _load_checkpoint_file, remap_video_checkpoint

            with torch.device("meta"):
                raw_model = build_sam3_video_model(
                    checkpoint_path=None,
                    load_from_HF=False,
                    bpe_path=str(_SAM3_NODES_DIR / "sam3" / "bpe_simple_vocab_16e6.txt.gz"),
                    enable_inst_interactivity=True,
                    compile=compile_model,
                    skip_checkpoint=True,
                )

            if progress_cb:
                progress_cb("加载模型权重...")

            # Load checkpoint
            if not _MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"模型文件未找到: {_MODEL_PATH}\n"
                    f"请先在ComfyUI中运行一次SAM3节点自动下载模型"
                )

            ckpt = _load_checkpoint_file(str(_MODEL_PATH))
            remapped = remap_video_checkpoint(ckpt, enable_inst_interactivity=True)
            del ckpt

            missing, unexpected = raw_model.load_state_dict(remapped, strict=False, assign=True)
            del remapped
            if missing:
                log.debug(f"SAM3: {len(missing)} missing keys")
            if unexpected:
                log.debug(f"SAM3: {len(unexpected)} unexpected keys")

            # Fix meta-device buffers
            for name, buf in list(raw_model.named_buffers()):
                if buf.device.type == "meta":
                    parts = name.split(".")
                    parent = raw_model
                    for p in parts[:-1]:
                        parent = getattr(parent, p)
                    attr_name = parts[-1]
                    if attr_name == "attn_mask" and hasattr(parent, "build_causal_mask"):
                        parent._buffers[attr_name] = parent.build_causal_mask()
                    else:
                        parent._buffers[attr_name] = torch.zeros_like(buf, device="cpu")

            raw_model.eval()

            if progress_cb:
                progress_cb("初始化推理引擎...")

            # Wrap in predictor
            from sam3.predictor import Sam3VideoPredictor
            self.model = Sam3VideoPredictor(
                bpe_path=str(_SAM3_NODES_DIR / "sam3" / "bpe_simple_vocab_16e6.txt.gz"),
                enable_inst_interactivity=True,
                compile=compile_model,
                model=raw_model,
            )

            # Set attention dtype
            from sam3.attention import set_sam3_dtype
            set_sam3_dtype(self.dtype if self.dtype != torch.float32 else None)

            # Selective weight casting
            if self.dtype != torch.float32:
                detector = self.model.model.detector
                for param in detector.backbone.parameters():
                    param.data = param.data.to(dtype=self.dtype)
                if detector.inst_interactive_predictor is not None:
                    for param in detector.inst_interactive_predictor.parameters():
                        param.data = param.data.to(dtype=self.dtype)

            self.detector = self.model.model.detector

            # Verify no parameters are still on meta device
            meta_params = []
            for name, param in raw_model.named_parameters():
                if param.device.type == "meta":
                    meta_params.append(name)
            if meta_params:
                log.warning(f"Found {len(meta_params)} parameters still on meta device! First 10: {meta_params[:10]}")
                # Replace each meta parameter with a zero-filled CPU parameter
                for name in meta_params:
                    # Navigate to the parent module and attribute
                    parts = name.rsplit(".", 1)
                    if len(parts) == 2:
                        parent_path, attr_name = parts
                        parent = raw_model
                        for p in parent_path.split("."):
                            parent = getattr(parent, p)
                        old_param = getattr(parent, attr_name)
                        new_param = nn.Parameter(
                            torch.zeros(old_param.shape, dtype=self.dtype, device="cpu")
                        )
                        setattr(parent, attr_name, new_param)
                        log.warning(f"  Fixed meta param: {name} shape={old_param.shape}")
                    else:
                        log.error(f"  Cannot fix top-level meta param: {name}")
                # Re-verify
                still_meta = sum(1 for _, p in raw_model.named_parameters() if p.device.type == "meta")
                log.warning(f"After fix: {still_meta} params still on meta device")

            self.detector = self.model.model.detector

            # Build processor
            from sam3.utils import Sam3Processor
            self.processor = Sam3Processor(
                model=self.detector,
                resolution=1008,
                device=self.device,
                confidence_threshold=0.2,
            )

            if progress_cb:
                progress_cb(f"模型就绪 ({self.model_size_mb:.0f} MB)")

            self._loaded = True
            log.info(f"SAM3 model loaded successfully ({self.model_size_mb:.1f} MB)")
            return True

        except Exception as e:
            log.exception("Failed to load SAM3 model")
            self._cleanup()
            raise

    def set_image(self, image: Image.Image | np.ndarray):
        """
        Set the current image for segmentation.
        Extracts backbone features (heavy computation).

        Args:
            image: PIL Image or numpy array (H, W, 3) RGB uint8
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        self._current_image = image

        # Move model to GPU
        self._move_to_device()

        # Clear old state
        if self._state is not None:
            del self._state
            gc.collect()
            torch.cuda.empty_cache()

        log.info(f"Extracting backbone features for image {image.size}...")
        self._state = self.processor.set_image(image)
        log.info(f"Image set: {image.size}, backbone features extracted, state keys: {list(self._state.keys())}")

    def segment_interactive(
        self,
        multi_prompts: List[Dict],
        img_w: int,
        img_h: int,
        refinement_iterations: int = 0,
        use_multimask: bool = True,
    ) -> Tuple[List[np.ndarray], List[float], np.ndarray]:
        """
        Run interactive segmentation for multiple prompt regions.

        Matches ComfyUI-SAM3's SAM3InteractiveCollector._run_prompts().

        Args:
            multi_prompts: List of prompt dicts from PromptManager.to_sam3_format()
            img_w, img_h: Image dimensions in pixels
            refinement_iterations: Number of mask refinement iterations (0-10)
            use_multimask: Use multi-mask output (3 candidates, pick best)

        Returns:
            (masks, scores, vis_image)
            - masks: List of (H, W) boolean numpy arrays
            - scores: List of float confidence scores
            - vis_image: (H, W, 3) RGB uint8 visualization
        """
        if not self._loaded or self._state is None:
            raise RuntimeError("Model not loaded or image not set")

        self._move_to_device()

        model = self.detector
        state = self._state

        all_masks = []
        all_scores = []

        for prompt in multi_prompts:
            # Build point arrays
            pts, labels = [], []
            for pt in prompt.get("positive_points", {}).get("points", []):
                pts.append([pt[0] * img_w, pt[1] * img_h])
                labels.append(1)
            for pt in prompt.get("negative_points", {}).get("points", []):
                pts.append([pt[0] * img_w, pt[1] * img_h])
                labels.append(0)

            point_coords = np.array(pts) if pts else None
            point_labels = np.array(labels) if labels else None

            # Build box array (use first positive box)
            box_array = None
            pos_boxes = prompt.get("positive_boxes", {}).get("boxes", [])
            if pos_boxes:
                cx, cy, w, h = pos_boxes[0]
                box_array = np.array([
                    (cx - w / 2) * img_w, (cy - h / 2) * img_h,
                    (cx + w / 2) * img_w, (cy + h / 2) * img_h,
                ])

            if point_coords is None and box_array is None:
                continue

            # Run predict_inst (interactive segmentation)
            masks_np, scores_np, _ = model.predict_inst(
                state,
                point_coords=point_coords,
                point_labels=point_labels,
                box=box_array,
                mask_input=None,
                multimask_output=use_multimask,
                normalize_coords=True,
            )

            # Select best mask
            best_idx = np.argmax(scores_np)
            best_mask = masks_np[best_idx]
            best_score = float(scores_np[best_idx])

            # Optional refinement iterations
            if refinement_iterations > 0:
                for _ in range(refinement_iterations):
                    # Feed low-res mask back as input
                    mask_input = masks_np[best_idx:best_idx+1]  # Use original low-res
                    masks_np2, scores_np2, _ = model.predict_inst(
                        state,
                        point_coords=point_coords,
                        point_labels=point_labels,
                        box=box_array,
                        mask_input=mask_input,
                        multimask_output=False,
                        normalize_coords=True,
                    )
                    best_mask = masks_np2[0]
                    best_score = float(scores_np2[0])

            all_masks.append(best_mask)
            all_scores.append(best_score)

        # Generate visualization
        vis_image = self._visualize(all_masks, all_scores, img_w, img_h)

        return all_masks, all_scores, vis_image

    def segment_text(
        self,
        text_prompt: str,
        img_w: int,
        img_h: int,
        confidence_threshold: float = 0.2,
        max_detections: int = -1,
        positive_boxes: Dict | None = None,
        negative_boxes: Dict | None = None,
    ) -> Tuple[List[np.ndarray], List[float], List, np.ndarray]:
        """
        Run text-based grounding segmentation.

        Matches ComfyUI-SAM3's SAM3Grounding.

        Args:
            text_prompt: Text description (e.g., "dog, cat")
            img_w, img_h: Image dimensions
            confidence_threshold: Minimum confidence score
            max_detections: Max results (-1 for all)
            positive_boxes: Optional positive box prompts
            negative_boxes: Optional negative box prompts

        Returns:
            (masks, scores, boxes, vis_image)
        """
        if not self._loaded or self._state is None:
            raise RuntimeError("Model not loaded or image not set")

        self._move_to_device()
        self.processor.set_confidence_threshold(confidence_threshold)

        state = self._state

        # Set text prompt
        if text_prompt and text_prompt.strip():
            state = self.processor.set_text_prompt(text_prompt.strip(), state)

        # Add box prompts if provided
        if positive_boxes and positive_boxes.get("boxes"):
            state = self.processor.add_multiple_box_prompts(
                positive_boxes["boxes"], positive_boxes["labels"], state
            )
        if negative_boxes and negative_boxes.get("boxes"):
            state = self.processor.add_multiple_box_prompts(
                negative_boxes["boxes"], negative_boxes["labels"], state
            )

        # Extract results
        masks = state.get("masks", None)
        boxes = state.get("boxes", None)
        scores = state.get("scores", None)

        if masks is None or len(masks) == 0:
            empty_vis = np.array(self._current_image) if self._current_image else np.zeros((img_h, img_w, 3), dtype=np.uint8)
            return [], [], [], empty_vis

        # Sort by score
        if scores is not None and len(scores) > 0:
            sorted_idx = torch.argsort(scores, descending=True)
            masks = masks[sorted_idx]
            boxes = boxes[sorted_idx] if boxes is not None else None
            scores = scores[sorted_idx]

        # Limit detections
        if max_detections > 0 and len(masks) > max_detections:
            masks = masks[:max_detections]
            boxes = boxes[:max_detections] if boxes is not None else None
            scores = scores[:max_detections] if scores is not None else None

        masks_list = [m.cpu().numpy() for m in masks]
        scores_list = [float(s) for s in scores]
        boxes_list = boxes.cpu().tolist() if boxes is not None else []

        vis_image = self._visualize(masks_list, scores_list, img_w, img_h, boxes_list)

        return masks_list, scores_list, boxes_list, vis_image

    def _visualize(self, masks: list, scores: list, img_w: int, img_h: int,
                   boxes=None) -> np.ndarray:
        """Create visualization with colored mask overlay."""
        from .prompt_manager import PROMPT_COLORS

        if self._current_image is None:
            return np.zeros((img_h, img_w, 3), dtype=np.uint8)

        vis = np.array(self._current_image).copy()

        if not masks:
            return vis

        for i, (mask, score) in enumerate(zip(masks, scores)):
            color_hex = PROMPT_COLORS[i % len(PROMPT_COLORS)]
            color_rgb = tuple(int(color_hex[j:j+2], 16) for j in (1, 3, 5))

            # Semi-transparent overlay
            mask_bool = mask > 0 if mask.dtype != bool else mask
            # Ensure mask is 2D (H, W) — may come as (H, W, 1) or (1, H, W)
            while mask_bool.ndim > 2:
                # Remove leading or trailing singleton dims
                if mask_bool.shape[0] == 1:
                    mask_bool = mask_bool[0]
                elif mask_bool.shape[-1] == 1:
                    mask_bool = mask_bool.squeeze(-1)
                else:
                    break
            # Resize mask to original image size if needed
            if mask_bool.shape != (img_h, img_w):
                import cv2
                mask_bool = cv2.resize(
                    mask_bool.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST
                ).astype(bool)

            vis[mask_bool] = (vis[mask_bool] * 0.5 + np.array(color_rgb) * 0.5).astype(np.uint8)

        return vis

    def _move_to_device(self):
        """Ensure model is on the correct device."""
        if self.model is not None:
            try:
                # Sam3VideoPredictor.model is the actual nn.Module
                self.model.model.to(self.device)
            except Exception:
                pass

    def _cleanup(self):
        """Free all GPU resources."""
        if self._state is not None:
            del self._state
        if self.model is not None:
            del self.model
        if self.processor is not None:
            del self.processor
        self.model = None
        self.processor = None
        self.detector = None
        self._state = None
        self._current_image = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload(self):
        """Unload model and free GPU memory."""
        log.info("Unloading SAM3 model...")
        self._cleanup()
        log.info("Model unloaded")
