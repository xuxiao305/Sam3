"""
SAM3 Segment Tool - Export Utilities

Export segmentation results as PNG masks, JSON prompts, and visualization images.
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from PIL import Image


def export_mask_png(masks: List[np.ndarray], output_path: str, img_w: int, img_h: int):
    """
    Export masks as a combined PNG (8-bit grayscale).
    Background = 0 (black), each region gets a distinct visible value.
    For single mask: 0 = background, 255 = foreground.
    For multiple masks: evenly spaced values so each region is distinguishable.
    """
    import cv2
    combined = np.zeros((img_h, img_w), dtype=np.uint8)
    n = len(masks)
    for i, mask in enumerate(masks):
        # Ensure mask is 2D (H, W) — may come as (1, H, W) or (H, W, 1)
        if mask.ndim > 2:
            mask = mask.squeeze()
        # Resize if needed
        if mask.shape != (img_h, img_w):
            mask_resized = cv2.resize(
                mask.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        else:
            mask_resized = mask.astype(bool) if mask.dtype != bool else mask
        # Assign visible value: single mask -> 255, multiple -> evenly spaced
        combined[mask_resized] = 255 if n == 1 else int(255 * (i + 1) / n)

    Image.fromarray(combined, mode='L').save(output_path)


def export_prompts_json(
    multi_prompts: List[Dict],
    image_path: Optional[str],
    img_w: int,
    img_h: int,
    output_path: str,
):
    """
    Export prompt data as JSON matching UI_DESIGN.md spec.
    """
    export = {
        "export_time": datetime.now().isoformat() + "Z",
        "image_info": {
            "filename": Path(image_path).name if image_path else "",
            "width": img_w,
            "height": img_h,
        },
        "prompts": multi_prompts,
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export, f, ensure_ascii=False, indent=2)


def export_visualization(vis_image: np.ndarray, output_path: str):
    """Export visualization image as PNG."""
    Image.fromarray(vis_image).save(output_path)


def export_individual_masks(masks: List[np.ndarray], output_dir: str, img_w: int, img_h: int):
    """Export each mask as a separate PNG file."""
    import cv2
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, mask in enumerate(masks):
        # Ensure mask is 2D (H, W) — may come as (1, H, W) or (H, W, 1)
        if mask.ndim > 2:
            mask = mask.squeeze()
        if mask.shape != (img_h, img_w):
            mask_resized = cv2.resize(
                mask.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        else:
            mask_resized = mask.astype(bool) if mask.dtype != bool else mask

        mask_uint8 = (mask_resized * 255).astype(np.uint8)
        Image.fromarray(mask_uint8, mode='L').save(str(output_dir / f"mask_region_{i+1}.png"))
