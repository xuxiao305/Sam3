"""
SAM3 Segment Tool - Prompt Manager

Manages multi-region prompt data model.
Each prompt region has: positive_points, negative_points, positive_boxes, negative_boxes.
Pixel coordinates stored internally; normalization happens at export time.

Matches ComfyUI-SAM3's SAM3_MULTI_PROMPTS format.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


# 8-color palette matching ComfyUI-SAM3 JS widgets
PROMPT_COLORS = [
    '#00ffff',  # Cyan - Region 1
    '#ffff00',  # Yellow - Region 2
    '#ff00ff',  # Magenta - Region 3
    '#00ff00',  # Lime - Region 4
    '#ff6600',  # Orange - Region 5
    '#ff0099',  # Pink - Region 6
    '#6666ff',  # Blue - Region 7
    '#00cccc',  # Teal - Region 8
]

MAX_PROMPTS = 8


@dataclass
class PromptRegion:
    """Single prompt region with points and boxes in pixel coordinates."""
    id: int = 0
    positive_points: List[Tuple[int, int]] = field(default_factory=list)
    negative_points: List[Tuple[int, int]] = field(default_factory=list)
    positive_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)  # (x1, y1, x2, y2)
    negative_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)  # (x1, y1, x2, y2)

    @property
    def color(self) -> str:
        return PROMPT_COLORS[self.id % len(PROMPT_COLORS)]

    @property
    def total_prompts(self) -> int:
        return (len(self.positive_points) + len(self.negative_points) +
                len(self.positive_boxes) + len(self.negative_boxes))

    def has_content(self) -> bool:
        return bool(self.positive_points or self.negative_points or
                     self.positive_boxes or self.negative_boxes)

    def clear(self):
        self.positive_points.clear()
        self.negative_points.clear()
        self.positive_boxes.clear()
        self.negative_boxes.clear()


class PromptManager:
    """Manages multiple prompt regions for SAM3 segmentation."""

    def __init__(self):
        self.regions: List[PromptRegion] = []
        self.active_index: int = 0
        self._add_region()  # Start with one region

    def _add_region(self) -> PromptRegion:
        region = PromptRegion(id=len(self.regions))
        self.regions.append(region)
        return region

    def add_region(self) -> Optional[PromptRegion]:
        """Add a new prompt region. Returns None if max reached."""
        if len(self.regions) >= MAX_PROMPTS:
            return None
        region = self._add_region()
        self.active_index = len(self.regions) - 1
        return region

    def remove_region(self, index: int) -> bool:
        """Remove a prompt region. Cannot remove if only 1 left."""
        if len(self.regions) <= 1:
            return False
        self.regions.pop(index)
        # Re-assign IDs and colors
        for i, r in enumerate(self.regions):
            r.id = i
        if self.active_index >= len(self.regions):
            self.active_index = len(self.regions) - 1
        return True

    def set_active(self, index: int):
        if 0 <= index < len(self.regions):
            self.active_index = index

    @property
    def active(self) -> PromptRegion:
        return self.regions[self.active_index]

    def add_positive_point(self, x: int, y: int):
        self.active.positive_points.append((x, y))

    def add_negative_point(self, x: int, y: int):
        self.active.negative_points.append((x, y))

    def add_positive_box(self, x1: int, y1: int, x2: int, y2: int):
        self.active.positive_boxes.append((x1, y1, x2, y2))

    def add_negative_box(self, x1: int, y1: int, x2: int, y2: int):
        self.active.negative_boxes.append((x1, y1, x2, y2))

    def remove_last_point(self, region_index: Optional[int] = None):
        """Remove the last added point from the specified (or active) region."""
        region = self.regions[region_index if region_index is not None else self.active_index]
        # Remove from the last non-empty list
        for lst in [region.positive_points, region.negative_points]:
            if lst:
                lst.pop()
                return

    def remove_point_at(self, region_index: int, x: int, y: int, radius: int = 10):
        """Remove a point near (x, y) within radius pixels."""
        region = self.regions[region_index]
        for lst in [region.positive_points, region.negative_points]:
            for i, (px, py) in enumerate(lst):
                if abs(px - x) <= radius and abs(py - y) <= radius:
                    lst.pop(i)
                    return True
        return False

    def remove_box_at(self, region_index: int, x: int, y: int):
        """Remove a box containing point (x, y)."""
        region = self.regions[region_index]
        for lst in [region.positive_boxes, region.negative_boxes]:
            for i, (x1, y1, x2, y2) in enumerate(lst):
                if x1 <= x <= x2 and y1 <= y <= y2:
                    lst.pop(i)
                    return True
        return False

    def clear_active(self):
        self.active.clear()

    def clear_all(self):
        for r in self.regions:
            r.clear()

    def has_content(self) -> bool:
        return any(r.has_content() for r in self.regions)

    def total_points(self) -> int:
        return sum(r.total_prompts for r in self.regions)

    def to_sam3_format(self, img_w: int, img_h: int) -> List[Dict]:
        """
        Export prompts in SAM3_MULTI_PROMPTS format (normalized coordinates).
        
        Matches ComfyUI-SAM3's SAM3MultiRegionCollector output format:
        - Points: normalized [0,1] with labels
        - Boxes: normalized [cx, cy, w, h] center format with labels
        """
        multi_prompts = []
        for region in self.regions:
            if not region.has_content():
                continue

            prompt = {
                "id": region.id,
                "positive_points": {"points": [], "labels": []},
                "negative_points": {"points": [], "labels": []},
                "positive_boxes": {"boxes": [], "labels": []},
                "negative_boxes": {"boxes": [], "labels": []},
            }

            # Normalize positive points
            for (px, py) in region.positive_points:
                prompt["positive_points"]["points"].append([px / img_w, py / img_h])
                prompt["positive_points"]["labels"].append(1)

            # Normalize negative points
            for (px, py) in region.negative_points:
                prompt["negative_points"]["points"].append([px / img_w, py / img_h])
                prompt["negative_points"]["labels"].append(0)

            # Normalize positive boxes (pixel x1,y1,x2,y2 → normalized cx,cy,w,h)
            for (x1, y1, x2, y2) in region.positive_boxes:
                x1n, y1n = x1 / img_w, y1 / img_h
                x2n, y2n = x2 / img_w, y2 / img_h
                prompt["positive_boxes"]["boxes"].append([
                    (x1n + x2n) / 2, (y1n + y2n) / 2, x2n - x1n, y2n - y1n
                ])
                prompt["positive_boxes"]["labels"].append(True)

            # Normalize negative boxes
            for (x1, y1, x2, y2) in region.negative_boxes:
                x1n, y1n = x1 / img_w, y1 / img_h
                x2n, y2n = x2 / img_w, y2 / img_h
                prompt["negative_boxes"]["boxes"].append([
                    (x1n + x2n) / 2, (y1n + y2n) / 2, x2n - x1n, y2n - y1n
                ])
                prompt["negative_boxes"]["labels"].append(False)

            multi_prompts.append(prompt)

        return multi_prompts
