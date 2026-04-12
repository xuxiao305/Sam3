#!/usr/bin/env python
"""
SAM3 Segment Tool — Entry Point

Standalone PyQt6 application for SAM3 image segmentation.
Supports point/box interactive prompts and text grounding.

Usage:
    python sam3_app/main.py
    # or from SAM3_Segment root:
    python -m sam3_app.main
"""

import logging
import sys
import os

# ─── Ensure project root is on sys.path ──────────────────────────────────────
# When run as `python sam3_app/main.py`, the parent directory won't be on
# sys.path by default, so relative/package imports fail. Add it explicitly.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sam3_app")


def main():
    # ─── Install ComfyUI shims FIRST (before any SAM3 imports) ─────────────
    from sam3_app.comfy_shim import install_shims
    install_shims()

    # ─── Add ComfyUI-SAM3 source to path ───────────────────────────────────
    # The backend.py imports from ComfyUI-SAM3's nodes/ directory.
    # Adjust this path if your ComfyUI installation is elsewhere.
    comfyui_sam3_path = os.environ.get(
        "SAM3_SOURCE_PATH",
        r"d:\AI\ComfyUI-Easy-Install\ComfyUI\custom_nodes\ComfyUI-SAM3-main",
    )
    if os.path.isdir(comfyui_sam3_path):
        comfyui_nodes = os.path.join(comfyui_sam3_path, "nodes")
        if comfyui_nodes not in sys.path:
            sys.path.insert(0, comfyui_nodes)
            log.info(f"Added SAM3 source: {comfyui_nodes}")
    else:
        log.warning(f"SAM3 source path not found: {comfyui_sam3_path}")

    # ─── Qt Application ─────────────────────────────────────────────────────
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt

    # High-DPI support
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("SAM3 Segment Tool")
    app.setStyle("Fusion")

    # Dark palette
    from PyQt6.QtGui import QPalette, QColor
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(224, 224, 224))
    palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(35, 35, 35))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(42, 42, 42))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(224, 224, 224))
    palette.setColor(QPalette.ColorRole.Text, QColor(224, 224, 224))
    palette.setColor(QPalette.ColorRole.Button, QColor(42, 42, 42))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(224, 224, 224))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 51, 51))
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)

    # ─── Main Window ────────────────────────────────────────────────────────
    from sam3_app.app import SAM3App

    window = SAM3App()
    window.show()

    log.info("SAM3 Segment Tool started")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
