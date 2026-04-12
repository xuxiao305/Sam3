# SAM3 Segment Tool - PyQt6 Desktop Application

Standalone SAM3 segmentation tool replicating ComfyUI-SAM3 functionality.

## Features
- **Image Segmentation**: Point/BBox/Text prompt modes
- **Interactive Canvas**: Click to add positive/negative points, drag to draw boxes
- **Multi-Region Prompts**: Up to 8 color-coded prompt regions
- **Live Segmentation**: Instant mask preview with Run button
- **Video Segmentation**: Click on first frame → track through video
- **Export**: Masks (PNG), Prompts (JSON), Visualization (PNG)

## Quick Start
```bash
python main.py
```

## Architecture
```
sam3_app/
├── main.py              # Entry point
├── app.py               # Main window & layout
├── canvas.py            # Interactive image canvas
├── preview.py           # Result preview canvas
├── prompt_manager.py    # Multi-region prompt data model
├── prompt_panel.py      # Prompt management UI panel
├── toolbar.py           # Top toolbar
├── status_bar.py        # Bottom status bar
├── backend.py           # SAM3 model loading & inference
├── video_dialog.py      # Video segmentation dialog
└── export.py            # Export utilities
```
