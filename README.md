# OcrModelNemberDetect

Aviation & Technical Chart Digit Extractor and Label Verification Studio.

## Overview
- **Hybrid OCR Engine**: Combines EasyOCR (CRAFT) full-scene detection with OpenCV morphological grid removal and ROI contour-based single-digit extraction.
- **Interactive Web Verification Studio**: Web UI for inspecting crops, verifying/correcting predicted digit values, and toggling entity types (NUMBER vs TEXT).
- **Label Studio Integration**: Export one-click pre-annotation JSON formatted directly for Label Studio object detection & OCR workflows.
- **MinIO S3 Support**: Cloud storage integration for streaming chart images and exporting annotations.

## Quick Start
```bash
pip install -r requirements.txt
python3 app.py
```
Open `http://127.0.0.1:5050` in browser.
