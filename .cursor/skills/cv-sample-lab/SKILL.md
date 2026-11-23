---
name: cv-sample-lab
description: >-
  Project conventions for cv_sample — a Python image-processing lab for verifying
  and developing classical algorithms, ML/DL models, and visual effects. Use when
  working in this repo, adding a new experiment under code/, reading images from
  sources/, or when the user asks for image-processing scripts, algorithm demos,
  model inference/training prototypes, or visual-effect experiments.
---

# cv_sample Lab

## Purpose

这个工程是专门针对图像处理做的，代码用Python运行，用于验证或开发很多图像处理算法、模型，或者达到某种效果。

Scope includes:
- Classical CV algorithms (filters, morphology, geometry, …)
- Models (ONNX / Torch / HF / custom nets — inference or small training loops)
- End visual effects / pipelines that combine either

## Layout

| Path | Role |
|------|------|
| `sources/` | 图片来源（输入素材；可按主题分子目录） |
| `code/` | 代码（按不同需求建不同文件夹，各自放脚本） |

```
cv_sample/
├── sources/                 # input images only
│   └── <topic>/             # e.g. exclude_crops/
└── code/
    └── <experiment>/        # one folder per algorithm / model / effect
        ├── *.py
        ├── weights/         # optional; local checkpoints (git-ignore large files)
        └── output/          # optional; generated results (git-ignore if large)
```

Do **not** dump scripts into `code/` root. Always create a dedicated subfolder.

## When adding a new experiment

1. Name the folder after the algorithm, model, or effect (`code/canny_edges/`, `code/sam_segment/`, `code/exclude_mask_blend/`).
2. Put runnable Python under that folder (prefer a single `main.py` or clearly named scripts).
3. Read inputs from `sources/...` using paths relative to the **repo root** (or resolve via `__file__` → repo root).
4. Write outputs under `code/<experiment>/output/` (create if missing). Never overwrite files in `sources/`.
5. Keep each script self-contained and runnable: `python code/<experiment>/main.py`.
6. For models: keep weights under `code/<experiment>/weights/` or document a download step; do not commit large binaries unless the user asks.

## Project toolkit (installed)

| Kind | Name | When |
|------|------|------|
| Skill | `cv-sample-lab` (this file) | Always — layout & workflow |
| Skill | `computer-vision-opencv` | Classical CV / OpenCV / detectors |
| Skill | `image-processing` | PIL/Pillow transforms & image I/O |
| Skill | `pytorch-patterns` | Torch model code patterns |
| Skill | `huggingface-local-models` | Local HF model download & run |
| Rule | `.cursor/rules/cv-lab.mdc` | Always-on project conventions |
| MCP | Context7 (`.cursor/mcp.json`) | Up-to-date library API docs |
| Deps | `requirements.txt` | `pip install -r requirements.txt` |

Skills live under `.agents/skills/` (project-scoped via `skills-lock.json`). Global MemPalace + Codegraph MCPs already cover memory and code navigation.

## Script conventions

- Stack: Python 3 + OpenCV (`cv2`), NumPy by default; add Pillow / scikit-image when needed; for models use PyTorch / onnxruntime / transformers as appropriate.
- Show or save results so the effect is inspectable (window +/or image files under `output/`).
- Prefer small, focused scripts over shared frameworks — this is a lab, not a product app.
- Classical CV → `computer-vision-opencv` / `image-processing`. Models → `pytorch-patterns` / `huggingface-local-models`. API uncertainty → Context7.

### Minimal template

```python
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[2]  # repo root from code/<exp>/script.py
SRC = ROOT / "sources" / "<topic>" / "<image>.png"
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)

img = cv2.imread(str(SRC), cv2.IMREAD_UNCHANGED)
# ... process ...
cv2.imwrite(str(OUT / "result.png"), result)
```

## Agent checklist

- [ ] New work → new `code/<experiment>/` folder
- [ ] Inputs from `sources/`; outputs to `code/<experiment>/output/`
- [ ] Script runs with plain `python` from repo root or script dir
- [ ] No mutation of source images
