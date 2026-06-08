# Image-to-Prompt Project

## Overview

This repository contains a pipeline that generates images from text prompts, evaluates them, and iteratively improves the prompts using a **VLM -> OPRO -> GA** workflow.  The main notebooks (`Code_automated.ipynb` and `Code_automated2.ipynb`) drive the entire process.

## Folder Structure

```
.
├── README.md                # THIS FILE
├── Code_automated.ipynb     # First version of the pipeline
├── Code_automated2.ipynb    # Updated version (SAGE)
├── tp2-chosen/              # Directory with target images (copy of tp2‑chosen)
├── outputs/                 # Generated outputs when running locally
├── GLOBAL_TOP_3_RESULTS/    # folder with results
├── utils/                   # Script to extract the results from json
└── src/                     # Custom modules loaded via Google Drive (fitness, vlm, OPRO, ga, etc.)
```

## Prerequisites & Dependencies

The notebooks install the required Python packages automatically.  For reference:

- `diffusers[torch]`
- `transformers`
- `accelerate`
- `safetensors`
- `matplotlib`
- `lpips`
- `pandas<3`
- `Pillow<12`
- `torch` (installed as a dependency of `diffusers`)

> **Important:** The notebooks pin `Pillow<12` and `pandas<3` to avoid compatibility issues on Colab/VS Code environments.

## Quick Start (Local / Colab)

1. **Open the notebook** (`Code_automated.ipynb` or `Code_automated2.ipynb`).
2. **Run the first cell** – it installs the dependencies and sets `PYTORCH_ALLOC_CONF` for better VRAM handling.
3. **Mount Google Drive (optional):**

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

   The notebooks will look for the target images in `MyDrive/GENAI_TP2/tp2‑chosen/` or in a local `tp2‑chosen/` folder.
4. **Provide target images**:

   - Place PNG/JPG files in `tp2‑chosen/` (or the equivalent Google‑Drive folder).
   - Alternatively, zip the folder and put `tp2‑chosen.zip` in the same location.
5. **Set the target** (edit the cell defining `TARGET_IMAGE` and `NUM_RUNS`).
6. **Run the notebook** from top to bottom.  It will:

   - Load the LCM diffusion model (`SimianLuo/LCM_Dreamshaper_v7`).
   - Generate initial candidates with a VLM.
   - Optimise prompts using OPRO (online prompt optimisation).
   - Run a genetic algorithm (GA) for further evolution.
   - Save best images and checkpoints under a timestamped run folder inside `outputs/`.

## Notes & Tips

- **VRAM Management:** The notebooks call `flush_vram()` after loading/unloading large models to free GPU memory.
- **Fitness Module:** `Code_automated.ipynb` uses `fitness.py` and `ga.py`; `Code_automated2.ipynb` swaps to `fitness2.py` and `ga2.py`. Ensure the corresponding module exists in the `src/` directory on Google Drive.
- **Results:** Each run creates a folder `run_<timestamp>_TARGET_<id>` inside `outputs/` containing sub‑folders for VLM, OPRO, and GA results, plus JSON checkpoints.
