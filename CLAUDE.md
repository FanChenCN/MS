# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
python -m venv venv/marigold
source venv/marigold/bin/activate
pip install -r requirements.txt          # inference only
pip install -r requirements+.txt         # + evaluation
pip install -r requirements++.txt        # + training
```

Checkpoint cache location is controlled by `HF_HOME`. To download weights locally:
```bash
bash script/download_weights.sh marigold-depth-v1-1
```

## Inference

```bash
python script/depth/run.py --checkpoint prs-eth/marigold-depth-v1-1 --input_rgb_dir input/ --output_dir output/ --fp16
python script/normals/run.py --checkpoint prs-eth/marigold-normals-v1-1 --input_rgb_dir input/ --output_dir output/ --fp16
python script/iid/run.py --checkpoint prs-eth/marigold-iid-appearance-v1-1 --input_rgb_dir input/ --output_dir output/ --fp16
```

Key inference flags: `--ensemble_size` (default 1), `--denoise_steps`, `--processing_res`, `--seed`.

## Evaluation

```bash
export BASE_DATA_DIR=<your_data_dir>
bash script/depth/eval/11_infer_nyu.sh && bash script/depth/eval/12_eval_nyu.sh
bash script/normals/eval/11_infer_scannet.sh && bash script/normals/eval/12_eval_scannet.sh
```

## Training

```bash
export BASE_DATA_DIR=YOUR_DATA_DIR
export BASE_CKPT_DIR=YOUR_CHECKPOINT_DIR  # place SD v2 checkpoint here

python script/depth/train.py --config config/train_marigold_depth.yaml
python script/normals/train.py --config config/train_marigold_normals.yaml
python script/iid/train.py --config config/train_marigold_iid_appearance.yaml

# Resume from checkpoint
python script/depth/train.py --resume_run output/marigold_base/checkpoint/latest
```

Use `config/train_debug_*.yaml` configs for quick debug runs.

## Architecture

The codebase has two layers:

**`marigold/`** — Public pipeline library (HuggingFace `diffusers`-compatible):
- `marigold_depth_pipeline.py`, `marigold_normals_pipeline.py`, `marigold_iid_pipeline.py` — each wraps a Stable Diffusion v2 UNet with task-specific encoding/decoding and ensemble logic.
- `util/` — shared image utilities, batch size auto-selection, ensemble aggregation.

**`src/`** — Training infrastructure:
- `trainer/` — one trainer per modality (`marigold_depth_trainer.py`, etc.), driven by YAML configs via Accelerate.
- `dataset/` — per-dataset loaders (Hypersim, KITTI, NYU, ScanNet, etc.) with `base_depth_dataset.py` / `base_normals_dataset.py` / `base_iid_dataset.py` as base classes; `mixed_sampler.py` handles multi-dataset sampling.

**`script/`** — CLI entry points organized by modality (`depth/`, `normals/`, `iid/`), each containing `run.py` (user-facing), `infer.py`, `train.py`, `eval.py`, and `eval/` bash scripts.

**`config/`** — YAML configs for training (`train_marigold_*.yaml`), datasets (`dataset_depth/`, `dataset_normals/`, `dataset_iid/`), and the base SD v2 model (`model_sdv2.yaml`).

The inference pipeline flow: RGB image → VAE encode → concatenate with noisy latent → UNet denoise (N steps) → VAE decode → task-specific normalization → optional ensemble over multiple passes.
