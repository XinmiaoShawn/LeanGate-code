# LeanGate code release

## Goal
This repository exposes the inference path of LeanGate.

1. download the released LeanGate checkpoint,
2. run LeanGate on prepared `TUM`, `7SCENES`, or `EUROC` scenes,
3. export a sparse RGB manifest,
4. optionally run MASt3R-SLAM on that sparse sequence.

## Demo
![LeanGate demo](viz/demo.gif)

## Contents
- [Goal](#goal)
- [Quick Start](#quick-start)
- [What The User Gets](#what-the-user-gets)
- [Supported Inputs](#supported-inputs)
- [Public Workflow](#public-workflow)
- [Outputs](#outputs)
- [What Is Deliberately Out Of Scope](#what-is-deliberately-out-of-scope)
- [Troubleshooting](#troubleshooting)
- [Third-party components](#third-party-components)
- [Repository Layout](#repository-layout)

## Quick Start
```bash
pip install -e .
pip install -e third_party/MASt3R-SLAM/thirdparty/mast3r
pip install -e third_party/MASt3R-SLAM/thirdparty/in3d
pip install --no-build-isolation -e third_party/MASt3R-SLAM

python3 scripts/download_checkpoints.py --output-root checkpoints --repo-id ShawnX98/LeanGate

python3 scripts/generate_rgb_lists.py \
  --dataset-type TUM \
  --dataset-root /data/tum \
  --output-root outputs/predictions \
  --device cuda:0
```

## What The User Gets
- A checkpoint download entrypoint: `scripts/download_checkpoints.py`
- A sparse RGB export entrypoint: `scripts/generate_rgb_lists.py`
- A single-scene MASt3R-SLAM wrapper: `scripts/run_slam_scene.py`
- A dataset-level MASt3R-SLAM wrapper: `scripts/run_slam_dataset.py`

The intended public workflow is:

```text
prepared scene directories
    -> LeanGate inference
    -> sparse rgb manifest
    -> staged sparse scene for MASt3R-SLAM
    -> trajectory / reconstruction outputs
```

## Supported Inputs
- `TUM`
- `7SCENES`
- `EUROC`

Only prepared layouts are supported. The exact expected structures are documented in [docs/dataset_layouts.md](docs/dataset_layouts.md).

## Public Workflow
### 1. Install
Use `python3` and install PyTorch matching your CUDA runtime first.

```bash
pip install -e .
pip install -e third_party/MASt3R-SLAM/thirdparty/mast3r
pip install -e third_party/MASt3R-SLAM/thirdparty/in3d
pip install --no-build-isolation -e third_party/MASt3R-SLAM
```

### 2. Prepare checkpoints
The public LeanGate checkpoint is hosted at:

- Repo: `ShawnX98/LeanGate`
- URL: `https://huggingface.co/ShawnX98/LeanGate`
- File: `https://huggingface.co/ShawnX98/LeanGate/resolve/main/leangate.pt`

Download it with:

```bash
python3 scripts/download_checkpoints.py --output-root checkpoints --repo-id ShawnX98/LeanGate
```

You also need FLARE's `geometry_pose.pth` at:

```text
third_party/FLARE/checkpoints/geometry_pose.pth
```

Official FLARE source:

- `https://huggingface.co/AntResearch/FLARE/resolve/main/geometry_pose.pth`

Checkpoint notes:
- The public LeanGate file is expected locally as `checkpoints/leangate.pt`.
- `scripts/generate_rgb_lists.py` loads `leangate.pt` directly; the released setup does not require separate `iter` or `dec` flags.
- FLARE's geometry checkpoint is not mirrored by this repo.
- LeanGate uses FLARE pretrain weights, which follow FLARE's upstream terms.
- MASt3R-SLAM code and any weights used with it follow MASt3R-SLAM's upstream terms.

### 3. Generate sparse RGB manifests
```bash
python3 scripts/generate_rgb_lists.py \
  --dataset-type TUM \
  --dataset-root /data/tum \
  --output-root outputs/predictions \
  --device cuda:0
```

Each scene produces a manifest containing the kept RGB frames in original scene-relative paths and timestamp order.

### 4. Launch MASt3R-SLAM on the sparse sequence
Single scene:

```bash
python3 scripts/run_slam_scene.py \
  --dataset-type TUM \
  --dataset-root /data/tum \
  --scene-id rgbd_dataset_freiburg1_desk \
  --predictions-root outputs/predictions \
  --output-root outputs/slam
```

Full dataset:

```bash
python3 scripts/run_slam_dataset.py \
  --dataset-type TUM \
  --dataset-root /data/tum \
  --predictions-root outputs/predictions \
  --output-root outputs/slam
```

The wrapper materializes a sparse scene under `outputs/mast3r_sparse_inputs/`, generates an `intrinsics.yaml` when a supported calibration file is available, and then calls the vendored MASt3R-SLAM entrypoint.

## Outputs
Sparse RGB generation:
- `outputs/predictions/<dataset_slug>/leangate/<scene>.txt`
- `outputs/predictions/<dataset_slug>/leangate/scores/<scene>_scores.csv`
- `outputs/predictions/<dataset_slug>/leangate/timings/<scene>.json`

MASt3R-SLAM wrapper:
- `outputs/slam/<dataset_slug>/leangate/<scene>/trajectory_keyframes.tum`
- `outputs/slam/<dataset_slug>/leangate/<scene>/reconstruction.ply` when MASt3R-SLAM saves one
- `outputs/slam/<dataset_slug>/leangate/<scene>/run_metadata.json`
- `outputs/slam/<dataset_slug>/leangate/summary.csv`
- `outputs/slam/<dataset_slug>/leangate/summary.json`

## What Is Deliberately Out Of Scope
- training code
- teacher labeling pipelines
- reinforcement learning / wandb pipelines
- raw dataset download and preprocessing
- support for datasets outside the three public benchmarks above

## Troubleshooting
- If `leangate.pt` is missing, run `python3 scripts/download_checkpoints.py --output-root checkpoints --repo-id ShawnX98/LeanGate`.
- If `geometry_pose.pth` is missing, download it from the official FLARE URL above.
- If scene discovery fails, the dataset layout likely does not match [docs/dataset_layouts.md](docs/dataset_layouts.md).
- If MASt3R-SLAM import fails, ensure its vendored packages were installed from `third_party/MASt3R-SLAM/`.

## Third-party components
- LeanGate uses FLARE pretrain weights from `third_party/FLARE/`.
- The optional SLAM stage uses MASt3R-SLAM from `third_party/MASt3R-SLAM/`.
- Please refer to the corresponding upstream repositories and license files when using or redistributing these components.

## Repository Layout
- `scripts/`: public CLI entrypoints
- `src/evaluate/`: public workflow logic
- `src/student/`: LeanGate model loading and inference
- `src/slam_prefilter/`: compatibility utilities for RGB sequence loading
- `third_party/FLARE/`: vendored FLARE dependency
- `third_party/MASt3R-SLAM/`: vendored MASt3R-SLAM dependency
