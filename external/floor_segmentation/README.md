# Floor Segmentation Worker

This worker keeps MMSegmentation out of the main AutoLabel Python 3.12
environment. It loads the three-class SegFormer checkpoint once per generation
run and returns deterministic 200x200 floor selections for all runnable images.

Expected labels:

- `0`: background
- `1`: line
- `2`: road

Create a Python 3.10 environment and install the training-compatible stack:

```powershell
conda create -n auto-label-mmseg310 python=3.10 -y
conda activate auto-label-mmseg310
pip install -r external/floor_segmentation/requirements.txt
mim install "mmcv==2.1.0"
```

Configure the production shell:

```powershell
$env:MMSEG_FLOOR_PYTHON="D:\envs\auto-label-mmseg310\python.exe"
$env:MMSEG_FLOOR_CHECKPOINT="D:\models\best_mIoU_iter_3000.pth"
$env:MMSEG_FLOOR_DEVICE="cuda:0"
```

The supplied generic Cityscapes PSPNet checkpoint is not compatible with this
worker. Preflight loads the configured checkpoint and rejects any model whose
decode head is not three-class.

