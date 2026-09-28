# Track 3：平台使用与基线复现

赛题以 [大作业 Track 3](../大作业%20track%203.md) 为准。开发集由课程平台提供，挂载路径取决于运行环境；模型权重需由使用者下载，并指定权重目录。GPU 编号与可写输出目录同样通过路径和环境变量配置。仓库内的 `data/`、`checkpoint/` 只是默认相对路径；下列流程不依赖它们。

## 1. 平台数据、权重与环境

从仓库的 `baseline/` 目录执行。将示例路径替换为平台实际提供的数据挂载点及可写工作目录：

```bash
export GPU_ID=0
export DEV_SET_ROOT=/path/to/platform-mounted/dev_set
export WORK_ROOT=/path/to/your-writable-workspace/track3
export WEIGHTS_ROOT="$WORK_ROOT/weights"
mkdir -p "$WORK_ROOT" "$WEIGHTS_ROOT"
```

`DEV_SET_ROOT` 指向包含 `metadata/`、`sound_event/`、`interference/` 等子目录的 **dev_set 本身**，数据挂载可为只读。`WEIGHTS_ROOT` 是可写的权重下载目录，下载完成后应包含 `m2dat_4c.ckpt`、`resunetk.ckpt` 和 M2D 权重目录；需预留数 GB 空间。`CUDA_VISIBLE_DEVICES="$GPU_ID"` 将分配的 GPU 映射为程序内的 `cuda:0`。

### 安装项目依赖

平台镜像已提供基础 PyTorch 环境时，在选定镜像的终端安装本项目依赖。依赖清单位于 [`pyproject.toml`](pyproject.toml)，不单独指定 Torch/CUDA 版本；推荐使用 Python 3.11。若平台要求隔离依赖，应先进入平台提供的个人虚拟环境。

```bash
# 在仓库的 baseline/ 目录运行。
python -m pip install -e .
python -m pip check
python -c 'import sofa, lightning.pytorch, pytorch_lightning, torchmetrics, timm, nnAudio.features, torchlibrosa.stft, librosa, soundfile, yaml; from src.models.m2dat.portable_m2d import PortableM2D; assert hasattr(sofa, "Database"); print("Track 3 imports: OK")'
```

这里不使用仓库原有的 `requirements.txt` 和 `environment.yml`，两者包含固定 CUDA/PyTorch 配置或整套导出环境。若依赖安装报错或 `python -m pip check` 报告冲突，应核对镜像的 Python 与已预装包版本。GPU 可用性可用 `CUDA_VISIBLE_DEVICES="$GPU_ID" python -c 'import torch; print(torch.cuda.is_available())'` 检查。

参考模型与 M2D 权重不由平台提供。运行以下命令下载至 `WEIGHTS_ROOT`（约需下载 1.9 GB 压缩包）：

```bash
python -m scripts.download_weights --weights-root "$WEIGHTS_ROOT"
```

脚本分别从参考模型发布包与 M2D 特征提取器发布包下载，支持断点续传和安全解压；提取时会按 ZIP 自带的 CRC 信息检查目标文件，已有文件也会核对后复用。先加 `--dry-run` 可查看 URL 与目标路径。若计算节点不能访问外网，可在可联网机器下载，再将权重目录复制或挂载到计算节点并相应设置 `WEIGHTS_ROOT`。大型权重不提交到 Git。

```bash
python verify.py --source_dir . \
  --data-root "$DEV_SET_ROOT" --checkpoint-root "$WEIGHTS_ROOT"
```

完成环境和权重准备后，再运行目录检查。GPU 驱动和 CUDA 版本以实例实际环境为准。

## 2. 数据检查与干扰音频

完整训练需要 `sound_event/{train,valid}`、`noise/{train,valid}`、`room_ir/{train,valid}`、`interference/{train,valid}` 和 `metadata/valid.json`。公开验证用混合音频在 `synthesized/test/soundscape`，参考目标在 `synthesized/test/oracle_target`；其中目录名 `test` 不代表课程最终盲测。缺少干扰音频时，可以先确认课程挂载是否完整；如课程要求自行补齐，见 [干扰音频准备脚本](scripts/prepare_interference.py) 和 [资产清单](ASSET_STATUS.md)。不要把原始音频或权重提交到 Git。

## 3. 两阶段训练

以下包装脚本只在 `WORK_ROOT` 生成替换挂载路径后的配置，不改仓库原始 YAML：

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" python -m scripts.train \
  --config config/label/m2dat_4c.yaml \
  --data-root "$DEV_SET_ROOT" --checkpoint-root "$WEIGHTS_ROOT" \
  --workspace "$WORK_ROOT/label_stage1"

# 查看上一阶段的 checkpoints/，选取需要继续训练的权重。
export STAGE1_CKPT=/path/to/your/label_stage1_checkpoint.ckpt
CUDA_VISIBLE_DEVICES="$GPU_ID" python -m scripts.train \
  --config config/label/m2dat_4c_2blks.yaml \
  --data-root "$DEV_SET_ROOT" --checkpoint-root "$WEIGHTS_ROOT" \
  --workspace "$WORK_ROOT/label_stage2" --resume "$STAGE1_CKPT"

CUDA_VISIBLE_DEVICES="$GPU_ID" python -m scripts.train \
  --config config/separation/resunetk_capisdr.yaml \
  --data-root "$DEV_SET_ROOT" --checkpoint-root "$WEIGHTS_ROOT" \
  --workspace "$WORK_ROOT/separation"
```

第一次启动可给命令加 `--dry-run`，检查生成配置里的路径是否指向自己的挂载点；该模式不执行训练。正式运行时删除 `--dry-run`。完整训练需较长时间；先用小规模数据确认环境、磁盘与 GPU 资源足够。

## 4. 公开验证与参考权重

以下示例使用前述步骤下载的参考权重；若使用自行训练的模型，将 `TAGGER_CKPT`、`SEPARATOR_CKPT` 指向相应文件。`FEATURE_WEIGHTS` 是 M2D 预训练特征提取器权重。

```bash
export TAGGER_CKPT="$WEIGHTS_ROOT/m2dat_4c.ckpt"
export SEPARATOR_CKPT="$WEIGHTS_ROOT/resunetk.ckpt"
export FEATURE_WEIGHTS="$WEIGHTS_ROOT/m2d_as_vit_base-80x1001p16x16p32k-240413_AS-FT_enconly/weights_ep69it3124-0.47998.pth"
export VALID_INPUT="$DEV_SET_ROOT/synthesized/test/soundscape"
export VALID_TARGET="$DEV_SET_ROOT/synthesized/test/oracle_target"

CUDA_VISIBLE_DEVICES="$GPU_ID" python -m scripts.infer \
  --input-dir "$VALID_INPUT" --output-dir "$WORK_ROOT/validation_predictions" \
  --tagger-ckpt "$TAGGER_CKPT" --separator-ckpt "$SEPARATOR_CKPT" \
  --feature-weights "$FEATURE_WEIGHTS" --device cuda
python -m scripts.validate_submission \
  --input-dir "$VALID_INPUT" --predictions "$WORK_ROOT/validation_predictions"
python -m scripts.evaluate_submission \
  --input-dir "$VALID_INPUT" --reference-dir "$VALID_TARGET" \
  --predictions "$WORK_ROOT/validation_predictions" \
  --output "$WORK_ROOT/validation_metrics.json"
```

输出为每条混合音频一个 JSON 和最多三条 32 kHz 单通道分离波形；格式检查不需要参考目标。已发布参考成绩见 [BASELINE_RESULTS.md](BASELINE_RESULTS.md)，比较时应确认数据版本与评价脚本一致。

## 5. 后续盲测

盲测挂载后，把 `--input-dir` 改为盲测混合音频目录、`--output-dir` 改为新的可写预测目录，运行 `scripts.infer` 和 `scripts.validate_submission`。不要在无参考目标的盲测上运行 `scripts.evaluate_submission`；课程组会统一评分。
