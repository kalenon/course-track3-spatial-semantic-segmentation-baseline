# Track 3 路径参数速查

在 `baseline/` 下，先设置自己的挂载点（示例值均需替换）：

```bash
export GPU_ID=0
export DEV_SET_ROOT=/path/to/mounted/dev_set
export WEIGHTS_ROOT=/path/to/mounted/checkpoint
export WORK_ROOT=/path/to/your-writable-workspace/track3
```

- 训练：`python -m scripts.train --config config/label/m2dat_4c.yaml --data-root "$DEV_SET_ROOT" --checkpoint-root "$WEIGHTS_ROOT" --workspace "$WORK_ROOT/label"`。其他阶段只换配置和工作目录；`--resume` 指定已有 checkpoint。`--dry-run` 生成配置但不训练。
- 推理：`python -m scripts.infer --input-dir "$DEV_SET_ROOT/synthesized/test/soundscape" --output-dir "$WORK_ROOT/predictions" --tagger-ckpt "$WEIGHTS_ROOT/m2dat_4c.ckpt" --separator-ckpt "$WEIGHTS_ROOT/resunetk.ckpt" --feature-weights "$WEIGHTS_ROOT/m2d_as_vit_base-80x1001p16x16p32k-240413_AS-FT_enconly/weights_ep69it3124-0.47998.pth"`。
- 校验：`python verify.py --source_dir . --data-root "$DEV_SET_ROOT" --checkpoint-root "$WEIGHTS_ROOT"`。

GPU 由平台分配，可在训练/推理命令前加 `CUDA_VISIBLE_DEVICES="$GPU_ID"`；不要将别人的 GPU 编号、挂载点或工作目录写入脚本。完整操作顺序见 [COURSE_GUIDE.md](COURSE_GUIDE.md)。
