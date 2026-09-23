# Track 3 Baseline

本目录提供 Track 3 的参考实现。系统采用两阶段结构：

1. 音频标注（AT）网络从四通道 FOA 混合音频预测目标事件类别；
2. 标签查询式源分离（SS）网络根据预测类别输出单通道目标波形。

训练阶段使用在线空间声景合成：从课程提供的干声事件、FOA 房间脉冲响应、背景噪声和干扰声中随机构造训练样本。`src/modules/spatial_audio_synthesizer/` 已随本项目提供，无须另行下载。

## 环境

```bash
conda env create -f environment.yml
conda activate ssp-track3
```

在 Linux 上，如音频依赖安装失败，可安装 SoX 开发包：

```bash
sudo apt-get install -y gcc g++ sox libsox-dev
```

## 数据放置

将课程提供的数据放在 `data/dev_set/`。其应包含 `config`、`metadata`、`noise`、`room_ir`、`sound_event`、`interference` 和 `synthesized/test` 等目录。可运行以下命令检查结构与预训练特征提取器权重：

```bash
python verify.py --source_dir .
```

预训练特征提取器和参考检查点不包含在仓库中；由课程发布渠道另行提供，或按课程说明自行准备。

## 训练

先训练四通道音频标注网络：

```bash
python -m src.train -c config/label/m2dat_4c.yaml -w workspace/label
```

选定最佳 epoch 后，继续微调特征提取器最后两个块：

```bash
python -m src.train -c config/label/m2dat_4c_2blks.yaml \
  -w workspace/label \
  -r workspace/label/m2dat_4c/checkpoints/epoch=BEST_EPOCH_NUMBER.ckpt
```

训练标签查询式源分离网络：

```bash
python -m src.train -c config/separation/resunetk_capisdr.yaml -w workspace/separation
```

训练日志和检查点默认写入 `workspace/`。完整训练需要较高的 GPU 算力；建议先缩短配置文件中的 `dataset_length`，确认数据和训练流程正确后再进行完整训练。

## 本地评测

为评测配置中的 `tagger_ckpt`、`separator_ckpt` 填入本地检查点路径，再执行：

```bash
python -m src.evaluation.evaluate \
  -c src/evaluation/eval_configs/m2dat_4c_resunetk.yaml \
  --result_dir workspace/evaluation
```

## 许可证

本目录以及 `third_party/SpAudSyn/` 中保留了相应的许可证文件。使用其中代码、数据或预训练模型时，请遵守各自许可证与课程要求。
