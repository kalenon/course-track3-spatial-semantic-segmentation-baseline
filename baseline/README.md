# Track 3 参考实现

本项目完成四通道空间音频的事件标注与标签条件源分离。[赛题说明](../大作业%20track%203.md) 给出任务要求；[平台使用与复现指南](COURSE_GUIDE.md) 说明如何使用平台提供的数据挂载、下载并指定模型权重，以及运行训练、验证和盲测推理。

方法分两阶段：四通道音频标注网络识别目标事件类别，标签查询式源分离网络据此输出目标波形。训练阶段从独立事件、干扰音频、背景音和空间脉冲响应在线合成声景。训练与评估的原始 YAML 保留在 `config/`、`src/evaluation/eval_configs/`；平台上推荐使用 `scripts.train` 包装脚本传入自己的挂载路径，不必编辑这些 YAML。

运行所需资源见 [资产清单](ASSET_STATUS.md)，路径参数见 [PATHS.md](PATHS.md)。模型权重由使用者下载，可按 [课程指南](COURSE_GUIDE.md) 运行 `scripts.download_weights`，并将下载目录指定为 `WEIGHTS_ROOT`。参考权重在公开验证集上的实测值见 [BASELINE_RESULTS.md](BASELINE_RESULTS.md)。代码仓库不包含课程音频、大型预训练权重或后期盲测标签。

`src/modules/spatial_audio_synthesizer/` 和 `third_party/SpAudSyn/` 已随本项目提供；使用代码与外部资源时请遵守仓库内许可文件及课程要求。
