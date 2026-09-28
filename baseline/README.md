# Track 3 参考实现

本项目完成四通道空间音频的事件标注与标签条件源分离。先阅读 [赛题说明](../大作业%20track%203.md)，再按 [平台使用与复现指南](COURSE_GUIDE.md) 挂载自己的数据和权重、选择 GPU、训练、验证及准备盲测提交。

方法分两阶段：四通道音频标注网络识别目标事件类别，标签查询式源分离网络据此输出目标波形。训练阶段从独立事件、干扰音频、背景音和空间脉冲响应在线合成声景。训练与评估的原始 YAML 保留在 `config/`、`src/evaluation/eval_configs/`；平台上推荐使用 `scripts.train` 包装脚本传入自己的挂载路径，不必编辑这些 YAML。

运行前请核对 [资产清单](ASSET_STATUS.md)，路径参数见 [PATHS.md](PATHS.md)。没有权重挂载时，可按 [课程指南](COURSE_GUIDE.md) 用 `scripts.download_weights` 从原发布处获取。参考权重在公开验证集上的实测值见 [BASELINE_RESULTS.md](BASELINE_RESULTS.md)。代码仓库不包含课程音频、大型预训练权重或后期盲测标签。

`src/modules/spatial_audio_synthesizer/` 和 `third_party/SpAudSyn/` 已随本项目提供；使用代码与外部资源时请遵守仓库内许可文件及课程要求。
