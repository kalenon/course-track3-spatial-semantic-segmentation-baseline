# Track 3 参考权重评测记录

下表是参考权重在公开验证划分 1,512 条混合音频上的记录，使用四通道标注权重 `m2dat_4c.ckpt` 与分离权重 `resunetk.ckpt`，不是最终盲测成绩。

| 指标 | 实测值 |
|---|---:|
| CAPI-SDRi | 8.4886 dB |
| 混合级标签 Accuracy | 60.7143% |
| 源级标签 Accuracy | 70.3939% |
| 模型总参数 | 126,434,854 |

按 [平台使用与复现指南](COURSE_GUIDE.md) 设置 `DEV_SET_ROOT`、`WEIGHTS_ROOT` 和 `WORK_ROOT`，运行 `scripts.infer`、`scripts.validate_submission`、`scripts.evaluate_submission` 可在自己的挂载数据上取得对应指标。下表记录来自参考评测流程；学生脚本和参考配置的统计细节若有差异，应以课程正式评测口径为准，并在报告中写明脚本与数据版本。只有使用相同数据版本、模型权重和指标口径，数值才可直接比较。
