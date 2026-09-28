# Track 3 运行前资产清单

运行所需资产如下。开发集由课程平台提供，模型权重由使用者下载；两者均不随代码仓库发布。

| 路径变量 | 必要内容 | 来源与用途 |
|---|---|---|
| `DEV_SET_ROOT` | `sound_event/`、`noise/`、`room_ir/`、`interference/` 的训练与验证子集，`metadata/valid.json` | 平台提供；用于在线声景合成与模型训练 |
| `DEV_SET_ROOT` | `synthesized/test/soundscape/`、`synthesized/test/oracle_target/` | 平台提供；用于公开验证 |
| `WEIGHTS_ROOT` | M2D 预训练权重、`m2dat_4c.ckpt`、`resunetk.ckpt` | 使用者下载并指定目录；用于特征提取与参考模型推理 |
| `WORK_ROOT` | 可写目录 | 存放配置副本、训练日志、检查点和预测 |

运行 `python verify.py --source_dir . --data-root "$DEV_SET_ROOT" --checkpoint-root "$WEIGHTS_ROOT"` 检查结构。若数据挂载缺少 `interference/train` 或 `interference/valid`，应先核对课程资源是否挂载完整；按课程要求需要补齐时，可运行 `python -m scripts.prepare_interference --dataset-root "$DEV_SET_ROOT" --storage-dir /path/to/large-raw-storage`。该脚本下载大型原始归档，运行前需确认网络、存储容量、目录写权限和语料许可。参考权重推理与两阶段从头训练所需资产并不完全相同；完整训练前应通过上述结构检查。
