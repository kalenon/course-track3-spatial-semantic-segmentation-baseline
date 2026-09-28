# Track 3 运行前资产清单

下列内容由你在课程平台挂载，不随代码仓库发布：

| 挂载 | 必要内容 | 用途 |
|---|---|---|
| `DEV_SET_ROOT` | `sound_event/`、`noise/`、`room_ir/`、`interference/` 的训练与验证子集，`metadata/valid.json` | 在线声景合成与模型训练 |
| `DEV_SET_ROOT` | `synthesized/test/soundscape/`、`synthesized/test/oracle_target/` | 公开验证的输入与参考目标 |
| `WEIGHTS_ROOT` | M2D 预训练权重、`m2dat_4c.ckpt`、`resunetk.ckpt` | 特征提取与参考模型推理；可挂载或用 `scripts.download_weights` 获取 |
| `WORK_ROOT` | 学生自选的可写目录 | 配置副本、训练日志、检查点和预测 |

运行 `python verify.py --source_dir . --data-root "$DEV_SET_ROOT" --checkpoint-root "$WEIGHTS_ROOT"` 检查结构。若挂载的开发集缺少 `interference/train` 或 `interference/valid`，请先核对课程资源是否挂载完整；需要自行生成时可用 `python -m scripts.prepare_interference --dataset-root "$DEV_SET_ROOT" --storage-dir /path/to/large-raw-storage`。脚本会下载大型原始归档，需确认网络、存储容量和语料许可。参考权重推理与两阶段从头训练所需资产并不完全相同；完整训练前应通过上述结构检查。
