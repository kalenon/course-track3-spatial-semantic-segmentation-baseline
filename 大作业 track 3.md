# 《语音信号处理》大作业赛题 3：多通道空间声景语义分割

## 1. 赛题背景

真实声景中，多个声源通常在不同方向同时活动。例如，室内可能同时出现说话声、敲击声和门铃声，且同一类声源也可能来自不同空间位置。要实现沉浸式通信、机器人听觉或智能声学感知，系统不仅要判断“有什么声音”，还需要从多通道录音中分离出每一个目标声源。

本赛题要求构建一个**多通道空间声景语义分割系统**：给定一段包含目标声源、干扰声和背景噪声的空间声景，系统需要识别其中的目标声音类别和数量，并为每一个被识别的目标输出对应的单通道分离波形。

本 Track 重点考查以下课程知识：

- 多通道语音 / 音频信号表示与空间声学；
- 声音事件分类与多标签检测；
- 盲源分离、条件源分离；
- 分离质量与检测性能的联合评价。

## 2. 任务定义

设输入为 $M=4$ 通道的声景混合波形：

$$
\boldsymbol{Y}=[\boldsymbol{y}^{(1)},\ldots,\boldsymbol{y}^{(M)}]^\top
\in\mathbb{R}^{M\times T}.
$$

其中，$T$ 为采样点数。混合声景最多包含 $K_{\max}=3$ 个目标声音事件，此外可包含非目标干扰事件和空间背景噪声。第 $m$ 个通道可概念化地表示为：

$$
\boldsymbol{y}^{(m)}=
\sum_{k=1}^{K}\boldsymbol{h}^{(m)}_k*\boldsymbol{s}_{c_k}
+
\left[
\sum_{j=1}^{J}\boldsymbol{h}^{(m)}_j*\boldsymbol{s}_{c_j}
+\boldsymbol{n}^{(m)}
\right]_{\mathrm{optional}},
$$

其中，$\boldsymbol{s}_{c_k}$ 表示目标事件波形，$c_k$ 表示其类别，$\boldsymbol{h}^{(m)}_k$ 表示到第 $m$ 个通道的房间脉冲响应，$\boldsymbol{n}^{(m)}$ 表示背景噪声。

系统输出预测事件集合：

$$
\hat{\mathcal{E}}=
\{(\hat{c}_q,\hat{\boldsymbol{s}}_q)\}_{q=1}^{\hat{K}},
\qquad \hat{K}\in\{0,1,2,3\},
$$

其中，$\hat{c}_q$ 是预测类别，$\hat{\boldsymbol{s}}_q$ 是对应的单通道分离波形。

需要特别注意：

- 同一类别的多个目标事件可以同时出现；它们来自不同空间方向，课程数据中同类目标的方向间隔不小于 $60^\circ$；
- 输入可能不包含任何目标事件；此时系统必须输出空事件集合，不能生成虚假的目标波形；
- 每个输出仅对应一个目标事件，不应把多个目标混合到同一输出中；
- 系统应利用四通道输入中的空间信息，但不限制采用波束形成、空间特征、端到端网络或其他方法。

### 2.1 目标声音类别

课程数据包含以下 18 个目标类别。输出标签必须严格使用表中的英文名称。

| 类别 | 标签 | 类别 | 标签 |
|---:|---|---:|---|
| 1 | `AlarmClock` | 10 | `FootSteps` |
| 2 | `BicycleBell` | 11 | `HairDryer` |
| 3 | `Blender` | 12 | `MechanicalFans` |
| 4 | `Buzzer` | 13 | `MusicalKeyboard` |
| 5 | `Clapping` | 14 | `Percussion` |
| 6 | `Cough` | 15 | `Pour` |
| 7 | `CupboardOpenClose` | 16 | `Speech` |
| 8 | `Dishes` | 17 | `Typing` |
| 9 | `Doorbell` | 18 | `VacuumCleaner` |

## 3. 输入与输出规范

### 3.1 输入音频

所有输入音频满足以下规范：

| 项目 | 规范 |
|---|---|
| 文件格式 | WAV，PCM 16 bit |
| 采样率 | 32,000 Hz |
| 声道数 | 4 |
| 通道格式 | 一阶 Ambisonics（FOA）B-format，通道顺序 `W, Y, Z, X` |
| 单条时长 | 10 s |
| 采样点数 | 每通道 320,000 |
| 输入文件名 | `<mixture_id>.wav` |

系统不得改变输入文件的时长或采样率后再作为最终输出参考。允许在模型内部使用重采样、STFT 或其他表示，但提交波形必须恢复为规定格式。

### 3.2 输出事件与分离波形

每个输入混合音频对应一个 JSON 元数据文件；系统检测出的每个目标事件对应一个单通道 WAV 文件。目录结构如下：

```text
predictions/
├── metadata/
│   └── <mixture_id>.json
└── audio/
    ├── <mixture_id>_0_<EventClass>.wav
    ├── <mixture_id>_1_<EventClass>.wav
    └── ...
```

`<mixture_id>.json` 的格式为：

```json
{
  "mixture_id": "example_0001",
  "events": [
    {
      "source_index": 0,
      "class": "Speech",
      "audio": "audio/example_0001_0_Speech.wav"
    }
  ]
}
```

其中：

- `events` 必须是列表，包含 $0$--$3$ 个事件；
- `source_index` 在同一混合音频内从 0 开始且不重复；
- `class` 必须是第 2.1 节规定的 18 个标签之一；
- `audio` 是相对于 `predictions/` 的路径；
- 对于没有目标事件的输入，必须提交 `"events": []`，且不能提供任何对应分离波形；
- 每个分离波形必须为单通道、32 kHz、10 s WAV，采样点数严格为 320,000；
- 分离波形表示课程规定的参考通道目标信号，不要求也不允许输出四通道波形。

课程将提供格式检查脚本。缺少 JSON、JSON 格式错误、类别名错误、无效 WAV、长度或采样率不符的输出，均视为无效预测处理。

## 4. 数据集

### 4.1 训练数据

课程将提供训练数据所需的孤立目标事件、干扰事件、空间房间脉冲响应和四通道背景噪声。训练混合音频可依据课程提供的配置在线合成；参赛队伍也可以构造自己的训练混合策略。

数据的基本属性如下：

| 项目 | 设定 |
|---|---|
| 目标事件类别数 | 18 |
| 每条混合音频中的目标数 | 0--3 |
| 每条混合音频中的干扰事件数 | 0--2 |
| 最大同时重叠事件数 | 3 |
| 目标事件相对背景噪声的 SNR | 均匀采样于 5--20 dB |
| 干扰事件相对背景噪声的 SNR | 均匀采样于 0--15 dB |
| 相同类别目标 | 允许同时出现；空间方向间隔不小于 $60^\circ$ |

训练中实际使用的所有数据、预训练模型和处理过程需在报告中声明。

### 4.2 公开验证集

课程将提供带参考目标波形和事件标签的公开验证集，用于模型选择、阈值调整和接口调试。

### 4.3 隐藏测试集

最终排名使用隐藏测试集。隐藏测试集仅提供四通道混合音频，不公开事件类别、事件数量和参考分离波形。

测试集中包含合成声景和真实录制声景，并覆盖零目标、同类目标重叠、非目标干扰和背景噪声条件。参赛队伍须使用冻结系统生成预测结果；课程组统一运行格式检查与评测程序。

## 5. 基线系统

课程将提供一个可执行的 Baseline，帮助学生跑通数据、训练、推理和评测流程：

1. **事件检测 / 音频标注模块**：从混合声景预测目标事件的类别和数量；
2. **标签条件源分离模块**：以输入声景和预测类别为条件，输出对应目标的单通道分离波形。

Baseline 将提供单通道输入与四通道 FOA 输入两个事件检测变体；分离模型使用置换不变训练处理同一类别多源的匹配歧义。

课程 Baseline 将包含：

- 数据检查和在线混合脚本；
- 事件检测、源分离、端到端推理与结果打包脚本；
- 公开验证集评测与输出格式检查脚本；
- 参考模型权重与配置；
- 计算量、参数量统计脚本。

## 6. 评价指标与排名

### 6.1 SDRi

对于成功匹配的预测波形 $\hat{\boldsymbol{s}}$ 与参考目标波形 $\boldsymbol{s}$，定义信号失真比提升为：

$$
\operatorname{SDRi}(\hat{\boldsymbol{s}},\boldsymbol{s},\boldsymbol{y}_{\mathrm{ref}})
=
\operatorname{SDR}(\hat{\boldsymbol{s}},\boldsymbol{s})
-
\operatorname{SDR}(\boldsymbol{y}_{\mathrm{ref}},\boldsymbol{s}),
$$

$$
\operatorname{SDR}(\hat{\boldsymbol{s}},\boldsymbol{s})
=10\log_{10}
\left(
\frac{\|\boldsymbol{s}\|^2}
{\|\boldsymbol{s}-\hat{\boldsymbol{s}}\|^2}
\right),
$$

其中 $\boldsymbol{y}_{\mathrm{ref}}$ 为课程评测指定的混合参考通道。

### 6.2 类别感知置换不变 SDRi

最终排名指标为**类别感知置换不变 SDRi**（class-aware permutation-invariant SDRi，简称 CAPI-SDRi）。该指标同时衡量事件检测是否正确、同类多源是否被正确分开，以及分离波形质量。

对于每个类别 $\bar{c}$，评测程序在该类别的预测源与参考源之间寻找使 SDRi 总和最大的匹配排列。记正确匹配数、漏检数和误检数分别为 $N^{\bar{c}}_{\mathrm{TP}}$、$N^{\bar{c}}_{\mathrm{FN}}$ 与 $N^{\bar{c}}_{\mathrm{FP}}$，则该类别的累计分数为：

$$
P^{\bar{c}}
=N^{\bar{c}}_{\mathrm{FN}}\mathcal{P}_{\mathrm{FN}}
+N^{\bar{c}}_{\mathrm{FP}}\mathcal{P}_{\mathrm{FP}}
+\max_{\sigma,\pi}
\sum_{i=1}^{N^{\bar{c}}_{\mathrm{TP}}}
\operatorname{SDRi}
\left(
\hat{\boldsymbol{s}}^{\bar{c}}_{\sigma(i)},
\boldsymbol{s}^{\bar{c}}_{\pi(i)},
\boldsymbol{y}_{\mathrm{ref}}
\right).
$$

本赛题中，漏检和误检的惩罚均固定为：

$$
\mathcal{P}_{\mathrm{FN}}=\mathcal{P}_{\mathrm{FP}}=0\ \mathrm{dB}.
$$

单个混合音频的 CAPI-SDRi 为所有出现于参考或预测集合中的类别分数的平均值：

$$
\mathrm{CAPI\!-\!SDRi}
=
\frac{
\sum_{\bar{c}\in\mathcal{C}\cup\hat{\mathcal{C}}}P^{\bar{c}}
}{
\sum_{\bar{c}\in\mathcal{C}\cup\hat{\mathcal{C}}}N^{\bar{c}}
}.
$$

对于零目标混合音频：

- 若系统正确输出空事件集合，该样本没有可评价的目标，因而不计入 CAPI-SDRi 的平均；
- 若系统错误输出任意事件，将作为误检受到惩罚。

最终排行榜按隐藏测试集全部有效混合音频的平均 CAPI-SDRi 从高到低排序。

### 6.3 辅助指标

课程组还将计算以下辅助指标，用于结果分析，不作为最终排名的直接依据：

- 事件标签的 Accuracy、Recall、Precision 与 F1；
- 语音类目标的 PESQ、STOI；
- 非语音类目标的感知质量指标；
- 类别感知源分离分析指标与典型样例试听结果。

## 7. 提交要求

每支队伍提交一个压缩包：

```text
team_<编号>/
├── README.md
├── requirements.txt
├── configs/
├── src/
├── scripts/
│   ├── train_tagger.py
│   ├── train_separator.py
│   ├── infer.py
│   ├── validate_submission.py
│   └── complexity.py
├── checkpoints/
├── predictions/
│   ├── metadata/
│   └── audio/
├── logs/
│   ├── validation_metrics.json
│   └── complexity.json
└── team_<编号>_report.pdf
```

其中：

- `README.md`：环境安装、模型下载与完整复现命令；
- `scripts/infer.py`：输入混合音频目录，输出符合第 3.2 节格式的预测目录；
- `checkpoints/`：最终系统使用的全部参数；
- `predictions/`：隐藏测试集的完整预测结果；
- `logs/complexity.json`：参数量、计算量和统计命令；
- `team_<编号>_report.pdf`：课程大作业报告。

## 8. 报告要求

报告正文建议包含：

1. 摘要；
2. 任务分析与系统总体流程；
3. 训练数据、空间混合策略与数据划分；
4. 多通道输入特征或空间建模方法；
5. 事件检测模块与零目标判定策略；
6. 标签条件源分离模块与同类多源处理方法；
7. 损失函数、训练设置与阈值选择；
8. 公开验证集结果，至少包含 CAPI-SDRi、检测指标和分条件分析；
9. 参数量、计算量、失败案例和局限；
10. 队伍成员分工、第三方数据 / 代码 / 模型声明及参考文献。

## 9. 课程评分建议

最终课程成绩建议由以下部分构成，课程发布时以授课教师公布的比例为准：

| 项目 | 建议比例 |
|---|---:|
| 隐藏测试集 CAPI-SDRi 排名表现 | 80% |
| 报告质量与成员答辩 | 20% |


## 附录 A：提交前自查表

- [ ] 输入按 `W, Y, Z, X` 顺序读取 4 通道 32 kHz FOA WAV；
- [ ] 每个混合音频预测 0--3 个目标事件；
- [ ] 输出类别名严格属于第 2.1 节的 18 个标签；
- [ ] 同类多个目标分别输出独立波形和独立事件条目；
- [ ] 零目标混合音频的 JSON 使用空 `events` 列表；
- [ ] 每个输出 WAV 均为单通道、32 kHz、10 s、320,000 个采样点；
- [ ] `metadata` JSON 与分离 WAV 的路径、编号和类别完全一致；
- [ ] 公开验证集和隐藏测试集未被用于训练或测试时自适应；
- [ ] 代码可在课程指定环境中从 checkpoint 完整复现；
- [ ] 报告完整声明数据、代码、模型和权重来源。
