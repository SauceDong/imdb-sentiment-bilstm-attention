# IMDB Sentiment Classification with BiLSTM-Attention

基于 PyTorch 的 IMDB 影评情感二分类课程项目。模型使用双向 LSTM 提取上下文特征，通过 Attention Pooling 聚合序列信息，再使用 LayerNorm 和三层全连接分类器完成预测。

## 模型结构

```text
Input [B, 200]
  → Embedding [B, 200, 256]
  → BiLSTM [B, 200, 512]
  → Attention Pooling [B, 512]
  → LayerNorm
  → FC 512 → 256 → 128 → 2
  → LogSoftmax
```

主要参数：

| 参数 | 数值 |
|---|---:|
| `MAX_WORDS` | 25,000（数据中的最大 token ID 为 9,999） |
| `MAX_LEN` | 200 |
| `BATCH_SIZE` | 256 |
| `EMB_SIZE` | 256 |
| `HID_SIZE` | 256 |
| 外部 Dropout | 0.5 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Early-stopping patience | 8 |

## 数据集

仓库中的 [`data/imdb_preprocessed_data.zip`](data/imdb_preprocessed_data.zip) 包含四个预处理后的 NumPy 文件：

| 文件 | 形状 | 说明 |
|---|---:|---|
| `x_train.npy` | `(31818, 200)` | 训练输入 |
| `y_train.npy` | `(31818,)` | 训练标签 |
| `x_val.npy` | `(13636, 200)` | 验证输入 |
| `y_val.npy` | `(13636,)` | 验证标签 |

标签 `0` 表示负面评价，`1` 表示正面评价。数据来源与发布说明见 [`data/README.md`](data/README.md)。

## 快速开始

```bash
git clone https://github.com/SauceDong/imdb-sentiment-bilstm-attention.git
cd imdb-sentiment-bilstm-attention

python scripts/prepare_data.py
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/imdb_sentiment_bilstm_attention.ipynb
```

Notebook 可在 CPU 或 CUDA GPU 上运行。训练产生的权重默认保存为 `models/model.pth`，该文件不会提交到 Git。

## 实验结果

最终 Notebook 的原始运行记录为：

- 最佳验证准确率：**88.80%**
- 正确分类：**12,109 / 13,636**
- 最佳轮次：第 11 轮
- 第 19 轮触发早停

汇报材料记录了另一轮实验的 88.92%（12,125 / 13,636，第 15 轮）以及 pooling/FC head 消融结果。由于对应的完整 checkpoint 与消融代码未保留，本仓库把 Notebook 中可追溯的 **88.80%** 作为主结果。具体差异见 [`report/NOTES.md`](report/NOTES.md)。

## 项目文件

```text
.
├── README.md
├── requirements.txt
├── data/
│   ├── README.md
│   └── imdb_preprocessed_data.zip
├── notebooks/
│   └── imdb_sentiment_bilstm_attention.ipynb
├── report/
│   ├── NOTES.md
│   └── presentation.pdf
├── scripts/
│   └── prepare_data.py
└── models/
    └── README.md
```

## 已知限制

- Attention 尚未屏蔽 padding token；可进一步结合 `padding_idx`、`pack_padded_sequence` 和 masked attention。
- 单层 LSTM 不使用 PyTorch LSTM 模块内部的 dropout；正则化由 Embedding、LSTM 输出和全连接层外部的 Dropout 完成。
- 训练结果可能因 PyTorch/CUDA 版本和硬件而有小幅波动。
- 当前未提供与本 Notebook 完全匹配的模型权重，需要重新训练。

## 环境

原始实验使用 Python 3.11.5、PyTorch 2.4.1 + CUDA 12.1。本仓库的依赖采用兼容范围，便于在较新的 CPU、CUDA 或 Apple Silicon 环境中安装。

