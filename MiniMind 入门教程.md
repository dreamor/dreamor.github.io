# MiniMind 入门教程

> 从零开始理解大语言模型训练

## 目录

1. [宏观理解：什么是 LLM](#1-宏观理解什么是-llm)
2. [代码整体架构](#2-代码整体架构)
3. [第一步：认识模型结构](#3第一步认识模型结构)
4. [第二步：理解数据处理](#4第二步理解数据处理)
5. [第三步：理解训练流程](#5第三步理解训练流程)
6. [第四步：运行第一个例子](#6第四步运行第一个例子)
7. [核心概念图解](#7核心概念图解)
8. [常见问题解答](#8常见问题解答)

---

## 1. 宏观理解：什么是 LLM

### 1.1 大语言模型本质

**一句话概括：LLM 是一个超级大的函数 F，输入一段文字，输出下一段文字。**

```
输入: "今天天气"
输出: "很好，适合出游"
```

这个函数 F 做的事情是：**根据前面的文字，预测下一个最可能的文字**。

### 1.2 LLM 训练的两个阶段

| 阶段 | 目的 | 比喻 |
|------|------|------|
| **预训练 (Pretrain)** | 让模型学会说话 | 博览群书 |
| **微调 (SFT)** | 让模型听指令 | 请家教 |

**预训练**：
- 输入：海量互联网文本
- 学习：给定前文，预测下一个词
- 结果：模型学会"补完句子"

**监督微调 (SFT)**：
- 输入：问答对 (问答格式的数据)
- 学习：根据问题，回答正确答案
- 结果：模型学会"回答问题"

---

## 2. 代码整体架构

```
minimind/
├── model/                    # 模型定义 (最重要！)
│   └── model_minimind.py     # 核心模型代码 (~2800 行)
│
├── dataset/                  # 数据处理
│   └── lm_dataset.py         # 数据集类
│
├── trainer/                  # 训练脚本
│   ├── train_pretrain.py     # 预训练
│   ├── train_full_sft.py     # 监督微调
│   └── ...
│
├── scripts/                  # 推理工具
│   ├── eval_llm.py           # 评测
│   ├── web_demo.py           # Web 界面
│   └── serve_openai_api.py   # API 服务
│
└── model/                    # 预训练好的 tokenizer
    ├── tokenizer.json        # 词表
    └── tokenizer_config.json # 配置
```

---

## 3. 第一步：认识模型结构

这是最核心的部分！建议反复阅读 `model/model_minimind.py`。

### 3.1 核心组件一览

```
MiniMindForCausalLM (完整模型)
├── MiniMindModel (主体)
│   ├── embed_tokens (词嵌入)
│   ├── layers (N个 Transformer 块)
│   │   └── MiniMindBlock
│   │       ├── self_attn (注意力)
│   │       ├── mlp (前馈网络)
│   │       └── rmsnorm (归一化)
│   └── norm (输出归一化)
│
└── lm_head (输出层)
```

### 3.2 Transformer 块 (Miniblock)

这是 LLM 的核心单元，结构如下：

```
输入 x:
    │
    ▼
┌─────────────────────────────┐
│  RMSNorm (归一化)           │  ← Pre-Norm: 先归一化再处理
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  Self-Attention (注意力)    │  ← 理解词与词的关系
│  - Q K V 投影               │
│  - RoPE 位置编码            │
│  - 注意力计算               │
└─────────────────────────────┘
    │
    ▼ 残差连接
    │
┌─────────────────────────────┐
│  RMSNorm (归一化)           │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  FeedForward (前馈网络)     │  ← 增强表达能力
│  (或 MOEFeedForward)        │     MoE = 多个专家
└─────────────────────────────┘
    │
    ▼ 残差连接
    │
输出 x'
```

### 3.3 注意力机制 (Self-Attention)

**作用：让每个词"看到"句子中的其他词**

```
例子: "猫 坐在 垫子 上"

"猫"  → 看到 "坐在垫子上" (关注这些词)
"坐在" → 看到 "猫垫子上" (关注这些词)
...
```

**实现细节：**

```python
class Attention(nn.Module):
    def __init__(self, config):
        # Q K V 投影层
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim)  # 查询
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim)  # 键
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim)  # 值
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size)  # 输出
```

**Q K V 是什么意思？**

- **Q (Query)**: 我要找什么？
- **K (Key)**: 我有什么？
- **V (Value)**: 实际传递的内容

就像搜索引擎：
- Q = 你搜索的关键词
- K = 网页的关键词
- V = 网页的实际内容

注意力分数 = Q 和 K 的匹配程度，然后用它来加权 V。

### 3.4 RoPE 旋转位置编码

**问题：模型怎么知道词的位置？**

"我打你" vs "你打我" — 意思完全不同！

**传统方法**：给每个位置一个固定的编号
**RoPE 方法**：把位置信息"旋转"到向量里

```python
def apply_rotary_pos_emb(q, k, cos, sin):
    # 旋转公式: q' = q * cos + rotate(q) * sin
    # rotate(x) 把向量的一半和另一半交换
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed
```

### 3.5 前馈网络 (FeedForward)

**作用：为每个词添加非线性变换，增强表达能力**

```python
class FeedForward(nn.Module):
    def __init__(self, config):
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)  # 门控
        self.up_proj = nn.Linear(hidden_size, intermediate_size)    # 上投影
        self.down_proj = nn.Linear(intermediate_size, hidden_size)  # 下投影

    def forward(self, x):
        # SwiGLU 激活函数
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

### 3.6 MoE (混合专家)

**目的：在不增加计算量的情况下增加模型容量**

```
普通 FFN: 一个专家，处理所有 token
MoE:     多个专家，每个 token 只选 top-k 个专家处理
```

```python
class MOEFeedForward(nn.Module):
    def __init__(self, config):
        self.gate = nn.Linear(hidden_size, num_experts)  # 路由器
        self.experts = nn.ModuleList([FeedForward(...) for _ in range(num_experts)])

    def forward(self, x):
        # 1. 路由器决定用哪些专家
        scores = F.softmax(self.gate(x), dim=-1)

        # 2. 选 top-k 专家
        topk_weight, topk_idx = torch.topk(scores, k=num_experts_per_tok)

        # 3. 专家处理并加权求和
        ...
```

---

## 4. 第二步：理解数据处理

### 4.1 分词 (Tokenization)

**问题：计算机怎么"看懂"文字？**

答案：把文字转换成数字

```
"你好世界" → [1024, 3056, 5678]
```

这个转换过程叫**分词 (Tokenization)**，使用的工具叫**分词器 (Tokenizer)**。

MiniMind 使用 BPE (Byte Pair Encoding) 分词器。

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('./model')

# 分词
ids = tokenizer("你好世界").input_ids  # [1024, 3056, 5678]

# 解码
text = tokenizer.decode(ids)  # "你好世界"
```

### 4.2 数据集类

**PretrainDataset** - 预训练数据：

```python
class PretrainDataset(Dataset):
    def __getitem__(self, index):
        # 原始文本
        text = self.samples[index]['text']
        # 转成 token ids
        tokens = tokenizer(text).input_ids
        # 添加特殊 token
        tokens = [bos_id] + tokens + [eos_id]
        # 补齐到固定长度
        input_ids = tokens + [pad_id] * (max_length - len(tokens))
        labels = input_ids.clone()
        return input_ids, labels
```

**目标：让模型学习 Predict Next Token**

```
input_ids:  [BOS] 今天 是    天气    [EOS] [PAD]
labels:     今天 是    天气    [EOS] [PAD] [PAD]
              ↑      ↑      ↑
          预测这个  预测这个  预测这个
```

**SFTDataset** - 监督微调数据：

```python
class SFTDataset(Dataset):
    def __getitem__(self, index):
        # 对话格式: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        conversations = sample['conversations']
        # 使用 chat template 组装
        prompt = tokenizer.apply_chat_template(conversations)
        # 只对 assistant 回复部分计算 loss
        labels = generate_labels(prompt)
        return input_ids, labels
```

---

## 5. 第三步：理解训练流程

### 5.1 训练循环

```python
def train_epoch(epoch, loader):
    for step, (input_ids, labels) in enumerate(loader):
        # 1. 前向传播
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss

        # 2. 反向传播
        loss.backward()

        # 3. 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 4. 参数更新
        optimizer.step()
        optimizer.zero_grad()
```

### 5.2 损失函数

**预训练任务：Next Token Prediction**

```python
def forward(self, input_ids, labels):
    # 得到预测 logits
    logits = self.model(input_ids).logits

    # 计算交叉熵损失
    # 目标: 预测下一个 token
    x = logits[:, :-1, :]  # 除了最后一个
    y = labels[:, 1:]       # 从第二个开始

    loss = F.cross_entropy(x.view(-1, vocab_size), y.view(-1))
    return loss
```

### 5.3 学习率调度

使用**余弦退火**：

```python
def get_lr(step, total_steps, lr):
    # 0.1 + 0.45 * (1 + cos(π * step / total))
    # 从 0.55*lr -> lr -> 0.1*lr 变化
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * step / total_steps)))
```

---

## 6. 第四步：运行第一个例子

### 6.1 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 验证环境
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available())"
```

### 6.2 运行推理

```bash
cd /path/to/minimind
python eval_llm.py --load_from ./model --weight full_sft_m4 --device mps
```

### 6.3 运行训练

```bash
cd trainer

# 预训练
python train_pretrain.py \
    --data_path ../dataset/pretrain_t2t_mini.jsonl \
    --batch_size 4 \
    --hidden_size 384 \
    --num_hidden_layers 4 \
    --save_weight pretrain_m4 \
    --device mps

# 监督微调
python train_full_sft.py \
    --from_weight ../out/pretrain_m4_384.pth \
    --data_path ../dataset/sft_t2t_mini.jsonl \
    --save_weight full_sft_m4 \
    --device mps
```

---

## 7. 核心概念图解

### 7.1 自回归生成

```
Step 1: 输入 "hello" → 模型预测 " world" → 输出 "hello world"
Step 2: 输入 "hello world" → 模型预测 "!" → 输出 "hello world!"
Step 3: 输入 "hello world!" → 模型预测 [EOS] → 结束
```

### 7.2 KV Cache 加速

推理时，保存之前的 K 和 V，避免重复计算：

```
不使用 Cache: 每个 step 都要重新计算所有历史
使用 Cache:   只计算新 token 的 K/V，历史直接复用
```

### 7.3 混合精度训练

```python
# 使用 float16 加速，减少显存
with torch.cuda.amp.autocast(dtype=torch.float16):
    outputs = model(input_ids)
    loss = outputs.loss

# GradScaler 处理梯度缩放
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## 8. 常见问题解答

### Q1: 为什么要用 RMSNorm 而不是 LayerNorm？

RMSNorm 更快（少算一次均值），效果相当。

### Q2: 什么是 GQA？

Grouped Query Attention：多个 Query 头共享一个 Key/Value 头，减少显存。

### Q3: 为什么用 SwiGLU？

比 ReLU/GELU 更平滑，效果更好。

### Q4: 训练时显存不够怎么办？

1. 减小 batch_size
2. 减小 hidden_size
3. 使用梯度累积
4. 使用 LoRA 微调

### Q5: 如何选择模型大小？

| 参数 | 隐藏层 | 层数 | 设备要求 |
|------|--------|------|----------|
| 超小 | 256 | 4 | CPU 也能跑 |
| 小型 | 384 | 4 | M1/M2 Mac |
| 标准 | 512 | 6 | 8G 显存 |
| 原版 | 768 | 8 | 16G+ 显存 |

---

## 推荐学习路径

### 第一阶段：跑通流程
1. 运行 eval_llm.py 测试推理
2. 运行 train_pretrain.py 完成预训练
3. 运行 train_full_sft.py 完成微调

### 第二阶段：理解原理
1. 仔细阅读 model_minimind.py
2. 理解 Attention 的 Q K V 机制
3. 理解 RoPE 位置编码

### 第三步：动手修改
1. 修改模型配置 (hidden_size, layers)
2. 尝试不同训练参数
3. 添加新功能

---

## 相关资源

- [MiniMind GitHub](https://github.com/jingyaogong/minimind)
- [代码说明文档](./代码说明文档.md)
- [M4 训练指南](./M4 训练指南.md)

---

**祝你学习愉快！** 🚀