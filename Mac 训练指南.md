# MiniMind M4 本地训练指南

## 环境配置（已完成）

✅ PyTorch 2.8.0 - 支持 MPS 加速
✅ Transformers 4.57.6
✅ Datasets 3.6.0
✅ 测试数据集已生成

---

## 快速开始

### 步骤 1：验证环境

```bash
cd /Users/scottwang/Documents/Automation/Local\ LLM/minimind
python3 -c "import torch; print('MPS:', torch.backends.mps.is_available())"
# 应输出：MPS: True
```

### 步骤 2：预训练（Pretrain）

```bash
cd trainer
python3 train_pretrain.py \
    --data_path ../dataset/pretrain_t2t_mini.jsonl \
    --batch_size 4 \
    --hidden_size 384 \
    --num_hidden_layers 4 \
    --epochs 2 \
    --device mps \
    --save_weight pretrain_m4 \
    --use_compile 0
```

**参数说明：**
- `--batch_size 4`：M4 16GB 内存建议值，可根据情况调整为 2 或 8
- `--hidden_size 384`：隐藏层维度（原默认 768）
- `--num_hidden_layers 4`：Transformer 层数（原默认 8）
- `--epochs 2`：训练轮数
- `--use_compile 0`：MPS 不支持 torch.compile，必须设为 0

**预计时间：** 2-4 小时
**输出文件：** `../out/pretrain_m4_384.pth`

---

### 步骤 3：监督微调（SFT）

```bash
python3 train_full_sft.py \
    --from_weight ../out/pretrain_m4_384.pth \
    --data_path ../dataset/sft_t2t_mini.jsonl \
    --batch_size 2 \
    --hidden_size 384 \
    --num_hidden_layers 4 \
    --epochs 3 \
    --device mps \
    --save_weight full_sft_m4 \
    --use_compile 0
```

**预计时间：** 3-5 小时
**输出文件：** `../out/full_sft_m4_384.pth`

---

### 步骤 4：推理测试

```bash
cd ..
python3 eval_llm.py \
    --load_from ./model \
    --weight full_sft_m4 \
    --hidden_size 384 \
    --num_hidden_layers 4 \
    --device mps
```

---

## 模型配置参考

| 配置 | hidden_size | num_hidden_layers | 参数量 | 适用场景 |
|------|-------------|-------------------|--------|----------|
| 超小 | 256 | 4 | ~15M | 快速测试 |
| 小型 | 384 | 4 | ~35M | 学习训练 ⭐ |
| 标准 | 512 | 6 | ~80M | 追求效果 |
| 原版 | 768 | 8 | ~64M* | 参考对比 |

*注：原版 64M 使用词表 6400，实际参数量因配置而异

---

## 训练进度监控

训练过程中会输出：
```
Epoch:[1/2](100/500), loss: 2.8453, logits_loss: 2.8401, aux_loss: 0.0052, lr: 0.00048523, epoch_time: 15.2min
```

- **loss**：损失值，越低越好（通常从 3.0+ 降至 1.5 以下）
- **lr**：当前学习率
- **epoch_time**：当前轮次预计剩余时间

---

## 常见问题

### Q1: 内存不足（OOM）
**解决方案：**
1. 减小 `batch_size`（4 → 2 → 1）
2. 减小模型维度（hidden_size 384 → 256）
3. 减少序列长度（在代码中修改 max_seq_len）

### Q2: 训练速度慢
**说明：** MPS 相比 CUDA 确实较慢，这是正常的
- 3090 GPU: 2 小时（原版配置）
- M4 MPS:  5-10 小时（缩小配置后）

### Q3:  loss 不下降
**可能原因：**
1. 学习率过大/过小
2. 数据量太少（目前的测试数据仅用于演示）
3. 模型配置过小

---

## 关键代码学习路径

### 第一阶段：理解模型结构
**文件：** `model/model_minimind.py`

1. **RMSNorm** (第 49-59 行) - 归一化层
2. **Attention** (第 90-132 行) - 注意力机制
3. **FeedForward** (第 134-144 行) - 前馈网络
4. **MOEFeedForward** (第 146-174 行) - 混合专家
5. **MiniMindBlock** (第 176-192 行) - Transformer 块
6. **MiniMindForCausalLM** (第 228-279 行) - 完整模型

### 第二阶段：理解数据处理
**文件：** `dataset/lm_dataset.py`

1. **PretrainDataset** (第 37-55 行) - 预训练数据
2. **SFTDataset** (第 58-119 行) - 指令微调数据

### 第三阶段：理解训练流程
**文件：** `trainer/train_pretrain.py`

1. **train_epoch** (第 23-80 行) - 训练循环
2. **主函数** (第 82-170 行) - 初始化、配置、执行

---

## 下一步学习

完成基础训练后，可以尝试：

1. **LoRA 微调** - `train_lora.py`
   - 参数高效微调，只更新少量参数
   
2. **DPO 偏好优化** - `train_dpo.py`
   - 让模型输出更符合人类偏好

3. **GRPO 强化学习** - `train_grpo.py`
   - DeepSeek-R1 同款算法

4. **模型蒸馏** - `train_distillation.py`
   - 从大模型学习知识到小模型

---

## 有用的命令

```bash
# 查看 GPU 使用（MPS 不适用，但可监控内存）
top -stats pid,cpu,mem

# 查看 Python 进程
ps aux | grep python3

# 中断训练
# 按 Ctrl+C

# 恢复训练（断点续训）
python3 train_pretrain.py --from_resume 1
```

---

**祝训练顺利！** 🚀

如有问题，请参考：
- 原项目 README.md
- 代码注释
- 日志输出