# MiniMind 大模型交互式学习网站

> 🌐 网站地址: https://dreamor.github.io

一个通过交互式动画和图解帮助理解大语言模型原理的教程网站。

## 学习路径

```
📝 分词 → 📊 词嵌入 → 📍 位置编码 → 🎯 注意力机制 → 📐 精度与量化 → 🔄 前馈网络
     ↓
🏗️ 模型架构 → 👥 MoE 混合专家
     ↓
📖 预训练 → 👨‍🏫 SFT监督微调 → 📝 LoRA高效微调 → 👍 DPO偏好优化 → 🎮 GRPO强化学习
     ↓
🔮 推理
```

## 教程内容

### 基础概念
- **分词** (`tokenization.html`) - BPE 算法原理演示
- **词嵌入** (`embedding.html`) - 查表操作与向量可视化
- **位置编码** (`position.html`) - RoPE 旋转编码动画
- **注意力机制** (`attention.html`) - QKV 四步计算动画
- **精度与量化** (`precision.html`) - FP32/FP16/INT8 对比
- **前馈网络** (`forward.html`) - SwiGLU 激活详解

### 模型架构
- **模型架构** (`model_architecture.html`) - Transformer 堆叠
- **MoE** (`moe.html`) - 混合专家机制

### 训练流程
- **预训练** (`pretrain.html`) - 完形填空任务
- **SFT** (`sft.html`) - 监督微调
- **LoRA** (`lora.html`) - 高效微调技术
- **DPO** (`dpo.html`) - 偏好优化
- **GRPO** (`grpo.html`) - 强化学习

### 推理流程
- **推理** (`inference.html`) - KV Cache 加速与自回归生成
- **全流程动画** (`full_pipeline.html`) - 交互式数据流演示

## 本地运行

```bash
# 克隆仓库
git clone https://github.com/dreamor/dreamor.github.io.git
cd dreamor.github.io

# 启动本地服务器
python3 -m http.server 8080

# 打开浏览器访问
open http://localhost:8080
```

## 项目特点

- 🎬 **交互式动画** - 关键概念配有动画演示
- 📖 **通俗易懂** - 生活化类比解释专业概念
- 💻 **代码示例** - PyTorch 代码实现
- 🔄 **完整流程** - 训练、调优、推理全覆盖

## 相关项目

- [MiniMind](https://github.com/jingyaogong/minimind) - 轻量级 LLM 训练框架

## 许可证

MIT License
