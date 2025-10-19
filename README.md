# 🎬 Xiaoice Video LLM: 与你的视频对话

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-ff69b4.svg)](https://streamlit.io/)

这是一个基于论文 《通过自监督时空语义特征聚类实现免训练的视频理解》 的交互式Web应用。它模拟了一个视频语言模型（Video LLM），允许用户上传视频，并能围绕视频内容进行智能对话。

本项目的核心思想是完全免训练。它不依赖任何标注过的视频数据集，而是巧妙地结合了强大的预训练视觉模型（如CLIP）与经典的无监督学习算法，自动地为视频生成结构化的语义摘要，并以此为基础赋能大语言模型（GPT-4o）进行视频内容的理解和问答。

## ⚙️ 技术实现

本项目严格遵循了原论文提出的四阶段分析流程：

1.  **语义特征轨迹提取 (Semantic Feature Trajectory Extraction)**
    *   使用 `OpenCV` 对上传的视频按指定帧率进行采样。
    *   利用 `sentence-transformers` 库加载预训练的 `CLIP` 模型，将每一帧图像编码为高维语义特征向量，形成一个时间序列。

2.  **事件片段识别 (Event Segment Identification)**
    *   实现了完整的核时序分割 (Kernel Temporal Segmentation, KTS) 算法。该算法通过动态规划在帧间的余弦相似度矩阵上寻找最佳分割点，将视频流切分为一系列语义上连贯的短片段。

3.  **场景发现 (Scene Discovery)**
    *   对每个事件片段的特征向量进行平均池化，得到其代表性向量。
    *   使用 `scikit-learn` 中的 DBSCAN 聚类算法对这些片段向量进行无监督聚类。每个生成的簇（Cluster）代表一个在视频中反复出现的宏观场景。

4.  **结构化摘要与对话 (Structured Summary & Dialogue)**
    *   为每个发现的场景，计算其聚类中心，并找到最能代表该场景的关键帧。
    *   利用 GPT-4o 的多模态能力，为每个关键帧生成简洁的文本描述。
    *   将所有场景的关键帧和描述整合成一份结构化的JSON摘要。
    *   当用户提问时，通过精心设计的提示工程，将这份摘要作为上下文注入到GPT-4o的提示中，引导模型扮演一个“已经观看过视频”的AI助手角色，并基于摘要内容进行回答。

## 🚀 快速开始

### 1. 环境准备

首先，克隆本仓库到本地：
```bash
git clone https://github.com/imbue-bit/Xiaoice.git
cd Xiaoice
```

然后，创建一个Python虚拟环境并安装所需的依赖包：
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install streamlit langchain langchain_openai openai opencv-python scikit-learn numpy sentence-transformers Pillow
```

### 2. 配置API密钥

本项目需要使用OpenAI的API。请将您的API密钥设置为环境变量。

**在Linux或macOS上:**
```bash
export OPENAI_API_KEY="sk-xxxxxxxx"
```

**在Windows上:**
```bash
set OPENAI_API_KEY="sk-xxxxxxxx"
```

### 3. 运行应用

在项目根目录下，运行以下命令：
```bash
streamlit run app.py
```

您的浏览器将自动打开一个新标签页，地址为 `http://localhost:8501`，即可开始使用。

## 📖 使用指南

1.  **上传视频**: 在左侧的侧边栏，点击“上传你的视频文件”按钮，选择一个 `.mp4`, `.mov`, 或 `.avi` 格式的视频。为了获得最佳的性能和体验，建议使用长度在1到5分钟之间的视频。
2.  **调整参数 (可选)**:
    *   **采样率**: 控制每秒分析的帧数。较高的值会更精确，但处理时间更长。
    *   **KTS期望片段数**: 设定你希望将视频分割成的基础事件数量。这个值会影响最终场景的数量和大小。
3.  **开始分析**: 点击“🚀 开始分析视频”按钮。应用会显示详细的处理进度。
4.  **查看摘要**: 分析完成后，主界面会以卡片形式展示所有发现的宏观场景、关键帧和AI生成的描述。
5.  **开始对话**: 在页面底部的聊天框中，输入你关于视频内容的问题（例如：“视频里出现了几个人？”或“视频的主要情绪是怎样的？”），然后按回车键，AI将会回答你。
