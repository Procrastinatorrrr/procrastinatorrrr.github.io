---
title: "About"
date: 2026-04-21T15:24:40+08:00
lastmod: 2026-04-21T15:24:40+08:00
author: ["Yijun Long"]

description: "" # 文章描述，与搜索优化相关
summary: "" # 文章简单描述，会展示在主页
weight: # 输入1可以顶置文章，用来给文章展示排序，不填就默认按时间排序
slug: ""
draft: false # 是否为草稿
comments: true
showToc: false # 显示目录
TocOpen: true # 自动展开目录
autonumbering: true # 目录自动编号
hidemeta: true # 是否隐藏文章的元信息，如发布日期、作者等
disableShare: true # 底部不显示分享栏
searchHidden: true # 该页面可以被搜索到
showbreadcrumbs: true #顶部显示当前路径
mermaid: true
cover:
    image: ""
    caption: ""
    alt: ""
    relative: false
---

**Long Yijun (龙奕均)** | Male | 22 years old | 📧 <procrastinator@mail.ustc.edu.cn> | [Github](https://github.com/Procrastinatorrrr)

## Education

**Master**: University of Science and Technology of China · Computer Science & Technology
<span style="float: right;">**Time**: 2026.09 --</span>

**Bachelor**：University of Science and Technology of China · Computer Science & Technology
<span style="float: right;">**Time**: 2022.09 -- 2026.08</span>

## Project Experience

### RL-based Search-Augmented LLM Training Framework

<span style="float: right;">**Time**: 2025.10 -- 2026.02</span>

**Tech Stack**: LLM, Agentic RL, RLVR, GRPO, veRL, Ray, FSDP, vLLM

Addressed LLM knowledge cutoff and hallucination issues by training LLMs to autonomously use search engines via reinforcement learning (RLVR), enabling multi-turn "think-search-reason" loop.

- Built distributed RLVR training architecture using veRL with Ray distributed scheduling, FSDP training, and vLLM inference
- Integrated search engine as external component into RL training with multi-dimensional reward mechanisms
- Added LLDS penalty to GRPO loss to stabilize Agentic-RL training and prevent entropy explosion
- Achieved 42.7% improvement over RAG baseline on Qwen2.5-3b-base model

### MedSketcher——Interactive Medical Image Segmentation Platform

<span style="float: right;">**Time**: 2024.05 -- 2025.10</span>

**Tech Stack**: PyTorch, SAM, FastAPI, Vue3, VTK.js, Cornerstone.js, WebGL

Developed a Human-in-the-loop intelligent annotation system to reduce medical image annotation cost and barrier.

- Fine-tuned MedSAM with LoRA for high-precision real-time segmentation
- Built 3D volume rendering engine with double-buffering and LUT optimization for low-latency interaction
- Developed FastAPI async inference service with DICOM/NIfTI parsing and task queue management

## Research Experience

### Tool Call Noise Injection for Robustness Evaluation of Code Generation Agents

<span style="float: right;">**Time**: 2025.12 -- 2026.03</span>

**First Author** | Under Review

Proposed tool call noise injection to evaluate code generation agents' robustness by injecting controllable failures and perturbations into tool execution. Experiments on SWE-bench-lite reveal significant performance degradation under tool noise.

## Skills

- **Languages**: CET-4 (570), CET-6 (626)
- **Programming**: Python, C/C++, JavaScript/TypeScript, SQL
- **AI/LLM**: LLM architecture; PyTorch, SFT, RL experience; PPO/DPO/GRPO/DAPO; FSDP/Deepspeed, vLLM/aglang, veRL/Slime
- **AI Agent**: RAG/MCP/Agent Skills
- **Web Full Stack**: Vue, FastAPI
- **AI Coding Tools**: Extensive vibe coding experience
