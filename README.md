# Multimodal LLMs to Classify Cancer Pathology Images

This project develops a custom multimodal large language model (LLM) from scratch to classify cancer pathology images using both visual and textual information. Built for an advanced machine learning course, the pipeline aligns with three major areas: Transformer-based LLMs, fine-tuning techniques, and multimodal modeling. We use the PatchCamelyon and MHIST Binary datasets to drive model development, data analysis, and evaluation.

## Project Goals
- Build a transformer-based language model from scratch for pathology-related tasks
- Design a multimodal system that fuses image features with textual reasoning
- Generate clinical-style textual descriptions of pathology slides to train and evaluate the LLM
- Apply fine-tuning and instruction tuning methods for downstream classification
- Explore retrieval-augmented generation (RAG) and advanced prompt engineering

## Key Features
- Full image and text preprocessing pipeline for multimodal model input
- Custom-built transformer-based LLM trained on synthetic and derived clinical captions
- Multimodal fusion with ViT-based or hybrid vision backbones
- Prompt engineering with at least four techniques, including instruction tuning
- Retrieval-Augmented Generation (RAG) with similarity-based caption retrieval
- Fine-tuning using LoRA and full-model adaptation strategies
- Model optimization through quantization, distillation, and architecture scaling

## Course Alignment

### Transformers & Language Models
- Custom LLM architecture using attention and transformer blocks
- Self-attention mechanisms and transformer encoder-decoder design
- Language model scaling, architecture depth, and tokenization strategies

### Fine-Tuning LLMs
- Supervised Fine-Tuning (SFT) on caption-to-class and image-to-text tasks
- LoRA and QLoRA for parameter-efficient adaptation
- Instruction tuning for structured multimodal reasoning
- Optional: RLHF, adapters, and cost optimization strategies

### Multimodal Models
- Vision encoders (ViT, ResNet) fused with custom LLM
- BLIP-style or dual-encoder fusion mechanisms
- Text generation from images using transformer decoders
- Optional augmentation with GANs or diffusion models for synthetic data

## Datasets
- [PatchCamelyon (PCam)](https://huggingface.co/datasets/1aurent/PatchCamelyon) – histopathologic scans of lymph node sections
- [MHIST Binary](https://huggingface.co/datasets/mamunrobi35/mhist_binary) – colon histology image classification (hyperplastic vs. SSA)

## Project Stages
1. Dataset ingestion and multimodal formatting
2. Image preprocessing and clinical-style caption generation
3. Language model development and training from scratch
4. Vision-language fusion and multimodal architecture design
5. Prompt engineering and RAG integration
6. Fine-tuning and evaluation
7. Model optimization and scaling experiments

## Requirements
- Python 3.10+
- PyTorch, HuggingFace Transformers, OpenCV, Scikit-learn
- GPU support for training custom LLM and vision models
