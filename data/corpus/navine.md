# Navine AI

Navine AI is a local multimodal artificial intelligence system. It runs on your own computer without sending data to the cloud.

## Text Model

The text model is a decoder-only transformer similar to GPT. It uses rotary position embeddings, RMS normalization, and SwiGLU feed-forward layers. The compact tier has about thirteen million parameters and trains on consumer GPUs.

## Training Pipeline

First collect markdown or plain text files in the corpus folder. Train a byte-pair encoding tokenizer on that text. Prepare pretraining and supervised fine-tuning datasets. Pretrain on raw text to learn language structure. Fine-tune on question and answer pairs so the model learns to follow instructions.

## Chat Format

Questions and answers use this format:

### User: What is Navine AI?
### Assistant: Navine AI is a local AI platform for text, image, and video generation.

### User: How do I train the model?
### Assistant: Run train tokenizer, prepare data, pretrain, then SFT fine-tune on chat examples.

## Capabilities

Navine AI can generate text, write code, create images, and produce short videos. Everything runs locally with PyTorch.
