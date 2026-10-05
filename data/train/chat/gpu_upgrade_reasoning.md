# Dense reasoning — GPU upgrade pack

User: Compare RoPE+SwiGLU enterprise text vs the archived 34M net.
Assistant: The live enterprise stack is 10 layers, d_model 640, SwiGLU FFN, RMSNorm, RoPE, and BPE 12k. The archive is 8x512 GELU/LN with a smaller word tokenizer. Prefer the live checkpoint for quality.

User: When should code mode use text_code?
Assistant: If checkpoints/text_code/latest.pt exists and the request is coding-focused, prefer that specialist fork. Otherwise use text_enterprise.
