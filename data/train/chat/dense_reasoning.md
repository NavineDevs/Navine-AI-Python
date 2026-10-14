### User: Walk through how hybrid GPU train works end to end.
### Assistant: CUDA holds model weights and matmuls. CPU DataLoader workers load tokens/images with pin_memory when CUDA is present. AMP uses float16 compute on GPU when mixed_precision is true. Checkpoints write to the modality namespace under checkpoints/. Evaluate on a small val split every eval_interval steps, keep best.pt, refresh latest.pt on save intervals.

### User: How should I reason about a bug report?
### Assistant: Restate the expected vs actual behavior. Name the failing surface (train, chat, image, video, assist). Find the last known good checkpoint or config change. Reproduce with the smallest command. Isolate data vs weights vs device mode. Change one variable, re-test, then document the fix.

### User: Explain grad accumulation for small VRAM.
### Assistant: With batch_size 2 and grad_accum_steps 8 you get an effective batch of 16 while only fitting two sequences on the GPU at a time. Loss is divided by the accumulate factor before backward so the summed gradients match full-batch scale. Step the optimizer only every N micro-batches.
