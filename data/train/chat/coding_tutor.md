### User: Teach me how to write a safe file append helper in Python.
### Assistant: Open the path with encoding utf-8 in append mode. Resolve the path under an allowed root. Reject `..` segments and absolute escapes. Write the text then flush. Raise clear errors on permission or missing parent directory instead of silent failure.

### User: Debug a TypeError: NoneType is not subscriptable in a dataloader.
### Assistant: Find where a field became None before indexing. Add an early continue when the sample is empty. Log the source path of bad rows. Prefer optional chaining patterns like `(entry or {}).get("code") or ""` so mixed JSONL remains train-safe.

### User: Write a minimal transformer LM train loop sketch.
### Assistant: Tokenize text to ids, pack into x and next-token y. Run model(x), cross-entropy against y with pad ignore_index. Clip grads, AdamW step, cosine LR after warmup. Every N steps evaluate val loss and save latest.pt. Prefer AMP and pin_memory when CUDA is available.
