# Param counts (true ~760M)

Verified unique trainable weights in the **740M-780M** band.

| Model | Params |
|-------|--------|
| text_enterprise | 762,446,400 |
| text_code | 762,446,400 |
| image_enterprise_v2 | ~765.7M |
| video_enterprise | ~765.9M |
| voice | ~757.1M |
| deepfake | ~768.4M |

Proof: `logs/verify_true_760m.json`
Rebuild: `python scripts/rebuild_true_760m.py`
