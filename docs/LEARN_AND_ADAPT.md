# Continuous learning / adaptation

After each solid chat or assist action:

1. Session log to `data/conversations/*.jsonl`
2. Trainable blocks to `data/train/chat/conversation_learned.txt` plus RAG index
3. Online micro-SFT in the background on enterprise text weights when new pairs accumulate
4. Checkpoint mtime reload so later answers use adapted weights

```text
python -m navine.cli memory status
python -m navine.cli memory adapt
python -m navine.cli memory ingest
```

Corrections like "Actually the answer is ..." are prioritised.
Garbled model outputs are not learned.
