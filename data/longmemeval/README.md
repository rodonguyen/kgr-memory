# LongMemEval

Official cleaned release: [xiaowu0162/longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned).

**S** is the small history setting from the paper (~40 sessions / ~115k tokens if concatenated). JSON is gitignored (~264 MB).

```bash
mkdir -p data/longmemeval
curl -L --fail -o data/longmemeval/longmemeval_s_cleaned.json \
  "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
```
