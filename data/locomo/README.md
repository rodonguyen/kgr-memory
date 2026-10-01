# LoCoMo

The published eval set is one file: 10 conversations (`locomo10.json`, about 2.7 MB). There is no separate small or medium split. Papers that report LoCoMo use this file. JSON is gitignored.

A 10-turn slice of session 1, for the chat button, is committed at `kgr_memory/fixtures/locomo_session1_10.json`.

```bash
mkdir -p data/locomo
curl -L --fail -o data/locomo/locomo10.json \
  "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
```

If that URL fails, copy `locomo10.json` from a local clone that already has it (AgenticMemory, MemMachine, MemoryOS).
