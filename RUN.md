# Run Guide

## 1) Install dependencies
```bash
python3 -m pip install -r requirements.txt
```

## 2) Validate LogMelFilterBanks
Random signal:
```bash
python3 scripts/check_logmel.py
```

With real wav:
```bash
python3 scripts/check_logmel.py --wav-path /absolute/path/to/audio.wav
```

## 3) Run experiments
Full run (downloads SpeechCommands on first run):
```bash
python3 scripts/run_experiments.py \
  --epochs 10 \
  --batch-size 64 \
  --n-mels-list 20,40,80 \
  --groups-list 2,4,8,16 \
  --baseline-n-mels 80 \
  --data-root data \
  --output-dir artifacts
```

Smoke run on a small subset (after dataset is already available):
```bash
python3 scripts/run_experiments.py \
  --epochs 1 \
  --batch-size 32 \
  --n-mels-list 20 \
  --groups-list 2 \
  --baseline-n-mels 20 \
  --max-train-samples 256 \
  --max-val-samples 128 \
  --max-test-samples 128 \
  --no-download \
  --data-root data \
  --output-dir artifacts/smoke
```

## 4) Build report plots
```bash
python3 scripts/plot_results.py --summary-csv artifacts/summary.csv
```

Generated outputs:
- logs: `artifacts/logs/*.csv`
- checkpoints: `artifacts/checkpoints/*.pt`
- summary: `artifacts/summary.csv`
- plots: `artifacts/plots/*.png`
