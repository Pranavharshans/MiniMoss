# MiniMoss-GD

Lean overfit experiment for a frozen Qwen2.5-0.5B temporal backbone and an
8-step grouped decoder over the 32 RVQ codebooks from MOSS-Audio-Tokenizer.

## Environment

Use Python 3.10 or newer on an A40/A100 VM:

```bash
git clone <YOUR_REPOSITORY_URL> MiniMoss
cd MiniMoss
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest -q
```

The official codec is loaded from
`OpenMOSS-Team/MOSS-Audio-Tokenizer` with Hugging Face remote code. The first
run downloads both that checkpoint and `Qwen/Qwen2.5-0.5B`.

## Dataset

Download ten clips from one LibriTTS-R speaker and create the manifest:

```bash
python -m minimoss.prepare_hf_dataset --output-dir data --count 10
```

This produces `data/manifest.jsonl` with one object per line:

```json
{"id":"utt_0001","wav":"/absolute/path/utt_0001.wav","text":"Hello world."}
```

Start with 1-10 clean, single-speaker utterances. Precompute MOSS tokens:

```bash
python -m minimoss.prepare_tokens \
  --manifest data/manifest.jsonl \
  --token-dir data/tokens \
  --device cuda
```

Listen to `data/tokens/_codec_check.wav` before training. Do not continue if
the reconstruction is wrong.

## One-Batch Gate

Run the smallest required training gate:

```bash
python -m minimoss.train_overfit \
  --manifest data/manifest.jsonl \
  --token-dir data/tokens \
  --output-dir checkpoints/one_batch \
  --batch-size 1 \
  --overfit-one-batch
```

Loss should fall strongly. Then decode teacher-forced predictions:

```bash
python -m minimoss.generate \
  --checkpoint checkpoints/one_batch/final.pt \
  --teacher-forced data/tokens/utt_0001.pt \
  --output teacher_forced.wav
```

Free generation has no learned stop token in this lean experiment. Pass the
known target frame count (`duration_seconds * 12.5`) with `--max-frames`:

```bash
python -m minimoss.generate \
  --checkpoint checkpoints/one_batch/final.pt \
  --text "Hello world." \
  --max-frames 50 \
  --output generated.wav
```

Qwen and MOSS-Audio-Tokenizer remain frozen. Training updates the previous-frame
conditioner, global-to-local projection, RVQ embeddings, grouped local decoder,
and output heads.

## 100-Utterance Validation Run

Create a reproducible 90/10 split from a single speaker with enough clips:

```bash
python -m minimoss.prepare_hf_dataset \
  --output-dir data100 \
  --count 100 \
  --speaker-id 84 \
  --validation-count 10 \
  --seed 42

python -m minimoss.prepare_tokens \
  --manifest data100/manifest.jsonl \
  --token-dir data100/tokens \
  --device cuda
```

Train only on the 90-item training manifest:

```bash
mkdir -p logs
python -u -m minimoss.train_overfit \
  --manifest data100/train_manifest.jsonl \
  --token-dir data100/tokens \
  --output-dir checkpoints/train_90 \
  --batch-size 1 \
  --max-steps 5000 \
  --device cuda 2>&1 | tee logs/train_90.log
```

Evaluate the held-out ten items in one process:

```bash
python -m minimoss.evaluate \
  --checkpoint checkpoints/train_90/final.pt \
  --manifest data100/validation_manifest.jsonl \
  --token-dir data100/tokens \
  --output-dir evaluation/validation_10 \
  --device cuda
```

The evaluator writes `01_ground_truth.wav`, `01_teacher.wav`, `01_free.wav`,
and `01.txt` through item 10, plus `evaluation.jsonl` and `summary.json`. Listen
to the ten `*_free.wav` files. Compare a corresponding ground-truth file only
when the free output sounds wrong. Token accuracy is a regression diagnostic,
not a perceptual audio score.

This lean model does not predict duration or EOS. Validation free generation
therefore uses each held-out sample's reference frame count. It tests acoustic
token generalization, but not duration prediction.
