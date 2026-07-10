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

Create `data/manifest.jsonl` with one object per line:

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
