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

## LJSpeech 1,000/100 Gate

This is the next generalization experiment after the tiny LibriTTS-R gates. It
keeps the architecture unchanged and uses 1,000 training plus 100 held-out
single-speaker LJSpeech utterances. The preparation command uses the
Parquet-backed `dinhbinh161/ljspeech` mirror because current `datasets`
versions no longer execute the legacy loader in `keithito/lj_speech`.

Prepare the deterministic split and verify its sizes:

```bash
python -m minimoss.prepare_ljspeech \
  --output-dir data_ljspeech_1100 \
  --train-count 1000 \
  --validation-count 100 \
  --seed 42

wc -l \
  data_ljspeech_1100/manifest.jsonl \
  data_ljspeech_1100/train_manifest.jsonl \
  data_ljspeech_1100/validation_manifest.jsonl
```

Tokenize all 1,100 clips once with the MOSS audio tokenizer:

```bash
python -u -m minimoss.prepare_tokens \
  --manifest data_ljspeech_1100/manifest.jsonl \
  --token-dir data_ljspeech_1100/tokens \
  --device cuda 2>&1 | tee logs/prepare_ljspeech_1100.log
```

Listen to `data_ljspeech_1100/tokens/_codec_check.wav` before training. Train
with full held-out validation every 1,000 steps:

```bash
python -u -m minimoss.train_overfit \
  --manifest data_ljspeech_1100/train_manifest.jsonl \
  --validation-manifest data_ljspeech_1100/validation_manifest.jsonl \
  --token-dir data_ljspeech_1100/tokens \
  --output-dir checkpoints/ljspeech_1000 \
  --batch-size 1 \
  --max-steps 20000 \
  --validate-every 1000 \
  --device cuda 2>&1 | tee logs/train_ljspeech_1000.log
```

The trainer writes `best_validation.pt` whenever held-out loss improves. To
resume an interrupted run, retain the original total `--max-steps`:

```bash
python -u -m minimoss.train_overfit \
  --manifest data_ljspeech_1100/train_manifest.jsonl \
  --validation-manifest data_ljspeech_1100/validation_manifest.jsonl \
  --token-dir data_ljspeech_1100/tokens \
  --output-dir checkpoints/ljspeech_1000 \
  --batch-size 1 \
  --max-steps 20000 \
  --validate-every 1000 \
  --resume checkpoints/ljspeech_1000/step_10000.pt \
  --device cuda 2>&1 | tee -a logs/train_ljspeech_1000.log
```

Generate numbered audio for the first 20 held-out clips using the best
validation checkpoint:

```bash
python -u -m minimoss.evaluate \
  --checkpoint checkpoints/ljspeech_1000/best_validation.pt \
  --manifest data_ljspeech_1100/validation_manifest.jsonl \
  --token-dir data_ljspeech_1100/tokens \
  --output-dir evaluation/ljspeech_validation_20 \
  --limit 20 \
  --device cuda 2>&1 | tee logs/evaluate_ljspeech_validation_20.log
```

Listen to `01_free.wav` through `20_free.wav`. Compare the corresponding
teacher and ground-truth files when a free output fails. A successful gate has
recognizable held-out target speech in teacher mode and at least partially
stable target speech in free-running mode. Exact validation RVQ accuracy is a
diagnostic, not a perceptual quality score.

## V2 Alignment Gate

Use this experiment when the fully frozen Qwen run memorizes training audio but
held-out validation remains near random. V2 keeps the grouped local decoder and
MOSS codec unchanged, while enabling two targeted alignment changes:

- Rank-8 LoRA updates on Qwen Q/K/V/O attention projections. Qwen base weights
  remain frozen and are not stored in MiniMoss checkpoints.
- A nonlinear conditioner that concatenates all 32 RVQ embeddings, mixes them
  with an MLP, normalizes them, and matches their scale to Qwen text embeddings.

Run V2 from scratch in a new output directory. Do not resume a V1 checkpoint:

```bash
python -u -m minimoss.train_overfit \
  --manifest data_ljspeech_1100/train_manifest.jsonl \
  --validation-manifest data_ljspeech_1100/validation_manifest.jsonl \
  --token-dir data_ljspeech_1100/tokens \
  --output-dir checkpoints/ljspeech_1000_v2 \
  --batch-size 4 \
  --max-steps 5000 \
  --validate-every 250 \
  --qwen-lora \
  --qwen-lora-rank 8 \
  --qwen-lora-alpha 16 \
  --nonlinear-frame-conditioner \
  --text-diagnostics \
  --early-stopping-patience 4 \
  --device cuda 2>&1 | tee logs/train_ljspeech_1000_v2.log
```

The text diagnostic compares normal validation loss with validation after
reversing text assignments within each batch. A positive and growing `delta`
indicates that predictions depend on the correct text. A near-zero delta means
the model is still ignoring text. Early stopping ends the run after four
validation checks without a new best loss.

Evaluate the best V2 checkpoint:

```bash
python -u -m minimoss.evaluate \
  --checkpoint checkpoints/ljspeech_1000_v2/best_validation.pt \
  --manifest data_ljspeech_1100/validation_manifest.jsonl \
  --token-dir data_ljspeech_1100/tokens \
  --output-dir evaluation/ljspeech_validation_v2 \
  --limit 20 \
  --device cuda 2>&1 | tee logs/evaluate_ljspeech_validation_v2.log
```

## V3 Forced-Alignment Curriculum

Use V3 when V2's shuffled-text loss is effectively identical to normal
validation loss. V3 prevents the model from relying exclusively on the correct
previous audio frame:

- Every frame receives a learned absolute frame-position embedding.
- Previous-audio context is replaced by a learned null vector for the first
  curriculum phase, then gradually restored.
- Validation reports normal, shuffled-text, and no-context losses.
- Early stopping does not begin until the alignment curriculum is nearly done.

On a 24 GB GPU where V2 batch 4 used about 10 GB, start with physical batch 8.
This keeps approximately the same 20,000 total sample exposures in 2,500
optimizer steps:

```bash
python -u -m minimoss.train_overfit \
  --manifest data_ljspeech_1100/train_manifest.jsonl \
  --validation-manifest data_ljspeech_1100/validation_manifest.jsonl \
  --token-dir data_ljspeech_1100/tokens \
  --output-dir checkpoints/ljspeech_1000_v3_b8 \
  --batch-size 8 \
  --max-steps 2500 \
  --validate-every 125 \
  --qwen-lora \
  --qwen-lora-rank 8 \
  --qwen-lora-alpha 16 \
  --nonlinear-frame-conditioner \
  --frame-position-embedding \
  --context-dropout-warmup-steps 500 \
  --context-dropout-decay-steps 1500 \
  --context-dropout-start 1.0 \
  --context-dropout-end 0.2 \
  --text-diagnostics \
  --early-stopping-start-step 2000 \
  --early-stopping-patience 4 \
  --device cuda 2>&1 | tee logs/train_ljspeech_1000_v3_b8.log
```

If batch 8 runs out of memory, use batch 6 with exposure-equivalent schedule:

```bash
python -u -m minimoss.train_overfit \
  --manifest data_ljspeech_1100/train_manifest.jsonl \
  --validation-manifest data_ljspeech_1100/validation_manifest.jsonl \
  --token-dir data_ljspeech_1100/tokens \
  --output-dir checkpoints/ljspeech_1000_v3_b6 \
  --batch-size 6 \
  --max-steps 3334 \
  --validate-every 167 \
  --qwen-lora \
  --qwen-lora-rank 8 \
  --qwen-lora-alpha 16 \
  --nonlinear-frame-conditioner \
  --frame-position-embedding \
  --context-dropout-warmup-steps 667 \
  --context-dropout-decay-steps 2000 \
  --context-dropout-start 1.0 \
  --context-dropout-end 0.2 \
  --text-diagnostics \
  --early-stopping-start-step 2667 \
  --early-stopping-patience 4 \
  --device cuda 2>&1 | tee logs/train_ljspeech_1000_v3_b6.log
```

For V3, a useful result requires the shuffled-text delta to become clearly
positive while no-context validation improves. A near-zero shuffled-text delta
after the full-context-drop phase means the architecture still has no usable
text-to-frame alignment signal.

## V4 Gated Group Curriculum

V4 isolates the text-dependent coarse RVQ prediction before allowing
teacher-forced refinement groups to influence optimization:

- Phase A, steps 1-750: group 1 only, weights `1,0,0,0,0,0,0,0`.
- Phase B, steps 751-1500: groups 1-4, weights `4,1,1,1,0,0,0,0`.
- Phase C, steps 1501-3000: all groups, weights `4,1,1,1,1,1,1,1`.
- Previous-frame context stays fully hidden through phases A and B.
- Phase checkpoints use no-context validation in A/B and normal inference
  context in C.
- Per-codebook validation exposes whether codebooks 1-4 can be grouped safely.

Run from scratch with batch 8:

```bash
python -u -m minimoss.train_overfit \
  --manifest data_ljspeech_1100/train_manifest.jsonl \
  --validation-manifest data_ljspeech_1100/validation_manifest.jsonl \
  --token-dir data_ljspeech_1100/tokens \
  --output-dir checkpoints/ljspeech_1000_v4_b8 \
  --batch-size 8 \
  --max-steps 3000 \
  --validate-every 125 \
  --qwen-lora \
  --qwen-lora-rank 8 \
  --qwen-lora-alpha 16 \
  --nonlinear-frame-conditioner \
  --frame-position-embedding \
  --context-dropout-warmup-steps 1500 \
  --context-dropout-decay-steps 1000 \
  --context-dropout-start 1.0 \
  --context-dropout-end 0.2 \
  --group-curriculum \
  --phase-a-end 750 \
  --phase-b-end 1500 \
  --phase-gate-min-improvement 0.02 \
  --phase-gate-min-text-delta 0.005 \
  --text-diagnostics \
  --early-stopping-start-step 1625 \
  --early-stopping-patience 4 \
  --device cuda 2>&1 | tee logs/train_ljspeech_1000_v4_b8.log
```

At each phase boundary, training emits `PHASE_A_PASS/FAIL` or
`PHASE_B_PASS/FAIL`. A failed phase stops immediately. Phase A writes
`best_phase_a.pt`, phase B writes `best_phase_b.pt`, and phase C writes
`best_validation.pt`. Do not decode phase-A audio because refinement heads are
intentionally untrained.

The improvement gate compares the first validation in a phase with that
phase's best checkpoint, not necessarily its final boundary value. To continue
an earlier V4 run that stopped at the step-750 phase-A boundary, resume its
`final.pt` with the same V4 arguments and the corrected `0.02` threshold; step
751 enters phase B directly.

## V5 One-Group Refinement Ladder

Use V5 when V4 phase A establishes strong coarse text conditioning but phase B
fails after introducing groups 2-4 simultaneously. V5 resumes the best phase-A
checkpoint and introduces exactly one refinement group every 375 steps:

```text
R2: 8,1,0,0,0,0,0,0
R3: 8,2,1,0,0,0,0,0
...
R8: 8,2,2,2,2,2,2,1
```

Each stage must improve its newly introduced group by at least `0.01`, preserve
group-1 validation within `0.03` of the refinement baseline, and retain a
positive group-1 shuffled-text delta. The resumed optimizer learning rate is
explicitly replaced with the new `3e-5` rate instead of inheriting V4's
`1e-4`.

For a V4 `best_phase_a.pt` saved at step 625, run through global step 3250:

```bash
python -u -m minimoss.train_overfit \
  --manifest data_ljspeech_1100/train_manifest.jsonl \
  --validation-manifest data_ljspeech_1100/validation_manifest.jsonl \
  --token-dir data_ljspeech_1100/tokens \
  --output-dir checkpoints/ljspeech_1000_v5_b8 \
  --batch-size 8 \
  --max-steps 3250 \
  --validate-every 125 \
  --lr 3e-5 \
  --qwen-lora \
  --qwen-lora-rank 8 \
  --qwen-lora-alpha 16 \
  --nonlinear-frame-conditioner \
  --frame-position-embedding \
  --context-dropout-warmup-steps 5000 \
  --context-dropout-decay-steps 0 \
  --context-dropout-start 1.0 \
  --context-dropout-end 1.0 \
  --refinement-curriculum \
  --refinement-stage-steps 375 \
  --refinement-min-improvement 0.01 \
  --refinement-max-g1-regression 0.03 \
  --phase-gate-min-text-delta 0.005 \
  --text-diagnostics \
  --resume checkpoints/ljspeech_1000_v4_b8/best_phase_a.pt \
  --device cuda 2>&1 | tee logs/train_ljspeech_1000_v5_b8.log
```

The trainer writes `best_r2.pt` through `best_r8.pt`. A failed stage emits
`R2_FAIL` through `R8_FAIL` and stops immediately. Previous-frame context stays
fully hidden throughout V5; restoring temporal context is a separate gate only
after all refinement groups generalize.
