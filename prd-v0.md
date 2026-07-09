# PRD: Mini-MOSS-GD Lean Overfit Viability Test

## 1. Goal

Build the smallest useful prototype to test whether a frozen Qwen2.5-0.5B backbone plus a grouped RVQ local decoder can learn to predict audio codec tokens on a tiny dataset.

This is not a full TTS training system. The goal is to answer:

> Can the proposed grouped local decoder learn a text-to-RVQ mapping well enough to overfit a tiny clean dataset and produce recognizable decoded audio?

## 2. Non-Goals

Do not implement:

- distributed training
- LoRA
- full Qwen fine-tuning
- speaker cloning
- duration control
- pronunciation control
- multilingual support
- web UI
- streaming server
- large-scale data pipeline
- production checkpoint format

## 3. Target Architecture

```text
text
  ↓
Qwen2.5-0.5B frozen backbone
  ↓
selected hidden states
  ↓
projection layer
  ↓
grouped local decoder
  ↓
16 RVQ tokens/frame
  ↓
codec decoder
  ↓
wav
```

Use:

```text
Backbone:
Qwen2.5-0.5B, frozen

Trainable:
- projection layer
- RVQ token embeddings
- grouped local decoder
- output heads

Codec:
pretrained codec producing 16 RVQ codebooks/frame
```

If the chosen codec produces more than 16 codebooks, use the first 16 for the lean test.

## 4. Grouped Local Decoder

For each audio frame `t`, predict 16 RVQ tokens in 4 autoregressive group steps:

```text
step 1 -> codebooks 1-4
step 2 -> codebooks 5-8, conditioned on group 1
step 3 -> codebooks 9-12, conditioned on groups 1-2
step 4 -> codebooks 13-16, conditioned on groups 1-3
```

Factorization:

```text
P(q1..q16 | h_t)
=
P(q1..q4 | h_t)
P(q5..q8 | h_t, q1..q4)
P(q9..q12 | h_t, q1..q8)
P(q13..q16 | h_t, q1..q12)
```

Within each group, 4 codebooks are predicted in parallel.

## 5. Minimal File Layout

```text
minimoss/
  config.py
  dataset.py
  codec.py
  model.py
  train_overfit.py
  generate.py
  prepare_tokens.py
  utils.py
```

Expected total size: about 1.2k-2.5k LOC.

## 6. Dataset Format

Use a simple JSONL manifest:

```json
{"id": "utt_0001", "wav": "/path/to/audio.wav", "text": "hello world"}
```

Requirements:

- single speaker preferred
- 10-100 utterances for first test
- 3-10 seconds per utterance
- clean audio
- accurate transcripts

## 7. Scripts

### `prepare_tokens.py`

Inputs:

```text
manifest.jsonl
audio files
pretrained codec
```

Outputs:

```text
token_manifest.jsonl
rvq token files
```

Each token file should contain:

```python
{
    "id": str,
    "text": str,
    "rvq": LongTensor[num_frames, 16],
}
```

Also verify:

- ground-truth RVQ tokens decode back to recognizable audio
- codebook count is correct
- frame count is nonzero

### `train_overfit.py`

Train only the projection plus grouped decoder.

Required features:

- load frozen Qwen
- freeze all Qwen parameters
- load tokenized dataset
- compute grouped cross-entropy loss
- print total loss and per-group loss
- checkpoint trainable modules
- support overfit-one-batch mode
- optionally generate a fixed sample during training

### `generate.py`

Inputs:

```text
checkpoint
text prompt from training set
optional max frames
```

Outputs:

```text
generated RVQ tokens
decoded wav
```

For debugging, support:

- greedy decoding
- temperature sampling
- saving generated token tensor
- decoding ground-truth tokens for comparison

## 8. Success Criteria

Minimum success:

```text
1. Ground-truth RVQ tokens decode to recognizable audio.
2. Training loss drops strongly on tiny dataset.
3. Model can overfit 10-100 utterances.
4. Teacher-forced predicted tokens decode into speech-like audio.
5. Free-running generation does not instantly collapse.
```

Stronger success:

```text
Generated audio from training text is recognizable and roughly matches the target utterance.
```

Do not require high naturalness or generalization.

## 9. Required Metrics

Print during training:

```text
total_loss
group_1_loss
group_2_loss
group_3_loss
group_4_loss
learning_rate
step_time
```

Optional but useful:

```text
per-codebook accuracy
token perplexity per group
generated sample every N steps
```

## 10. Possible Outcomes And Meaning

### Outcome A: Codec ground-truth decode is bad

Meaning:

```text
Codec/tokenizer setup is wrong or codec is unsuitable.
```

Action:

```text
Do not train.
Fix codec loading, sample rate, codebook order, or choose another codec.
```

### Outcome B: Loss does not decrease

Meaning:

```text
Likely implementation bug.
```

Check:

```text
label shift
padding mask
frame alignment
wrong target frame
optimizer params
learning rate
frozen/trainable flags
RVQ codebook vocab sizes
```

Action:

```text
Overfit a single batch.
If one batch does not overfit, fix code before proceeding.
```

### Outcome C: Total loss drops, but group 1 loss stays high

Meaning:

```text
The model is not learning coarse acoustic tokens.
This is serious because early RVQ codebooks carry the most important information.
```

Action:

```text
Check frame/text alignment.
Increase group 1 loss weight.
Try making q1 standalone:
q1 -> q2-q4 -> q5-q8 -> q9-q12 -> q13-q16.
```

### Outcome D: Group 1 learns, later groups do not

Meaning:

```text
The model learns coarse audio but not fine residual detail.
This is acceptable for the first test.
```

Action:

```text
Train longer.
Add loss weighting for later groups.
Try smaller groups for later codebooks if artifacts are severe.
```

### Outcome E: Teacher-forced audio is speech-like, free generation is bad

Meaning:

```text
Training path works, inference loop or autoregressive stability is the issue.
```

Action:

```text
Debug generation one frame at a time.
Use greedy decoding.
Compare teacher-forced vs generated token distributions.
Check group-conditioning order.
```

### Outcome F: Model overfits train data but validation is bad

Meaning:

```text
Architecture can learn.
Generalization/data/conditioning is weak.
```

Action:

```text
This is a valid positive result for the lean test.
Next phase would add more data, LoRA, or better conditioning.
```

### Outcome G: Generated audio is intelligible but robotic

Meaning:

```text
Core mapping works, but prosody/detail is weak.
```

Action:

```text
For lean test, mark as success.
Future work: LoRA Qwen, more data, duration/prosody conditioning.
```

### Outcome H: Generated audio has codec artifacts

Meaning:

```text
Later RVQ groups may be weak, sampling may be too hot, or group predictions conflict.
```

Action:

```text
Decode ground-truth tokens first.
Try greedy decoding.
Lower temperature.
Inspect per-group losses.
Consider smaller groups for q1-q8.
```

### Outcome I: Tiny overfit works cleanly

Meaning:

```text
Mini-MOSS-GD architecture is viable enough to scale.
```

Action:

```text
Proceed to 500-1000 utterances.
Then consider LoRA Qwen.
```

## 11. Implementation Notes

Keep the first version simple:

```text
No KV cache required.
No streaming required.
No distributed training.
No mixed precision complexity unless needed.
No fancy sampling until greedy works.
```

Use greedy generation first. Sampling can hide bugs.

The most important sanity check:

```text
Can the model overfit 10 utterances?
```

If not, do not scale.

## 12. Expected Engineering Effort

Estimated implementation:

```text
Lean token-prediction-only version: 700-1200 LOC
Full overfit test with decode/generate: 1200-2500 LOC
```

Estimated time:

```text
Best case: 2-3 days
Normal case: 4-7 days
Messy codec/data case: 1-2 weeks
```

## 13. Compute Estimate

For the lean viability test:

```text
A40: 6-24 GPU-hours
A100: 3-16 GPU-hours
```

This assumes the codec and dataset are already ready.

## 14. Final Decision Rule

The architecture is considered worth continuing if:

```text
1. one-batch overfit succeeds
2. tiny-dataset loss drops clearly
3. ground-truth codec decode works
4. generated or teacher-forced predictions decode to recognizable speech-like audio
```

If all four pass, move to the next phase.

If any fail, debug that failure before scaling.
