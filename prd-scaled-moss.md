# PRD: Scaled MOSS-TTS Local-Transformer Student

## 0. Document Status

```text
Status: implementation-ready draft
Scope: architecture, data, training, evaluation, and release gates
Primary target: faster and smaller MOSS-compatible TTS model
Language scope: preserve existing teacher capability; no new language expansion
Supersedes: none
Historical predecessor: prd-v0.md grouped-decoder feasibility experiment
```

This PRD defines the first serious compressed MOSS experiment after the grouped
decoder experiments. It deliberately preserves the original MOSS probability
factorization and changes model capacity only. The objective is to determine
whether the released MOSS-TTS Local-Transformer can be distilled into a roughly
700M-850M student with lower time to first audio (TTFA), lower memory use, and
higher streaming throughput without introducing grouped-RVQ independence
assumptions.

## 1. Product Goal

Build a smaller, faster version of MOSS-TTS-Local-Transformer that:

- initializes its global transformer from `Qwen/Qwen3-0.6B`;
- uses the official MOSS text/message format and MOSS Audio Tokenizer;
- preserves all 33 output channels and their autoregressive ordering;
- reduces global and local hidden/FFN dimensions;
- trains the global and local student jointly through ground-truth supervision
  and teacher distillation;
- generates complete speech autoregressively, feeding student-generated frames
  back into the student global transformer;
- supports the official 8/16/32-RVQ inference operating points;
- exposes reproducible quality, TTFA, real-time factor, and memory benchmarks.

The first successful release is an architecture-validation checkpoint, not an
Indic-specialized checkpoint. Language expansion begins only after this PRD's
quality and performance gates pass.

## 2. Core Product Question

> Can the MOSS global-latent plus local-transformer architecture retain useful
> synthesis quality when both transformers are scaled down, while keeping the
> original 33-step conditional factorization unchanged?

This PRD must answer that question independently of grouped prediction,
MaskGIT, frame stacking, extreme quantization, or new-language data.

## 3. Architectural Constitution

These requirements are invariants for the first model. An implementation that
violates one of them is a different experiment and must not reuse its results.

### 3.1 Must remain unchanged

- MOSS Audio Tokenizer at 24 kHz and 12.5 frames per second.
- 32 RVQ audio codebooks with vocabulary size 1024.
- One channel-0 text/pad/EOS prediction followed by 32 audio predictions.
- Autoregressive conditioning of every RVQ channel on all earlier channels in
  the same frame.
- Summed multi-channel audio embeddings at the global-transformer input.
- No temporal delay pattern for the local-transformer architecture.
- No local positional encoding, matching the released MOSS implementation.
- MOSS processor message packing and special-token semantics.
- Causal global generation and generated-audio feedback between frames.
- Variable-RVQ inference with 8, 16, and 32 codebooks.

### 3.2 May change in this experiment

- Global hidden size and FFN size.
- Global pretrained initialization.
- Local hidden size and FFN size.
- Bridge MLP dimensions.
- Parameter initialization and distillation projections.
- Training data volume and curriculum.
- Runtime implementation, provided token-level parity tests cover unchanged
  behavior.

### 3.3 Explicitly forbidden in the first model

- Predicting multiple RVQ channels independently from one hidden state.
- Grouping codebooks into four-way or other parallel AR groups.
- Replacing the MOSS codec with DAC, Encodec, Mimi, or another codec.
- Freezing the Qwen3-0.6B backbone for the full training run.
- Evaluating only against cached ground-truth global states.
- Claiming success from training loss or teacher-forced audio alone.
- Adding new language support before the architecture gate passes.
- One-bit/two-bit training, frame stacking, or a MaskGIT decoder in this phase.

## 4. Reference And Student Architecture

### 4.1 Side-by-side configuration

| Component | Official MOSS teacher | Scaled student |
| --- | ---: | ---: |
| Global initialization | Released MOSS checkpoint | `Qwen/Qwen3-0.6B` |
| Global layers | 28 | 28 |
| Global hidden size | 2048 | 1024 |
| Global FFN size | 6144 | 3072 |
| Global query heads | 16 | 16 |
| Global KV heads | 8 | 8 |
| Global head dimension | 128 | 128 |
| Text vocabulary | 155648 | 155648 after expansion |
| Local layers | 4 | 4 |
| Local hidden size | 1536 | 1024 |
| Local FFN size | 8960 | 4096 |
| Local query heads | 16 | 16 |
| Local KV heads | 8 | 8 |
| Local positional encoding | disabled | disabled |
| Bridge FFN size | 2048 | 1024 |
| Output channels | 33 | 33 |
| Audio codebooks | 32 | 32 |
| Audio vocabulary | 1024 plus pad | 1024 plus pad |
| Local iterations/frame | 33 | 33 |
| Codec | MOSS Audio Tokenizer | same |

The student attention configuration intentionally follows native Qwen3-0.6B:
hidden size 1024, 16 query heads, 8 KV heads, and head dimension 128. The query
projection therefore has a larger projected dimension than the residual hidden
size. Do not silently replace this with an 8-query/4-KV configuration in the
first experiment.

### 4.2 Student flow

```text
text and optional voice prompt
        |
official MOSS processor packing
        |
Qwen3-0.6B-initialized global speech transformer (trainable)
        |
one 1024-dimensional global latent per aligned step
        |
global-to-local SwiGLU bridge
        |
4-layer local transformer, hidden 1024, FFN 4096
        |
channel 0: text/pad/EOS
        |
RVQ 1 -> RVQ 2 -> ... -> RVQ 32
        |
MOSS Audio Tokenizer causal decoder
        |
24 kHz waveform
```

### 4.3 Exact factorization

For frame `t`, the student must implement:

```text
p(y_0,t | text, all previous frames)
* p(y_1,t | text, previous frames, y_0,t)
* p(y_2,t | text, previous frames, y_0:1,t)
...
* p(y_32,t | text, previous frames, y_0:31,t)
```

Codebooks within a frame are never treated as conditionally independent.

### 4.4 Vocabulary expansion

Qwen3-0.6B has a 151936-token vocabulary; the released MOSS configuration uses
155648. The implementation must:

1. Instantiate the student with the MOSS vocabulary size.
2. Copy all overlapping Qwen token embeddings and tied text-head weights.
3. Initialize additional rows with the Qwen initializer distribution.
4. Verify every MOSS special token ID is in range and has a dedicated test.
5. Preserve weight tying for the student text embedding and text output head
   unless an ablation explicitly disables it.

## 5. Parameter And Performance Targets

These are targets, not claims. The implementation must print exact parameter
counts by subsystem before training.

| Metric | Target |
| --- | ---: |
| Total student parameters, codec excluded | 650M-850M |
| Global subsystem | approximately 600M class |
| Complete local path and bridges | 100M-220M |
| BF16 trainable weight memory | at most 1.7 GB |
| Relative parameter count | at most 50% of released teacher |
| Local iterations | exactly 33 |
| End-to-end warm throughput | at least 1.5x teacher on same GPU |
| Warm TTFA | at least 25% lower than teacher |
| Peak inference VRAM | at least 30% lower than teacher |

Parameter totals must distinguish:

- global transformer;
- text and audio embeddings;
- global-to-local bridge;
- local transformer core;
- per-channel local-to-global bridges;
- channel norms and heads.

## 6. User And Deployment Scenario

The first target is batch-size-one streaming TTS for a realtime voice agent.
The benchmark scenario is:

```text
one fixed system/voice prompt
one user text request
one synthesized assistant response
first PCM chunk emitted as soon as causally decodable
generation continues while earlier waveform is played
```

The first release does not need a production server, but the generation API
must expose timestamps required to measure this path.

## 7. Data Requirements

### 7.1 Dataset contract

Each source item must contain:

```json
{
  "id": "unique-id",
  "audio": "/absolute/or-resolved/path.wav",
  "text": "verbatim normalized transcript",
  "language": "en",
  "speaker_id": "optional-stable-id",
  "duration_seconds": 4.2,
  "split": "train"
}
```

Required quality rules:

- one dominant speaker;
- no overlapping speech;
- no music for the architecture-validation corpus;
- accurate transcript and language ID;
- 24 kHz or resampleable source audio;
- no duplicate utterance across train, validation, and test;
- speaker-disjoint test subset when speaker metadata exists;
- duration and text-length sanity checks;
- codec reconstruction manually checked on every new data source.

### 7.2 Dataset scales

| Tier | Audio | Purpose | Decision allowed |
| --- | ---: | --- | --- |
| S0 | 10-30 minutes | code and one-batch debugging | implementation correctness only |
| S1 | 1-5 hours | overfit and short held-out gate | reject broken architecture/training |
| S2 | 50-100 hours | pilot distillation | compare quality and capacity trends |
| S3 | 500-1000+ hours | serious architecture validation | release-candidate decision |

No generalization claim may be made from S0 or S1. The previous 1000-utterance
experiment was an implementation probe, not enough evidence for a 700M-class
speech model.

### 7.3 Splits

Use deterministic, persisted split manifests:

```text
train:      98% by utterance
validation: 1% by utterance, at least 500 items at S2/S3
test:       1% by utterance, at least 500 items at S2/S3
```

Where possible, reserve an additional speaker-disjoint test set. Do not tune
sampling parameters on the final test set.

### 7.4 Precomputed artifacts

Training must not run the audio codec in the optimization loop. Precompute:

```python
{
    "id": str,
    "text": str,
    "language": str,
    "rvq": Int16Tensor[num_frames, 32],
    "sample_rate": 24000,
    "codec_revision": str,
    "source_sha256": str,
}
```

Store artifacts in sharded files rather than one file per utterance at S2/S3.
Each shard requires a checksum and resumable generation marker.

## 8. Teacher Contract And Parity Gate

Before student training, capture the official teacher path directly from
`model.generate()`.

For fixed prompts and seeds, record:

- packed input IDs and attention mask;
- every generated 33-channel frame;
- global last-layer state used by the local decoder;
- local input sequence for one or more frames;
- per-channel logits before sampling;
- logits-processor settings;
- sampled token after processing;
- decoded waveform and timing trace.

The standalone teacher replay must reproduce:

- identical packed inputs;
- identical greedy tokens;
- sampled tokens under an explicitly controlled generator where supported;
- per-channel logits within numerical tolerance;
- the same start/end frame boundaries.

This parity gate exists because the earlier teacher-replay audio did not match
the quality of untouched official generation. Student training must not begin
until the mismatch is understood or the replay is removed from the training
path.

## 9. Teacher Cache Design

### 9.1 Why a new cache format is required

The previous cache retained top-k logits and renormalized them as if they were
the complete teacher distribution. That changes the KD objective. The new cache
must preserve tail probability mass.

### 9.2 Per-frame cache

```python
{
    "global_hidden": Float16Tensor[2048],
    "selected_global_layers": Float16Tensor[num_layers, 2048],
    "local_topk_indices": Int16Tensor[32, K],
    "local_topk_logits": Float16Tensor[32, K],
    "local_logsumexp": Float16Tensor[32],
    "local_target_logits": Float16Tensor[32],
    "teacher_sampled_tokens": Int16Tensor[32],
}
```

Recommended `K=64` for the pilot. The KD implementation must use the stored
log-sum-exp to represent omitted probability as an explicit tail bucket rather
than renormalizing the top-k entries to one.

For S3, teacher cache generation must be sharded and streamable. Retaining every
intermediate global state may be too expensive; the implementation must support
configuring selected teacher layers and deleting completed cache shards only
after their student-training checkpoint is durable.

### 9.3 Offline versus online teacher

```text
24-48 GB GPU: offline teacher-cache generation is the default
80 GB GPU: online teacher execution may be benchmarked
multi-GPU: teacher and student may run on separate devices
```

Online and offline KD must have a numerical equivalence test on a small batch.

## 10. Initialization

### 10.1 Global student

1. Load `Qwen/Qwen3-0.6B` at a pinned revision.
2. Expand the vocabulary to 155648.
3. Copy overlapping text embeddings and transformer weights exactly.
4. Initialize additional vocabulary rows normally.
5. Add 32 audio embedding tables plus channel-0/MOSS special-token support.
6. Initialize audio embeddings from a shared teacher-embedding projection when
   available; otherwise use standard initialization and rely on distillation.
7. Keep all global parameters trainable unless a warmup phase explicitly
   freezes lower layers for a bounded number of steps.

### 10.2 Local student

The local student has the same four-layer topology but different dimensions.
Support two initialization modes:

```text
random: standard Qwen/MOSS initialization
svd: truncated-SVD/PCA projection of compatible teacher matrices
```

The pilot must compare both modes on S1. The selected mode must be recorded in
checkpoint metadata.

### 10.3 Distillation projections

Use trainable, checkpointed projections only for loss computation:

```text
global: student 1024 -> teacher 2048
local:  student 1024 -> teacher 1536
```

These projections are discarded at inference and excluded from deployed-model
parameter counts.

## 11. Training Objectives

### 11.1 Ground-truth weighted cross entropy

Use the MOSS channel weighting pattern:

```text
channel 0: weight 1
RVQ 1-3:  weight 3
RVQ 4-6:  weight 2
RVQ 7-32: weight 1
```

All channel losses must be logged separately. Padding and invalid codec tokens
must be masked before reduction.

### 11.2 Teacher distribution loss

Compute temperature-scaled KD for:

- channel-0 text/pad/EOS logits;
- all 32 local audio channels;
- optional global next-step auxiliary heads if available.

Use top-k plus tail-bucket KL when full logits are not stored. Never compute KL
against a top-k distribution renormalized without its omitted mass.

### 11.3 Hidden-state loss

At selected global layers and every local layer, combine:

```text
normalized MSE(project(student_hidden), teacher_hidden)
+ cosine-distance loss
```

Hidden losses are auxiliary. They must not dominate ground-truth token loss.

### 11.4 Sequence rollout loss

Teacher forcing proves local conditional learning but not complete synthesis.
Later training must include generated-history examples:

- student-generated prior audio frames are fed into the student global model;
- student-generated earlier RVQ channels are fed into its local decoder;
- teacher logits are queried or cached for the corresponding trajectory when
  practical;
- ground-truth CE remains active to prevent drift.

Do not mutate integer context tensors after they have been used as embedding
indices in the same autograd graph.

### 11.5 Recommended initial loss schedule

| Phase | Ground truth | Local KD | Global hidden | Local hidden | Rollout |
| --- | ---: | ---: | ---: | ---: | ---: |
| T0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| T1 | 0.5 | 0.35 | 0.10 | 0.05 | 0.0 |
| T2 | 0.45 | 0.30 | 0.10 | 0.05 | 0.10 |
| T3 | 0.45 | 0.25 | 0.05 | 0.05 | 0.20 |

These are starting values. The trainer must serialize the exact resolved loss
configuration. Changes require a new experiment ID.

## 12. Training Curriculum

### Stage P0: official parity and instrumentation

Purpose:

- establish trusted teacher behavior;
- establish timing and quality baselines;
- verify codec and processor revisions.

Exit gate:

- untouched official audio is clean;
- teacher replay parity passes;
- 8/16/32-RVQ controls decode;
- profiler reports global, local, codec, and total timing separately.

### Stage P1: one-batch student overfit

Train one packed batch until:

- total and all coarse-channel losses collapse;
- teacher-forced student audio is recognizable;
- integrated free-running generation produces the memorized sentence;
- gradients reach global audio embeddings, global transformer, both bridges,
  local transformer, and all channel heads.

Failure means implementation error. Do not tune architecture around a failed
one-batch gate.

### Stage P2: S1 initialization comparison

Run `random` and `svd` initialization on the same 1-5-hour split and seed.

Compare:

- convergence speed;
- validation CE by channel;
- teacher agreement;
- integrated free-running intelligibility;
- onset stability and long-utterance drift.

Select one initialization before S2.

### Stage P3: S2 pilot distillation

Train on 50-100 hours with a held-out split. Enable T0 then T1 losses. Do not
enable rollout until teacher-forced validation and integrated short generation
are stable.

Exit gate:

- held-out audio is intelligible;
- outputs are not identical across prompts;
- no progressive buzz/jitter across at least 20 samples;
- global-state shuffled-text diagnostic degrades measurably;
- student beats unigram and text-shuffled baselines by a wide margin;
- quality trend justifies S3 cost.

### Stage P4: S3 full architecture validation

Train on 500-1000+ hours. Enable T2/T3 rollout curriculum only after T1
validation plateaus. Use checkpoint averaging or EMA only if configured before
the run and represented in metadata.

Exit gate is the release-candidate gate in Section 16.

## 13. Optimization Defaults

Initial defaults, subject to S1 calibration:

```text
precision: bf16
optimizer: AdamW
global learning rate: 5e-5
new-module learning rate: 2e-4
distillation-projection learning rate: 2e-4
weight decay: 0.1 for matrix weights, 0 for norms/biases
gradient clipping: 1.0
schedule: warmup-stable-decay
warmup: 2% of planned optimizer steps
gradient checkpointing: enabled
sequence packing: enabled
dropout: 0.0 initially, then tune only from validation evidence
seed set: 42, 43, 44 for decision runs
```

Use parameter groups. Do not apply the new-module learning rate to pretrained
Qwen weights.

Effective batch size must be defined in audio frames and packed tokens, not only
utterance count. Log both.

## 14. Checkpoint And Resume Contract

Every checkpoint must contain:

```python
{
    "format_version": int,
    "student_config": dict,
    "model_state_dict": dict,
    "optimizer_state_dict": dict,
    "scheduler_state_dict": dict,
    "scaler_state_dict": dict | None,
    "global_step": int,
    "samples_seen": int,
    "audio_frames_seen": int,
    "rng_state": dict,
    "data_cursor": dict,
    "loss_config": dict,
    "teacher_revision": str,
    "qwen_revision": str,
    "codec_revision": str,
    "git_commit": str,
    "metrics": dict,
}
```

Resume must restore optimizer, scheduler, scaler, RNG, and data position. A
weights-only warm start must use a different flag and print that optimizer state
is not restored.

## 15. Evaluation

### 15.1 Required audio sets

For every milestone checkpoint, generate numbered files for at least:

```text
20 train prompts
50 held-out in-domain prompts
20 speaker-disjoint prompts where available
10 long prompts over 20 seconds
10 punctuation/number/abbreviation stress prompts
```

For each prompt save:

```text
01_ground_truth.wav
01_teacher.wav
01_student_greedy.wav
01_student_sampled.wav
01.txt
```

Do not use reference frame counts in the primary free-running evaluation. A
secondary oracle-duration diagnostic may do so, but must be labeled clearly.

### 15.2 Objective quality metrics

- WER/CER using a pinned ASR model.
- Speaker similarity using a pinned speaker encoder.
- Duration error and EOS failure rate.
- UTMOS or another pinned neural MOS proxy.
- Codec-space Frechet distance if implemented reproducibly.
- Repetition, truncation, empty-output, and runaway-generation rates.
- Per-channel CE and perplexity.
- Teacher-student KL with tail mass.

Exact RVQ token accuracy is a debugging metric, not the quality acceptance
metric.

### 15.3 Human listening rubric

Score blind A/B samples for:

- intelligibility;
- speaker identity;
- naturalness;
- pitch/prosody stability;
- onset cleanliness;
- end-of-utterance cleanliness;
- progressive jitter or degradation;
- preference versus teacher.

Ground-truth codec reconstruction must be included to distinguish model errors
from codec artifacts.

### 15.4 Performance metrics

Measure batch size one with the same device, software stack, prompts, and RVQ
count for teacher and student.

Definitions:

```text
cold TTFA: process start/model load to first PCM chunk
warm TTFA: request accepted to first PCM chunk, model already resident
prefill time: request accepted to first generated frame decision
first-frame local time: all required local channel decisions for frame one
codec first-chunk time: first complete codes to first PCM chunk
RTF: wall-clock generation time / generated audio duration
steady throughput: generated audio seconds / wall-clock second after first chunk
```

Report p50 and p95 over at least 100 prompts after warmup. Also report peak VRAM,
model-load time, and prompt length.

## 16. Release-Candidate Gates

The architecture passes only if all required gates pass on the same checkpoint.

### 16.1 Correctness

- No target leakage.
- No reference duration in primary generation.
- EOS terminates at least 99% of normal test prompts before the safety cap.
- Codec reconstruction controls are clean.
- Saved and resumed training matches uninterrupted training within tolerance.

### 16.2 Quality

- Student held-out WER/CER is no worse than 25% relative over the teacher.
- Mean speaker-similarity decrease is no more than 0.05 absolute.
- No catastrophic buzz, identical-output collapse, or progressive degradation
  in the required listening set.
- At least 80% of blind samples are rated acceptable for agent conversation.
- Long-form failure rate remains below 5% on the specified stress set.

These thresholds are initial engineering gates, not publication claims.

### 16.3 Performance

- Exact deployed parameters are at most 850M, codec excluded.
- Warm TTFA is at least 25% lower than teacher at equal RVQ count.
- End-to-end steady throughput is at least 1.5x teacher at equal RVQ count.
- Peak inference VRAM is at least 30% lower.
- The student streams a first PCM chunk without waiting for the full utterance.

### 16.4 Decision rule

```text
quality pass + performance pass -> proceed to language expansion
quality pass + performance fail -> optimize runtime before architecture changes
quality fail + training healthy -> increase data/capacity or improve distillation
training/parity fail -> fix implementation; do not scale data
```

## 17. Failure Outcomes And Actions

| Outcome | Meaning | Required action |
| --- | --- | --- |
| Official controls are bad | Environment, revision, or codec failure | Stop before training |
| Teacher replay differs from `generate()` | Teacher contract is not understood | Instrument official path; do not cache KD targets |
| One batch does not overfit | Shift, mask, gradient, or packing bug | Fix implementation |
| One batch overfits but integrated audio fails | Generation feedback/EOS bug | Compare frame-by-frame against teacher |
| Train loss falls, validation stays uniform | Insufficient data or shortcut memorization | Check splits, text dependence, conditioning |
| Shuffled text has no effect | Global student ignores text | Audit packing and curriculum; do not add local capacity |
| Coarse RVQs fail first | Global planning/alignment failure | Increase global KD and inspect temporal shift |
| Coarse works, fine RVQs fail | Local capacity/distillation issue | Increase local capacity or KD quality |
| Teacher-forced good, free-running bad | Exposure bias | Enable bounded rollout curriculum |
| All outputs sound alike | Mode/collapse or conditioning failure | Inspect token entropy, speaker/text diagnostics |
| Audio starts clean then degrades | generated-history instability | Train longer rollouts and inspect EOS/context feedback |
| Student quality good but not faster | Runtime or memory-bound bottleneck | Profile before changing architecture |
| 32 RVQ good, 8/16 bad | Variable-bitrate training insufficient | Add progressive sequence dropout training |
| Resume changes training abruptly | Incomplete checkpoint state | Fix resume contract |

## 18. GPU Planning

GPU-hour estimates must be derived from a calibration run on the target machine.
The training CLI must support:

```text
--calibration-steps 200
--estimate-total-steps
--estimate-cache-bytes
--estimate-gpu-hours
```

Planning envelopes, including tokenization and teacher-cache generation:

| Tier | A100 80 GB rough envelope | A40 48 GB rough envelope |
| --- | ---: | ---: |
| P0/P1 | 4-16 GPU-hours | 8-30 GPU-hours |
| S1 | 10-40 GPU-hours | 20-80 GPU-hours |
| S2 | 50-200 GPU-hours | 100-400 GPU-hours |
| S3 | 300-1200+ GPU-hours | 600-2400+ GPU-hours |

These are budgeting ranges, not promises. The 200-step calibration output
replaces them for an actual run. If the measured S3 budget is unacceptable, the
team must reduce data/epochs or secure more GPUs rather than pretending an S1
result proves generalization.

## 19. Required Software Modules

Recommended new modules:

```text
minimoss/scaled_config.py
minimoss/scaled_model.py
minimoss/teacher_trace.py
minimoss/teacher_cache.py
minimoss/prepare_shards.py
minimoss/train_scaled.py
minimoss/generate_scaled.py
minimoss/evaluate_scaled.py
minimoss/benchmark_scaled.py
minimoss/compare_scaled_runs.py
```

Do not retrofit the old grouped model until its behavior becomes ambiguous.
Shared codec and manifest utilities may be reused after tests confirm the same
contracts.

## 20. CLI Acceptance Surface

Illustrative commands the implementation must support:

```bash
python -m minimoss.teacher_trace \
  --model OpenMOSS-Team/MOSS-TTS-Local-Transformer \
  --output evaluation/teacher_parity \
  --device cuda

python -m minimoss.prepare_shards \
  --manifest data/train.jsonl \
  --output-dir data/scaled_shards \
  --codec OpenMOSS-Team/MOSS-Audio-Tokenizer \
  --device cuda

python -m minimoss.teacher_cache \
  --shard-dir data/scaled_shards \
  --output-dir data/teacher_cache \
  --top-k 64 \
  --device cuda

python -u -m minimoss.train_scaled \
  --config configs/scaled_moss_06b.json \
  --train-shards data/scaled_shards/train \
  --validation-shards data/scaled_shards/validation \
  --teacher-cache data/teacher_cache \
  --output-dir checkpoints/scaled_moss_06b \
  --device cuda

python -m minimoss.evaluate_scaled \
  --checkpoint checkpoints/scaled_moss_06b/best.pt \
  --manifest data/test.jsonl \
  --output-dir evaluation/scaled_moss_06b \
  --device cuda

python -m minimoss.benchmark_scaled \
  --teacher OpenMOSS-Team/MOSS-TTS-Local-Transformer \
  --student checkpoints/scaled_moss_06b/best.pt \
  --manifest data/benchmark.jsonl \
  --rvq 32 \
  --device cuda
```

Exact CLI names may change during implementation, but every capability above is
required and must be documented.

## 21. Test Requirements

### Unit tests

- vocabulary expansion and weight copying;
- all special-token IDs in range;
- 33-channel input/output shapes;
- local causal masking and no future-channel leakage;
- global temporal shift and no current-frame leakage;
- top-k plus tail-bucket KD correctness against full-logit KL;
- channel weighting and padding masks;
- parameter-group learning rates;
- checkpoint/resume state restoration;
- deterministic sampling with a passed generator;
- 8/16/32-RVQ output shapes.

### Integration tests

- official processor pack/unpack round trip;
- teacher trace versus official generation;
- MOSS codec encode/decode reconstruction;
- one optimizer step with nonzero gradients in every required subsystem;
- one-batch overfit fixture;
- save/load generation parity;
- interruption and maximum-length safety stop;
- first PCM chunk emitted before utterance completion.

### Regression tests

- no in-place mutation of embedding index tensors;
- no accidental frozen global backbone;
- no reference-frame-count use in primary evaluation;
- no top-k-only probability renormalization in KD;
- no grouped codebook prediction in the scaled baseline.

## 22. Observability

Training logs must include:

```text
step and phase
audio frames and packed tokens seen
ground-truth total loss
channel-0 and all 32 RVQ losses
global/local KD loss
global/local hidden loss
rollout loss and rollout probability
gradient norm by subsystem
learning rate by parameter group
tokens/second and audio-frames/second
GPU memory allocated/reserved
validation metrics and checkpoint decision
```

Write machine-readable JSONL alongside human-readable logs. Every generated
evaluation directory must contain the resolved configuration and source commit.

## 23. Reproducibility And Supply Chain

- Pin teacher, Qwen, codec, and remote-code revisions.
- Review downloaded remote code before execution.
- Save package versions and CUDA/PyTorch environment.
- Save manifest and shard checksums.
- Record random seeds and deterministic settings.
- Never overwrite a completed experiment directory.
- Use unique experiment IDs derived from config plus git commit.
- Store licenses and provenance for every training dataset.

## 24. Implementation Work Packages

The work should be implemented and reviewed in this order:

1. Teacher trace and token-level parity harness.
2. Performance instrumentation for untouched teacher.
3. Scaled model configuration and exact 33-channel forward pass.
4. Qwen3-0.6B vocabulary expansion and initialization.
5. Teacher-cache format with tail-aware KD.
6. Sharded codec-token data pipeline.
7. Ground-truth trainer and one-batch gate.
8. Hidden/logit distillation losses.
9. Full integrated student generation.
10. Rollout curriculum.
11. Objective and listening evaluation suite.
12. Teacher-versus-student performance benchmark.
13. S1 initialization experiment.
14. S2 pilot run.
15. S3 architecture-validation run.

Every work package must have independent tests and a command that demonstrates
its definition of done.

## 25. Post-Gate Roadmap

Only after the release-candidate gates pass:

1. Add local KV caching and fused incremental inference.
2. Evaluate channel-0 direct-head prediction.
3. Test fewer default RVQs and progressive sequence dropout.
4. Explore coarse-AR plus iterative fine-codebook refinement.
5. Explore asymmetric frame stacking for steady-state throughput.
6. Add Indic language data, text normalization, phonetic conditioning, and
   language-specific evaluation.
7. Evaluate 8-bit/4-bit deployment quantization.

Each item is a separate controlled experiment. The scaled baseline remains the
comparison anchor.

## 26. Final Definition Of Done

This PRD is complete when the repository contains:

- a reproducible scaled student implementation;
- a verified official-teacher parity trace;
- deterministic sharded data and teacher-cache pipelines;
- resumable joint global/local training;
- integrated free-running generation with the MOSS codec;
- objective, listening, and performance evaluations;
- exact parameter counts and subsystem timing;
- a checkpoint passing Section 16 on S3-scale data;
- a written go/no-go decision for language expansion.

Until those conditions are met, the result is an experiment, not a faster MOSS
replacement.

## 27. Primary References

- MOSS-TTS Technical Report: <https://arxiv.org/abs/2603.18090>
- MOSS-TTS Local-Transformer checkpoint and code:
  <https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer>
- MOSS Audio Tokenizer:
  <https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer>
- Qwen3-0.6B:
  <https://huggingface.co/Qwen/Qwen3-0.6B>

