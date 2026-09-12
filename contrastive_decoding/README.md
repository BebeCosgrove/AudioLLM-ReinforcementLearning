# Contrastive decoding on the DCASE 2025 multiple-choice splits

Audio-Aware Decoding (AAD) contrastive decoding as a **training-free baseline**
alongside the mDPO runs, evaluated with the *exact* multiple-choice protocol from
`af3_evaluation.py` and `qwen2_evaluation.py`.

```
modified_logits = (1 + alpha) * clean_logits - alpha * negative_logits
```

The clean branch sees the real audio; the negative branch sees the same prompt with a
perturbed waveform. Ported from the AdaptivePerturbation project's
`adaptive_perturbation/run_{qwen,af3}.py`, matching the reference implementation at
[GillbertHsu/Audio-Aware-Decoding](https://github.com/GillbertHsu/Audio-Aware-Decoding).

## Experiment grid

4 perturbations x 2 alphas x 2 models x 2 splits = **32 runs**, plus 4 no-CD baselines.

| Perturbation | Setting | Parameters |
|---|---|---|
| `no_audio` | — | `np.zeros_like(audio)` — a **silent waveform**, audio tokens still present |
| `reverse` | `full` | `audio[::-1]` |
| `time_mask` | `light` | `n_masks=3, max_width=0.08` |
| `noise` | `weak` | `sigma=0.02` |

Alphas: **0.5** and **1.0**. Models: **base** Qwen2-Audio-7B-Instruct and
nvidia/audio-flamingo-3-hf — no mDPO adapter, so these are a training-free
alternative to mDPO rather than an addition to it.

Settings are read from the repository-root `perturbations.py`, the same module that
generated the mDPO perturbed training audio, so the contrastive branch and that
training data come from identical code.

> **`no_audio` is silent audio, not audio removal.** `perturbations.py` defines
> `NO_AUDIO` as `np.zeros_like(audio)`, and the `perturbed_audio/dcase_no_audio/*.wav`
> files were generated that way. The AAD paper prefers dropping the audio tokens
> entirely, and the AdaptivePerturbation runners do that. Staying consistent with this
> repository's pipeline was a deliberate choice — worth stating in any write-up, since
> the two variants give different numbers.

## Files

| File | Purpose |
|---|---|
| `cd_common.py` | Dataset loading + audio path remapping, perturbation resolution, and the evaluation code copied verbatim from her scripts |
| `cd_processor.py` | `ContrastiveAudioLogitsProcessor` — the AAD logits processor |
| `run_cd_qwen2.py` | Qwen2-Audio runner |
| `run_cd_af3.py` | Audio Flamingo 3 runner |
| `verify_setup.py` | Pre-flight check: no GPU or model weights needed |
| `summarize_results.py` | Collate all runs into one table |
| `results/` | Generated: `results/<model>/<split>/cd_<perturbation>_alpha_<alpha>.json` |

## Running

Always start with the pre-flight check — it catches every path problem before a job
reaches a GPU:

```bash
python contrastive_decoding/verify_setup.py
```

Full sweep for one model, one split (model loads once, all 8 conditions run against it):

```bash
accelerate launch contrastive_decoding/run_cd_qwen2.py --split test --baseline
accelerate launch contrastive_decoding/run_cd_qwen2.py --split validation --baseline
accelerate launch contrastive_decoding/run_cd_af3.py   --split test --baseline
accelerate launch contrastive_decoding/run_cd_af3.py   --split validation --baseline
```

A single condition, or a quick smoke test:

```bash
python contrastive_decoding/run_cd_qwen2.py --split test --perturbation noise --alpha 0.5
python contrastive_decoding/run_cd_qwen2.py --split test --perturbation reverse --alpha 1.0 --limit 8
```

Completed conditions are skipped, so re-running the same command resumes an interrupted
sweep. Pass `--overwrite` to redo them. Pin weights with `--revision <commit>`; the
resolved revision is recorded either way.

### What "skip" actually checks

A result file is only reused when it is **complete** *and* **measures the same thing**:

| Existing file | Behaviour |
|---|---|
| Absent, or `--overwrite` given | Run it |
| Truncated or unparseable (killed mid-write) | Re-run it, with a note |
| Complete, metadata matches | Skip |
| Complete, metadata differs | **Raise**, naming the differing fields |

That last case is deliberate: a stale file from an earlier experiment is never silently
reported as current, and never silently overwritten either. Identity is
model / model path / revision / weights / split / perturbation / alpha / seed /
max_new_tokens / dataset size / dataset fingerprint. Batch size and process count are
excluded — they change padding and runtime, not the quantity being measured.

### Integrity guarantees

- **Writes are atomic.** Output goes to a temporary file in the destination directory
  and is `os.replace`d into position only once complete, so an interrupted job can never
  leave a truncated file behind for a later run to mistake for a finished one.
- **Short runs are refused.** Before writing, the deduplicated `_row_index` set must be
  exactly `range(len(data))` — nothing dropped by a failed shard, nothing invented.
  A run that lost rows raises instead of writing a file that looks successful.
- **Decode count is checked per batch.** If the model returns fewer predictions than
  examples, the run fails rather than letting `zip()` silently truncate the batch.
- **Every result records a dataset fingerprint** (SHA-256 of the split file) and the
  resolved model revision, so any number can be traced to the exact data and weights
  that produced it.

```bash
python contrastive_decoding/summarize_results.py
python contrastive_decoding/summarize_results.py --dedup --csv summary.csv
```

## Dataset paths

The split JSONs are **never modified**. They store absolute cluster paths:

```
/data/not_backed_up/cosgrv/af3_project/dcase_2025/2025_DCASE_AudioQA/local_audio_path/dev/audio_00505.wav
```

`load_split()` uses each `audio_url` as stored when the file exists, and otherwise
rebases it **by basename** onto an audio root. So on the cluster nothing needs to be
passed; locally the flat `dcase_dev_audio_files/` directory is found automatically.
Override with `--data-root` / `--audio-root`, or `DCASE_DATA_ROOT` / `DCASE_AUDIO_ROOT`.

| | test | validation |
|---|---|---|
| Examples | 1122 | 1120 |
| Complex / Temporal | 817 / 305 | 816 / 304 |
| Audio resolved locally | 1122/1122 | 1120/1120 |

There are **no Bio examples** in either split, so `domain_average_accuracy` is a macro
average over Complex and Temporal only. That matches her result files.

## Reproducibility

`time_mask` and `noise` are stochastic. `perturb_for_example()` seeds numpy from the
example's own id rather than from call order, so a given example gets the same perturbed
waveform regardless of batch size, GPU count, or shard boundary. A 1-GPU run and an
8-GPU run produce identical perturbations.

## Row counts: 1136 vs 1122

Her result files report `total=1136` for the 1122-example test split. `accelerate`'s
distributed sampler pads the final batch by repeating examples, and 11 ids genuinely
repeat in the dataset. Each output file therefore carries **both**:

- top-level `overall` / `domain_average_accuracy` — as-run, padding included, directly
  comparable to her numbers
- `deduplicated` — the same run with padding rows removed (keyed on the example's
  position in the dataset file, so genuine repeats survive)

Use the top-level block to compare against her mDPO results, and `deduplicated` to
compare runs made with different GPU counts.

## Verified

- **Metrics are exact.** Feeding her `dcase_qwen_baseline_results.json` predictions
  through `cd_common.py`'s extraction and metrics reproduces her overall accuracy
  (637/1136), domain average (0.5075), and every per-subset and per-question-type cell
  bit-for-bit.
- **Prompt alignment holds.** For all four perturbations, the Qwen2-Audio processor
  produces *bit-identical* `input_ids` for the clean and negative branches — every
  perturbation preserves waveform length, so the audio-token expansion is unchanged.
  The runners assert this at runtime and fail loudly rather than silently subtracting
  misaligned logits.
- **The processor math is exact** at alpha 0.5 and 1.0, the negative branch grows by
  exactly one token per decode step (branch sync), alpha=0 is the identity, and the
  shape-mismatch guard fires.
- Qwen2-Audio's tokenizer pads **left**, so appending generated-token embeddings to the
  end of the negative sequence is correct.
- **The resumability machinery works**: a short run is refused, a padded run is accepted,
  an interrupted write leaves no file and no scratch file, a truncated file is re-run,
  and a stale-configuration file raises with the differing fields named.

Not verified locally: a full forward/generate pass. The local GPU has 4 GB and the
installed transformers (4.57.0) has no `AudioFlamingo3ForConditionalGeneration`. The
first cluster run should be `--limit 8` before launching a full sweep.

## Known costs and deviations

- **The negative branch keeps no KV cache.** Every decode step re-forwards the whole
  negative sequence: 16 extra forward passes per Qwen batch, 4 per AF3 batch. This
  matches the AdaptivePerturbation implementation exactly and was left alone rather than
  optimised, because caching here is an unverified change to a numerically sensitive
  path. It dominates runtime.
- **AF3 pads to `max_length`.** Her config, unchanged. Combined with the uncached
  negative branch this could be slow; the runner prints the real padded sequence length
  on the first batch so the cost is visible immediately.
- **The model is not passed through `accelerator.prepare()`** — it is moved with
  `.to(device)` instead. Her scripts prepare the model and then call
  `accelerator.unwrap_model(...)` to generate, bypassing the DDP wrapper anyway, so this
  is numerically identical for inference-only work and avoids wrapping overhead. The
  *DataLoader* is still prepared, which is what produces the sharding and padding
  behaviour behind the 1136 row count.
