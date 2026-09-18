# Contrastive decoding on the DCASE 2025 multiple-choice splits

Audio-Aware Decoding as a **training-free baseline** next to the mDPO runs, using the
exact evaluation protocol from `af3_evaluation.py` and `qwen2_evaluation.py`.

```text
modified_logits = (1 + alpha) * clean_logits - alpha * negative_logits
```

The clean branch sees the real audio, the negative branch the same prompt with a
perturbed waveform. Everything is in one file: [run_cd.py](run_cd.py).

## Grid

4 perturbations x 2 alphas x 2 models x 2 splits = 32 runs, plus 4 no-CD baselines.

| Perturbation | Setting |
|---|---|
| `no_audio` | `np.zeros_like(audio)` — silent audio, audio tokens still present |
| `reverse` | `audio[::-1]` |
| `time_mask` | `n_masks=3, max_width=0.08` |
| `noise` | `sigma=0.02` |

Alphas 0.5 and 1.0, on **base** Qwen2-Audio-7B-Instruct and audio-flamingo-3-hf (no mDPO
adapter). Settings resolve through the repo-root `perturbations.py` — the same module
that produced the mDPO training audio.

> `no_audio` is **silent audio, not audio removal**. That is how `perturbations.py`
> defines it and how `perturbed_audio/dcase_no_audio/*.wav` were made. The AAD paper
> prefers dropping the audio tokens entirely. Worth stating in any write-up.

## Commands

Put `contrastive_decoding/` next to `perturbations.py` at the repo root.

```bash
# 1. preflight - no GPU, no weights. If this fails nothing else will work.
python contrastive_decoding/run_cd.py --check

# 2. smoke test - first time the contrastive branch actually runs
python contrastive_decoding/run_cd.py --model qwen2 --split test --perturbation noise --alpha 0.5 --limit 8

# 3. the sweep - each command loads the model once and runs all 8 conditions + baseline
accelerate launch --num_processes 4 contrastive_decoding/run_cd.py --model qwen2 --split test       --baseline
accelerate launch --num_processes 4 contrastive_decoding/run_cd.py --model qwen2 --split validation --baseline
accelerate launch --num_processes 4 contrastive_decoding/run_cd.py --model af3   --split test       --baseline
accelerate launch --num_processes 4 contrastive_decoding/run_cd.py --model af3   --split validation --baseline

# 4. results table
python contrastive_decoding/run_cd.py --summary
```

Results land in `results/<model>/<split>/`. Re-running a command skips finished
conditions, so an interrupted sweep resumes; `--overwrite` redoes them.

**`--num_processes 4` matters.** At batch size 4 it reproduces her exact 1136-row totals
(accelerate pads the last batch). Other process counts give valid accuracies but totals
that don't line up with her tables — each result file also carries a `deduplicated` block
that is GPU-count independent.

## Dataset paths

The split JSONs are never modified. They hold absolute cluster paths; `load_split()` uses
them as-is when they exist and otherwise looks each file up **by basename** in an audio
root. So the cluster needs no flags and a local checkout works too. Override with
`--data-root` / `--audio-root` or `DCASE_DATA_ROOT` / `DCASE_AUDIO_ROOT`.

| | test | validation |
|---|---|---|
| Examples | 1122 | 1120 |
| Complex / Temporal | 817 / 305 | 816 / 304 |

There are **no Bio examples** in either split, so `domain_average_accuracy` is a macro
average over two groups. That matches her result files.

## Verified

- **Metrics are exact.** Her `dcase_qwen_baseline_results.json` predictions, run through
  this file's extraction and scoring, reproduce her overall accuracy (637/1136), domain
  average (0.5075), and every per-subset and per-question-type cell bit-for-bit.
- **Prompt alignment holds.** For all four perturbations the Qwen2-Audio processor emits
  *bit-identical* `input_ids` for the clean and negative branches — every perturbation
  preserves waveform length, so audio-token expansion is unchanged. Asserted at runtime.
- **The formula is exact** at both alphas, and the negative branch grows by exactly one
  token per decode step (branch sync).
- Qwen2-Audio pads **left**, so appending generated-token embeddings at the end of the
  negative sequence is correct.
- Short runs are refused before writing; writes are atomic; a truncated file from a
  killed job is re-run rather than skipped.

Not verified locally: a full forward/generate pass — the local GPU has 4 GB and
transformers 4.57.0 has no AF3 class. Run the `--limit 8` smoke test first.

## Notes

- `time_mask` and `noise` are stochastic, seeded per example id, so perturbations are
  identical regardless of batch size or GPU count.
- The negative branch keeps **no KV cache** — it re-forwards the whole sequence each
  step. This matches the AdaptivePerturbation implementation and dominates runtime.
- AF3 pads to `max_length` (her config, unchanged), which with the uncached negative
  branch may be slow. The smoke test will show it.
- Answers are free-form text parsed with `\b([A-D])\b`, not constrained decoding. That is
  her protocol; constraining it would measure something her baselines don't.
