"""
run_af3.py
----------
Run Audio Flamingo 3 evaluations for adaptive audio contrastive decoding.

This module is the AF3-specific front end for the repository's inference-time
perturbation experiments. It loads a yes/no audio question-answering dataset,
builds Audio Flamingo 3 chat conversations, optionally constructs a negative
audio view, and runs generation while a custom LogitsProcessor performs the
contrastive logit update.

Goal
----
The goal is to force an audio-language model to commit to answers that are
grounded in acoustic evidence rather than in the model's learned text priors.
Audio-Aware Decoding (AAD, arXiv 2506.07233) introduces a parallel "negative"
branch alongside the normal clean-audio branch. The original AAD paper evaluates
two kinds of negative input: a blank (silent, all-zeros) waveform and a fully
audio-removed prompt (no audio tokens at all). Both estimate the model's language
prior in the absence of real audio content; audio removal outperforms silent
audio in AAD's own experiments. At every generation step the two branches each
produce a next-token logit vector combined with an alpha-weighted contrastive
update:

    modified_logits = (1 + alpha) * clean_logits - alpha * negative_logits

Tokens whose probability rises specifically because of the clean audio are
promoted; tokens the model would predict even without audio are penalised. This
implementation covers both AAD negative modes (NO_AUDIO) and extends the idea
to waveform-level perturbations — masking, band-filtering, time-reversal,
additive noise, and other transforms from perturbations.py. These are
deliberately destructive inference probes, not training augmentation, chosen to
isolate which parts of a prediction are truly audio-grounded.

Audio Flamingo 3 input assembly
--------------------------------
AF3 is built on the Gemma-3 language model backbone and uses its own AF-Whisper
audio encoder, followed by learnable audio adaptor layers that project pooled
audio frame embeddings into the LM embedding space. The HuggingFace AF3
processor inserts audio placeholder tokens into the token sequence; during the
forward pass, the model replaces those placeholder positions in-place with the
projected audio frame embeddings ("replace-in-place audio/text fusion"). This
approach preserves exact sequence length regardless of audio duration and avoids
any padding artefacts.

Negative-branch synchronization
---------------------------------
For perturbed-audio runs this runner first performs a full forward pass on the
negative conversation (with output_hidden_states=True). It stores hidden_states[0]
— the LM's input embedding layer output — which is the combined sequence of
projected audio frame embeddings and text token embeddings before any transformer
attention is applied. During clean-audio generation with generate(), the
AudioLogitsProcessor appends each newly generated token's embedding to that
stored negative sequence at every decode step, so both branches always condition
on the same partial answer prefix y_{<t}.

Diagnostics and VACoDe-style analysis
---------------------------------------
Step 0 of each decode stores yes/no logits, target-token probability shifts, and
softmax distances between the clean and negative distributions. VACoDe
(arXiv 2408.05337) selects the perturbation that maximises L2 distance between
softmax distributions; our compute_softmax_distances function reports that L2
along with L1, L3, L-infinity, cosine distance, and KL divergence for broader
comparative analysis. The runner also checkpoints long jobs, resumes incomplete
evaluations, writes final RunResult/SampleResult records, and updates per-
directory summaries without retraining the model.

Time-usage profiling
-----------------------
Pass --profile (optionally --profile-max-batches) to measure exactly where
time and GPU/CPU memory go, without changing any model behavior or output.
Writes a JSON + human-readable .log report under <repo root>/profiler_reports/.
See helpers/profiling.py. Disabled by default and a true no-op on that path.
"""

import json
import librosa
import torch
import gc
import os
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor, LogitsProcessor
from adaptive_perturbation.perturbations import (
    get_perturbation,
    get_perturbation_class,
    iter_perturbation_configs,
)
from helpers.config import Config, make_config
from helpers.process_results import SampleResult, RunResult, save_run_result
from helpers.run_helpers import *
from helpers.profiling import Profiler, get_profile_report_path

# --- Configuration ---
Config.model = "af3"
Config._recompute_derived_fields()

MODEL_ID = Config.model_name


TOP_K_LOGGING = 10
BATCH_SIZE = Config.default_batch_size
MAX_NEW_TOKENS = Config.default_max_new_tokens
PREFIX_PROMPT = Config.run_prompt
CHECKPOINT_INTERVAL = 10

# Shared profiler instance — disabled (true no-op) unless a run explicitly passes
# profile=True. Reassigned per-run in main() / spot_check_run() / run_audit() once
# the --profile flag is known, then referenced directly as a module global from
# process_batch()/AudioLogitsProcessor so no function signatures need to change.
PROFILER = Profiler(enabled=False)


set_random_seed(42)


class AudioLogitsProcessor(LogitsProcessor):
    def __init__(self, model, tokenizer, embeds_neg, atts_neg, target_token_ids, alpha=None, top_k=10):
        self.model = model
        self.tokenizer = tokenizer
        self.alpha = Config.alpha if alpha is None else alpha
        self.top_k = top_k
        self.embeds_neg = embeds_neg
        self.atts_neg = atts_neg
        self.target_token_ids = target_token_ids

        self.first_call = True
        self.batch_size = embeds_neg.shape[0]
        self.batch_logs = [[] for _ in range(self.batch_size)]
        self.step0_metrics = [{} for _ in range(self.batch_size)]
        self.softmax_distances: list = [None] * self.batch_size
        self.yes_token_ids = self._token_ids_by_text(["Yes", "yes"])
        self.no_token_ids = self._token_ids_by_text(["No", "no"])
        self._step_idx = 0  # decode-step counter, used to tag profiler samples

    def _first_token_id(self, text):
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            raise ValueError(f"Could not encode token text: {text!r}")
        return token_ids[0]

    def _token_ids_by_text(self, texts):
        return {text: self._first_token_id(text) for text in texts}

    def _get_yes_no_logits(self, logits_row):
        probs = torch.softmax(logits_row, dim=-1)
        out = {}
        for text, token_id in {**self.yes_token_ids, **self.no_token_ids}.items():
            out[text] = {
                "logit": round(logits_row[token_id].item(), 4),
                "prob": round(probs[token_id].item(), 6),
            }
        return out

    def _get_top(self, logits_row):
        probs = torch.softmax(logits_row, dim=-1)
        vals, idxs = torch.topk(logits_row, self.top_k)
        return [{"token": self.tokenizer.decode([i]), "score": round(v.item(), 4), "prob": round(probs[i].item(), 6)}
                for v, i in zip(vals, idxs)]

    def __call__(self, input_ids, scores):
        # `scores` = raw logits for the next token from the CLEAN audio forward pass,
        # provided by HuggingFace's generate() loop at each decoding step.
        target_device = scores.device
        step = self._step_idx

        with torch.no_grad():
            # Isolated from the forward pass below: this is the growing-tensor
            # embed/concat cost (O(sequence length) tensor copy every step).
            with PROFILER.section("aad_step_embed_concat", step=step, batch_size=self.batch_size):
                if self.first_call:
                    # On step 0 the negative embeddings are the full prompt (audio + text).
                    # Move them to the same device as the clean branch.
                    self.embeds_neg = self.embeds_neg.to(target_device)
                    self.atts_neg = self.atts_neg.to(target_device)
                else:
                    # On subsequent steps append the last generated token to the negative
                    # sequence so both branches stay in sync token-by-token.
                    new_tokens = input_ids[:, -1:].to(target_device)
                    new_embeds = self.model.get_input_embeddings()(new_tokens).to(target_device)
                    self.embeds_neg = torch.cat([self.embeds_neg, new_embeds], dim=1)
                    new_atts = (new_tokens != self.tokenizer.eos_token_id).to(dtype=self.atts_neg.dtype).to(target_device)
                    self.atts_neg = torch.cat([self.atts_neg, new_atts], dim=1)

            # Run the model on the NEGATIVE (perturbed) audio embeddings to get its
            # next-token distribution. We only need the last position's logits.
            # This is the prime suspect for O(N^2) decode cost: no past_key_values
            # is passed/reused, so every step reprocesses the full growing sequence.
            with PROFILER.section(
                "aad_negative_forward", step=step, batch_size=self.batch_size,
                seq_len=int(self.embeds_neg.shape[1]),
            ):
                out_neg = self.model(inputs_embeds=self.embeds_neg, attention_mask=self.atts_neg)
            logits_neg = out_neg.logits[:, -1, :]

        # AAD contrastive formula: amplify the clean signal and subtract the perturbed one.
        # alpha=1.0 -> modified = 2*original - negative (equal-weight contrast).
        modified_logits = (1 + self.alpha) * scores - self.alpha * logits_neg

        # Isolated separately: this "just logging" block does many .item()/tokenizer.decode()
        # calls per step (each .item() forces a GPU sync) — tests whether diagnostics
        # collection itself, not just the negative forward pass, is a real cost driver.
        with PROFILER.section("aad_step_diagnostics", step=step, batch_size=self.batch_size):
            for i in range(self.batch_size):
                if self.first_call:
                    # Compute and store softmax distances for VACoDe-style selection.
                    # Done only at step 0: that's when the audio representation matters most
                    # (subsequent steps are just token predictions conditioned on the answer so far).
                    # scores[i] = clean logits, logits_neg[i] = perturbed logits, both [vocab_size].
                    self.softmax_distances[i] = compute_softmax_distances(scores[i], logits_neg[i])
                    tid = self.target_token_ids[i]
                    orig_l, mod_l = scores[i, tid].item(), modified_logits[i, tid].item()
                    neg_l = logits_neg[i, tid].item()
                    orig_p = torch.softmax(scores[i], dim=-1)[tid].item()
                    mod_p = torch.softmax(modified_logits[i], dim=-1)[tid].item()
                    neg_p = torch.softmax(logits_neg[i], dim=-1)[tid].item()

                    self.step0_metrics[i] = {
                        "token": self.tokenizer.decode([tid]),
                        "original_logit": round(orig_l, 4),
                        "negative_logit": round(neg_l, 4),
                        "modified_logit": round(mod_l, 4),
                        "logit_delta": round(mod_l - orig_l, 4),
                        "original_prob": round(orig_p, 6),
                        "negative_prob": round(neg_p, 6),
                        "modified_prob": round(mod_p, 6),
                        "prob_delta": round(mod_p - orig_p, 6)
                    }

                step_log = {
                    "step": len(self.batch_logs[i]),
                    "original_top10": self._get_top(scores[i]),
                    "negative_top10": self._get_top(logits_neg[i]),
                    "modified_top10": self._get_top(modified_logits[i])
                }
                if self.first_call:
                    step_log["yes_no_logits"] = {
                        "original": self._get_yes_no_logits(scores[i]),
                        "negative": self._get_yes_no_logits(logits_neg[i]),
                        "modified": self._get_yes_no_logits(modified_logits[i]),
                    }
                self.batch_logs[i].append(step_log)

        self.first_call = False
        self._step_idx += 1
        return modified_logits


def _cast_inputs(inputs: dict, model) -> dict:
    """Cast floating-point tensors to the model's dtype (e.g. bfloat16).
    Integer tensors (input_ids, attention_mask) are left untouched.
    """
    dtype = next(model.parameters()).dtype
    return {k: v.to(dtype) if torch.is_floating_point(v) else v for k, v in inputs.items()}


def _make_conversation(item, audio=None):
    """Build an AF3-style single-turn conversation dict.

    audio: numpy array (or None for text-only). AF3's processor accepts numpy arrays
    directly in the 'path' field of audio content blocks.
    """
    content = [{"type": "text", "text": PREFIX_PROMPT + item["Q"]}]
    if audio is not None:
        content.append({"type": "audio", "path": audio})
    return [{"role": "user", "content": content}]


def process_batch(batch, model, processor, perturbation_type, perturbation_setting, alpha):
    sr = processor.feature_extractor.sampling_rate
    audios, items, target_ids = [], [], []

    for item in batch:
        if not os.path.exists(item["path"]):
            continue
        with PROFILER.section("audio_load", track_key=str(item["path"])):
            audio, _ = librosa.load(item["path"], sr=sr, mono=True)
        audios.append(audio)
        items.append(item)

        gt_token = item["text"].lower().strip()
        encoded = processor.tokenizer.encode(gt_token, add_special_tokens=False)
        target_ids.append(encoded[0] if encoded else 0)

    if not audios:
        return []

    # AF3 processes conversations directly via apply_chat_template (numpy arrays accepted as 'path').
    conversations_clean = [_make_conversation(item, audio) for item, audio in zip(items, audios)]

    # --- Clean (positive) branch ---
    with PROFILER.section("clean_template_encode", batch_size=len(items)):
        inputs_clean = _cast_inputs(
            processor.apply_chat_template(
                conversations_clean,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
            ).to(model.device),
            model,
        )

    # --- Negative branch & AAD initialization ---
    if perturbation_type == "ORIGINAL":
        logits_processor_list = []
        aad_proc = None

    elif perturbation_type == "NO_AUDIO":
        # Text-only negative branch
        conversations_no_audio = [_make_conversation(item, audio=None) for item in items]
        with PROFILER.section("negative_template_encode_no_audio", batch_size=len(items)):
            inputs_neg = _cast_inputs(
                processor.apply_chat_template(
                    conversations_no_audio,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                ).to(model.device),
                model,
            )

        with PROFILER.section("negative_embed_lookup", batch_size=len(items)):
            neg_embeds = model.get_input_embeddings()(inputs_neg["input_ids"])
        atts_neg = inputs_neg["attention_mask"]

        aad_proc = AudioLogitsProcessor(
            model, processor.tokenizer, neg_embeds, atts_neg, target_ids,
            alpha=alpha, top_k=TOP_K_LOGGING
        )
        logits_processor_list = [aad_proc]

    else:
        # Perturbed audio branch
        with PROFILER.section("perturbation_apply", batch_size=len(items)):
            pert_cls = get_perturbation_class(perturbation_type)
            pert_fn = get_perturbation(pert_cls, perturbation_setting, sr=sr)
            perturbed = [pert_fn(a) for a in audios]

        conversations_neg = [_make_conversation(item, p) for item, p in zip(items, perturbed)]
        with PROFILER.section("negative_template_encode_perturbed", batch_size=len(items)):
            inputs_neg = _cast_inputs(
                processor.apply_chat_template(
                    conversations_neg,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                ).to(model.device),
                model,
            )

        with torch.no_grad():
            # One full forward pass per batch (not per decode step) — contrast this
            # section's cost against aad_negative_forward's (once-per-step) cost.
            with PROFILER.section("negative_forward_full", batch_size=len(items)):
                out_neg = model(
                    **inputs_neg,
                    output_hidden_states=True,
                    return_dict=True
                )
            # hidden_states[0]: input embeddings (text + projected audio) before first attn layer
            neg_embeds = out_neg.hidden_states[0]
        atts_neg = inputs_neg["attention_mask"]

        assert neg_embeds.shape[1] == atts_neg.shape[1], "Shape mismatch!"

        aad_proc = AudioLogitsProcessor(
            model, processor.tokenizer, neg_embeds, atts_neg, target_ids,
            alpha=alpha, top_k=TOP_K_LOGGING
        )
        logits_processor_list = [aad_proc]

    # --- Generation ---
    # Wraps all the internal per-step AudioLogitsProcessor calls above: generate_total's
    # total_s minus the sum of the aad_* sections isolates HF's own clean-branch decode cost.
    with torch.no_grad():
        with PROFILER.section("generate_total", batch_size=len(items), max_new_tokens=MAX_NEW_TOKENS):
            output_ids = model.generate(
                **inputs_clean,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                logits_processor=logits_processor_list,
                pad_token_id=processor.tokenizer.pad_token_id
            )

    with PROFILER.section("decode_extract", batch_size=len(items)):
        response_ids = output_ids[:, inputs_clean["input_ids"].size(1):]
        responses = processor.batch_decode(response_ids, skip_special_tokens=True)

        results: list[SampleResult] = []
        for i, (item, resp) in enumerate(zip(items, responses)):
            trace = aad_proc.batch_logs[i] if aad_proc else None
            sd = aad_proc.softmax_distances[i] if aad_proc else None
            extracted = extract_answer_with_config(resp, trace)
            gt = item["text"].lower().strip()
            is_correct = tokens_match_answer(extracted, gt)
            results.append(
                SampleResult(
                    format="freeform",
                    audio_file=item["path"],
                    question=item["Q"],
                    ground_truth=gt,
                    model_response=resp,
                    extracted_answer=extracted,
                    is_correct=is_correct,
                    aad_enabled=perturbation_type != "ORIGINAL",
                    aad_alpha=alpha,
                    logit_trace=trace,
                    softmax_distance=sd,
                )
            )

    return results


def main(perturbation_type="NO_AUDIO", perturbation_setting=None, alpha=None, results_dir=None, data_path=None, dataset=None, append_softmax_distance=False,
         profile=False, profile_max_batches=None):
    global MAX_NEW_TOKENS, BATCH_SIZE, PREFIX_PROMPT, PROFILER

    if alpha is None:
        alpha = Config.alpha

    cfg = make_config(variant=dataset, model="af3", alpha=alpha)
    MAX_NEW_TOKENS = cfg.default_max_new_tokens
    BATCH_SIZE = cfg.default_batch_size
    PREFIX_PROMPT = cfg.run_prompt

    dataset_path = Path(data_path) if data_path else cfg.curr_dataset_json

    if results_dir:
        results_directory = Path(results_dir)
    else:
        results_directory = cfg.curr_results_dir

    # ORIGINAL has no negative branch so it never gets softmax_distance — treat it
    # as done on file existence even in append mode.
    if append_softmax_distance and perturbation_type.upper() != "ORIGINAL":
        if has_softmax_distance(perturbation_type, alpha, perturbation_setting, results_directory):
            print(f"✓ softmax_distance already present, skipping.")
            return
        # File may exist but lacks the field — fall through to rerun
    elif check_existing_results(perturbation_type, alpha, perturbation_setting, results_directory):
        return

    results_directory.mkdir(parents=True, exist_ok=True)
    output_path = get_output_filename(perturbation_type, alpha, perturbation_setting, results_directory)

    all_results, last_batch_idx = load_checkpoint(
        perturbation_type, alpha, perturbation_setting, results_directory
    )
    start_batch_idx = last_batch_idx + 1

    print(f"Starting evaluation (Audio Flamingo 3)...")
    print(f"  Model: {MODEL_ID}")
    print(f"  Perturbation: {perturbation_type}")
    print(f"  Perturbation Setting: {perturbation_setting}")
    print(f"  Alpha: {alpha}")
    print(f"  Dataset: {dataset_path}")
    print(f"  Output: {output_path}")
    if start_batch_idx > 0:
        print(f"  Resuming from batch: {start_batch_idx}")

    PROFILER = Profiler(enabled=profile)
    if profile:
        print(f"  Profiling: enabled (max_batches={profile_max_batches})")
        PROFILER.set_meta(
            mode="run",
            perturbation_type=perturbation_type,
            perturbation_setting=perturbation_setting,
            alpha=alpha,
            batch_size=BATCH_SIZE,
            max_new_tokens=MAX_NEW_TOKENS,
            model_id=MODEL_ID,
        )

    with PROFILER.section("processor_load"):
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        processor.tokenizer.padding_side = "left"

    with PROFILER.section("model_load"):
        model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            MODEL_ID,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )

    if profile:
        PROFILER.set_meta(
            device_map=getattr(model, "hf_device_map", None),
            gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
        )
        PROFILER.snapshot_memory("after_model_load")

    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    total_batches = (len(data) + BATCH_SIZE - 1) // BATCH_SIZE
    batch_indices = list(range(0, len(data), BATCH_SIZE))

    if profile:
        PROFILER.set_meta(num_batches_total=total_batches)

    truncated = False
    profiled_batches = 0

    for batch_idx, i in enumerate(tqdm(batch_indices, desc="Processing batches", initial=start_batch_idx, total=total_batches)):
        if batch_idx < start_batch_idx:
            continue

        batch = data[i : i + BATCH_SIZE]

        with PROFILER.section("batch_total"):
            batch_results = process_batch(
                batch=batch,
                model=model,
                processor=processor,
                perturbation_type=perturbation_type,
                perturbation_setting=perturbation_setting,
                alpha=alpha
            )

        all_results.extend(batch_results)

        if profile:
            profiled_batches += 1
            PROFILER.snapshot_memory(f"after_batch_{batch_idx}")

        if (batch_idx + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(
                all_results, batch_idx,
                perturbation_type, alpha, perturbation_setting, results_directory
            )
            if profile:
                # Safety-net flush so a long profiled run that gets killed still leaves a report.
                PROFILER.write_report(get_profile_report_path("run", perturbation_type=perturbation_type, alpha=alpha, perturbation_setting=perturbation_setting))

        gc.collect()
        torch.cuda.empty_cache()

        if profile and profile_max_batches is not None and profiled_batches >= profile_max_batches:
            truncated = True
            break

    if profile:
        PROFILER.set_meta(num_batches_profiled=profiled_batches, truncated=truncated)
        PROFILER.write_report(get_profile_report_path("run", perturbation_type=perturbation_type, alpha=alpha, perturbation_setting=perturbation_setting))

    if truncated:
        print(f"\n{'='*60}")
        print(f"PROFILING RUN COMPLETE — partial results NOT saved (--profile-max-batches={profile_max_batches}).")
        print(f"Profiled {profiled_batches} batch(es). See the time-usage report above for the breakdown.")
        print(f"{'='*60}")
        return

    correct = sum(1 for r in all_results if r.is_correct)
    total = len(all_results)
    accuracy = correct / total if total > 0 else 0

    gt_counter = Counter(classify_answer(r.ground_truth) for r in all_results)
    pred_counter = Counter(classify_answer(r.extracted_answer) for r in all_results)

    gt_yes_no_ratio = (
        round(gt_counter.get("yes", 0) / gt_counter.get("no", 0), 3)
        if gt_counter.get("no", 0) > 0
        else float("inf")
    )
    pred_yes_no_ratio = (
        round(pred_counter.get("yes", 0) / pred_counter.get("no", 0), 3)
        if pred_counter.get("no", 0) > 0
        else float("inf")
    )

    flip_analysis = analyze_flips(all_results)
    balance_analysis = analyze_dataset_balance(all_results)

    run_result = RunResult(
        format="freeform",
        timestamp=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        model=MODEL_ID,
        dataset=str(dataset_path),
        total_samples=total,
        perturbation_type=perturbation_type,
        perturbation_setting=perturbation_setting,
        aad_enabled=perturbation_type != "ORIGINAL",
        aad_alpha=alpha,
        max_new_tokens=MAX_NEW_TOKENS,
        batch_size=BATCH_SIZE,
        answer_extraction_uses_step0_yes_no_logits=Config.use_step0_yes_no_logits_extraction,
        ground_truth_yes_no_ratio=gt_yes_no_ratio,
        extracted_prediction_yes_no_ratio=pred_yes_no_ratio,
        ground_truth_non_yes_no_count=gt_counter.get("other", 0),
        extracted_prediction_non_yes_no_count=pred_counter.get("other", 0),
        performance_metrics={
            "accuracy": round(accuracy, 4),
            "correct": correct,
            "total": total,
        },
        flip_analysis=flip_analysis,
        dataset_balance_analysis=balance_analysis,
        results=all_results,
    )

    save_run_result(run_result, output_path)
    delete_checkpoint(perturbation_type, alpha, perturbation_setting, results_directory)

    summary_data = build_summary_from_output(run_result.to_dict())
    update_summaries_file(results_directory, summary_data)

    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy*100:.2f}% ({correct}/{total})")

    def print_dir_stats(label, stats):
        print(f"\n{label}:")
        print(f"  Total:     {stats['total']}")
        print(f"  Yes -> No: {stats['yes_to_no']} ({stats['yes_to_no_pct']}%)")
        print(f"  No -> Yes: {stats['no_to_yes']} ({stats['no_to_yes_pct']}%)")
        print(f"  Other:     {stats['other']} ({stats['other_pct']}%)")

    w2r = flip_analysis['wrong_to_right']['directionality']
    r2w = flip_analysis['right_to_wrong']['directionality']

    print_dir_stats("Helpful Flips (Wrong -> Right)", w2r)
    print_dir_stats("Harmful Flips (Right -> Wrong)", r2w)

    print(f"\nNet Benefit: {flip_analysis['net_benefit']:+d}")
    print(f"\nBias Analysis:")
    bias = balance_analysis['bias_analysis']
    print(f"  Original Yes Bias: {bias['original_yes_bias']:+.2f}%")
    print(f"  Modified Yes Bias: {bias['modified_yes_bias']:+.2f}%")
    print(f"  Bias Reduction: {bias['bias_reduction']:.2f}%")
    print(f"\nResults saved to: {output_path}")
    print(f"{'='*60}")


def spot_check_run(perturbation_type, perturbation_setting, alpha, results_dir, data_path, dataset=None,
                    profile=False, profile_max_batches=None):
    """Re-run samples where step-0 modified top token is not yes/no, using 256 tokens."""
    global MAX_NEW_TOKENS, PREFIX_PROMPT, PROFILER

    cfg = make_config(variant=dataset, model="af3", alpha=alpha)
    results_directory = Path(results_dir) if results_dir else cfg.curr_results_dir
    dataset_path = Path(data_path) if data_path else cfg.curr_dataset_json
    output_path = get_output_filename(perturbation_type, alpha, perturbation_setting, results_directory)

    fpath = output_path if output_path.exists() else None
    if fpath is None:
        alt = Path(str(output_path).removesuffix(".gz")) if str(output_path).endswith(".gz") else Path(str(output_path) + ".gz")
        fpath = alt if alt.exists() else None
    if fpath is None:
        print(f"No results file found for spot-check: {output_path}")
        return

    from helpers.process_results import load_run_result, save_run_result as _save
    run_result = load_run_result(fpath)
    samples = run_result.results

    flagged_indices = [i for i, s in enumerate(samples) if needs_spot_check(s)]

    total = len(samples)
    correct_before = sum(1 for r in samples if r.is_correct)
    accuracy_before = correct_before / total if total else 0.0

    if not flagged_indices:
        print(f"No samples need spot-check: {perturbation_type} α={alpha}")
        return

    print(f"Spot-checking {len(flagged_indices)}/{len(samples)} samples: {perturbation_type} α={alpha}")

    with dataset_path.open("r", encoding="utf-8") as f:
        dataset_items = json.load(f)
    audio_to_item = {item["path"]: item for item in dataset_items}

    # spot_check_run forces max_new_tokens=256 below — the worst case for the suspected
    # O(N^2) AAD negative-branch cost, so this is the highest-value entry point to profile.
    PROFILER = Profiler(enabled=profile)
    if profile:
        print(f"  Profiling: enabled (max_batches={profile_max_batches})")
        PROFILER.set_meta(
            mode="spotcheck",
            perturbation_type=perturbation_type,
            perturbation_setting=perturbation_setting,
            alpha=alpha,
            model_id=MODEL_ID,
            max_new_tokens=256,
            batch_size=cfg.default_batch_size,
            num_batches_total=(len(flagged_indices) + cfg.default_batch_size - 1) // cfg.default_batch_size,
        )

    with PROFILER.section("processor_load"):
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        processor.tokenizer.padding_side = "left"
    with PROFILER.section("model_load"):
        model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16
        )

    if profile:
        PROFILER.set_meta(
            device_map=getattr(model, "hf_device_map", None),
            gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
        )
        PROFILER.snapshot_memory("after_model_load")

    saved_max = MAX_NEW_TOKENS
    saved_prefix = PREFIX_PROMPT
    MAX_NEW_TOKENS = 256
    PREFIX_PROMPT = cfg.run_prompt

    change_records = []
    truncated = False
    profiled_batches = 0
    try:
        batch_sz = cfg.default_batch_size

        for batch_start in range(0, len(flagged_indices), batch_sz):
            slice_idx = flagged_indices[batch_start:batch_start + batch_sz]
            batch_pairs = []
            for idx in slice_idx:
                item = audio_to_item.get(samples[idx].audio_file)
                if item is not None:
                    batch_pairs.append((idx, item))

            if not batch_pairs:
                continue

            with PROFILER.section("batch_total"):
                batch_results = process_batch(
                    batch=[item for _, item in batch_pairs],
                    model=model,
                    processor=processor,
                    perturbation_type=perturbation_type,
                    perturbation_setting=perturbation_setting,
                    alpha=alpha,
                )
            if profile:
                profiled_batches += 1
                PROFILER.snapshot_memory(f"after_batch_{profiled_batches}")

            new_by_audio = {r.audio_file: r for r in batch_results}

            for idx, item in batch_pairs:
                s = samples[idx]
                new_r = new_by_audio.get(s.audio_file)
                if new_r is None:
                    continue

                old_ans = s.extracted_answer
                old_correct = s.is_correct
                old_response = s.model_response

                new_ans = aad_extract_yes_no(new_r.model_response)
                method = "aad_text"
                if new_ans is None:
                    new_ans = extract_answer_from_step0_yes_no_logits(new_r.logit_trace, new_r.model_response)
                    method = "step0_logit"
                if new_ans is None:
                    new_ans = s.extracted_answer
                    method = "original"

                s.extracted_answer = new_ans
                s.is_correct = tokens_match_answer(new_ans, s.ground_truth)
                s.spot_checked = True
                s.spot_check_method = method
                change_records.append({
                    "audio_file": s.audio_file,
                    "ground_truth": s.ground_truth,
                    "old_answer": old_ans,
                    "new_answer": new_ans,
                    "method": method,
                    "was_correct": old_correct,
                    "is_correct": s.is_correct,
                    "old_model_response": old_response,
                    "new_model_response": new_r.model_response,
                })

            if profile and profile_max_batches is not None and profiled_batches >= profile_max_batches:
                truncated = True
                break
    finally:
        MAX_NEW_TOKENS = saved_max
        PREFIX_PROMPT = saved_prefix

    if profile:
        PROFILER.set_meta(num_batches_profiled=profiled_batches, truncated=truncated)
        PROFILER.write_report(get_profile_report_path("spotcheck", perturbation_type=perturbation_type, alpha=alpha, perturbation_setting=perturbation_setting))

    if truncated:
        print(f"  PROFILING RUN COMPLETE — partial spot-check results NOT saved "
              f"({profiled_batches} batch(es) profiled, {len(change_records)} sample(s) touched in memory only).")
        return

    n_changed = sum(1 for c in change_records if c["old_answer"] != c["new_answer"])
    print(f"  Spot-checked {len(change_records)} samples, {n_changed} answers changed.")

    correct = sum(1 for r in samples if r.is_correct)
    total = len(samples)
    pred_counter = Counter(classify_answer(r.extracted_answer) for r in samples)

    run_result.performance_metrics = {
        "accuracy": round(correct / total, 4) if total else 0.0,
        "correct": correct,
        "total": total,
    }
    run_result.flip_analysis = analyze_flips(samples)
    run_result.dataset_balance_analysis = analyze_dataset_balance(samples)
    run_result.extracted_prediction_yes_no_ratio = (
        round(pred_counter.get("yes", 0) / pred_counter.get("no", 0), 3)
        if pred_counter.get("no", 0) > 0 else float("inf")
    )

    _save(run_result, fpath)
    summary_data = build_summary_from_output(run_result.to_dict())
    replace_summary_in_file(results_directory, summary_data)

    accuracy_after = correct / total if total else 0.0
    report_path = append_spot_check_report(
        fpath, MODEL_ID, total, len(flagged_indices),
        accuracy_before, accuracy_after, change_records
    )
    print(f"  Accuracy after spot-check: {accuracy_after*100:.2f}% ({correct}/{total})")
    print(f"  Report: {report_path}")


def run_experiment(perturbation_type, perturbation_setting, alpha, results_dir, data_path, dataset=None, append_softmax_distance=False,
                    profile=False, profile_max_batches=None):
    main(
        perturbation_type=perturbation_type,
        perturbation_setting=perturbation_setting,
        alpha=alpha,
        results_dir=results_dir,
        data_path=data_path,
        dataset=dataset,
        append_softmax_distance=append_softmax_distance,
        profile=profile,
        profile_max_batches=profile_max_batches,
    )


def run_audit(perturbation_type, perturbation_setting, alpha, results_dir, data_path, dataset=None,
              audit_n=50, audit_seed=42, profile=False, profile_max_batches=None):
    import gzip
    import random

    cfg = make_config(variant=dataset, model="af3", alpha=alpha)
    results_directory = Path(results_dir) if results_dir else cfg.curr_results_dir
    output_path = get_output_filename(perturbation_type, alpha, perturbation_setting, results_directory)

    # Accept either .json.gz or plain .json
    if not output_path.exists():
        alt = Path(str(output_path).removesuffix(".gz")) if str(output_path).endswith(".gz") else Path(str(output_path) + ".gz")
        if alt.exists():
            output_path = alt
        else:
            print(f"ERROR: No results file found.\n  Tried: {output_path}\n  Tried: {alt}")
            return 1

    print(f"Auditing: {output_path}")

    opener = gzip.open if str(output_path).endswith(".gz") else open
    with opener(output_path, "rt", encoding="utf-8") as f:
        run_data = json.load(f)

    stored_samples = run_data.get("results", [])
    if not stored_samples:
        print("ERROR: No results found in file.")
        return 1

    meta = run_data.get("metadata", {})
    orig_batch_size = int(meta.get("batch_size") or cfg.default_batch_size)

    random.seed(audit_seed)
    n = min(audit_n, len(stored_samples))
    sampled_indices = sorted(random.sample(range(len(stored_samples)), n))
    print(f"  Total stored: {len(stored_samples)}  |  Auditing: {n}  |  Batch size: {orig_batch_size}")

    dataset_path = Path(data_path) if data_path else cfg.curr_dataset_json
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Group sampled indices by their original batch so we re-run full batches.
    batches_to_audit: dict[int, list[int]] = {}  # batch_start -> [sample_indices]
    for idx in sampled_indices:
        batch_start = (idx // orig_batch_size) * orig_batch_size
        batches_to_audit.setdefault(batch_start, []).append(idx)

    global MAX_NEW_TOKENS, BATCH_SIZE, PREFIX_PROMPT, PROFILER
    MAX_NEW_TOKENS = cfg.default_max_new_tokens
    BATCH_SIZE = orig_batch_size
    PREFIX_PROMPT = cfg.run_prompt

    PROFILER = Profiler(enabled=profile)
    if profile:
        print(f"  Profiling: enabled (max_batches={profile_max_batches})")
        PROFILER.set_meta(
            mode="audit",
            perturbation_type=perturbation_type,
            perturbation_setting=perturbation_setting,
            alpha=alpha,
            batch_size=BATCH_SIZE,
            max_new_tokens=MAX_NEW_TOKENS,
            model_id=MODEL_ID,
            num_batches_total=len(batches_to_audit),
        )

    with PROFILER.section("processor_load"):
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        processor.tokenizer.padding_side = "left"
    with PROFILER.section("model_load"):
        model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16
        )

    if profile:
        PROFILER.set_meta(
            device_map=getattr(model, "hf_device_map", None),
            gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
        )
        PROFILER.snapshot_memory("after_model_load")

    mismatches = []
    skipped = 0
    profiled_batches = 0
    truncated = False

    for batch_start, target_indices in sorted(batches_to_audit.items()):
        batch = data[batch_start : batch_start + orig_batch_size]
        with PROFILER.section("batch_total"):
            batch_results = process_batch(
                batch=batch,
                model=model,
                processor=processor,
                perturbation_type=perturbation_type,
                perturbation_setting=perturbation_setting,
                alpha=alpha,
            )
        if profile:
            profiled_batches += 1
            PROFILER.snapshot_memory(f"after_batch_{profiled_batches}")

        # batch_results may be shorter than batch if some audio files are missing.
        # Map back by position within the batch (process_batch skips missing files,
        # so we must align by audio_file rather than raw index).
        new_by_audio = {r.audio_file: r for r in batch_results}

        for idx in target_indices:
            stored_dict = stored_samples[idx]
            audio_file = str(stored_dict.get("audio_file", ""))
            question = str(stored_dict.get("question", ""))
            new = new_by_audio.get(audio_file)
            if new is None:
                print(f"  WARN: re-run produced no result for {audio_file!r}")
                skipped += 1
                continue

            stored = SampleResult.from_dict(stored_dict, default_format="freeform")

            diffs: dict = {}
            if new.model_response != stored.model_response:
                diffs["model_response"] = (stored.model_response, new.model_response)
            if new.extracted_answer != stored.extracted_answer:
                diffs["extracted_answer"] = (stored.extracted_answer, new.extracted_answer)
            if new.is_correct != stored.is_correct:
                diffs["is_correct"] = (stored.is_correct, new.is_correct)

            # Compare step-0 logits from all three branches
            s0_stored = (stored.logit_trace or [{}])[0]
            s0_new    = (new.logit_trace    or [{}])[0]

            # yes_no_logits: {original/negative/modified: {Yes/yes/No/no: {logit, prob}}}
            yn_stored = s0_stored.get("yes_no_logits") or {}
            yn_new    = s0_new.get("yes_no_logits")    or {}
            for branch in ("original", "negative", "modified"):
                for label in ("Yes", "yes", "No", "no"):
                    sv = (yn_stored.get(branch) or {}).get(label) or {}
                    nv = (yn_new.get(branch)    or {}).get(label) or {}
                    if sv.get("logit") != nv.get("logit"):
                        diffs[f"yes_no_logits.{branch}.{label}.logit"] = (sv.get("logit"), nv.get("logit"))

            # top-1 token + score for all three branches
            for branch in ("original_top10", "negative_top10", "modified_top10"):
                s_top = (s0_stored.get(branch) or [{}])[0]
                n_top = (s0_new.get(branch)    or [{}])[0]
                if s_top.get("token") != n_top.get("token"):
                    diffs[f"{branch}[0].token"] = (s_top.get("token"), n_top.get("token"))
                if s_top.get("score") != n_top.get("score"):
                    diffs[f"{branch}[0].score"] = (s_top.get("score"), n_top.get("score"))

            if diffs:
                mismatches.append({"audio_file": audio_file, "question": question[:60], "diffs": diffs})

        if profile and profile_max_batches is not None and profiled_batches >= profile_max_batches:
            truncated = True
            break

    if profile:
        PROFILER.set_meta(num_batches_profiled=profiled_batches, truncated=truncated)
        PROFILER.write_report(get_profile_report_path("audit", perturbation_type=perturbation_type, alpha=alpha, perturbation_setting=perturbation_setting))

    audited = n - skipped
    print(f"\n{'='*60}")
    print(f"AUDIT: {perturbation_type}{'/' + perturbation_setting if perturbation_setting else ''} α={alpha}")
    if truncated:
        print(f"  NOTE: profiling run truncated after {profiled_batches} batch(es) (--profile-max-batches={profile_max_batches})")
    print(f"  Audited:    {audited}")
    print(f"  Skipped:    {skipped}")
    print(f"  Matches:    {audited - len(mismatches)}")
    print(f"  Mismatches: {len(mismatches)}")
    if mismatches:
        print(f"\n  --- MISMATCHES (first 20) ---")
        for m in mismatches[:20]:
            print(f"  {m['audio_file']}")
            print(f"    Q: {m['question']}")
            for field, (old, new_val) in m["diffs"].items():
                print(f"    {field}: stored={old!r}  re-run={new_val!r}")
    else:
        print(f"  ✅ All audited samples match stored results.")
    print(f"{'='*60}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AudioFlamingo3 - Adaptive Perturbation Evaluation")
    parser.add_argument("--perturbation", "-p", type=str, default="NO_AUDIO",
                        help="Perturbation type (e.g., NO_AUDIO, NOISE, SILENCE, SHUFFLE, ORIGINAL)")
    parser.add_argument("--setting", "-s", type=str, default=None,
                        help="Perturbation setting")
    parser.add_argument("--alpha", "-a", type=float, default=Config.alpha,
                        help=f"Contrastive decoding alpha (default: {Config.alpha})")
    parser.add_argument("--results-dir", "-r", type=str, default=None,
                        help="Directory to save results (default: results/af3/...)")
    parser.add_argument("--data-path", "-d", type=str, default=None,
                        help="Path to dataset JSON file")
    parser.add_argument("--dataset", type=str, nargs="+", default=None,
                        metavar="DATASET",
                        help=(
                            "One or more config variants to run sequentially "
                            "(e.g. ah_existence, 1word_train). "
                            "Use run_parallel.py -d clotho_1word to run all Clotho splits."
                        ))
    parser.add_argument("--model", "-m", type=str, default="af3", choices=["af3"],
                        help="Model slug for this runner (fixed: af3)")
    parser.add_argument("--append-softmax-distance", "--append_softmax_distance",
                        action="store_true",
                        help="Rerun only files missing softmax_distance field; skip files that already have it")
    parser.add_argument("--spot-check", "--spot_check", action="store_true",
                        help="Re-run samples where step-0 modified top token is not yes/no using 256 tokens")
    parser.add_argument("--all", action="store_true",
                        help="Run all perturbation experiments")
    parser.add_argument("--audit", action="store_true",
                        help="Audit mode: re-run random samples from existing results and compare")
    parser.add_argument("--audit-n", type=int, default=50,
                        help="Number of samples to audit per results file (default: 50)")
    parser.add_argument("--audit-seed", type=int, default=42,
                        help="Random seed for audit sample selection (default: 42)")
    parser.add_argument("--profile", action="store_true",
                        help="Enable time/memory profiling (records the full detailed report); "
                             "writes a report under <repo root>/profiler_reports/")
    parser.add_argument("--profile-max-batches", "--profile_max_batches", type=int, default=None,
                        help="Stop after N batches when profiling (fast diagnostic run — "
                             "nothing is written to results/checkpoints in this mode)")

    args = parser.parse_args()

    datasets = args.dataset if args.dataset is not None else [None]
    audit_exit_code = 0
    for _dataset in datasets:
        args.dataset = _dataset

        if args.spot_check:
            if args.all:
                for perturbation_type, setting in iter_perturbation_configs():
                    spot_check_run(
                        perturbation_type=perturbation_type,
                        perturbation_setting=setting,
                        alpha=args.alpha,
                        results_dir=args.results_dir,
                        data_path=args.data_path,
                        dataset=args.dataset,
                        profile=args.profile,
                        profile_max_batches=args.profile_max_batches,
                    )
            else:
                spot_check_run(
                    perturbation_type=args.perturbation,
                    perturbation_setting=args.setting,
                    alpha=args.alpha,
                    results_dir=args.results_dir,
                    data_path=args.data_path,
                    dataset=args.dataset,
                    profile=args.profile,
                    profile_max_batches=args.profile_max_batches,
                )
            continue

        if args.audit:
            if args.all:
                for perturbation_type, setting in iter_perturbation_configs():
                    code = run_audit(
                        perturbation_type=perturbation_type,
                        perturbation_setting=setting,
                        alpha=args.alpha,
                        results_dir=args.results_dir,
                        data_path=args.data_path,
                        dataset=args.dataset,
                        audit_n=args.audit_n,
                        audit_seed=args.audit_seed,
                        profile=args.profile,
                        profile_max_batches=args.profile_max_batches,
                    )
                    audit_exit_code = max(audit_exit_code, code)
            else:
                code = run_audit(
                    perturbation_type=args.perturbation,
                    perturbation_setting=args.setting,
                    alpha=args.alpha,
                    results_dir=args.results_dir,
                    data_path=args.data_path,
                    dataset=args.dataset,
                    audit_n=args.audit_n,
                    audit_seed=args.audit_seed,
                    profile=args.profile,
                    profile_max_batches=args.profile_max_batches,
                )
                audit_exit_code = max(audit_exit_code, code)
            continue  # skip non-audit block for this dataset

        asd = args.append_softmax_distance
        if args.all:
            for perturbation_type, setting in iter_perturbation_configs():
                run_experiment(
                    perturbation_type=perturbation_type,
                    perturbation_setting=setting,
                    alpha=args.alpha,
                    results_dir=args.results_dir,
                    data_path=args.data_path,
                    dataset=args.dataset,
                    append_softmax_distance=asd,
                    profile=args.profile,
                    profile_max_batches=args.profile_max_batches,
                )
        else:
            run_experiment(
                perturbation_type=args.perturbation,
                perturbation_setting=args.setting,
                alpha=args.alpha,
                results_dir=args.results_dir,
                data_path=args.data_path,
                dataset=args.dataset,
                append_softmax_distance=asd,
                profile=args.profile,
                profile_max_batches=args.profile_max_batches,
            )

    if args.audit:
        import sys; sys.exit(audit_exit_code)
