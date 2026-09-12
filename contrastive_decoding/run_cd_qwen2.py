"""
run_cd_qwen2.py
---------------
Contrastive decoding for Qwen2-Audio on the DCASE 2025 multiple-choice splits.

This is the colleague's qwen2_evaluation.py evaluation protocol -- same system
prompt, same question assembly, same conversation layout, same
max_new_tokens=16 / do_sample=False, same extract_choice_letter(), same
subset inference, same metrics -- with an AAD contrastive branch inserted at
generation time via a logits processor. Nothing about how answers are produced
from logits or scored afterwards differs from her baseline.

Runs against the BASE Qwen2-Audio-7B-Instruct weights, no mDPO adapter, so the
numbers sit next to her dcase_qwen_baseline_results.json as a training-free
alternative to mDPO.

The model is loaded once and every (perturbation, alpha) combination is run
against it. Existing result files are skipped, so an interrupted sweep resumes
by re-running the same command.

Usage
-----
    # everything: 4 perturbations x 2 alphas, plus the no-CD baseline
    accelerate launch contrastive_decoding/run_cd_qwen2.py --split test --baseline

    # one condition
    python contrastive_decoding/run_cd_qwen2.py \
        --split test --perturbation noise --alpha 0.5

    # locally, where the audio lives in a flat directory
    python contrastive_decoding/run_cd_qwen2.py --split test \
        --audio-root "dcase might not be the same as on the remote/dcase_dev_audio_files"
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import librosa
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contrastive_decoding.cd_common import (  # noqa: E402
    DEFAULT_ALPHAS,
    PERTURBATION_PARAMS,
    PERTURBATIONS,
    SYSTEM_PROMPT,
    build_question,
    build_result_row,
    dataset_fingerprint,
    deduplicate,
    load_split,
    perturb_for_example,
    print_summary,
    result_path,
    resolve_perturbation,
    resolve_revision,
    should_skip,
    write_results,
)
from contrastive_decoding.cd_processor import ContrastiveAudioLogitsProcessor  # noqa: E402

# Her settings, unchanged.
BATCH_SIZE = 4
MAX_NEW_TOKENS = 16
MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"

# Cluster locations from her qwen2_evaluation.py. Used when present; otherwise
# the run falls back to the hub id and the default cache.
CLUSTER_MODEL_PATH = (
    "/data/not_backed_up/cosgrv/huggingface_cache/hub/"
    "models--Qwen--Qwen2-Audio-7B-Instruct/snapshots/"
    "0a095220c30b7b31434169c3086508ef3ea5bf0a"
)
CLUSTER_CACHE_DIR = "/data/not_backed_up/cosgrv/huggingface_cache/hub"


def build_conversation(example, audio, model_name="qwen2"):
    """Her exact Qwen2 conversation layout: audio block first, then text."""
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": build_question(example, model_name)},
            ],
        },
    ]


def evaluate(
    unwrapped_model,
    processor,
    loader,
    accelerator,
    label,
    perturbation=None,
    alpha=None,
    seed=42,
):
    """Run one condition. perturbation=None means plain generation (baseline)."""
    unwrapped_model.eval()

    sampling_rate = processor.feature_extractor.sampling_rate
    pert_fn = resolve_perturbation(perturbation, sr=sampling_rate) if perturbation else None

    correct = 0
    results = []

    for batch_index, batch in enumerate(loader):
        conversations = []
        audios = []
        examples = []

        load_start = time.time()
        for example in batch:
            audio = librosa.load(example["audio_url"], sr=sampling_rate, mono=True)[0]
            conversations.append(build_conversation(example, audio))
            audios.append(audio)
            examples.append(example)

        if accelerator.is_main_process:
            print(
                f"{label} batch {batch_index}: loaded audio in "
                f"{time.time() - load_start:.2f}s",
                flush=True,
            )

        # The chat template only inserts audio placeholders; the waveforms are
        # passed separately below. So clean and negative branches share one
        # formatted text, and -- because every perturbation preserves waveform
        # length -- they expand to identical token counts.
        formatted_texts = processor.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = processor(
            text=formatted_texts,
            audio=audios,
            sampling_rate=sampling_rate,
            padding=True,
            return_tensors="pt",
        )
        inputs = {
            key: (value.to(accelerator.device) if isinstance(value, torch.Tensor) else value)
            for key, value in inputs.items()
        }

        logits_processor_list = None

        if pert_fn is not None:
            perturbed = [
                perturb_for_example(audio, pert_fn, example["id"], seed=seed)
                for audio, example in zip(audios, examples)
            ]

            inputs_neg = processor(
                text=formatted_texts,
                audio=perturbed,
                sampling_rate=sampling_rate,
                padding=True,
                return_tensors="pt",
            )
            inputs_neg = {
                key: (value.to(accelerator.device) if isinstance(value, torch.Tensor) else value)
                for key, value in inputs_neg.items()
            }

            with torch.inference_mode():
                # hidden_states[0] is the LM input embedding sequence: projected
                # audio frames merged with text token embeddings, before any
                # transformer layer.
                out_neg = unwrapped_model(
                    **inputs_neg,
                    output_hidden_states=True,
                    return_dict=True,
                )
                neg_embeds = out_neg.hidden_states[0].detach().clone()

            if neg_embeds.shape[1] != inputs["input_ids"].shape[1]:
                raise RuntimeError(
                    "Clean and negative branches have different sequence lengths "
                    f"({inputs['input_ids'].shape[1]} vs {neg_embeds.shape[1]}). "
                    "The contrastive subtraction assumes aligned prompts."
                )

            logits_processor_list = [
                ContrastiveAudioLogitsProcessor(
                    model=unwrapped_model,
                    embeds_neg=neg_embeds,
                    atts_neg=inputs_neg["attention_mask"],
                    alpha=alpha,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )
            ]

            del out_neg, inputs_neg

        generation_start = time.time()
        with torch.inference_mode():
            output_ids = unwrapped_model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                logits_processor=logits_processor_list,
            )

        if accelerator.is_main_process:
            print(
                f"{label} batch {batch_index}: generation took "
                f"{time.time() - generation_start:.2f}s",
                flush=True,
            )

        prompt_length = inputs["input_ids"].shape[1]
        predictions = processor.batch_decode(
            output_ids[:, prompt_length:],
            skip_special_tokens=True,
        )

        # zip() would silently truncate the batch and produce a short result
        # file that still looks successful. Fail loudly instead.
        if len(predictions) != len(examples):
            raise RuntimeError(
                f"Batch {batch_index}: decoded {len(predictions)} predictions for "
                f"{len(examples)} examples. Refusing to drop rows."
            )

        for example, prediction in zip(examples, predictions):
            row = build_result_row(example, prediction)
            results.append(row)
            if row["correct"]:
                correct += 1

    local_correct = torch.tensor([correct], device=accelerator.device, dtype=torch.long)
    local_total = torch.tensor([len(results)], device=accelerator.device, dtype=torch.long)
    global_correct = accelerator.gather(local_correct).sum().item()
    global_total = accelerator.gather(local_total).sum().item()
    all_results = gather_object(results)

    return all_results, global_correct, global_total


def create_loader(data):
    return DataLoader(
        data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda examples: examples,
        num_workers=0,
    )


def main():
    global BATCH_SIZE

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=["test", "validation"])
    parser.add_argument(
        "--perturbation",
        action="append",
        choices=sorted(PERTURBATIONS),
        help="Repeatable. Defaults to all four.",
    )
    parser.add_argument(
        "--alpha",
        action="append",
        type=float,
        help="Repeatable. Defaults to 0.5 and 1.0.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Also run plain generation with no contrastive branch.",
    )
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--audio-root", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--revision",
        default=None,
        help="Pin a model revision (branch, tag, or commit). Recorded in results.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Debug: first N examples.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size

    perturbations = args.perturbation or sorted(PERTURBATIONS)
    alphas = args.alpha or list(DEFAULT_ALPHAS)

    conditions = []
    if args.baseline:
        conditions.append((None, None))
    for perturbation in perturbations:
        for alpha in alphas:
            conditions.append((perturbation, alpha))

    accelerator = Accelerator(mixed_precision="bf16")

    data = load_split(args.split, data_root=args.data_root, audio_root=args.audio_root)
    if args.limit:
        data = data[: args.limit]
    accelerator.print(f"Loaded {len(data)} examples from the {args.split} split")

    model_path = args.model_path or (
        CLUSTER_MODEL_PATH if os.path.isdir(CLUSTER_MODEL_PATH) else MODEL_ID
    )
    cache_dir = args.cache_dir or (
        CLUSTER_CACHE_DIR if os.path.isdir(CLUSTER_CACHE_DIR) else None
    )
    accelerator.print(f"Loading base Qwen2-Audio from {model_path}")

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_path,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    model.config.use_cache = True

    processor = AutoProcessor.from_pretrained(
        model_path,
        revision=args.revision,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    revision = resolve_revision(model, model_path, args.revision)
    accelerator.print(f"Resolved model revision: {revision}")

    model = model.to(accelerator.device)
    unwrapped_model = accelerator.unwrap_model(model)

    fingerprint = dataset_fingerprint(args.split, data_root=args.data_root)
    accelerator.print(f"Dataset fingerprint: {fingerprint}")

    for perturbation, alpha in conditions:
        path = result_path("qwen2", args.split, perturbation, alpha)
        metadata = {
            "model": "qwen2",
            "model_path": model_path,
            "revision": revision,
            "weights": "base (no mDPO adapter)",
            "split": args.split,
            "perturbation": perturbation,
            "perturbation_params": PERTURBATION_PARAMS.get(perturbation, {}),
            "alpha": alpha,
            "max_new_tokens": MAX_NEW_TOKENS,
            "batch_size": BATCH_SIZE,
            "seed": args.seed,
            "num_processes": accelerator.num_processes,
            "dataset_examples": len(data),
            "dataset_fingerprint": fingerprint,
        }

        if should_skip(path, metadata, args.overwrite):
            accelerator.print(f"Skipping completed {path.name} (use --overwrite to redo)")
            continue

        label = "baseline" if perturbation is None else f"cd_{perturbation}_alpha_{alpha}"
        accelerator.print(f"\n=== {label} | qwen2 | {args.split} ===")

        loader = accelerator.prepare(create_loader(data))

        all_results, global_correct, global_total = evaluate(
            unwrapped_model=unwrapped_model,
            processor=processor,
            loader=loader,
            accelerator=accelerator,
            label=label,
            perturbation=perturbation,
            alpha=alpha,
            seed=args.seed,
        )

        if accelerator.is_main_process:
            summary = write_results(path, metadata, all_results, expected=len(data))
            print_summary(label, summary, deduped_total=len(deduplicate(all_results)))
            print(f"\nSaved results to {path}", flush=True)

        accelerator.wait_for_everyone()
        del loader
        torch.cuda.empty_cache()

    accelerator.wait_for_everyone()
    accelerator.print("\nAll requested conditions complete.")


if __name__ == "__main__":
    main()
