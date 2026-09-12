"""
run_cd_af3.py
-------------
Contrastive decoding for Audio Flamingo 3 on the DCASE 2025 multiple-choice
splits.

This is the colleague's af3_evaluation.py protocol -- same system prompt, same
question assembly (note: AF3's has NO trailing "answer with one letter" line,
unlike her Qwen script), same conversation layout with the text block before
the audio block, same padding="max_length", same bfloat16 cast of
input_features, same autocast generate at max_new_tokens=4, same
extract_choice_letter() and metrics -- with an AAD contrastive branch inserted
via a logits processor.

Runs against base nvidia/audio-flamingo-3-hf, no mDPO adapter.

Known cost, flagged rather than changed
---------------------------------------
Her AF3 config pads to "max_length", and the contrastive negative branch
re-forwards the entire padded sequence at every decode step with no KV cache.
That is up to 4 full forward passes over a long padded sequence per batch, on
top of clean generation. The runner prints the actual padded sequence length on
the first batch so the cost is visible immediately. Her generation config is
left exactly as written; if the padded length turns out to be large enough to
make this impractical, that is a decision to take deliberately, not silently.

Usage
-----
    accelerate launch contrastive_decoding/run_cd_af3.py --split test --baseline

    python contrastive_decoding/run_cd_af3.py --split test \
        --perturbation reverse --alpha 1.0 --limit 8
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import librosa
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from torch.utils.data import DataLoader
import transformers
from transformers import AutoProcessor

try:
    from transformers import AudioFlamingo3ForConditionalGeneration
except ImportError:  # checked in main() so --help still works
    AudioFlamingo3ForConditionalGeneration = None

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
MAX_NEW_TOKENS = 4
SAMPLING_RATE = 16000
MODEL_ID = "nvidia/audio-flamingo-3-hf"


def build_conversation(example, audio):
    """Her exact AF3 conversation layout: text block first, then audio.

    AF3's processor accepts a numpy waveform directly in the audio block's
    "path" field.
    """
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": build_question(example, "af3")},
                {"type": "audio", "path": audio},
            ],
        },
    ]


def prepare_inputs(processor, conversations, device):
    """Tokenize a batch of AF3 conversations exactly as her script does."""
    inputs = processor.apply_chat_template(
        conversations,
        tokenize=True,
        return_dict=True,
        add_generation_prompt=True,
        processor_kwargs={
            "padding": "max_length",
            "return_tensors": "pt",
        },
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    inputs["input_features"] = inputs["input_features"].to(torch.bfloat16)
    return inputs


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

    pert_fn = resolve_perturbation(perturbation, sr=SAMPLING_RATE) if perturbation else None

    correct = 0
    results = []
    reported_shape = False

    for batch_index, batch in enumerate(loader):
        conversations = []
        audios = []
        examples = []

        load_start = time.time()
        for example in batch:
            audio = librosa.load(example["audio_url"], sr=SAMPLING_RATE, mono=True)[0]
            conversations.append(build_conversation(example, audio))
            audios.append(audio)
            examples.append(example)

        if accelerator.is_main_process:
            print(
                f"{label} batch {batch_index}: loaded audio in "
                f"{time.time() - load_start:.2f}s",
                flush=True,
            )

        inputs = prepare_inputs(processor, conversations, accelerator.device)

        if accelerator.is_main_process and not reported_shape:
            # padding="max_length" -- surface the real padded length up front,
            # since the negative branch re-forwards this whole sequence per step.
            print(
                f"{label}: padded input_ids shape {tuple(inputs['input_ids'].shape)}",
                flush=True,
            )
            reported_shape = True

        logits_processor_list = None

        if pert_fn is not None:
            perturbed = [
                perturb_for_example(audio, pert_fn, example["id"], seed=seed)
                for audio, example in zip(audios, examples)
            ]
            conversations_neg = [
                build_conversation(example, audio)
                for example, audio in zip(examples, perturbed)
            ]
            inputs_neg = prepare_inputs(processor, conversations_neg, accelerator.device)

            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    # hidden_states[0]: text embeddings with the projected audio
                    # frames already merged in, before the first attention layer.
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
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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
        predictions = processor.tokenizer.batch_decode(
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
    parser.add_argument("--model-path", default=MODEL_ID)
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

    if AudioFlamingo3ForConditionalGeneration is None:
        raise SystemExit(
            "This runner needs a transformers build providing "
            "AudioFlamingo3ForConditionalGeneration -- the same class "
            "af3_evaluation.py imports. Installed: transformers "
            f"{transformers.__version__}, which does not have it.\n"
            "Upgrade transformers in this environment, or run on the cluster "
            "environment the baseline scripts already use."
        )

    # Her af3_evaluation.py constructs Accelerator() with no mixed_precision
    # and handles dtype explicitly instead. Kept as-is.
    accelerator = Accelerator()

    data = load_split(args.split, data_root=args.data_root, audio_root=args.audio_root)
    if args.limit:
        data = data[: args.limit]
    accelerator.print(f"Loaded {len(data)} examples from the {args.split} split")

    accelerator.print(f"Loading base Audio Flamingo 3 from {args.model_path}")
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        args.model_path,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = True

    processor = AutoProcessor.from_pretrained(args.model_path, revision=args.revision)

    revision = resolve_revision(model, args.model_path, args.revision)
    accelerator.print(f"Resolved model revision: {revision}")

    model = model.to(accelerator.device)
    unwrapped_model = accelerator.unwrap_model(model)

    fingerprint = dataset_fingerprint(args.split, data_root=args.data_root)
    accelerator.print(f"Dataset fingerprint: {fingerprint}")

    for perturbation, alpha in conditions:
        path = result_path("af3", args.split, perturbation, alpha)
        metadata = {
            "model": "af3",
            "model_path": args.model_path,
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
        accelerator.print(f"\n=== {label} | af3 | {args.split} ===")

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
