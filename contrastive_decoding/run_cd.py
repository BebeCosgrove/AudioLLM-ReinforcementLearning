"""
Contrastive decoding (AAD) on the DCASE 2025 multiple-choice splits,
for Qwen2-Audio and Audio Flamingo 3.

    modified_logits = (1 + alpha) * clean_logits - alpha * negative_logits

The clean branch sees the real audio; the negative branch sees the same prompt
with a perturbed waveform. Everything about how an answer is produced from text
and scored is copied from af3_evaluation.py / qwen2_evaluation.py and must stay
that way, or the numbers stop being comparable to the mDPO baselines.

    python run_cd.py --check                             # preflight, no GPU
    python run_cd.py --model qwen2 --split test --limit 8
    accelerate launch --num_processes 4 run_cd.py --model qwen2 --split test --baseline
    python run_cd.py --summary
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from torch.utils.data import DataLoader
from transformers import AutoProcessor, LogitsProcessor, Qwen2AudioForConditionalGeneration

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from perturbations import Perturbation, get_perturbation  # noqa: E402

SR = 16000
BATCH_SIZE = 4

# The four perturbations at the settings used for the mDPO training data.
# Resolved through the repo-root perturbations.py so both come from one source.
# NO_AUDIO there is np.zeros_like(audio) -- silent audio, audio tokens still
# present -- not the audio-removed prompt the AAD paper prefers.
PERTURBATIONS = {
    "no_audio":  (Perturbation.NO_AUDIO,  None),
    "reverse":   (Perturbation.REVERSE,   "full"),
    "time_mask": (Perturbation.TIME_MASK, "light"),   # n_masks=3, max_width=0.08
    "noise":     (Perturbation.NOISE,     "weak"),    # sigma=0.02
}
ALPHAS = (0.5, 1.0)

MODELS = {
    "qwen2": {
        "hub_id": "Qwen/Qwen2-Audio-7B-Instruct",
        "max_new_tokens": 16,
        # Her Qwen prompt has this trailing line; her AF3 prompt does not.
        "suffix": "\nAnswer with exactly one letter: A, B, C, or D.",
    },
    "af3": {
        "hub_id": "nvidia/audio-flamingo-3-hf",
        "max_new_tokens": 4,
        "suffix": "",
    },
}

SYSTEM_PROMPT = (
    "Focus on the given audio and answer the following multiple-choice "
    "question. Respond with the letter of the correct answer (A, B, C, or D)."
)

CLUSTER = "/data/not_backed_up/cosgrv/af3_project/dcase_2025/2025_DCASE_AudioQA"
DATA_ROOTS = [f"{CLUSTER}/combined_json", str(HERE.parent / "dcase might not be the same as on the remote")]
AUDIO_ROOTS = [f"{CLUSTER}/local_audio_path/dev", str(HERE.parent / "dcase might not be the same as on the remote" / "dcase_dev_audio_files")]
SPLITS = {"test": "dcase_split_test.json", "validation": "dcase_split_validation.json"}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_split(split, data_root=None, audio_root=None):
    """Load a split, rebasing audio paths by basename when needed.

    The split JSONs are read-only and store absolute cluster paths. On the
    cluster those resolve as-is; elsewhere each file is looked up by name in
    audio_root. Nothing is written back.
    """
    roots = [data_root] if data_root else [os.environ.get("DCASE_DATA_ROOT")] + DATA_ROOTS
    path = next((Path(r) / SPLITS[split] for r in roots if r and (Path(r) / SPLITS[split]).exists()), None)
    if path is None:
        raise SystemExit(f"Could not find {SPLITS[split]}. Pass --data-root. Searched: {DATA_ROOTS}")

    data = json.loads(path.read_text(encoding="utf-8"))

    audio_roots = [audio_root] if audio_root else [os.environ.get("DCASE_AUDIO_ROOT")] + AUDIO_ROOTS
    found = next((Path(r) for r in audio_roots if r and Path(r).is_dir()), None)

    missing = 0
    for i, ex in enumerate(data):
        ex["row"] = i  # used to drop accelerate's padding duplicates later
        if os.path.exists(ex["audio_url"]):
            continue
        rebased = found / Path(ex["audio_url"].replace("\\", "/")).name if found else None
        if rebased and rebased.exists():
            ex["audio_url"] = str(rebased)
        else:
            missing += 1
    if missing:
        raise SystemExit(f"{missing}/{len(data)} audio files not found for {split} (audio root: {found})")

    return data, path


def perturb(audio, name, example_id):
    """Apply a perturbation, seeded from the example id.

    Seeding per example rather than per call keeps time_mask and noise
    identical regardless of batch size, GPU count, or shard boundaries.
    """
    pert_type, setting = PERTURBATIONS[name]
    fn = get_perturbation(pert_type, setting, sr=SR)
    state = np.random.get_state()
    try:
        np.random.seed(zlib.crc32(example_id.encode()) & 0xFFFFFFFF)
        return fn(audio)
    finally:
        np.random.set_state(state)


# --------------------------------------------------------------------------
# Her evaluation protocol -- copied verbatim, do not "improve"
# --------------------------------------------------------------------------

def build_question(example, model):
    return f"{example['question']}\n" + "\n".join(example["choice"]) + MODELS[model]["suffix"]


def extract_choice_letter(text):
    match = re.search(r"\b([A-D])\b", text.strip().upper())
    return match.group(1) if match else None


def infer_subset(example):
    """fold... -> Temporal, audio... -> Complex, anything else -> Bio."""
    name = Path(str(example.get("audio_url", ""))).name.lower()
    return "Temporal" if name.startswith("fold") else "Complex" if name.startswith("audio") else "Bio"


def score(rows):
    """Overall accuracy, per-subset, per-question-type, and the macro average.

    Only Complex and Temporal occur in these splits -- there are no Bio
    examples -- so the macro average is over two groups in practice.
    """
    def table(key):
        stats = defaultdict(lambda: [0, 0])
        for r in rows:
            stats[r[key]][0] += r["correct"]
            stats[r[key]][1] += 1
        return {k: {"accuracy": c / t if t else 0.0, "correct": c, "total": t} for k, (c, t) in stats.items()}

    by_subset = table("subset")
    present = [by_subset[s]["accuracy"] for s in ("Bio", "Temporal", "Complex") if s in by_subset]
    correct = sum(r["correct"] for r in rows)
    return {
        "overall": {"accuracy": correct / len(rows) if rows else 0.0, "correct": correct, "total": len(rows)},
        "domain_average_accuracy": sum(present) / len(present) if present else 0.0,
        "accuracy_by_subset": by_subset,
        "accuracy_by_question_type": table("question_type"),
    }


# --------------------------------------------------------------------------
# Contrastive decoding
# --------------------------------------------------------------------------

class ContrastiveLogits(LogitsProcessor):
    """Applies (1 + alpha) * clean - alpha * negative at each decoding step.

    Seeded with the negative prompt's input embeddings (hidden_states[0]: the
    projected audio frames already merged with the text tokens). At each step
    the token the clean branch just emitted is appended, so both branches stay
    conditioned on the same prefix. No KV cache on the negative branch -- it
    re-forwards the whole sequence each step, which is the dominant cost.
    """

    def __init__(self, model, embeds, mask, alpha, eos_id, autocast):
        self.model, self.embeds, self.mask = model, embeds, mask
        self.alpha, self.eos_id, self.autocast = float(alpha), eos_id, autocast
        self.first = True

    def __call__(self, input_ids, scores):
        with torch.no_grad(), self.autocast():
            if self.first:
                self.embeds = self.embeds.to(scores.device)
                self.mask = self.mask.to(scores.device)
                self.first = False
            else:
                new = input_ids[:, -1:].to(scores.device)
                emb = self.model.get_input_embeddings()(new).to(self.embeds.dtype)
                self.embeds = torch.cat([self.embeds, emb], dim=1)
                self.mask = torch.cat([self.mask, (new != self.eos_id).to(self.mask.dtype)], dim=1)

            negative = self.model(inputs_embeds=self.embeds, attention_mask=self.mask).logits[:, -1, :]

        return (1 + self.alpha) * scores - self.alpha * negative.to(scores.dtype)


# --------------------------------------------------------------------------
# Model-specific input/output
# --------------------------------------------------------------------------

def load_model(model_name, model_path):
    if model_name == "qwen2":
        cls = Qwen2AudioForConditionalGeneration
    else:
        try:
            from transformers import AudioFlamingo3ForConditionalGeneration as cls
        except ImportError:
            import transformers
            raise SystemExit(
                "Audio Flamingo 3 needs a transformers build providing "
                "AudioFlamingo3ForConditionalGeneration (the class af3_evaluation.py "
                f"imports). Installed: {transformers.__version__}."
            )
    model = cls.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.config.use_cache = True
    return model, AutoProcessor.from_pretrained(model_path, trust_remote_code=True)


def encode(model_name, processor, examples, audios, device):
    """Build model inputs. The two models want genuinely different shapes."""
    if model_name == "qwen2":
        # Audio block first, then text; waveforms passed separately to the
        # processor, so clean and negative branches share one formatted text.
        convs = [
            [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
             {"role": "user", "content": [{"type": "audio", "audio": a},
                                          {"type": "text", "text": build_question(ex, "qwen2")}]}]
            for ex, a in zip(examples, audios)
        ]
        texts = processor.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=texts, audio=audios, sampling_rate=SR, padding=True, return_tensors="pt")
    else:
        # Text block first, then audio; AF3 accepts a numpy array as "path".
        convs = [
            [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
             {"role": "user", "content": [{"type": "text", "text": build_question(ex, "af3")},
                                          {"type": "audio", "path": a}]}]
            for ex, a in zip(examples, audios)
        ]
        inputs = processor.apply_chat_template(
            convs, tokenize=True, return_dict=True, add_generation_prompt=True,
            processor_kwargs={"padding": "max_length", "return_tensors": "pt"},
        )

    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    if "input_features" in inputs:
        inputs["input_features"] = inputs["input_features"].to(torch.bfloat16)
    return inputs


def autocast_for(model_name):
    """AF3 generates under autocast in her script; Qwen2 does not."""
    if model_name == "af3":
        return lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------

def run_condition(model_name, model, processor, loader, acc, perturbation, alpha, label):
    cfg = MODELS[model_name]
    autocast = autocast_for(model_name)
    rows = []
    started = time.time()

    for i, batch in enumerate(loader):
        audios = [librosa.load(ex["audio_url"], sr=SR, mono=True)[0] for ex in batch]
        inputs = encode(model_name, processor, batch, audios, acc.device)

        processors = None
        if perturbation:
            perturbed = [perturb(a, perturbation, ex["id"]) for a, ex in zip(audios, batch)]
            neg = encode(model_name, processor, batch, perturbed, acc.device)
            with torch.inference_mode(), autocast():
                embeds = model(**neg, output_hidden_states=True, return_dict=True).hidden_states[0]
            assert embeds.shape[1] == inputs["input_ids"].shape[1], "clean/negative prompt lengths differ"
            processors = [ContrastiveLogits(
                model, embeds.detach().clone(), neg["attention_mask"],
                alpha, processor.tokenizer.eos_token_id, autocast,
            )]
            del neg

        with torch.inference_mode(), autocast():
            out = model.generate(**inputs, max_new_tokens=cfg["max_new_tokens"],
                                 do_sample=False, logits_processor=processors)

        preds = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        assert len(preds) == len(batch), f"batch {i}: {len(preds)} predictions for {len(batch)} examples"

        for ex, pred in zip(batch, preds):
            rows.append({
                "id": ex["id"], "question": ex["question"],
                "question_type": ex.get("question_type", "unknown"), "subset": infer_subset(ex),
                "answer": ex["answer"], "prediction": pred,
                "pred_letter": extract_choice_letter(pred),
                "gt_letter": extract_choice_letter(ex["answer"]),
                "correct": extract_choice_letter(pred) == extract_choice_letter(ex["answer"]),
                "audio_url": ex["audio_url"], "row": ex["row"],
            })

        if acc.is_main_process and (i == 0 or i % 10 == 0):
            done = sum(r["correct"] for r in rows)
            rate = (time.time() - started) / (i + 1)
            # Batch 0 also reports the padded prompt length: AF3 pads to
            # max_length, and the negative branch re-forwards all of it every
            # step, so this is what makes an AF3 sweep cheap or expensive.
            shape = f"  seq={inputs['input_ids'].shape[1]}" if i == 0 else ""
            print(f"  {label} batch {i}/{len(loader)}: {done}/{len(rows)} correct  "
                  f"{rate:.1f}s/batch  eta {rate * (len(loader) - i - 1) / 60:.0f}m{shape}", flush=True)

    return gather_object(rows)


def write(path, rows, meta, expected):
    """Write one run's results, atomically so a killed job leaves no half-file."""
    unique = list({r["row"]: r for r in rows}.values())
    assert len(unique) == expected, f"expected {expected} unique rows, got {len(unique)}"

    payload = {"metadata": meta, **score(rows), "deduplicated": score(unique), "results": rows}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)  # atomic
    return payload


def summary():
    """Print every result file found, with its delta against that group's baseline."""
    files = sorted((HERE / "results").glob("*/*/*.json"))
    if not files:
        print("No results yet.")
        return

    runs = []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        m = d.get("metadata", {})
        runs.append((m.get("model"), m.get("split"), m.get("perturbation") or "-",
                     m.get("alpha"), d["overall"]["accuracy"], d["domain_average_accuracy"],
                     d["overall"]["total"]))

    base = {(r[0], r[1]): r[4] for r in runs if r[2] == "-"}
    print(f"{'model':6} {'split':11} {'perturbation':13} {'alpha':>5} {'acc':>7} {'d-acc':>7} {'vs base':>9} {'n':>6}")
    print("-" * 70)
    for model, split, pert, alpha, acc, dacc, n in sorted(runs, key=lambda r: (r[0], r[1], r[2], r[3] or 0)):
        delta = f"{acc - base[(model, split)]:+.4f}" if pert != "-" and (model, split) in base else "-"
        print(f"{model:6} {split:11} {pert:13} {'-' if alpha is None else alpha:>5} "
              f"{acc:7.4f} {dacc:7.4f} {delta:>9} {n:6d}")


def check(args):
    """Preflight: no GPU, no model weights. Run this before queueing anything."""
    for split in SPLITS:
        data, path = load_split(split, args.data_root, args.audio_root)
        subsets = defaultdict(int)
        for ex in data:
            subsets[infer_subset(ex)] += 1
        unparsed = sum(extract_choice_letter(ex["answer"]) is None for ex in data)
        print(f"{split:11} {len(data):5d} examples  audio all resolved  {dict(subsets)}  "
              f"unparsable answers: {unparsed}")
        print(f"{'':11} {path}")

    audio = librosa.load(data[0]["audio_url"], sr=SR, mono=True)[0]
    print(f"\nprobe: {Path(data[0]['audio_url']).name}  {len(audio)} samples")
    for name in PERTURBATIONS:
        out = perturb(audio, name, data[0]["id"])
        again = perturb(audio, name, data[0]["id"])
        # Length must be preserved or the audio-token count changes and the
        # clean/negative prompts stop lining up.
        assert len(out) == len(audio), f"{name} changed waveform length"
        assert np.array_equal(out, again), f"{name} is not deterministic"
        print(f"  {name:10} length preserved  deterministic  "
              f"rms delta {float(np.sqrt(np.mean((out - audio) ** 2))):.4f}")

    print("\nPrompts:")
    for m in MODELS:
        print(f"  --- {m} ---")
        for line in build_question(data[0], m).splitlines():
            print(f"    {line}")
    print("\nAll checks passed.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=sorted(MODELS))
    p.add_argument("--split", default="test", choices=sorted(SPLITS))
    p.add_argument("--perturbation", action="append", choices=sorted(PERTURBATIONS))
    p.add_argument("--alpha", action="append", type=float)
    p.add_argument("--baseline", action="store_true", help="Also run with no contrastive branch.")
    p.add_argument("--check", action="store_true", help="Preflight only; no GPU needed.")
    p.add_argument("--summary", action="store_true", help="Print the results table and exit.")
    p.add_argument("--data-root")
    p.add_argument("--audio-root")
    p.add_argument("--model-path", help="Defaults to the hub id for --model.")
    p.add_argument("--limit", type=int, help="Debug: first N examples.")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.check:
        return check(args)
    if args.summary:
        return summary()
    if not args.model:
        p.error("--model is required (or use --check / --summary)")

    conditions = ([(None, None)] if args.baseline else []) + [
        (pert, alpha)
        for pert in (args.perturbation or sorted(PERTURBATIONS))
        for alpha in (args.alpha or ALPHAS)
    ]

    acc = Accelerator()
    data, _ = load_split(args.split, args.data_root, args.audio_root)
    if args.limit:
        data = data[: args.limit]
    acc.print(f"{len(data)} examples from the {args.split} split")

    model_path = args.model_path or MODELS[args.model]["hub_id"]
    acc.print(f"Loading {args.model} from {model_path}")
    model, processor = load_model(args.model, model_path)
    model = model.to(acc.device).eval()

    out_dir = HERE / "results" / args.model / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    for perturbation, alpha in conditions:
        label = "baseline" if perturbation is None else f"cd_{perturbation}_alpha_{alpha}"
        path = out_dir / f"{label}.json"

        # Re-running the same command resumes: finished conditions are skipped.
        # A file that won't parse is a half-written one from a killed job, so redo it.
        if path.exists() and not args.overwrite:
            try:
                json.loads(path.read_text(encoding="utf-8"))
                acc.print(f"Skipping {label} (already done)")
                continue
            except json.JSONDecodeError:
                acc.print(f"{label} is truncated, re-running")

        acc.print(f"\n=== {label} | {args.model} | {args.split} ===")
        loader = acc.prepare(DataLoader(data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=list))
        rows = run_condition(args.model, model, processor, loader, acc, perturbation, alpha, label)

        if acc.is_main_process:
            meta = {"model": args.model, "model_path": model_path, "split": args.split,
                    "perturbation": perturbation, "alpha": alpha,
                    "weights": "base (no mDPO adapter)", "processes": acc.num_processes}
            result = write(path, rows, meta, expected=len(data))
            o = result["overall"]
            print(f"{label}: {o['accuracy']:.4f} ({o['correct']}/{o['total']})  "
                  f"domain avg {result['domain_average_accuracy']:.4f}  -> {path.name}", flush=True)

        acc.wait_for_everyone()
        torch.cuda.empty_cache()

    if acc.is_main_process:
        print()
        summary()


if __name__ == "__main__":
    main()
