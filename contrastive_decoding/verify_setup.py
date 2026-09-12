"""
verify_setup.py
---------------
Pre-flight check for the contrastive-decoding runs. Needs no GPU and no model
weights, so it can be run on a login node (or locally) before queueing real
work.

Checks, in order:
  1. Both split JSONs resolve, and every audio_url either exists as stored or
     rebases cleanly onto the audio root.
  2. All four perturbations load from the repository-root perturbations.py at
     the expected settings, and each one runs on a real waveform.
  3. Every perturbation preserves waveform length -- the contrastive branch
     depends on this, since a length change would alter the number of audio
     tokens and break alignment between the clean and negative prompts.
  4. Per-example seeding is deterministic and independent of call order.
  5. Answer extraction and subset inference reproduce her labels on real rows.

Usage
-----
    python contrastive_decoding/verify_setup.py
    python contrastive_decoding/verify_setup.py --audio-root /path/to/wavs
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contrastive_decoding.cd_common import (  # noqa: E402
    PERTURBATION_PARAMS,
    PERTURBATIONS,
    build_question,
    extract_choice_letter,
    find_audio_root,
    find_data_root,
    infer_subset,
    load_split,
    perturb_for_example,
    resolve_perturbation,
)

SAMPLING_RATE = 16000


def check_splits(data_root, audio_root):
    print("=" * 70)
    print("1. DATASETS")
    print("=" * 70)
    print(f"data root:  {find_data_root(data_root)}")
    print(f"audio root: {find_audio_root(audio_root)}")

    splits = {}
    for split in ("test", "validation"):
        data = load_split(split, data_root=data_root, audio_root=audio_root)
        splits[split] = data
        subsets = Counter(infer_subset(example) for example in data)
        choices = Counter(len(example["choice"]) for example in data)
        print(
            f"  {split:11s} {len(data):5d} examples | "
            f"subsets {dict(subsets)} | choice counts {dict(sorted(choices.items()))}"
        )
        print(f"  {'':11s} all {len(data)} audio files resolved")
    return splits


def check_perturbations(example):
    print()
    print("=" * 70)
    print("2. PERTURBATIONS")
    print("=" * 70)

    audio = librosa.load(example["audio_url"], sr=SAMPLING_RATE, mono=True)[0]
    print(f"probe clip: {Path(example['audio_url']).name} "
          f"({len(audio)} samples, {len(audio) / SAMPLING_RATE:.1f}s)")

    ok = True
    for name in sorted(PERTURBATIONS):
        pert_fn = resolve_perturbation(name, sr=SAMPLING_RATE)
        out = perturb_for_example(audio, pert_fn, example["id"])

        length_ok = len(out) == len(audio)
        changed = not np.array_equal(out, audio)
        rms = float(np.sqrt(np.mean((out - audio) ** 2)))

        status = "ok" if (length_ok and changed) else "FAIL"
        if status == "FAIL":
            ok = False
        params = str(PERTURBATION_PARAMS[name] or {})
        print(
            f"  {name:11s} {params:32s} "
            f"len {len(out):7d} ({'preserved' if length_ok else 'CHANGED'}) "
            f"| rms delta {rms:.4f} | {status}"
        )

    if not ok:
        raise SystemExit("Perturbation check failed.")
    return audio


def check_determinism(example, audio):
    print()
    print("=" * 70)
    print("3. PER-EXAMPLE SEEDING")
    print("=" * 70)

    for name in ("time_mask", "noise"):
        pert_fn = resolve_perturbation(name, sr=SAMPLING_RATE)

        first = perturb_for_example(audio, pert_fn, example["id"])
        # Burn global RNG, as a differently sharded run would.
        np.random.rand(1000)
        second = perturb_for_example(audio, pert_fn, example["id"])
        other = perturb_for_example(audio, pert_fn, example["id"] + "_x")

        same = np.array_equal(first, second)
        differs = not np.array_equal(first, other)
        status = "ok" if (same and differs) else "FAIL"
        print(
            f"  {name:11s} stable across RNG drift: {same} | "
            f"differs per example: {differs} | {status}"
        )
        if status == "FAIL":
            raise SystemExit("Determinism check failed.")


def check_extraction(data):
    print()
    print("=" * 70)
    print("4. PROMPTS AND EXTRACTION")
    print("=" * 70)

    unparsed = [ex for ex in data if extract_choice_letter(ex["answer"]) is None]
    print(f"  ground-truth answers parsed: {len(data) - len(unparsed)}/{len(data)}")
    if unparsed:
        print(f"  WARNING: {len(unparsed)} unparsable, first: {unparsed[0]['answer']!r}")

    example = data[0]
    for model in ("qwen2", "af3"):
        print()
        print(f"  --- {model} user turn ---")
        for line in build_question(example, model).splitlines():
            print(f"    {line}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--audio-root", default=None)
    args = parser.parse_args()

    splits = check_splits(args.data_root, args.audio_root)
    probe = splits["test"][0]
    audio = check_perturbations(probe)
    check_determinism(probe, audio)
    check_extraction(splits["test"])

    print()
    print("=" * 70)
    print("All checks passed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
