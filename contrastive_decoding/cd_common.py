"""
cd_common.py
------------
Shared pieces for the DCASE multiple-choice contrastive-decoding runs.

Two things live here, and the split between them is the whole point of this
module:

1. EVALUATION CODE COPIED VERBATIM from the colleague's baseline scripts
   (af3_evaluation.py / qwen2_evaluation.py) -- the system prompt, the
   question assembly, extract_choice_letter(), infer_subset(), and the
   accuracy/domain-average aggregation. These must stay byte-identical in
   behaviour to hers or the contrastive-decoding numbers aren't comparable
   to her mDPO baselines. Do not "improve" anything in this section.

2. NEW PLUMBING that the CD runs need and hers didn't: dataset location,
   audio-path remapping, perturbation resolution, and result writing.

Audio path remapping
--------------------
The split JSONs store ABSOLUTE cluster paths, e.g.
    /data/not_backed_up/cosgrv/af3_project/dcase_2025/2025_DCASE_AudioQA/local_audio_path/dev/audio_00505.wav
The datasets are read-only, so nothing rewrites them on disk. Instead
load_split() optionally rebases each audio_url onto a local directory by
BASENAME. On the cluster, pass no audio root and the original paths are used
untouched; locally, point --audio-root at the flat dcase_dev_audio_files/
directory. Verified 1122/1122 (test) and 1120/1120 (validation) basenames
resolve against that local directory.

Perturbation determinism
------------------------
TIME_MASK and NOISE are stochastic. perturb_for_example() seeds numpy from
the example's own id (not from call order), so a given example gets the same
perturbed waveform regardless of batch size, GPU count, or shard boundaries.
Without this, distributed runs would not be reproducible against single-GPU
ones.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# perturbations.py lives at the repository root, one level up from this package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from perturbations import Perturbation, get_perturbation  # noqa: E402


# =============================================================================
# PERTURBATIONS
#
# The four the colleague used, at her settings. Values are read out of the
# repository-root perturbations.py -- the same module that generated her mDPO
# training data -- so the contrastive branch and her perturbed training audio
# come from identical code.
#
# NO_AUDIO is a SILENT waveform (np.zeros_like), not an audio-removed prompt.
# That is how perturbations.py defines it and how her perturbed_audio/
# dcase_no_audio/*.wav files were generated. It differs from the AAD paper's
# preferred "drop the audio tokens entirely" variant; this was a deliberate
# choice to stay consistent with her pipeline.
# =============================================================================

PERTURBATIONS: Dict[str, Tuple[Perturbation, Optional[str]]] = {
    "no_audio":  (Perturbation.NO_AUDIO,  None),      # np.zeros_like(audio)
    "reverse":   (Perturbation.REVERSE,   "full"),    # audio[::-1]
    "time_mask": (Perturbation.TIME_MASK, "light"),   # n_masks=3, max_width=0.08
    "noise":     (Perturbation.NOISE,     "weak"),    # sigma=0.02
}

#: Documented for the results metadata so a run's exact parameters are
#: recoverable from its output file alone.
PERTURBATION_PARAMS: Dict[str, Dict[str, Any]] = {
    "no_audio":  {},
    "reverse":   {},
    "time_mask": {"n_masks": 3, "max_width": 0.08},
    "noise":     {"sigma": 0.02},
}

DEFAULT_ALPHAS: Tuple[float, ...] = (0.5, 1.0)


def resolve_perturbation(name: str, sr: int = 16000) -> Callable[[np.ndarray], np.ndarray]:
    """Return the waveform->waveform function for one of the four names above."""
    if name not in PERTURBATIONS:
        raise ValueError(
            f"Unknown perturbation {name!r}. Available: {sorted(PERTURBATIONS)}"
        )
    pert_type, setting = PERTURBATIONS[name]
    return get_perturbation(pert_type, setting, sr=sr)


def perturb_for_example(
    audio: np.ndarray,
    pert_fn: Callable[[np.ndarray], np.ndarray],
    example_id: str,
    seed: int = 42,
) -> np.ndarray:
    """Apply pert_fn with numpy seeded from example_id, restoring global RNG state.

    Makes stochastic perturbations (TIME_MASK, NOISE) a pure function of the
    example, so results do not depend on batch size or process count.
    """
    state = np.random.get_state()
    try:
        np.random.seed((zlib.crc32(example_id.encode("utf-8")) ^ seed) & 0xFFFFFFFF)
        return pert_fn(audio)
    finally:
        np.random.set_state(state)


# =============================================================================
# DATASET LOCATION AND LOADING
# =============================================================================

SPLIT_FILENAMES: Dict[str, str] = {
    "test": "dcase_split_test.json",
    "validation": "dcase_split_validation.json",
}

#: Searched in order for the split JSONs. The cluster path first so runs there
#: need no flags; the local checkout second so the same command works here.
CANDIDATE_DATA_ROOTS: Tuple[str, ...] = (
    "/data/not_backed_up/cosgrv/af3_project/dcase_2025/2025_DCASE_AudioQA/combined_json",
    str(_REPO_ROOT / "dcase might not be the same as on the remote"),
)

#: Searched in order for the audio files, used only when the audio_url stored
#: in the dataset does not exist as written.
CANDIDATE_AUDIO_ROOTS: Tuple[str, ...] = (
    "/data/not_backed_up/cosgrv/af3_project/dcase_2025/2025_DCASE_AudioQA/local_audio_path/dev",
    str(_REPO_ROOT / "dcase might not be the same as on the remote" / "dcase_dev_audio_files"),
)


def find_data_root(explicit: Optional[str] = None) -> Path:
    """Locate the directory holding the split JSONs."""
    if explicit:
        root = Path(explicit)
        if not root.is_dir():
            raise FileNotFoundError(f"--data-root does not exist: {root}")
        return root
    env = os.environ.get("DCASE_DATA_ROOT")
    if env:
        return Path(env)
    for candidate in CANDIDATE_DATA_ROOTS:
        if (Path(candidate) / SPLIT_FILENAMES["test"]).exists():
            return Path(candidate)
    raise FileNotFoundError(
        "Could not locate the DCASE split JSONs. Pass --data-root or set "
        f"DCASE_DATA_ROOT. Searched: {CANDIDATE_DATA_ROOTS}"
    )


def find_audio_root(explicit: Optional[str] = None) -> Optional[Path]:
    """Locate the directory holding the wav files, or None to use paths as stored."""
    if explicit:
        root = Path(explicit)
        if not root.is_dir():
            raise FileNotFoundError(f"--audio-root does not exist: {root}")
        return root
    env = os.environ.get("DCASE_AUDIO_ROOT")
    if env:
        return Path(env)
    for candidate in CANDIDATE_AUDIO_ROOTS:
        if Path(candidate).is_dir():
            return Path(candidate)
    return None


def split_path(split: str, data_root: Optional[str] = None) -> Path:
    """Absolute path to one split's JSON file."""
    if split not in SPLIT_FILENAMES:
        raise ValueError(f"Unknown split {split!r}. Available: {sorted(SPLIT_FILENAMES)}")
    return find_data_root(data_root) / SPLIT_FILENAMES[split]


def dataset_fingerprint(split: str, data_root: Optional[str] = None) -> str:
    """Short SHA-256 of the split file's bytes.

    Recorded in every result file so a run can be tied to the exact dataset
    revision it was produced from. The cluster and local copies of a split are
    expected to match; if they ever diverge, this is what surfaces it instead
    of two incomparable result sets quietly sitting side by side.
    """
    digest = hashlib.sha256()
    with open(split_path(split, data_root), "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def load_split(
    split: str,
    data_root: Optional[str] = None,
    audio_root: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load one split, rebasing audio paths onto audio_root when needed.

    The dataset file itself is never written to. Each example gains a
    "_row_index" key used later to strip the duplicate rows that
    accelerate's distributed sampler pads the final batch with.
    """
    path = split_path(split, data_root)
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    root = find_audio_root(audio_root)
    missing: List[str] = []

    for index, example in enumerate(data):
        example["_row_index"] = index
        stored = example["audio_url"]
        if os.path.exists(stored):
            continue
        if root is not None:
            rebased = root / Path(stored.replace("\\", "/")).name
            if rebased.exists():
                example["audio_url"] = str(rebased)
                continue
        missing.append(stored)

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(data)} audio files could not be resolved for "
            f"split {split!r} (audio root: {root}). First missing: {missing[0]}"
        )

    return data


# =============================================================================
# PROMPT CONSTRUCTION AND ANSWER EXTRACTION
#
# VERBATIM from af3_evaluation.py / qwen2_evaluation.py. The only intentional
# difference between the two models is the trailing instruction line, which
# her Qwen script has and her AF3 script does not -- preserved exactly.
# =============================================================================

SYSTEM_PROMPT = (
    "Focus on the given audio and answer the "
    "following multiple-choice question. "
    "Respond with the letter of the correct "
    "answer (A, B, C, or D)."
)


def build_question(example: Dict[str, Any], model: str) -> str:
    """Assemble the user-turn text. model is "qwen2" or "af3"."""
    choices_text = "\n".join(example["choice"])
    if model == "qwen2":
        return (
            f"{example['question']}\n"
            f"{choices_text}\n"
            "Answer with exactly one letter: "
            "A, B, C, or D."
        )
    if model == "af3":
        return f"{example['question']}\n{choices_text}"
    raise ValueError(f"Unknown model {model!r}")


def extract_choice_letter(text: str) -> Optional[str]:
    """Extract A, B, C, or D from an answer string."""
    match = re.search(r"\b([A-D])\b", text.strip().upper())
    return match.group(1) if match else None


def infer_subset(example: Dict[str, Any]) -> str:
    """
    Infer the dataset subset from the beginning of the audio filename.

        fold...  -> Temporal
        audio... -> Complex
        anything else -> Bio
    """
    audio_path = str(example.get("audio_url", ""))
    audio_name = Path(audio_path).name.lower()

    if audio_name.startswith("fold"):
        return "Temporal"

    if audio_name.startswith("audio"):
        return "Complex"

    return "Bio"


EXPECTED_SUBSETS: Tuple[str, ...] = ("Bio", "Temporal", "Complex")


# =============================================================================
# METRICS
# =============================================================================

def _accuracy_table(results: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for result in results:
        bucket = result.get(key, "unknown")
        stats[bucket]["total"] += 1
        if result["correct"]:
            stats[bucket]["correct"] += 1

    return {
        bucket: {
            "accuracy": counts["correct"] / counts["total"] if counts["total"] else 0.0,
            "correct": counts["correct"],
            "total": counts["total"],
        }
        for bucket, counts in stats.items()
    }


def _domain_average(accuracy_by_subset: Dict[str, Dict[str, Any]]) -> float:
    """Macro average over the subsets that are actually present.

    Note the DCASE dev splits contain only Complex and Temporal examples --
    no filename starts with anything that maps to Bio -- so this is a
    two-way macro average in practice, exactly as in her results files.
    """
    available = [
        accuracy_by_subset[subset]["accuracy"]
        for subset in EXPECTED_SUBSETS
        if subset in accuracy_by_subset
    ]
    return sum(available) / len(available) if available else 0.0


def summarize(results: List[Dict[str, Any]], correct: int, total: int) -> Dict[str, Any]:
    """Build her results dict: overall, domain average, and both breakdowns."""
    accuracy_by_subset = _accuracy_table(results, "subset")
    return {
        "overall": {
            "accuracy": correct / total if total else 0.0,
            "correct": correct,
            "total": total,
        },
        "domain_average_accuracy": _domain_average(accuracy_by_subset),
        "accuracy_by_subset": accuracy_by_subset,
        "accuracy_by_question_type": _accuracy_table(results, "question_type"),
    }


def deduplicate(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop the rows accelerate's sampler duplicated to pad the final batch.

    Keyed on _row_index (the example's position in the dataset file), so
    genuinely repeated ids -- the test split has 11 -- are preserved while
    padding artefacts are removed. Her reported totals are the PADDED ones
    (1136 rows for a 1122-example split), so both are written out.
    """
    seen = set()
    unique = []
    for result in results:
        row = result.get("_row_index")
        if row in seen:
            continue
        seen.add(row)
        unique.append(result)
    return unique


# =============================================================================
# RESULT FILES
# =============================================================================

def results_dir(model: str, split: str) -> Path:
    path = Path(__file__).resolve().parent / "results" / model / split
    path.mkdir(parents=True, exist_ok=True)
    return path


def result_path(model: str, split: str, perturbation: Optional[str], alpha: Optional[float]) -> Path:
    if perturbation is None:
        return results_dir(model, split) / "baseline.json"
    return results_dir(model, split) / f"cd_{perturbation}_alpha_{alpha}.json"


def resolve_revision(model, model_path: str, requested: Optional[str] = None) -> str:
    """Best-effort identification of the exact weights a run loaded.

    Preference order:
      1. the revision explicitly requested on the command line
      2. the commit hash transformers resolved and stamped on the config
      3. the snapshot directory name, when loading from a local hub snapshot
         (this is what her qwen2_evaluation.py pins by path)
      4. "unpinned", recorded honestly rather than guessed

    Written into every result file so a number can always be traced back to the
    weights that produced it.
    """
    if requested:
        return requested

    commit = getattr(getattr(model, "config", None), "_commit_hash", None)
    if commit:
        return str(commit)

    path = Path(model_path)
    if path.is_dir() and path.parent.name == "snapshots":
        return path.name

    return "unpinned"


#: Metadata fields that identify WHAT a result file measures. An existing file
#: is only reusable if every one of these matches the run being requested.
#: num_processes and batch_size are deliberately excluded -- they change the row
#: padding and the runtime, not the quantity being measured.
IDENTITY_FIELDS: Tuple[str, ...] = (
    "model",
    "model_path",
    "revision",
    "weights",
    "split",
    "perturbation",
    "alpha",
    "seed",
    "max_new_tokens",
    "dataset_examples",
    "dataset_fingerprint",
)


class IncompatibleResult(RuntimeError):
    """An existing result file measures something other than the current run."""


def validate_coverage(results: List[Dict[str, Any]], expected: int) -> List[Dict[str, Any]]:
    """Check every dataset row was processed exactly once after deduplication.

    Distributed runs are gathered from every process and padded by accelerate's
    sampler, so the raw row count is expected to exceed `expected`. What must
    hold is that the set of _row_index values, once deduplicated, is exactly
    range(expected): nothing dropped by a failed shard, nothing invented.

    Raises rather than writing a file that looks successful but is short rows.
    """
    unique = deduplicate(results)
    seen = {row.get("_row_index") for row in unique}

    if None in seen:
        raise RuntimeError(
            "Some result rows carry no _row_index; cannot verify coverage. "
            "This means examples reached the runner without going through load_split()."
        )

    expected_set = set(range(expected))
    missing = sorted(expected_set - seen)
    unexpected = sorted(seen - expected_set)

    if missing or unexpected:
        raise RuntimeError(
            f"Incomplete run: expected {expected} unique rows, got {len(seen)}. "
            f"Missing {len(missing)} (first: {missing[:5]}); "
            f"unexpected {len(unexpected)} (first: {unexpected[:5]}). "
            "Refusing to write a result file that would look complete."
        )

    return unique


def should_skip(path: Path, metadata: Dict[str, Any], overwrite: bool) -> bool:
    """Decide whether an existing result file can stand in for this run.

    Three outcomes:
      - file absent, or --overwrite given -> run it
      - file present, unreadable or truncated -> run it (a partial write from an
        interrupted job is worthless, and silently skipping it was the original
        resumability bug)
      - file present and complete but measuring a different configuration ->
        raise, rather than overwrite someone's results or report stale ones
    """
    if overwrite or not path.exists():
        return False

    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  {path.name}: unreadable ({exc.__class__.__name__}), re-running", flush=True)
        return False

    if not isinstance(payload, dict) or "results" not in payload or "overall" not in payload:
        print(f"  {path.name}: incomplete structure, re-running", flush=True)
        return False

    existing = payload.get("metadata", {})
    differences = [
        f"{field}: existing={existing.get(field)!r} requested={metadata.get(field)!r}"
        for field in IDENTITY_FIELDS
        if existing.get(field) != metadata.get(field)
    ]
    if differences:
        raise IncompatibleResult(
            f"{path} already exists but measures a different configuration:\n  "
            + "\n  ".join(differences)
            + "\n\nMove or delete the file, or pass --overwrite to replace it."
        )

    return True


def write_results(
    path: Path,
    metadata: Dict[str, Any],
    results: List[Dict[str, Any]],
    expected: Optional[int] = None,
) -> Dict[str, Any]:
    """Validate, then atomically write one run's output file.

    The write goes to a temporary file in the same directory and is renamed into
    place only once it is complete, so an interrupted job can never leave a
    truncated file that a later run would mistake for a finished one.
    """
    unique = deduplicate(results) if expected is None else validate_coverage(results, expected)

    padded = summarize(
        results,
        correct=sum(1 for r in results if r["correct"]),
        total=len(results),
    )
    deduped = summarize(
        unique,
        correct=sum(1 for r in unique if r["correct"]),
        total=len(unique),
    )

    payload = {
        "metadata": metadata,
        # Comparable to her result files, which include accelerate's padding rows.
        **padded,
        # Same run with padding duplicates removed; use this for like-for-like
        # comparisons across different GPU counts.
        "deduplicated": deduped,
        "results": results,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=f".{path.stem}.",
        suffix=".tmp",
        delete=False,
    )
    try:
        with handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    except BaseException:
        # Includes KeyboardInterrupt: never leave the scratch file behind.
        try:
            os.unlink(handle.name)
        except OSError:
            pass
        raise

    return padded


def print_summary(prefix: str, summary: Dict[str, Any], deduped_total: Optional[int] = None) -> None:
    overall = summary["overall"]
    print(
        f"\n{prefix} overall accuracy: {overall['accuracy']:.4f} "
        f"({overall['correct']}/{overall['total']})",
        flush=True,
    )
    print(
        f"{prefix} domain-average accuracy: {summary['domain_average_accuracy']:.4f}",
        flush=True,
    )
    print(f"\n{prefix} accuracy by subset:", flush=True)
    for subset in EXPECTED_SUBSETS:
        if subset not in summary["accuracy_by_subset"]:
            print(f"  {subset}: no examples found", flush=True)
            continue
        stats = summary["accuracy_by_subset"][subset]
        print(
            f"  {subset}: {stats['accuracy']:.4f} "
            f"({stats['correct']}/{stats['total']})",
            flush=True,
        )
    print(f"\n{prefix} accuracy by question type:", flush=True)
    for question_type in sorted(summary["accuracy_by_question_type"]):
        stats = summary["accuracy_by_question_type"][question_type]
        print(
            f"  {question_type}: {stats['accuracy']:.4f} "
            f"({stats['correct']}/{stats['total']})",
            flush=True,
        )
    if deduped_total is not None:
        print(f"\n{prefix} rows after removing padding duplicates: {deduped_total}", flush=True)


def build_result_row(example: Dict[str, Any], prediction: str) -> Dict[str, Any]:
    """One results entry, in her schema plus _row_index for deduplication."""
    ground_truth = example["answer"]
    predicted_letter = extract_choice_letter(prediction)
    ground_truth_letter = extract_choice_letter(ground_truth)
    return {
        "id": example["id"],
        "question": example["question"],
        "question_type": example.get("question_type", "unknown"),
        "subset": infer_subset(example),
        "answer": ground_truth,
        "prediction": prediction,
        "pred_letter": predicted_letter,
        "gt_letter": ground_truth_letter,
        "correct": predicted_letter == ground_truth_letter,
        "audio_url": example["audio_url"],
        "_row_index": example.get("_row_index"),
    }
