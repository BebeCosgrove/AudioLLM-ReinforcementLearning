"""
run_helpers.py
--------------
Shared utilities for audio contrastive-decoding evaluation runs.

The runner modules for Qwen2-Audio and Audio Flamingo 3 keep model-specific
input construction local, but they share a common set of mechanics for
interpreting outputs and managing experiment files. This module provides that
shared layer: answer extraction, yes/no token handling, checkpoint paths,
resumable checkpoint IO, result-existence checks, flip and bias analyses,
summary-file updates, reproducibility helpers, and softmax-distance metrics.

AAD contrastive decoding
-------------------------
The experiments are built around inference-time contrastive decoding (not model
training). In each AAD-style run (arXiv 2506.07233), a clean branch produces
next-token logits from the real audio and prompt, while a negative branch
produces logits from the same prompt with a degraded or absent audio signal.
The original AAD paper evaluates two negative modes — blank (silent, all-zeros)
audio and fully audio-removed prompts — with audio removal outperforming silence
in their experiments. This codebase supports both modes plus arbitrary waveform
perturbations. The runners combine branches as:

    modified_logits = (1 + alpha) * clean_logits - alpha * negative_logits

The update promotes tokens whose probability is elevated specifically by the
presence of real audio, and penalises tokens the model would predict without
any audio. Because the project evaluates yes/no audio QA and
hallucination benchmarks, this module includes conservative answer parsing and
step-0 yes/no logit extraction. The first-token distribution is often the
cleanest view of whether the contrastive intervention shifted "yes" or "no"
when generation is constrained to a single output token.

Flip analysis and bias detection
----------------------------------
The flip analysis measures how contrastive decoding changes the top predicted
token at step 0 relative to the ground-truth answer. A wrong_to_right flip
means the modified top token matches the answer when the unmodified top token
did not; a right_to_wrong flip indicates harm introduced by the intervention.
The dataset-balance analysis checks whether contrastive decoding reduces
systematic yes/no prediction bias rather than merely shifting overall accuracy.

Softmax-distance metrics and VACoDe-style selection
-----------------------------------------------------
compute_softmax_distances takes two 1-D raw logit vectors (clean and negative
branches, shape [vocab_size]) and converts both to full-vocabulary softmax
probability distributions in float32 before computing any distance. This
prevents bfloat16 rounding errors from corrupting small-probability tokens.

VACoDe (arXiv 2408.05337) identifies the best contrastive augmentation per
sample by selecting the one that maximises L2 distance between the original and
augmented softmax distributions. Our function returns that canonical L2 metric
along with L1, L3, L-infinity, cosine distance, and KL divergence for broader
comparative analysis. Larger distances indicate that the negative view shifted
the next-token distribution farther from the clean distribution, making it a
stronger candidate for contrastive subtraction. Earth Mover's Distance is
intentionally omitted: vocabulary token indices carry no meaningful ground
distance between tokens, so an EMD over the vocabulary dimension is not
interpretable.
"""

from __future__ import annotations

from collections import Counter
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from helpers.config import Config
from helpers.process_results import SampleResult


def get_selected_perturbation_filter(dataset: str, alpha: float, model: str) -> list[str]:
    """Read best_run_specs.json for dataset/model/alpha and return perturbation filter tokens.

    Returns a list of 'TYPE:setting' or 'TYPE' strings ready to drop into perturbation_filter.
    Uses 'selected' entry when present, falls back to 'recommended'.
    """
    alpha_str = str(float(alpha))
    path = Config.selector_data_root / model / dataset / alpha_str / "best_run_specs.json"
    if not path.exists():
        raise FileNotFoundError(
            f"best_run_specs.json not found: {path}\n"
            "Run spec selection for this dataset before using -p selected."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    entry = data.get("selected") or data["recommended"]
    tokens = []
    for spec in entry["specs"]:
        parts = {kv.split("=")[0]: kv.split("=")[1] for kv in spec.split("|")}
        t, s = parts["t"], parts.get("s", "None")
        tokens.append(t if s == "None" else f"{t}:{s.lower()}")
    return tokens


def limit_gpu_memory(fraction: float = 0.2) -> None:
    """Limit each visible GPU to a fraction of total VRAM. Call before any CUDA tensors are created."""
    import torch
    assert 0 < fraction <= 1.0
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    for gpu_id in range(torch.cuda.device_count()):
        torch.cuda.set_per_process_memory_fraction(fraction, gpu_id)
    print(f"✅ GPU memory limited to {fraction * 100:.0f}% per GPU per process")


def extract_answer(response: str) -> str:
    """Extract yes/no answer from model response."""
    response_lower = response.lower().strip()
    first_word = response_lower.split()[0] if response_lower.split() else ""
    first_word_clean = "".join(c for c in first_word if c.isalnum())
    if first_word_clean.startswith("yes"):
        return "yes"
    if first_word_clean.startswith("no"):
        return "no"
    if "yes" in response_lower:
        return "yes"
    if "no" in response_lower:
        return "no"
    return response_lower[:20]


def extract_answer_from_step0_yes_no_logits(logit_trace, response=None) -> Optional[str]:
    """Extract yes/no from step-0 modified logits, using decoded text to break ties."""
    if not isinstance(logit_trace, list) or not logit_trace:
        return None
    step0 = logit_trace[0]
    if not isinstance(step0, dict):
        return None
    yes_no = step0.get("yes_no_logits")
    if not isinstance(yes_no, dict):
        return None
    modified = yes_no.get("modified")
    if not isinstance(modified, dict):
        return None

    def best_logit(labels):
        vals = [
            info["logit"]
            for label in labels
            for info in [modified.get(label)]
            if isinstance(info, dict) and isinstance(info.get("logit"), (int, float))
        ]
        return max(vals) if vals else None

    yes_logit = best_logit(("Yes", "yes"))
    no_logit = best_logit(("No", "no"))
    if yes_logit is None or no_logit is None:
        return None
    if yes_logit > no_logit:
        return "yes"
    if no_logit > yes_logit:
        return "no"
    if response is not None:
        text_answer = extract_answer(response)
        if text_answer in {"yes", "no"}:
            return text_answer
    return "yes"


def extract_answer_with_config(response: str, logit_trace) -> str:
    """Extract answer using configured strategy (step-0 logits or text fallback)."""
    if Config.use_step0_yes_no_logits_extraction:
        extracted = extract_answer_from_step0_yes_no_logits(logit_trace, response=response)
        if extracted is not None:
            return extracted
    return extract_answer(response)


def set_random_seed(seed: int) -> None:
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_output_filename(
    perturbation_type: str,
    alpha: float,
    perturbation_setting: Optional[str] = None,
    results_dir: Optional[Path | str] = None,
) -> Path:
    directory = Path(results_dir) if results_dir else Path(Config.results_train_dir)
    name_part = perturbation_type.lower()
    if perturbation_setting:
        name_part = f"{name_part}_{perturbation_setting.lower()}"
    base_name = f"eval_{name_part}_alpha_{alpha}"
    return directory / f"{base_name}.json.gz"


def get_checkpoint_filename(
    perturbation_type: str,
    alpha: float,
    perturbation_setting: Optional[str] = None,
    results_dir: Optional[Path | str] = None,
) -> Path:
    directory = Path(results_dir) if results_dir else Path(Config.results_train_dir)
    name_part = perturbation_type.lower()
    if perturbation_setting:
        name_part = f"{name_part}_{perturbation_setting.lower()}"
    base_name = f"checkpoint_{name_part}_alpha_{alpha}"
    return directory / f"{base_name}.json"


def check_existing_results(
    perturbation_type: str,
    alpha: float,
    perturbation_setting: Optional[str] = None,
    results_dir: Optional[Path | str] = None,
) -> bool:
    output_path = get_output_filename(perturbation_type, alpha, perturbation_setting, results_dir)
    if output_path.exists() or Path(str(output_path) + ".gz").exists():
        print(f"✓ Results already exist: {output_path}")
        print("  Skipping evaluation. Delete the file to re-run.")
        return True
    return False


def load_checkpoint(
    perturbation_type: str,
    alpha: float,
    perturbation_setting: Optional[str] = None,
    results_dir: Optional[Path | str] = None,
) -> tuple[list[SampleResult], int]:
    checkpoint_path = get_checkpoint_filename(perturbation_type, alpha, perturbation_setting, results_dir)
    if checkpoint_path.exists():
        try:
            with checkpoint_path.open("r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
            raw_results = checkpoint_data.get("results", [])
            results: list[SampleResult] = []
            for item in raw_results:
                if isinstance(item, dict):
                    try:
                        results.append(SampleResult.from_dict(item, default_format="freeform"))
                    except ValueError:
                        continue
            last_batch_idx = checkpoint_data.get("last_batch_idx", -1)
            print(f"✓ Loaded checkpoint: {checkpoint_path}")
            print(f"  Resuming from batch {last_batch_idx + 1} ({len(results)} samples processed)")
            return results, last_batch_idx
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠ Checkpoint file corrupted, starting fresh: {e}")
            return [], -1
    return [], -1


def save_checkpoint(
    results: list[SampleResult],
    last_batch_idx: int,
    perturbation_type: str,
    alpha: float,
    perturbation_setting: Optional[str] = None,
    results_dir: Optional[Path | str] = None,
) -> None:
    checkpoint_path = get_checkpoint_filename(perturbation_type, alpha, perturbation_setting, results_dir)
    checkpoint_data = {
        "last_batch_idx": last_batch_idx,
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "results": [r.to_dict() for r in results],
    }
    with checkpoint_path.open("w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=2)
    print(f"  💾 Checkpoint saved at batch {last_batch_idx} ({len(results)} samples)")


def delete_checkpoint(
    perturbation_type: str,
    alpha: float,
    perturbation_setting: Optional[str] = None,
    results_dir: Optional[Path | str] = None,
) -> None:
    checkpoint_path = get_checkpoint_filename(perturbation_type, alpha, perturbation_setting, results_dir)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"  🗑️ Checkpoint deleted: {checkpoint_path}")


def normalize_token(token: str) -> str:
    token = token.strip()
    if token:
        return token[0].upper() + token[1:] if len(token) > 1 else token.upper()
    return token


def classify_answer(token: str) -> str:
    token_clean = "".join(c for c in token if c.isalnum()).upper()
    if token_clean.startswith("YES") or token_clean == "Y":
        return "yes"
    if token_clean.startswith("NO") or token_clean == "N":
        return "no"
    return "other"


def tokens_match_answer(token: str, ground_truth: str) -> bool:
    token_norm = token.strip().upper()
    gt_norm = ground_truth.strip().upper()

    if token_norm == gt_norm:
        return True

    token_clean = "".join(c for c in token_norm if c.isalnum())
    gt_clean = "".join(c for c in gt_norm if c.isalnum())
    if token_clean == gt_clean:
        return True

    if gt_norm in {"YES", "NO"}:
        if token_clean.startswith(gt_norm) or gt_norm.startswith(token_clean):
            return True

    return False


def _extract_trace(entry: Any) -> Optional[list]:
    if isinstance(entry, SampleResult):
        return entry.logit_trace
    if isinstance(entry, dict):
        return entry.get("logit_trace")
    return None


def _extract_ground_truth(entry: Any) -> str:
    if isinstance(entry, SampleResult):
        return entry.ground_truth.strip()
    if isinstance(entry, dict):
        return str(entry.get("ground_truth", "")).strip()
    return ""


def determine_flip_type(entry: Any) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    trace = _extract_trace(entry)
    if not trace:
        return None, None, None

    step0 = trace[0]
    original_top10 = step0.get("original_top10", [])
    modified_top10 = step0.get("modified_top10", [])
    if not original_top10 or not modified_top10:
        return None, None, None

    ground_truth = _extract_ground_truth(entry)
    original_top_token = original_top10[0].get("token", "").strip()
    modified_top_token = modified_top10[0].get("token", "").strip()

    original_normalized = normalize_token(original_top_token)
    modified_normalized = normalize_token(modified_top_token)

    if original_top_token.upper() == modified_top_token.upper():
        return None, None, None

    original_correct = tokens_match_answer(original_top_token, ground_truth)
    modified_correct = tokens_match_answer(modified_top_token, ground_truth)

    if not original_correct and modified_correct:
        return "wrong_to_right", original_normalized, modified_normalized
    if original_correct and not modified_correct:
        return "right_to_wrong", original_normalized, modified_normalized
    return None, None, None


def analyze_flips(results: Iterable[Any]) -> Dict[str, Any]:
    results_list = list(results)
    wrong_to_right = []
    right_to_wrong = []

    for entry in results_list:
        flip_type, orig_token, mod_token = determine_flip_type(entry)
        if flip_type == "wrong_to_right":
            wrong_to_right.append((orig_token, mod_token))
        elif flip_type == "right_to_wrong":
            right_to_wrong.append((orig_token, mod_token))

    def build_transition_stats(transitions):
        if not transitions:
            return {"total": 0, "transitions": []}
        transition_counter = Counter()
        for orig, mod in transitions:
            transition_counter[f"{orig} -> {mod}"] += 1
        total = len(transitions)
        transition_list = []
        for transition, count in transition_counter.most_common():
            transition_list.append(
                {
                    "transition": transition,
                    "count": count,
                    "percentage": round(count / total * 100, 2),
                }
            )
        return {"total": total, "transitions": transition_list}

    def analyze_directional_stats(transitions):
        stats = {"yes_to_no": 0, "no_to_yes": 0, "other": 0, "total": len(transitions)}
        if stats["total"] == 0:
            stats.update({"yes_to_no_pct": 0, "no_to_yes_pct": 0, "other_pct": 0})
            return stats
        for orig, mod in transitions:
            orig_cls = classify_answer(orig)
            mod_cls = classify_answer(mod)
            if orig_cls == "yes" and mod_cls == "no":
                stats["yes_to_no"] += 1
            elif orig_cls == "no" and mod_cls == "yes":
                stats["no_to_yes"] += 1
            else:
                stats["other"] += 1
        stats["yes_to_no_pct"] = round((stats["yes_to_no"] / stats["total"]) * 100, 2)
        stats["no_to_yes_pct"] = round((stats["no_to_yes"] / stats["total"]) * 100, 2)
        stats["other_pct"] = round((stats["other"] / stats["total"]) * 100, 2)
        return stats

    total_entries = len(results_list)
    w2r_direction = analyze_directional_stats(wrong_to_right)
    r2w_direction = analyze_directional_stats(right_to_wrong)

    return {
        "total_entries": total_entries,
        "wrong_to_right": {
            "count": len(wrong_to_right),
            "percentage": round(len(wrong_to_right) / total_entries * 100, 2) if total_entries > 0 else 0,
            "directionality": w2r_direction,
        },
        "right_to_wrong": {
            "count": len(right_to_wrong),
            "percentage": round(len(right_to_wrong) / total_entries * 100, 2) if total_entries > 0 else 0,
            "directionality": r2w_direction,
        },
        "no_flip": {
            "count": total_entries - len(wrong_to_right) - len(right_to_wrong),
            "percentage": round(
                (total_entries - len(wrong_to_right) - len(right_to_wrong)) / total_entries * 100,
                2,
            )
            if total_entries > 0
            else 0,
        },
        "net_benefit": len(wrong_to_right) - len(right_to_wrong),
        "wrong_to_right_transitions": build_transition_stats(wrong_to_right),
        "right_to_wrong_transitions": build_transition_stats(right_to_wrong),
    }


def analyze_dataset_balance(results: Iterable[Any]) -> Dict[str, Any]:
    gt_counter = Counter()
    original_pred_counter = Counter()
    modified_pred_counter = Counter()

    results_list = list(results)
    for entry in results_list:
        gt = _extract_ground_truth(entry)
        gt_counter[classify_answer(gt)] += 1

        trace = _extract_trace(entry)
        if not trace:
            continue
        step0 = trace[0]
        orig_top10 = step0.get("original_top10", [])
        mod_top10 = step0.get("modified_top10", [])
        if orig_top10:
            original_pred_counter[classify_answer(orig_top10[0].get("token", ""))] += 1
        if mod_top10:
            modified_pred_counter[classify_answer(mod_top10[0].get("token", ""))] += 1

    total = sum(gt_counter.values())

    def make_stats(counter, count_total):
        return {
            "yes": {"count": counter.get("yes", 0), "percentage": round(counter.get("yes", 0) / count_total * 100, 2) if count_total > 0 else 0},
            "no": {"count": counter.get("no", 0), "percentage": round(counter.get("no", 0) / count_total * 100, 2) if count_total > 0 else 0},
            "other": {"count": counter.get("other", 0), "percentage": round(counter.get("other", 0) / count_total * 100, 2) if count_total > 0 else 0},
            "total": count_total,
        }

    gt_stats = make_stats(gt_counter, total)
    orig_stats = make_stats(original_pred_counter, sum(original_pred_counter.values()))
    mod_stats = make_stats(modified_pred_counter, sum(modified_pred_counter.values()))

    gt_yes_pct = gt_stats["yes"]["percentage"]
    orig_yes_pct = orig_stats["yes"]["percentage"]
    mod_yes_pct = mod_stats["yes"]["percentage"]

    original_yes_bias = orig_yes_pct - gt_yes_pct
    modified_yes_bias = mod_yes_pct - gt_yes_pct

    return {
        "ground_truth_distribution": gt_stats,
        "original_prediction_distribution": orig_stats,
        "modified_prediction_distribution": mod_stats,
        "bias_analysis": {
            "ground_truth_yes_no_ratio": round(gt_counter.get("yes", 0) / gt_counter.get("no", 1), 3)
            if gt_counter.get("no", 0) > 0
            else float("inf"),
            "original_yes_no_ratio": round(original_pred_counter.get("yes", 0) / original_pred_counter.get("no", 1), 3)
            if original_pred_counter.get("no", 0) > 0
            else -1.0,
            "modified_yes_no_ratio": round(modified_pred_counter.get("yes", 0) / modified_pred_counter.get("no", 1), 3)
            if modified_pred_counter.get("no", 0) > 0
            else -1.0,
            "original_yes_bias": round(original_yes_bias, 2),
            "modified_yes_bias": round(modified_yes_bias, 2),
            "bias_reduction": round(abs(original_yes_bias) - abs(modified_yes_bias), 2),
            "is_dataset_balanced": abs(gt_stats["yes"]["percentage"] - gt_stats["no"]["percentage"]) < 10,
            "did_cd_reduce_bias": abs(modified_yes_bias) < abs(original_yes_bias),
        },
    }


def aad_extract_yes_no(text: str) -> Optional[str]:
    """
    Text-based yes/no extraction with negation-aware heuristics (AAD paper style).
    Returns "yes", "no", or None if extraction is inconclusive.
    Used in spot-check mode when step-0 top token is not yes/no.
    """
    import re
    if not text:
        return None
    t = text.lower().strip()

    # First word — strongest signal
    first_word = "".join(c for c in t.split()[0] if c.isalnum()) if t.split() else ""
    if first_word == "yes":
        return "yes"
    if first_word == "no":
        return "no"

    # Negation patterns → "no"
    neg_patterns = [
        r"\bthere is no\b",
        r"\bthere are no\b",
        r"\bdoes not\b",
        r"\bdo not\b",
        r"\bcannot\b",
        r"\bnot present\b",
        r"\bnot audible\b",
        r"\bnot heard\b",
        r"\bnot detected\b",
        r"\bunable to\b",
    ]
    for pat in neg_patterns:
        if re.search(pat, t):
            return "no"

    # Affirmation patterns → "yes"
    aff_patterns = [
        r"\bcontains\b",
        r"\bcontain\b",
        r"\bcan be heard\b",
        r"\bis present\b",
        r"\bis audible\b",
    ]
    for pat in aff_patterns:
        if re.search(pat, t):
            return "yes"

    # Word-boundary substring fallback
    if re.search(r"\byes\b", t):
        return "yes"
    if re.search(r"\bno\b", t):
        return "no"

    return None


def needs_spot_check(sample: SampleResult) -> bool:
    """Return True if this sample's step-0 modified top token is not yes/no
    and has not already been spot-checked."""
    if sample.spot_checked:
        return False
    if not isinstance(sample.logit_trace, list) or not sample.logit_trace:
        return False
    step0 = sample.logit_trace[0]
    if not isinstance(step0, dict):
        return False
    mod_top10 = step0.get("modified_top10", [])
    if not mod_top10:
        return False
    first = mod_top10[0]
    if not isinstance(first, dict):
        return False
    token = str(first.get("token", "")).strip().lower()
    clean = "".join(c for c in token if c.isalnum())
    return clean not in ("yes", "no")


def replace_summary_in_file(results_dir: Path | str, summary_data: Dict[str, Any]) -> None:
    """Replace the matching summary entry in summaries.json (match by perturbation_type/setting/alpha)."""
    summaries_path = Path(results_dir) / Config.summaries_file_name

    if not summaries_path.exists():
        update_summaries_file(results_dir, summary_data)
        return

    with summaries_path.open("r", encoding="utf-8") as f:
        all_summaries = json.load(f)

    new_meta = summary_data.get("metadata", {})
    pert_type = new_meta.get("perturbation_type")
    pert_setting = new_meta.get("perturbation_setting")
    alpha = new_meta.get("aad_alpha")

    replaced = False
    for i, entry in enumerate(all_summaries):
        meta = entry.get("metadata", {})
        if (meta.get("perturbation_type") == pert_type
                and meta.get("perturbation_setting") == pert_setting
                and meta.get("aad_alpha") == alpha):
            all_summaries[i] = summary_data
            replaced = True
            break

    if not replaced:
        all_summaries.append(summary_data)

    with summaries_path.open("w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"Updated summaries file: {summaries_path}")


SPOT_CHECK_OBS_DIR = Path("observations/spot_check_reports")


def _spot_check_report_path(result_fpath: Path) -> Path:
    """Derive the per-file report path from the result file path.

    Works for both relative paths (results/...) and absolute paths
    (/data/.../AdaptivePerturbation/results/...) by scanning for the
    'results' component and taking everything after it.
    """
    all_parts = result_fpath.parts
    try:
        results_idx = next(i for i, p in enumerate(all_parts) if p == "results")
        rel_parts = all_parts[results_idx + 1:]
    except StopIteration:
        rel_parts = (result_fpath.name,)

    stem = rel_parts[-1] if rel_parts else result_fpath.name
    for ext in (".json.gz", ".json"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break

    parts_to_join = list(rel_parts[:-1]) + [stem]
    return SPOT_CHECK_OBS_DIR / ("__".join(parts_to_join) + ".txt")


def append_spot_check_report(
    result_fpath: Path,
    model_name: str,
    total_samples: int,
    n_flagged: int,
    accuracy_before: float,
    accuracy_after: float,
    changes: "list[dict]",
) -> Path:
    """
    Append a timestamped spot-check block to the per-file report.
    Only called when n_flagged > 0.

    Each dict in `changes` has keys:
      audio_file, ground_truth, old_answer, new_answer, method,
      was_correct, is_correct, old_model_response, new_model_response.
    Returns the report path.
    """
    report_path = _spot_check_report_path(result_fpath)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    method_counts: Dict[str, int] = {}
    answer_changed = [c for c in changes if c["old_answer"] != c["new_answer"]]
    answer_same    = [c for c in changes if c["old_answer"] == c["new_answer"]]
    for c in changes:
        method_counts[c["method"]] = method_counts.get(c["method"], 0) + 1

    w2r = sum(1 for c in answer_changed if not c["was_correct"] and c["is_correct"])
    r2w = sum(1 for c in answer_changed if c["was_correct"] and not c["is_correct"])

    lines = []
    w = lines.append
    sep = "=" * 70

    w(sep)
    w(f"  SPOT-CHECK RUN  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"  File:   {result_fpath}")
    w(f"  Model:  {model_name}")
    w(sep)
    w("")
    w("  SUMMARY")
    w(f"  Total samples               {total_samples}")
    w(f"  Flagged (non-yes/no top-0)  {n_flagged}")
    w(f"  Total spot-checked          {len(changes)}")
    w(f"  Answer changed              {len(answer_changed)}  (wrong→right: {w2r}, right→wrong: {r2w})")
    w(f"  Answer unchanged            {len(answer_same)}")
    w("")
    w(f"  Accuracy before  {accuracy_before * 100:.2f}%")
    w(f"  Accuracy after   {accuracy_after * 100:.2f}%")
    w(f"  Delta            {(accuracy_after - accuracy_before) * 100:+.2f}%")
    w("")
    w("  METHOD BREAKDOWN")
    for method, count in sorted(method_counts.items()):
        w(f"    {method:<20} {count}")
    w("")

    # ---- Per-sample detail blocks ----------------------------------------
    if answer_changed:
        w(sep)
        w(f"  CHANGED ANSWERS  ({len(answer_changed)} total)")
        w(sep)
        for i, c in enumerate(answer_changed, 1):
            af = Path(c["audio_file"]).name
            verdict = ("wrong→right" if not c["was_correct"] and c["is_correct"]
                       else "right→wrong" if c["was_correct"] and not c["is_correct"]
                       else "same correctness")
            w(f"  [{i}/{len(answer_changed)}] {af}")
            w(f"    GT       : {c['ground_truth']}")
            w(f"    Old ans  : {c['old_answer']}  ({'correct' if c['was_correct'] else 'wrong'})")
            w(f"    New ans  : {c['new_answer']}  ({'correct' if c['is_correct'] else 'wrong'})  [{verdict}]")
            w(f"    Method   : {c['method']}")
            w(f"    Orig resp: {c['old_model_response']!r}")
            w(f"    New resp : {c['new_model_response']!r}")
            w("")

    if answer_same:
        w(sep)
        w(f"  UNCHANGED ANSWERS  ({len(answer_same)} total — spot-check confirmed original)")
        w(sep)
        for i, c in enumerate(answer_same, 1):
            af = Path(c["audio_file"]).name
            w(f"  [{i}/{len(answer_same)}] {af}")
            w(f"    GT       : {c['ground_truth']}")
            w(f"    Answer   : {c['new_answer']}  ({'correct' if c['is_correct'] else 'wrong'})  [unchanged]")
            w(f"    Method   : {c['method']}")
            w(f"    Orig resp: {c['old_model_response']!r}")
            w(f"    New resp : {c['new_model_response']!r}")
            w("")

    w(sep)
    w("")

    with report_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return report_path


def update_summaries_file(results_dir: Path | str, summary_data: Dict[str, Any]) -> None:
    summaries_path = Path(results_dir) / Config.summaries_file_name

    if summaries_path.exists():
        with summaries_path.open("r", encoding="utf-8") as f:
            all_summaries = json.load(f)
    else:
        all_summaries = []

    all_summaries.append(summary_data)

    with summaries_path.open("w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"Updated summaries file: {summaries_path}")


SOFTMAX_DISTANCE_KEYS = frozenset({"l1", "l2", "l3", "linf", "cosine", "kl"})


def has_softmax_distance(
    perturbation_type: str,
    alpha: float,
    perturbation_setting: Optional[str] = None,
    results_dir: Optional[Path | str] = None,
) -> bool:
    """Return True if every sample in the result file has a complete softmax_distance dict."""
    import gzip as _gzip
    output_path = get_output_filename(perturbation_type, alpha, perturbation_setting, results_dir)
    fpath = (
        output_path if output_path.exists()
        else Path(str(output_path) + ".gz") if Path(str(output_path) + ".gz").exists()
        else None
    )
    if fpath is None:
        return False
    try:
        opener = _gzip.open if str(fpath).endswith(".gz") else open
        with opener(fpath, "rt", encoding="utf-8") as f:
            data = json.load(f)
        results = data.get("results", [])
        if not results:
            return False
        return all(
            isinstance(r.get("softmax_distance"), dict)
            and SOFTMAX_DISTANCE_KEYS.issubset(r["softmax_distance"])
            and all(
                isinstance(r["softmax_distance"][k], (int, float))
                for k in SOFTMAX_DISTANCE_KEYS
            )
            for r in results
        )
    except Exception:
        return False


def avg_softmax_distances(results: Iterable[Any]) -> Optional[Dict[str, float]]:
    """Compute per-metric mean of softmax_distance across all samples that have each key."""
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for r in results:
        sd = r.softmax_distance if hasattr(r, "softmax_distance") else (
            r.get("softmax_distance") if isinstance(r, dict) else None
        )
        if not isinstance(sd, dict):
            continue
        for k, v in sd.items():
            if k not in SOFTMAX_DISTANCE_KEYS:
                continue
            if not isinstance(v, (int, float)):
                continue
            sums[k] = sums.get(k, 0.0) + float(v)
            counts[k] = counts.get(k, 0) + 1
    if not sums:
        return None
    return {k: round(sums[k] / counts[k], 6) for k in sums}


def build_summary_from_output(output: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "metadata": output["metadata"],
        "performance_metrics": output["performance_metrics"],
        "flip_analysis": {
            "wrong_to_right": output["flip_analysis"]["wrong_to_right"],
            "right_to_wrong": output["flip_analysis"]["right_to_wrong"],
            "net_benefit": output["flip_analysis"]["net_benefit"],
        },
        "dataset_balance_analysis": {
            "bias_analysis": output["dataset_balance_analysis"]["bias_analysis"]
        },
    }
    # Include average softmax distances if present
    results = output.get("results", [])
    avg_sd = avg_softmax_distances(results)
    if avg_sd is not None:
        summary["avg_softmax_distances"] = avg_sd
    return summary


# ---------------------------------------------------------------------------
# VACoDe-style softmax distance metrics between two logit vectors.
#
# Computes distances over the full vocabulary softmax distribution.
# Earth Mover's Distance is omitted — it requires a ground distance
# between tokens, which has no natural definition for vocabulary indices.
# ---------------------------------------------------------------------------

def compute_softmax_distances(
    logits_orig: "torch.Tensor",
    logits_neg: "torch.Tensor",
) -> Dict[str, float]:
    """
    Compute VACoDe-style distances between two next-token distributions.

    Inputs are 1-D raw logit vectors of shape [vocab_size] — one from the
    clean audio branch (logits_orig) and one from the perturbed/negative
    branch (logits_neg). Both are converted to full-vocabulary softmax
    distributions in float32 before any distance is computed, so bfloat16
    model outputs don't accumulate rounding error.

    The returned distances measure how different the model's next-token
    distribution is when it hears the perturbed audio vs. the clean audio.
    A higher distance means the perturbation pushed the distribution further
    from the original — VACoDe picks the highest-distance perturbation per
    sample so that contrastive decoding subtracts the most divergent signal.
    """
    import torch
    import torch.nn.functional as F

    # Convert raw logits → proper probability distributions over the full vocab.
    # float() upcasts from bfloat16/float16 so small-prob tokens aren't zeroed out.
    p_orig = torch.softmax(logits_orig.float(), dim=-1)
    p_neg  = torch.softmax(logits_neg.float(),  dim=-1)
    diff   = p_orig - p_neg  # element-wise difference, shape [vocab_size]

    # Lp norms of the probability difference vector
    l1   = diff.abs().sum().item()                    # sum of |p_orig - p_neg| over all tokens
    l2   = diff.norm(p=2).item()                     # sqrt(sum of (p_orig - p_neg)^2) — the VACoDe metric
    l3   = diff.abs().pow(3).sum().pow(1.0 / 3.0).item()
    linf = diff.abs().max().item()                   # largest single-token shift

    # Cosine distance: treats each distribution as a vector in vocab-dim space.
    # 0 = identical distributions, 1 = orthogonal (no token overlap in mass).
    cosine_sim = F.cosine_similarity(
        p_orig.unsqueeze(0), p_neg.unsqueeze(0)
    ).item()
    cosine = 1.0 - cosine_sim

    # KL divergence KL(p_orig || p_neg): how many extra nats are needed to encode
    # p_orig samples using a code optimised for p_neg. Asymmetric — measures how
    # surprising the negative branch is when seen through the original's lens.
    # p_neg is clamped to avoid log(0) on zero-probability tokens.
    kl = F.kl_div(
        p_neg.clamp(min=1e-10).log(),
        p_orig,
        reduction="sum",
    ).item()

    return {
        "l1":     round(l1,     6),
        "l2":     round(l2,     6),
        "l3":     round(l3,     6),
        "linf":   round(linf,   6),
        "cosine": round(cosine, 6),
        "kl":     round(kl,     6),
    }
