from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import gzip
import json

from helpers.config import Config


@dataclass
class SampleResult:
    # Schema format for this sample: "freeform" or "yes_no"
    format: str = "freeform"

    # Common identity fields
    audio_file: str = ""
    question: str = ""
    ground_truth: str = ""

    # Freeform-style fields
    model_response: str = ""
    extracted_answer: str = ""
    is_correct: bool = False
    aad_enabled: bool = False
    aad_alpha: float = 0.0

    # New trace format
    logit_trace: Optional[List[Dict[str, Any]]] = None

    # VACoDe-style softmax distances (L1/L2/L3/Linf/cosine/KL) between
    # original and negative branch at step 0, computed over full vocabulary.
    softmax_distance: Optional[Dict[str, float]] = None

    # yes_no-style fields
    pred: Optional[Dict[str, Any]] = None
    aad: Optional[Dict[str, Any]] = None
    step0: Optional[Dict[str, Any]] = None

    # Optional per-sample metadata
    pruned_at_step: Optional[int] = None

    # Spot-check provenance (set when a re-run with verbose generation updated this sample)
    spot_checked: bool = False
    spot_check_method: Optional[str] = None

    # Any extra unknown fields to preserve round-trip
    extra: Dict[str, Any] = field(default_factory=dict)

    def _validate_output_strict(self) -> None:
        if self.extra:
            raise ValueError(
                f"Unexpected sample extra fields for format '{self.format}': {sorted(self.extra.keys())}"
            )

        if self.format == "yes_no":
            # yes_no should not emit freeform-only payload fields.
            freeform_payload_present = (
                bool(self.model_response)
                or bool(self.extracted_answer)
                or bool(self.is_correct)
                or bool(self.aad_enabled)
                or bool(self.aad_alpha)
            )
            if freeform_payload_present:
                raise ValueError(
                    "yes_no output received unexpected freeform fields "
                    "(model_response/extracted_answer/is_correct/aad_*)."
                )

            if self.pred is None or self.aad is None:
                raise ValueError("yes_no output requires both 'pred' and 'aad'.")

            # yes_no schema should not carry trace/pruning fields.
            if self.logit_trace is not None:
                raise ValueError("yes_no output received unexpected field: logit_trace")
            if self.pruned_at_step is not None:
                raise ValueError("yes_no output received unexpected field: _pruned_at_step")

        elif self.format == "freeform":
            # freeform should not emit yes_no-only nested blocks.
            if self.pred is not None or self.aad is not None or self.step0 is not None:
                raise ValueError("freeform output received unexpected yes_no fields (pred/aad/step0).")
        else:
            raise ValueError(f"Unknown sample format for output: {self.format}")

    @staticmethod
    def from_dict(data: Dict[str, Any], default_format: str = "freeform") -> "SampleResult":
        sample_format = str(data.get("format") or default_format or "freeform")

        known = {
            "format",
            "audio_file",
            "question",
            "ground_truth",
            "model_response",
            "extracted_answer",
            "is_correct",
            "aad_enabled",
            "aad_alpha",
            "negative_branch_mode",
            "logit_trace",
            "softmax_distance",
            "xai_history",
            "xai_step0",
            "_pruned_at_step",
            "pred",
            "aad",
            "step0",
            "spot_checked",
            "spot_check_method",
        }

        trace = data.get("logit_trace")

        extra = {k: v for k, v in data.items() if k not in known}

        pred = data.get("pred") if isinstance(data.get("pred"), dict) else None
        aad = data.get("aad") if isinstance(data.get("aad"), dict) else None
        step0 = data.get("step0") if isinstance(data.get("step0"), dict) else None
        if isinstance(aad, dict) and "negative_branch_mode" in aad:
            aad = dict(aad)
            aad.pop("negative_branch_mode", None)

        # yes_no format uses nested pred/aad blocks
        extracted_answer = data.get("extracted_answer", "")
        is_correct = bool(data.get("is_correct", False))
        aad_enabled = bool(data.get("aad_enabled", False))
        aad_alpha = float(data.get("aad_alpha", 0.0) or 0.0)

        if sample_format == "yes_no":
            if pred:
                extracted_answer = str(pred.get("label", extracted_answer))
                is_correct = bool(pred.get("is_correct", is_correct))
            if aad:
                aad_enabled = bool(aad.get("enabled", aad_enabled))
                aad_alpha = float(aad.get("alpha", aad_alpha) or 0.0)

        sd = data.get("softmax_distance")
        return SampleResult(
            format=sample_format,
            audio_file=str(data.get("audio_file", "")),
            question=str(data.get("question", "")),
            ground_truth=str(data.get("ground_truth", "")),
            model_response=str(data.get("model_response", "")),
            extracted_answer=str(extracted_answer),
            is_correct=is_correct,
            aad_enabled=aad_enabled,
            aad_alpha=aad_alpha,
            logit_trace=trace if isinstance(trace, list) else None,
            softmax_distance=sd if isinstance(sd, dict) else None,
            pred=pred,
            aad=aad,
            step0=step0,
            pruned_at_step=data.get("_pruned_at_step"),
            spot_checked=bool(data.get("spot_checked", False)),
            spot_check_method=data.get("spot_check_method"),
            extra=extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        self._validate_output_strict()

        if self.format == "yes_no":
            pred = dict(self.pred) if isinstance(self.pred, dict) else {}
            pred.setdefault("token", pred.get("token", self.extracted_answer))
            pred.setdefault("label", pred.get("label", self.extracted_answer))
            pred.setdefault("is_correct", pred.get("is_correct", self.is_correct))

            aad = dict(self.aad) if isinstance(self.aad, dict) else {}
            aad.setdefault("enabled", aad.get("enabled", self.aad_enabled))
            aad.setdefault("alpha", aad.get("alpha", self.aad_alpha))

            out: Dict[str, Any] = {
                "format": "yes_no",
                "audio_file": self.audio_file,
                "question": self.question,
                "ground_truth": self.ground_truth,
                "pred": pred,
                "aad": aad,
            }

            if self.step0 is not None:
                out["step0"] = self.step0

            out.update(self.extra)
            return out

        out: Dict[str, Any] = {
            "audio_file": self.audio_file,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "model_response": self.model_response,
            "extracted_answer": self.extracted_answer,
            "is_correct": self.is_correct,
            "aad_enabled": self.aad_enabled,
            "aad_alpha": self.aad_alpha,
        }

        if self.logit_trace is not None:
            out["logit_trace"] = self.logit_trace
        if self.softmax_distance is not None:
            out["softmax_distance"] = self.softmax_distance
        if self.pruned_at_step is not None:
            out["_pruned_at_step"] = self.pruned_at_step
        if self.spot_checked:
            out["spot_checked"] = True
        if self.spot_check_method is not None:
            out["spot_check_method"] = self.spot_check_method

        out.update(self.extra)
        return out


@dataclass
class RunResult:
    # Run format. If missing in source, defaults to freeform.
    format: str = "freeform"

    # Core metadata
    timestamp: Optional[str] = None
    model: str = Config.model_name
    dataset: Optional[str] = None
    total_samples: Optional[int] = None
    perturbation_type: Optional[str] = None
    perturbation_setting: Optional[str] = None
    aad_enabled: Optional[bool] = None
    aad_alpha: Optional[float] = None
    max_new_tokens: Optional[int] = Config.default_max_new_tokens
    batch_size: Optional[int] = Config.default_batch_size
    answer_extraction_uses_step0_yes_no_logits: Optional[bool] = None
    ground_truth_yes_no_ratio: Optional[float] = None
    extracted_prediction_yes_no_ratio: Optional[float] = None
    ground_truth_non_yes_no_count: Optional[int] = None
    extracted_prediction_non_yes_no_count: Optional[int] = None

    # Optional pruning metadata
    pruned: Optional[bool] = None
    pruned_at: Optional[str] = None
    pruning_strategy: Optional[str] = None
    total_original_steps: Optional[int] = None
    total_kept_steps: Optional[int] = None

    # Main payload sections
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    flip_analysis: Dict[str, Any] = field(default_factory=dict)
    dataset_balance_analysis: Dict[str, Any] = field(default_factory=dict)
    results: List[SampleResult] = field(default_factory=list)

    # Preserve unknown metadata/top-level fields for safe round-trip
    metadata_extra: Dict[str, Any] = field(default_factory=dict)
    top_level_extra: Dict[str, Any] = field(default_factory=dict)

    def _validate_output_strict(self) -> None:
        if self.metadata_extra:
            raise ValueError(
                f"Unexpected metadata extra fields for format '{self.format}': "
                f"{sorted(self.metadata_extra.keys())}"
            )
        if self.top_level_extra:
            raise ValueError(
                f"Unexpected top-level extra fields for format '{self.format}': "
                f"{sorted(self.top_level_extra.keys())}"
            )
        for i, sample in enumerate(self.results):
            if sample.format != self.format:
                raise ValueError(
                    f"Sample format mismatch at index {i}: run format '{self.format}', "
                    f"sample format '{sample.format}'"
                )

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "RunResult":
        meta = data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {}

        known_meta = {
            "timestamp",
            "model",
            "dataset",
            "total_samples",
            "perturbation_type",
            "perturbation_setting",
            "aad_enabled",
            "aad_alpha",
            "max_new_tokens",
            "batch_size",
            "answer_extraction_uses_step0_yes_no_logits",
            "ground_truth_yes_no_ratio",
            "extracted_prediction_yes_no_ratio",
            "ground_truth_non_yes_no_count",
            "extracted_prediction_non_yes_no_count",
            "pruned",
            "pruned_at",
            "pruning_strategy",
            "total_original_steps",
            "total_kept_steps",
        }
        metadata_extra = {k: v for k, v in meta.items() if k not in known_meta}

        known_top = {
            "format",
            "metadata",
            "performance_metrics",
            "flip_analysis",
            "dataset_balance_analysis",
            "results",
        }
        top_level_extra = {k: v for k, v in data.items() if k not in known_top}

        sample_objs = []
        run_format = str(data.get("format", "freeform") or "freeform")
        for item in data.get("results", []) if isinstance(data.get("results"), list) else []:
            if isinstance(item, dict):
                sample_objs.append(SampleResult.from_dict(item, default_format=run_format))

        return RunResult(
            format=run_format,
            timestamp=meta.get("timestamp"),
            model=str(meta.get("model", Config.model_name)),
            dataset=meta.get("dataset"),
            total_samples=meta.get("total_samples"),
            perturbation_type=meta.get("perturbation_type"),
            perturbation_setting=meta.get("perturbation_setting"),
            aad_enabled=meta.get("aad_enabled"),
            aad_alpha=meta.get("aad_alpha"),
            max_new_tokens=meta.get("max_new_tokens"),
            batch_size=meta.get("batch_size"),
            answer_extraction_uses_step0_yes_no_logits=meta.get("answer_extraction_uses_step0_yes_no_logits"),
            ground_truth_yes_no_ratio=meta.get("ground_truth_yes_no_ratio"),
            extracted_prediction_yes_no_ratio=meta.get("extracted_prediction_yes_no_ratio"),
            ground_truth_non_yes_no_count=meta.get("ground_truth_non_yes_no_count"),
            extracted_prediction_non_yes_no_count=meta.get("extracted_prediction_non_yes_no_count"),
            pruned=meta.get("pruned"),
            pruned_at=meta.get("pruned_at"),
            pruning_strategy=meta.get("pruning_strategy"),
            total_original_steps=meta.get("total_original_steps"),
            total_kept_steps=meta.get("total_kept_steps"),
            performance_metrics=data.get("performance_metrics", {}) if isinstance(data.get("performance_metrics"), dict) else {},
            flip_analysis=data.get("flip_analysis", {}) if isinstance(data.get("flip_analysis"), dict) else {},
            dataset_balance_analysis=data.get("dataset_balance_analysis", {}) if isinstance(data.get("dataset_balance_analysis"), dict) else {},
            results=sample_objs,
            metadata_extra=metadata_extra,
            top_level_extra=top_level_extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        self._validate_output_strict()

        meta: Dict[str, Any] = {
            "model": self.model,
        }

        if self.timestamp is not None:
            meta["timestamp"] = self.timestamp
        if self.dataset is not None:
            meta["dataset"] = self.dataset
        if self.total_samples is not None:
            meta["total_samples"] = self.total_samples
        if self.perturbation_type is not None:
            meta["perturbation_type"] = self.perturbation_type
        if self.perturbation_setting is not None:
            meta["perturbation_setting"] = self.perturbation_setting
        if self.aad_enabled is not None:
            meta["aad_enabled"] = self.aad_enabled
        if self.aad_alpha is not None:
            meta["aad_alpha"] = self.aad_alpha
        if self.max_new_tokens is not None:
            meta["max_new_tokens"] = self.max_new_tokens
        if self.batch_size is not None:
            meta["batch_size"] = self.batch_size
        if self.answer_extraction_uses_step0_yes_no_logits is not None:
            meta["answer_extraction_uses_step0_yes_no_logits"] = self.answer_extraction_uses_step0_yes_no_logits
        if self.ground_truth_yes_no_ratio is not None:
            meta["ground_truth_yes_no_ratio"] = self.ground_truth_yes_no_ratio
        if self.extracted_prediction_yes_no_ratio is not None:
            meta["extracted_prediction_yes_no_ratio"] = self.extracted_prediction_yes_no_ratio
        if self.ground_truth_non_yes_no_count is not None:
            meta["ground_truth_non_yes_no_count"] = self.ground_truth_non_yes_no_count
        if self.extracted_prediction_non_yes_no_count is not None:
            meta["extracted_prediction_non_yes_no_count"] = self.extracted_prediction_non_yes_no_count

        if self.pruned is not None:
            meta["pruned"] = self.pruned
        if self.pruned_at is not None:
            meta["pruned_at"] = self.pruned_at
        if self.pruning_strategy is not None:
            meta["pruning_strategy"] = self.pruning_strategy
        if self.total_original_steps is not None:
            meta["total_original_steps"] = self.total_original_steps
        if self.total_kept_steps is not None:
            meta["total_kept_steps"] = self.total_kept_steps

        meta.update(self.metadata_extra)

        out: Dict[str, Any] = {
            "metadata": meta,
            "performance_metrics": self.performance_metrics,
            "flip_analysis": self.flip_analysis,
            "dataset_balance_analysis": self.dataset_balance_analysis,
            "results": [r.to_dict() for r in self.results],
        }
        if self.format != "freeform":
            out["format"] = self.format
        return out

    def refresh_total_samples(self) -> None:
        self.total_samples = len(self.results)


# -----------------------------
# JSON conversion helpers
# -----------------------------

def run_result_from_dict(data: Dict[str, Any]) -> RunResult:
    return RunResult.from_dict(data)


def run_result_to_dict(run: RunResult) -> Dict[str, Any]:
    return run.to_dict()


def load_oracle_json(path: Path | str) -> dict:
    """Load an oracle JSON file, supporting both .json and .json.gz."""
    p = Path(path)
    if p.suffix == ".gz":
        with gzip.open(p, "rt", encoding="utf-8") as f:
            return json.load(f)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_combined_clotho_oracle(model: str = "qwen2", alpha: float = 1.0) -> dict:
    """Merge the clotho_1word train/val/test oracle files into one flat oracle dict.

    Returns {"examples": [...]} with all fields intact (audio_file, question,
    answer, worked_perturbations, …).  Order is train → val → test, matching
    the concatenation order used by load_clotho_bundle / load_clotho_rebalanced_bundle
    so that split indices produced by create_clotho_rebalanced_splits.py align
    correctly with the feature tensors at training time.
    """
    from helpers.config import Config, CONFIG_VARIANTS, apply_variant

    all_examples: List[dict] = []
    for split in ("train", "val", "test"):
        variant_key = f"1word_{split}"
        apply_variant(CONFIG_VARIANTS[variant_key])
        Config.model = model
        Config.alpha = float(alpha)
        Config._recompute_derived_fields()
        oracle_path = Config.curr_oracle_file
        if not oracle_path.exists():
            raise FileNotFoundError(
                f"clotho_1word oracle not found for split={split!r}: {oracle_path}\n"
                "Run the oracle generation step for clotho_1word before creating splits."
            )
        oracle = load_oracle_json(oracle_path)
        all_examples.extend(oracle.get("examples", []))

    return {"examples": all_examples}


def load_run_result(path: Path | str) -> RunResult:
    p = Path(path)
    if p.suffix == ".gz":
        with gzip.open(p, "rt", encoding="utf-8") as f:
            data = json.load(f)
    else:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    return RunResult.from_dict(data)


def save_run_result(run: RunResult, path: Path | str, indent: int = 2) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix == ".gz":
        with gzip.open(p, "wt", encoding="utf-8") as f:
            json.dump(run.to_dict(), f, indent=indent)
    else:
        with p.open("w", encoding="utf-8") as f:
            json.dump(run.to_dict(), f, indent=indent)


def load_sample_results(path: Path | str) -> List[SampleResult]:
    return load_run_result(path).results


def save_sample_results(
    samples: List[SampleResult],
    path: Path | str,
    template_run: Optional[RunResult] = None,
    indent: int = 2,
) -> None:
    run = template_run if template_run is not None else RunResult()
    run.results = samples
    run.refresh_total_samples()
    save_run_result(run, path, indent=indent)


# -----------------------------
# File discovery / bulk loading
# -----------------------------

def iter_eval_files(
    roots: Optional[Iterable[Path | str]] = None,
    glob_pattern: Optional[str] = None,
    recursive: bool = True,
) -> List[Path]:
    roots_list = [Path(r) for r in roots] if roots is not None else [Config.results_train_dir, Config.results_val_dir]
    pattern = glob_pattern or Config.eval_file_glob

    files: List[Path] = []
    for root in roots_list:
        if not root.exists():
            continue
        if recursive:
            files.extend(sorted(root.rglob(pattern)))
        else:
            files.extend(sorted(root.glob(pattern)))
    return files


def load_run_results_from_files(files: Iterable[Path | str]) -> List[RunResult]:
    runs: List[RunResult] = []
    for fp in files:
        runs.append(load_run_result(fp))
    return runs


def load_run_results(
    roots: Optional[Iterable[Path | str]] = None,
    glob_pattern: Optional[str] = None,
    recursive: bool = True,
) -> List[RunResult]:
    return load_run_results_from_files(iter_eval_files(roots=roots, glob_pattern=glob_pattern, recursive=recursive))


def load_all_sample_results(
    roots: Optional[Iterable[Path | str]] = None,
    glob_pattern: Optional[str] = None,
    recursive: bool = True,
) -> List[SampleResult]:
    out: List[SampleResult] = []
    for run in load_run_results(roots=roots, glob_pattern=glob_pattern, recursive=recursive):
        out.extend(run.results)
    return out


# -----------------------------
# Convenience constructors
# -----------------------------

def new_run_result(
    dataset: Optional[str] = None,
    perturbation_type: Optional[str] = None,
    perturbation_setting: Optional[str] = None,
    aad_alpha: Optional[float] = None,
    aad_enabled: Optional[bool] = None,
) -> RunResult:
    return RunResult(
        model=Config.model_name,
        dataset=dataset,
        perturbation_type=perturbation_type,
        perturbation_setting=perturbation_setting,
        aad_alpha=aad_alpha,
        aad_enabled=aad_enabled,
        max_new_tokens=Config.default_max_new_tokens,
        batch_size=Config.default_batch_size,
    )
