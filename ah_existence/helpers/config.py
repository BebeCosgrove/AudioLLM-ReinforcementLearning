from pathlib import Path
from typing import Any, Dict, Optional, Union

# Kept for backward compatibility — code that imports CURRENT_RUN directly still works.
# New code should use per-instance Config.split instead.
CURRENT_RUN = None

# ─────────────────────────────────────────────────────────────────────────────
# ACTIVE VARIANT — change this line to switch the default configuration
# ─────────────────────────────────────────────────────────────────────────────
ACTIVE_VARIANT = "ah_existence"
DEFAULT_ALPHA = 1.0

# ─────────────────────────────────────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

MODEL_METADATA: Dict[str, str] = {
    "qwen2": "Qwen/Qwen2-Audio-7B-Instruct",
    "af3":   "nvidia/audio-flamingo-3-hf",
}

# ─────────────────────────────────────────────────────────────────────────────
# VARIANT DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

# Clotho-AQA base configs — not added to CONFIG_VARIANTS directly.
# Keys become the CLI dataset name suffix: clotho_{key} → "clotho_1word", "clotho_oldprompt".
# selector_key controls the cache/selector_data directory name (separate from dataset_variant,
# which controls the /results/ directory and is left as the original long-form name).
# Split-specific variants (1word_train, oldprompt_train, etc.) are generated below.
_CLOTHO_BASES: Dict[str, Dict[str, Any]] = {
    "oldprompt": {
        "dataset_name": "clotho_aqa",
        "dataset_variant": "clotho_aqa_old_prompt",
        "selector_key": "clotho_oldprompt",
        "run_prompt": "Focus on the given audio and answer the following question.",
        "default_max_new_tokens": 20,
        "use_step0_yes_no_logits_extraction": False,
    },
    "1word": {
        "dataset_name": "clotho_aqa",
        "dataset_variant": "clotho_aqa_1word_prompt",
        "selector_key": "clotho_1word",
        "run_prompt": "Focus on the given audio and answer the following question with exactly one word: yes or no.",
        "default_max_new_tokens": 1,
        "use_step0_yes_no_logits_extraction": True,
    },
}

CONFIG_VARIANTS: Dict[str, Dict[str, Any]] = {}

# Clotho: three explicit split variants per prompt style (internal use by load_clotho_bundle)
for _base_key, _base_val in _CLOTHO_BASES.items():
    for _split in ("train", "val", "test"):
        CONFIG_VARIANTS[f"{_base_key}_{_split}"] = {**_base_val, "split": _split}

# Bundle-level Clotho dataset keys accepted by the architecture exploration CLI.
# These load all three splits internally via load_clotho_bundle.
CLOTHO_BUNDLE_DATASETS: list = [f"clotho_{k}" for k in _CLOTHO_BASES]

# clotho_rebalanced: same 1word prompt as clotho_1word, but uses custom random splits
# (created by scripts/create_clotho_rebalanced_splits.py) instead of the canonical
# train/val/test boundary.  Cache lives under clotho_1word/ (shared; prompt is identical).
CONFIG_VARIANTS["clotho_rebalanced"] = {
    "dataset_name": "clotho_aqa",
    "dataset_variant": "clotho_aqa_1word_prompt",
    "selector_key": "clotho_rebalanced",
    "run_prompt": "Focus on the given audio and answer the following question with exactly one word: yes or no.",
    "default_max_new_tokens": 1,
    "use_step0_yes_no_logits_extraction": True,
    "split": None,   # explicit reset — apply_variant() only patches listed fields, so without
                     # this, any prior 1word_train/val/test variant would leave split="train" etc.
}


def clotho_prompt_style_from_bundle(dataset_key: str) -> str:
    """Return the prompt-style key from a bundle dataset key like 'clotho_1word'."""
    prefix = "clotho_"
    if not dataset_key.startswith(prefix):
        raise ValueError(f"Not a Clotho bundle dataset key: {dataset_key!r}")
    prompt_style = dataset_key[len(prefix):]
    if prompt_style not in _CLOTHO_BASES:
        raise ValueError(
            f"Unknown Clotho prompt style: {prompt_style!r}. "
            f"Valid bundle keys: {CLOTHO_BUNDLE_DATASETS}"
        )
    return prompt_style


def clotho_split_variants(dataset_key: str) -> list[str]:
    """Expand 'clotho_1word' / 'clotho_oldprompt' to their train/val/test variants."""
    prompt_style = clotho_prompt_style_from_bundle(dataset_key)
    return [f"{prompt_style}_{split}" for split in ("train", "val", "test")]


def clotho_representative_variant(dataset_key: str) -> str:
    """Return a split variant suitable for shared Clotho config/caching setup."""
    return clotho_split_variants(dataset_key)[0]


def results_dataset_dir_name(dataset_key: str) -> str:
    """Map a user-facing dataset key to the directory name used under /results."""
    if dataset_key in CLOTHO_BUNDLE_DATASETS:
        prompt_style = clotho_prompt_style_from_bundle(dataset_key)
        return _CLOTHO_BASES[prompt_style]["dataset_variant"]
    if dataset_key in CONFIG_VARIANTS:
        v = CONFIG_VARIANTS[dataset_key]
        return v.get("dataset_variant") or v.get("dataset_name", dataset_key)
    return dataset_key

# Audio Hallucination benchmark (Kuan & Lee, ICASSP 2025)
# No train/val/test split — single dataset file per variant.
CONFIG_VARIANTS.update({
    "ah_existence": {
        "dataset_name": "ah_existence",
        "dataset_variant": None,
        "run_prompt": "Focus on the given audio and answer the following question with exactly one word: yes or no.",
        "default_max_new_tokens": 1,
        "use_step0_yes_no_logits_extraction": True,
    },
    "ah_order": {
        "dataset_name": "ah_order",
        "dataset_variant": None,
        "run_prompt": "Focus on the given audio and answer the following question with exactly one word: yes or no.",
        "default_max_new_tokens": 1,
        "use_step0_yes_no_logits_extraction": True,
    },
    "ah_attribute": {
        "dataset_name": "ah_attribute",
        "dataset_variant": None,
        "run_prompt": "Focus on the given audio and answer the following question with exactly one word: yes or no.",
        "default_max_new_tokens": 1,
        "use_step0_yes_no_logits_extraction": True,
    },
    "ah_extended": {
        "dataset_name": "ah_extended",
        "dataset_variant": None,
        "run_prompt": "Focus on the given audio and answer the following question with exactly one word: yes or no.",
        "default_max_new_tokens": 1,
        "use_step0_yes_no_logits_extraction": True,
    },
    "ah_targeted": {
        "dataset_name": "ah_targeted",
        "dataset_variant": None,
        "run_prompt": "Focus on the given audio and answer the following question with exactly one word: yes or no.",
        "default_max_new_tokens": 1,
        "use_step0_yes_no_logits_extraction": True,
    },
})

# AH datasets with the general (non-1word) prompt — results go under results/{model}/aad_prompt/
AH_AAD_DATASETS: list = []
for _ah_base in ["ah_existence", "ah_order", "ah_attribute", "ah_extended", "ah_targeted"]:
    _aad_key = f"{_ah_base}_aad"
    CONFIG_VARIANTS[_aad_key] = {
        "dataset_name": _ah_base,
        "dataset_variant": f"aad_prompt/{_ah_base}",
        "run_prompt": "Focus on the given audio and answer the following question.",
        "default_max_new_tokens": 20,
        "use_step0_yes_no_logits_extraction": False,
    }
    AH_AAD_DATASETS.append(_aad_key)


def _fmt_alpha(alpha: Union[float, str]) -> str:
    """Format alpha for path names: 0.5 -> '0.5', 1.0 -> '1.0'."""
    return str(float(alpha))


class _Config:
    """
    Configuration object. Instantiate directly or use make_config() factory.

    model  - model slug ("qwen2" or "af3"); first subfolder under results/
             selector_data/ and architecture_exploration/.
    alpha  - contrastive decoding alpha; scopes all results, selector data, and
             ablation paths (float 0.5 / 1.0).
    split  - dataset split for this instance ("train", "val", "test", or None).
             Clotho split variants bake this in via CONFIG_VARIANTS["split"].
             AH datasets have no split (None).
    """

    # Fixed project-level paths (class-level, same for all instances)
    project_root       = Path(__file__).resolve().parent.parent
    selector_data_root = project_root / "selector_data"
    results_root       = project_root / "results"

    def __init__(
        self,
        variant: str = ACTIVE_VARIANT,
        model: str = "qwen2",
        alpha: Union[float, str] = DEFAULT_ALPHA,
        split: Optional[str] = None,
    ):
        v = CONFIG_VARIANTS.get(variant, {})

        # From variant
        self.dataset_name: str = v.get("dataset_name", "clotho_aqa")
        self.dataset_variant: Optional[str] = v.get("dataset_variant", "clotho_aqa_old_prompt")
        # selector_key: short name used for selector_data/ cache directories.
        # Falls back to dataset_variant (or dataset_name for AH) when not set.
        self.selector_key: Optional[str] = v.get("selector_key", None)
        self.run_prompt: str = v.get(
            "run_prompt",
            "Focus on the given audio and answer the following question.",
        )
        self.default_max_new_tokens: int = v.get("default_max_new_tokens", 1)
        self.use_step0_yes_no_logits_extraction: bool = v.get(
            "use_step0_yes_no_logits_extraction", False
        )

        # Model
        assert model in MODEL_METADATA, (
            f"model={model!r} is not supported; must be one of {list(MODEL_METADATA)}"
        )
        self.model: str = model
        self.model_name: str = MODEL_METADATA[model]

        self.default_batch_size: int = 16
        self.summaries_file_name: str = "summaries.json"

        # Normalize alpha and validate {0.5, 1.0}
        if isinstance(alpha, str):
            assert alpha in {"0.5", "1.0"}, (
                f"alpha={alpha!r} is not supported; must be '0.5' or '1.0'"
            )
            alpha = float(alpha)
        assert alpha in {0.5, 1.0}, (
            f"alpha={alpha} is not supported; must be 0.5 or 1.0"
        )
        self.alpha: float = float(alpha)

        # Split: explicit arg > variant-defined split > module-level CURRENT_RUN (compat)
        self.split: Optional[str] = split if split is not None else v.get("split", CURRENT_RUN)

        # Dataset analysis
        self.open_ended_dataset_analysis: bool = False

        # Derived paths (populated by _recompute)
        self._recompute_derived_fields()

    def _recompute_derived_fields(self) -> None:
        """Recompute all path fields from current scalar attributes."""
        assert float(self.alpha) in {0.5, 1.0}, (
            f"alpha={self.alpha} is not supported; must be 0.5 or 1.0"
        )
        self.alpha = float(self.alpha)
        _split = self.split  # None for AH / unsplit datasets
        _model = self.model
        self.model_name = MODEL_METADATA[_model]

        # Dataset paths (not model-scoped — datasets are shared across models)
        self.dataset_location   = self.project_root / "datasets" / self.dataset_name
        self.dataset_audio_dir  = self.dataset_location / "audio_files"
        self.dataset_train_json = self.dataset_location / f"{self.dataset_name}_train.json"
        self.dataset_val_json   = self.dataset_location / f"{self.dataset_name}_val.json"
        self.dataset_test_json  = self.dataset_location / f"{self.dataset_name}_test.json"
        self.dataset_eval_json  = self.dataset_location / f"{self.dataset_name}_eval.json"

        if _split is None:
            self.curr_dataset_json = self.dataset_location / f"{self.dataset_name}.json"
        else:
            self.curr_dataset_json = (
                self.dataset_location / f"{self.dataset_name}_{_split}.json"
            )

        # Results and selector paths — model is first subfolder, alpha scopes all subdirs
        _variant_dir = self.dataset_variant if self.dataset_variant is not None else self.dataset_name
        _alpha_str   = _fmt_alpha(self.alpha)

        _results_model_root        = self.results_root / _model
        self.results_dataset_dir       = _results_model_root / _variant_dir
        self.results_dataset_alpha_dir = self.results_dataset_dir / _alpha_str
        self.results_train_dir         = self.results_dataset_dir / "train" / _alpha_str
        self.results_val_dir           = self.results_dataset_dir / "val" / _alpha_str

        if _split is None:
            self.curr_results_dir = self.results_dataset_alpha_dir
        else:
            self.curr_results_dir = self.results_dataset_dir / _split / _alpha_str

        # Legacy Clotho-style ablation results (not alpha-scoped).
        # The historical ablation tree now lives under scripts/unused/.
        self.ablation_results_dir = (
            self.project_root / "scripts" / "unused" / "ablation_tests" / "results" / _variant_dir
        )

        # Selector paths — model is first subfolder.
        # selector_key overrides _variant_dir for the cache directory name so that
        # clotho_1word / clotho_oldprompt are used instead of the long dataset_variant names,
        # while /results/ paths above continue to use the original dataset_variant names.
        _selector_model_root  = self.selector_data_root / _model
        _selector_dir = self.selector_key if self.selector_key is not None else _variant_dir
        self.selector_dataset_base_dir  = _selector_model_root / _selector_dir
        self.selector_cache_dir         = self.selector_dataset_base_dir / "cached_hidden_states"
        self.selector_train_cache_dir   = self.selector_dataset_base_dir / "train" / "cached_hidden_states"
        self.selector_val_cache_dir     = self.selector_dataset_base_dir / "val" / "cached_hidden_states"

        # Alpha-scoped dir — used for oracle files and ablation results
        self.selector_dataset_dir    = _selector_model_root / _selector_dir / _alpha_str
        self.selector_train_dir      = self.selector_dataset_dir / "train"
        self.selector_validation_dir = self.selector_dataset_dir / "val"

        self.oracle_train_file = self.selector_train_dir / "oracle_train.json.gz"
        self.oracle_val_file   = self.selector_validation_dir / "oracle_val.json.gz"

        _oracle_name = (
            "oracle.json.gz" if _split is None else f"oracle_{_split}.json.gz"
        )
        self.curr_oracle_file_name = _oracle_name
        self.curr_oracle_file      = self.selector_dataset_dir / _oracle_name

        if _split is None:
            self.curr_selector_dir    = self.selector_dataset_dir
            self.curr_selector_suffix = ""
        elif _split == "train":
            self.curr_selector_dir    = self.selector_train_dir
            self.curr_selector_suffix = "_train"
        else:
            self.curr_selector_dir    = self.selector_validation_dir
            self.curr_selector_suffix = f"_{_split}"

        self.effective_perturb_registry_global = (
            _selector_model_root / "effective_perturb_spec_registry_global.json"
        )
        self.effective_perturb_registry_train = (
            self.selector_train_dir / "effective_perturb_spec_registry_train.json"
        )
        self.effective_perturb_registry_val = (
            self.selector_validation_dir / "effective_perturb_spec_registry_val.json"
        )
        self.evaluation_pool_registry_val = (
            self.selector_validation_dir / "evaluation_pool_registry_val.json"
        )

        self.selector_training_data_train = (
            self.selector_train_dir / "selector_training_data_train.json"
        )
        self.selector_training_data_val = (
            self.selector_validation_dir / "selector_training_data_val.json"
        )

        # Architecture exploration paths — model then alpha
        self.ah_ablation_results_dir = (
            self.project_root / "architecture_exploration" / _model / _alpha_str / "results"
        )
        self.ah_ablation_logs_dir = (
            self.project_root / "architecture_exploration" / _model / _alpha_str / "logs"
        )

        # Dataset analysis
        self.dataset_analysis_suffix = (
            "open_ended" if self.open_ended_dataset_analysis else "close_ended"
        )
        self.dataset_analysis_reports_dir = (
            self.project_root / "dataset_analysis" / "reports"
        )

    def apply_variant(self, overrides: Dict[str, Any]) -> None:
        """Patch specific fields (partial override), then recompute derived fields."""
        for k, v in overrides.items():
            setattr(self, k, v)
        self._recompute_derived_fields()


# ─────────────────────────────────────────────────────────────────────────────
# FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def make_config(
    variant: Optional[str] = None,
    model: Optional[str] = None,
    alpha: Optional[Union[float, str]] = None,
    split: Optional[str] = None,
) -> _Config:
    """
    Factory: create a fresh Config instance with optional overrides.

    variant - CONFIG_VARIANTS key (e.g. "ah_attribute", "1word_train").
              Defaults to ACTIVE_VARIANT.
    model   - model slug ("qwen2" or "af3"). Defaults to "qwen2".
    alpha   - contrastive decoding alpha; scopes all results and selector data
              (0.5 or 1.0, default 1.0).
    split   - override the split for this instance ("train", "val", "test", None).
              Usually comes from the variant itself; only needed for one-off overrides.
    """
    v = variant if variant is not None else ACTIVE_VARIANT
    m = model   if model   is not None else "qwen2"
    a = alpha   if alpha   is not None else DEFAULT_ALPHA
    return _Config(variant=v, model=m, alpha=a, split=split)


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL SINGLETON  (backward-compatible: Config.xxx still works everywhere)
# ─────────────────────────────────────────────────────────────────────────────

Config = _Config()


def apply_variant(overrides: Dict[str, Any]) -> None:
    """Apply partial overrides to the global Config singleton and recompute paths."""
    Config.apply_variant(overrides)
