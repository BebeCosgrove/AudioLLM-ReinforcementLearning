"""
profiling.py
------------
Opt-in, minimally invasive wall-clock and GPU/CPU memory profiler for the
Audio Flamingo 3 evaluation pipeline (adaptive_perturbation/run_af3.py).

Disabled by default and a true no-op on that path — `Profiler.section()`
returns before touching `time`/`torch.cuda` at all when `enabled=False` — so
it cannot affect timing or behavior unless a run explicitly passes
`--profile`. When enabled it only wraps existing code blocks with
`with PROFILER.section("name", **tags):`; it never reorders or changes what
those blocks compute. There is exactly one profiling mode: `--profile` always
records the full detailed report (per-call sample series included).

Produces three things per profiled run, all under <repo root>/profiler_reports/:
  * a JSON report with a "summary" rollup and a "detailed" section (full
    per-section stats, per-step sample series for the hot AAD loop so cost
    growth across decode steps is visible directly, slowest-N outliers, and
    the full memory-snapshot timeline)
  * a plain-text ".log" sibling laid out as human-readable tables, with plain-
    English labels for every section, a "biggest time sink" callout, and a
    growth-signal flag for sections whose cost varies a lot call-to-call —
    the "neat log" you can skim without touching JSON
  * the same text printed to stdout at the end of the run
"""

from __future__ import annotations

import heapq
import json
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

# name -> (short human label used in the table, longer description used in the legend)
SECTION_INFO: dict[str, tuple[str, str]] = {
    "processor_load": (
        "Load processor",
        "AutoProcessor.from_pretrained(...) — tokenizer/feature-extractor setup, once per run.",
    ),
    "model_load": (
        "Load model",
        "AudioFlamingo3ForConditionalGeneration.from_pretrained(..., device_map='auto') — once per run.",
    ),
    "batch_total": (
        "Process one batch (end-to-end)",
        "The full process_batch() call: audio load + encode + generate + decode for one batch.",
    ),
    "audio_load": (
        "Load+resample one audio file",
        "librosa.load() for a single file, timed individually (not just the whole batch loop).",
    ),
    "clean_template_encode": (
        "Encode clean-audio prompt",
        "processor.apply_chat_template() for the real (clean) audio + question.",
    ),
    "negative_template_encode_no_audio": (
        "Encode no-audio negative prompt",
        "processor.apply_chat_template() for the text-only negative branch (AAD, NO_AUDIO mode).",
    ),
    "negative_embed_lookup": (
        "Embed no-audio negative prompt",
        "model.get_input_embeddings() on the no-audio negative branch's token ids.",
    ),
    "perturbation_apply": (
        "Apply audio perturbation",
        "Waveform perturbation (noise/mask/reverse/etc.) applied to build the negative branch's audio.",
    ),
    "negative_template_encode_perturbed": (
        "Encode perturbed-audio prompt",
        "processor.apply_chat_template() for the perturbed-audio negative branch.",
    ),
    "negative_forward_full": (
        "Negative branch: full forward (per batch)",
        "One full model forward pass over the perturbed-audio negative prompt — once per BATCH, not per step.",
    ),
    "generate_total": (
        "generate() — full decode loop",
        "model.generate() for the clean branch. Wraps every decode step, including all AAD sections below — "
        "generate_total minus the sum of the aad_* rows below is roughly HF's own clean-branch decode cost.",
    ),
    "aad_step_embed_concat": (
        "AAD: grow negative sequence",
        "AudioLogitsProcessor: embed the newest token and append it to the negative sequence — once per decode step.",
    ),
    "aad_negative_forward": (
        "AAD: negative-branch forward (per step)",
        "AudioLogitsProcessor: full model forward pass on the negative branch — once per decode step, "
        "WITH NO KV CACHE (reprocesses the whole growing sequence every time). Prime suspect for slow runs.",
    ),
    "aad_step_diagnostics": (
        "AAD: per-step logging/metrics",
        "AudioLogitsProcessor: builds the logit_trace/step0_metrics (many .item()/tokenizer.decode() calls, "
        "each one a GPU sync) — once per decode step.",
    ),
    "decode_extract": (
        "Decode + extract answer",
        "processor.batch_decode() plus building the SampleResult objects for a batch.",
    ),
}


def _label(name: str) -> str:
    return SECTION_INFO.get(name, (name, ""))[0]


def _description(name: str) -> str:
    return SECTION_INFO.get(name, (name, "(no description on file for this section)"))[1]


def _now_tag() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _human_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


class Profiler:
    """Section-based wall-clock + memory profiler. A true no-op unless enabled=True.

    There is a single profiling mode: whenever enabled, the full detailed report
    (including per-call sample series) is recorded — no separate "detailed" flag.
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

        self._stats: dict[str, dict] = defaultdict(
            lambda: {"count": 0, "total_s": 0.0, "min_s": float("inf"), "max_s": 0.0}
        )
        self._samples: dict[str, list] = defaultdict(list)
        self._slowest: dict[str, list] = defaultdict(list)  # name -> heap of (duration_s, key)
        self.memory_snapshots: list[dict] = []
        self.meta: dict = {}
        self._start_time = time.perf_counter()

    # --- core timing ---------------------------------------------------------

    def _sync(self):
        # The model is sharded across every visible GPU via device_map="auto",
        # so a forward pass can still be running on cuda:3 after cuda:0 is idle.
        # Syncing only the default device would under-count wall time.
        for i in range(self._device_count):
            torch.cuda.synchronize(i)

    @contextmanager
    def section(self, name: str, track_key: Optional[str] = None, track_n: int = 5, **tags):
        if not self.enabled:
            # True no-op: no time.perf_counter(), no cuda sync, nothing allocated.
            yield
            return
        self._sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            dt = time.perf_counter() - t0
            s = self._stats[name]
            s["count"] += 1
            s["total_s"] += dt
            s["min_s"] = min(s["min_s"], dt)
            s["max_s"] = max(s["max_s"], dt)
            self._samples[name].append({"duration_s": round(dt, 6), **tags})
            if track_key is not None:
                self.track_slowest(name, key=track_key, duration_s=dt, n=track_n)

    def track_slowest(self, name: str, key: str, duration_s: float, n: int = 5):
        """Keep the N slowest individual calls for a section without storing every one."""
        if not self.enabled:
            return
        heap = self._slowest[name]
        entry = (duration_s, key)
        if len(heap) < n:
            heapq.heappush(heap, entry)
        elif duration_s > heap[0][0]:
            heapq.heapreplace(heap, entry)

    # --- memory ----------------------------------------------------------------

    def snapshot_memory(self, label: str):
        if not self.enabled:
            return
        snap = {"label": label, "wall_s": round(time.perf_counter() - self._start_time, 3)}
        if self._device_count:
            snap["gpu"] = {}
            for i in range(self._device_count):
                snap["gpu"][f"cuda:{i}"] = {
                    "allocated_gb": round(torch.cuda.memory_allocated(i) / 1e9, 3),
                    "reserved_gb": round(torch.cuda.memory_reserved(i) / 1e9, 3),
                    "max_allocated_gb": round(torch.cuda.max_memory_allocated(i) / 1e9, 3),
                }
        try:
            import psutil
            snap["cpu_rss_gb"] = round(psutil.Process().memory_info().rss / 1e9, 3)
        except ImportError:
            pass  # psutil isn't a declared project dependency — degrade silently.
        self.memory_snapshots.append(snap)

    # --- metadata ----------------------------------------------------------------

    def set_meta(self, **kwargs):
        self.meta.update(kwargs)

    # --- reporting ----------------------------------------------------------------

    def summary_dict(self) -> dict:
        total_wall_s = round(time.perf_counter() - self._start_time, 3)
        sections_sorted = sorted(self._stats.items(), key=lambda kv: -kv[1]["total_s"])

        top_sections = []
        for name, s in sections_sorted[:10]:
            pct = (s["total_s"] / total_wall_s * 100) if total_wall_s else 0.0
            top_sections.append({
                "name": name,
                "label": _label(name),
                "calls": s["count"],
                "total_s": round(s["total_s"], 3),
                "avg_s": round(s["total_s"] / s["count"], 6) if s["count"] else 0.0,
                "pct_of_run": round(pct, 2),
            })

        peak_gpu = {}
        if self.memory_snapshots:
            last_gpu = self.memory_snapshots[-1].get("gpu", {})
            for dev, vals in last_gpu.items():
                peak_gpu[dev] = vals.get("max_allocated_gb")

        slowest_audio = sorted(self._slowest.get("audio_load", []), reverse=True)
        slowest_audio = [{"file": k, "duration_s": round(d, 3)} for d, k in slowest_audio]

        summary = {
            "total_profiled_wall_s": total_wall_s,
            "model_load_s": round(self._stats.get("model_load", {}).get("total_s", 0.0), 3),
            "top_sections_by_total_time": top_sections,
            "peak_gpu_memory_gb": peak_gpu,
            "slowest_audio_loads": slowest_audio,
        }

        num_batches_total = self.meta.get("num_batches_total")
        batch_total = self._stats.get("batch_total")
        if num_batches_total and batch_total and batch_total["count"]:
            avg_batch_s = batch_total["total_s"] / batch_total["count"]
            est_s = summary["model_load_s"] + avg_batch_s * num_batches_total
            summary["estimated_full_run_s"] = round(est_s, 1)
            summary["estimated_full_run_human"] = _human_duration(est_s)

        return summary

    def detailed_dict(self) -> dict:
        sections = {}
        for name, s in self._stats.items():
            sections[name] = {
                "label": _label(name),
                "description": _description(name),
                "count": s["count"],
                "total_s": round(s["total_s"], 4),
                "avg_s": round(s["total_s"] / s["count"], 6) if s["count"] else 0.0,
                "min_s": round(s["min_s"], 6) if s["count"] else None,
                "max_s": round(s["max_s"], 6),
            }
        slowest = {
            name: [{"key": k, "duration_s": round(d, 4)} for d, k in sorted(heap, reverse=True)]
            for name, heap in self._slowest.items()
        }
        return {
            "sections": sections,
            # Per-call sample series — this is what lets you plot duration vs. decode-step-index
            # and see linear vs. quadratic growth directly instead of inferring it from an average.
            "per_step_samples": dict(self._samples),
            "slowest_calls": slowest,
            "memory_snapshots": self.memory_snapshots,
        }

    def to_dict(self) -> dict:
        return {
            "meta": self.meta,
            "summary": self.summary_dict(),
            "detailed": self.detailed_dict(),
        }

    def _build_report_text(self) -> str:
        summary = self.summary_dict()
        lines: list[str] = []
        W = 90
        lines.append("=" * W)
        lines.append("TIME USAGE PROFILE — Audio Flamingo 3 (run_af3.py)")
        lines.append("=" * W)

        # --- run info -----------------------------------------------------------
        run_bits = []
        for k in ("mode", "perturbation_type", "perturbation_setting", "alpha", "batch_size", "max_new_tokens"):
            if self.meta.get(k) is not None:
                run_bits.append(f"{k}={self.meta[k]}")
        if run_bits:
            lines.append("Run:      " + "  ".join(run_bits))
        if self.meta.get("model_id"):
            lines.append(f"Model:    {self.meta['model_id']}")
        if self.meta.get("gpu_names"):
            gpu_names = self.meta["gpu_names"]
            lines.append(f"GPUs:     {len(gpu_names)}x {gpu_names[0]}" if gpu_names else "GPUs:     (none detected)")
        if self.meta.get("device_map"):
            dm = self.meta["device_map"]
            devices = sorted(set(dm.values())) if isinstance(dm, dict) else None
            if devices:
                lines.append(f"Model is spread across devices: {devices}"
                              + ("  <- device_map='auto' is fragmenting it across multiple GPUs" if len(devices) > 1 else ""))
        if "num_batches_profiled" in self.meta:
            total_b = self.meta.get("num_batches_total", "?")
            trunc = "  (diagnostic run — nothing saved to results/checkpoints)" if self.meta.get("truncated") else ""
            lines.append(f"Batches:  profiled {self.meta['num_batches_profiled']} of {total_b}{trunc}")
        lines.append("")

        # --- biggest time sink callout, in plain English -------------------------
        top = summary["top_sections_by_total_time"]
        if top:
            biggest = top[0]
            lines.append("BIGGEST TIME SINK")
            lines.append(
                f"  {biggest['label']} — {biggest['pct_of_run']:.1f}% of profiled time "
                f"({biggest['total_s']:.2f}s across {biggest['calls']} call(s), "
                f"avg {biggest['avg_s'] * 1000:.1f}ms/call)"
            )
            lines.append(f"  {_description(biggest['name'])}")
            lines.append("")

        # --- wall time summary ----------------------------------------------------
        lines.append(f"Total profiled wall time : {summary['total_profiled_wall_s']:.2f}s")
        lines.append(f"Model load time          : {summary['model_load_s']:.2f}s  (one-time cost, not per-batch)")
        if "estimated_full_run_human" in summary:
            lines.append(
                f"Estimated FULL run time  : {summary['estimated_full_run_human']} "
                f"({summary['estimated_full_run_s']:.0f}s) — extrapolated from the batches profiled here"
            )
        lines.append("")

        # --- main ranked table, human labels, ms avg ------------------------------
        all_sections = sorted(self._stats.items(), key=lambda kv: -kv[1]["total_s"])
        total_wall = summary["total_profiled_wall_s"] or 1e-9
        lines.append("WHERE THE TIME GOES (most expensive first)")
        header = f"{'What':<40}{'Calls':>7}{'Total(s)':>10}{'Avg(ms)':>10}{'% of run':>10}"
        lines.append(header)
        lines.append("-" * len(header))
        growth_flags = []
        for name, s in all_sections:
            avg_ms = (s["total_s"] / s["count"] * 1000) if s["count"] else 0.0
            pct = s["total_s"] / total_wall * 100
            lines.append(f"{_label(name):<40}{s['count']:>7}{s['total_s']:>10.3f}{avg_ms:>10.2f}{pct:>9.1f}%")
            # High max/min spread with enough calls is the signature of a per-step cost that
            # grows over the run (e.g. an uncached forward pass over a growing sequence).
            if s["count"] >= 4 and s["min_s"] > 0 and s["max_s"] / s["min_s"] >= 3:
                growth_flags.append((name, s))
        lines.append("")

        # --- legend: what each row actually measures -------------------------------
        lines.append("WHAT EACH ROW MEASURES")
        for name, _s in all_sections:
            lines.append(f"  {_label(name)}: {_description(name)}")
        lines.append("")

        # --- growth signal: cost that isn't flat across calls ------------------------
        if growth_flags:
            lines.append("GROWING COST WARNING (fastest call vs. slowest call differs by 3x+)")
            lines.append("  This is the signature of per-call cost that scales with something that grows")
            lines.append("  over the run (e.g. sequence length) rather than being constant per call:")
            for name, s in growth_flags:
                lines.append(
                    f"  {_label(name)}: {s['min_s'] * 1000:.2f}ms (fastest) -> {s['max_s'] * 1000:.2f}ms (slowest)"
                )
            lines.append("  See detailed.per_step_samples in the JSON report for the full per-call series.")
            lines.append("")

        # --- memory -----------------------------------------------------------------
        if summary["peak_gpu_memory_gb"]:
            lines.append("Peak GPU memory (max_allocated) — confirms/refutes device_map fragmentation:")
            for dev, gb in summary["peak_gpu_memory_gb"].items():
                lines.append(f"  {dev}: {gb:.2f} GB")
            lines.append("")

        # --- slowest individual calls -------------------------------------------------
        for name, heap in self._slowest.items():
            if not heap:
                continue
            lines.append(f"Slowest '{_label(name)}' calls:")
            for d, k in sorted(heap, reverse=True):
                lines.append(f"  {d:.3f}s  {k}")
            lines.append("")

        lines.append("=" * W)
        return "\n".join(lines)

    def print_summary(self):
        if not self.enabled:
            return
        print("\n" + self._build_report_text())

    def write_report(self, path: "Path | str"):
        """Writes the JSON report plus a plain-text '.log' sibling, and prints the summary."""
        if not self.enabled:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

        log_path = path.with_suffix(".log")
        text = self._build_report_text()
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")

        print("\n" + text)
        print(f"\nDetailed JSON report: {path}")
        print(f"Neat text log:        {log_path}")


def get_profile_report_path(
    mode: str,
    *,
    tag: Optional[str] = None,
    perturbation_type: Optional[str] = None,
    alpha: Optional[float] = None,
    perturbation_setting: Optional[str] = None,
    reports_dir: Optional["Path | str"] = None,
) -> Path:
    """Report path under <repo root>/profiler_reports/ — repo root is resolved relative to
    this file (ah_existence/helpers/profiling.py, two levels up), so this works from any
    script in the repo, not just ones that import helpers.config.

    Either pass `tag` directly (any short descriptive string, e.g. "mdpo_beta_0.1"), or
    pass `perturbation_type`/`alpha`/`perturbation_setting` to build the tag the way
    run_af3.py's eval/spot-check/audit runs do.
    """
    directory = Path(reports_dir) if reports_dir else (Path(__file__).resolve().parent.parent.parent / "profiler_reports")
    if tag is None:
        name_part = (perturbation_type or mode).lower()
        if perturbation_setting:
            name_part = f"{name_part}_{perturbation_setting.lower()}"
        tag = f"{name_part}_alpha_{alpha}" if alpha is not None else name_part
    base_name = f"profile_{mode}_{tag}_{_now_tag()}"
    return directory / f"{base_name}.json"
