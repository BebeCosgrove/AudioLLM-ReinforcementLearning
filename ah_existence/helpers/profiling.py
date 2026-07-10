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
those blocks compute.

Produces three things per profiled run, all under reports/time_usage/:
  * a JSON report with a "summary" rollup and a "detailed" section (full
    per-section stats, per-step sample series for the hot AAD loop so cost
    growth across decode steps is visible directly, slowest-N outliers, and
    the full memory-snapshot timeline)
  * a plain-text ".log" sibling with the same information laid out as
    readable tables — the "neat log" you can skim without touching JSON
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
    """Section-based wall-clock + memory profiler. A true no-op unless enabled=True."""

    def __init__(self, enabled: bool = False, detailed: bool = False):
        self.enabled = enabled
        # per-step sample recording only ever makes sense (and only ever costs anything)
        # when the profiler itself is enabled.
        self.detailed = bool(detailed) and enabled
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
            if self.detailed:
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
            # Per-call sample series (present only with --profile-detailed) — this is what
            # lets you plot duration vs. decode-step-index and see linear vs. quadratic growth
            # directly instead of inferring it from an average.
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
        lines.append("=" * 78)
        lines.append("TIME USAGE PROFILE — Audio Flamingo 3 (run_af3.py)")
        lines.append("=" * 78)

        if self.meta:
            for k, v in self.meta.items():
                if isinstance(v, (list, dict)):
                    continue
                lines.append(f"  {k}: {v}")
        if self.meta.get("gpu_names"):
            lines.append(f"  gpu_names: {', '.join(self.meta['gpu_names'])}")
        if self.meta.get("device_map"):
            devices = sorted(set(self.meta["device_map"].values())) if isinstance(self.meta["device_map"], dict) else None
            if devices:
                lines.append(f"  device_map spans devices: {devices}")

        lines.append("")
        lines.append(f"Total profiled wall time : {summary['total_profiled_wall_s']:.2f}s")
        lines.append(f"Model load time          : {summary['model_load_s']:.2f}s")
        if "estimated_full_run_human" in summary:
            lines.append(
                f"Estimated full-run time  : {summary['estimated_full_run_human']} "
                f"({summary['estimated_full_run_s']:.0f}s) "
                f"[extrapolated from {self.meta.get('num_batches_profiled', '?')} of "
                f"{self.meta.get('num_batches_total', '?')} batches]"
            )
        lines.append("")

        # Full per-function/section table, every section (not just the top 10 in "summary").
        all_sections = sorted(self._stats.items(), key=lambda kv: -kv[1]["total_s"])
        total_wall = summary["total_profiled_wall_s"] or 1e-9
        header = f"{'Section (what/where)':<34}{'Calls':>8}{'Total(s)':>12}{'Avg(s)':>12}{'Min(s)':>10}{'Max(s)':>10}{'% run':>8}"
        lines.append(header)
        lines.append("-" * len(header))
        for name, s in all_sections:
            avg = s["total_s"] / s["count"] if s["count"] else 0.0
            pct = s["total_s"] / total_wall * 100
            lines.append(
                f"{name:<34}{s['count']:>8}{s['total_s']:>12.3f}{avg:>12.6f}"
                f"{s['min_s']:>10.4f}{s['max_s']:>10.4f}{pct:>7.1f}%"
            )

        if summary["peak_gpu_memory_gb"]:
            lines.append("")
            lines.append("Peak GPU memory (max_allocated), confirms/refutes device_map fragmentation:")
            for dev, gb in summary["peak_gpu_memory_gb"].items():
                lines.append(f"  {dev}: {gb:.2f} GB")

        for name, heap in self._slowest.items():
            if not heap:
                continue
            lines.append("")
            lines.append(f"Slowest '{name}' calls:")
            for d, k in sorted(heap, reverse=True):
                lines.append(f"  {d:.3f}s  {k}")

        lines.append("")
        lines.append("=" * 78)
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
    perturbation_type: str,
    alpha: float,
    perturbation_setting: Optional[str] = None,
    reports_dir: Optional["Path | str"] = None,
) -> Path:
    """Mirrors helpers.run_helpers.get_output_filename's naming shape, rooted at reports/time_usage/."""
    from helpers.config import Config

    directory = Path(reports_dir) if reports_dir else (Config.project_root / "reports" / "time_usage")
    name_part = perturbation_type.lower()
    if perturbation_setting:
        name_part = f"{name_part}_{perturbation_setting.lower()}"
    base_name = f"profile_{mode}_{name_part}_alpha_{alpha}_{_now_tag()}"
    return directory / f"{base_name}.json"
