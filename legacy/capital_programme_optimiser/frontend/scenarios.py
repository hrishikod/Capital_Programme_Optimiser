"""Scenario management utilities for the Streamlit front-end."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from capital_programme_optimiser.config import Settings
from capital_programme_optimiser.optimisation import solver_core

ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"
MANIFEST_NAME = "manifest.json"


@dataclass
class ScenarioFolder:
    name: str
    path: Path
    kind: str  # "preset" or "saved"
    is_default: bool = False
    metadata: Optional[Dict[str, object]] = None


@dataclass
class ForcedStartInput:
    include: Optional[bool] = None
    start: Optional[int] = None

    def to_solver_spec(self) -> Dict[str, Optional[int]]:
        spec: Dict[str, Optional[int]] = {}
        if self.include is not None:
            spec["include"] = self.include
        if self.start is not None:
            spec["start"] = int(self.start)
        return spec


@dataclass
class OptimiserRunConfig:
    cost_types: List[str]
    scenario_keys: List[str]
    objective_dims: List[str]
    surplus_options_m: Dict[str, float]
    plusminus_levels_m: List[float]
    start_fy: int
    years: int
    run_plusminus: bool = True
    forced_start: Dict[str, ForcedStartInput] = field(default_factory=dict)
    time_limit: Optional[int] = None
    run_prefix: Optional[str] = None

    def serializable(self) -> Dict[str, object]:
        return {
            "cost_types": list(self.cost_types),
            "scenario_keys": list(self.scenario_keys),
            "objective_dims": list(self.objective_dims),
            "surplus_options_m": {str(k): float(v) for k, v in self.surplus_options_m.items()},
            "plusminus_levels_m": [float(v) for v in self.plusminus_levels_m],
            "start_fy": int(self.start_fy),
            "years": int(self.years),
            "run_plusminus": bool(self.run_plusminus),
            "forced_start": {
                name: fs.to_solver_spec() for name, fs in self.forced_start.items() if fs.to_solver_spec()
            },
            "time_limit": self.time_limit,
            "run_prefix": self.run_prefix,
        }


@dataclass
class ProgressSnapshot:
    stage: str
    completed: int
    total: int
    percent: float
    eta_seconds: Optional[float]
    payload: Dict[str, object]


@dataclass
class RunSummary:
    folder: ScenarioFolder
    started_at: datetime
    finished_at: datetime
    elapsed_seconds: float
    output_files: List[Path]

    def serializable(self) -> Dict[str, object]:
        return {
            "folder": {
                "name": self.folder.name,
                "kind": self.folder.kind,
            },
            "started_at": self.started_at.strftime(ISO_FMT),
            "finished_at": self.finished_at.strftime(ISO_FMT),
            "elapsed_seconds": self.elapsed_seconds,
            "output_files": [f.name for f in self.output_files],
        }


class ProgressTracker:
    """Translate solver progress callbacks into coarse progress snapshots."""

    def __init__(self, callback: Optional[Callable[[ProgressSnapshot], None]] = None) -> None:
        self.callback = callback
        self.total_steps = 0
        self.completed_steps = 0
        self._start_time = time.time()
        self._current_step_started: Optional[float] = None
        self._family_step_weight: Dict[Tuple[str, str], int] = {}

    def listener(self, stage: str, payload: Dict[str, object]) -> None:
        if stage == "family_start":
            key = (str(payload.get("cost_type")), str(payload.get("scenario_key")))
            dims = payload.get("dimensions") or []
            surplus = payload.get("surplus_keys") or []
            plus_levels = payload.get("plus_levels") or []
            combos_per_dim = max(1, len(list(surplus))) * max(1, len(list(plus_levels)))
            family_total = len(list(dims)) * combos_per_dim
            self._family_step_weight[key] = combos_per_dim
            self.total_steps += family_total
            self._emit(stage, payload)
            return

        if stage == "dimension_skipped":
            key = (str(payload.get("cost_type")), str(payload.get("scenario_key")))
            combos = self._family_step_weight.get(key, 0)
            if combos:
                self.total_steps = max(0, self.total_steps - combos)
            self._emit(stage, payload)
            return

        if stage == "solve_start":
            self._current_step_started = time.time()
            self._emit(stage, payload)
            return

        if stage == "solve_complete":
            self.completed_steps += 1
            self._emit(stage, payload)
            return

        if stage == "family_complete":
            self._emit(stage, payload)
            return

        self._emit(stage, payload)

    def finalize(self, status: str = "completed") -> None:
        self._emit(status, {})

    def _emit(self, stage: str, payload: Dict[str, object]) -> None:
        if self.callback is None:
            return
        total = max(self.total_steps, 1)
        completed = min(self.completed_steps, total)
        percent = completed / total if total else 0.0
        remaining = max(total - completed, 0)
        eta: Optional[float] = None
        if completed > 0 and remaining > 0:
            elapsed = max(time.time() - self._start_time, 0.0)
            avg = elapsed / completed if completed else 0.0
            if avg > 0:
                eta = avg * remaining
        snapshot = ProgressSnapshot(
            stage=stage,
            completed=completed,
            total=total,
            percent=min(percent, 1.0),
            eta_seconds=eta,
            payload=payload,
        )
        self.callback(snapshot)


def ensure_scenario_roots(settings: Settings) -> Tuple[Path, Path]:
    """Resolve preset/saved directories with backwards compatibility."""

    preset_dir_callable = getattr(settings, "preset_dir", None)
    saved_dir_callable = getattr(settings, "saved_dir", None)

    if callable(preset_dir_callable) and callable(saved_dir_callable):
        preset_dir = Path(preset_dir_callable())
        saved_dir = Path(saved_dir_callable())
    else:  # fall back to derived paths for older Settings objects
        paths = getattr(settings, "paths", None)
        root = getattr(settings, "root", Path.cwd())
        preset_rel = getattr(paths, "preset_dir", Path("scenario_sets/presets"))
        saved_rel = getattr(paths, "saved_dir", Path("scenario_sets/saved_runs"))
        preset_dir = Path(root) / Path(preset_rel)
        saved_dir = Path(root) / Path(saved_rel)

    preset_dir = preset_dir.resolve()
    saved_dir = saved_dir.resolve()
    preset_dir.mkdir(parents=True, exist_ok=True)
    saved_dir.mkdir(parents=True, exist_ok=True)
    return preset_dir, saved_dir


def _slugify(name: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9]+", name)
    if not tokens:
        return "scenario"
    return "_".join(tokens)


def _manifest_path(folder: Path) -> Path:
    return folder / MANIFEST_NAME


def load_manifest(folder: Path) -> Optional[Dict[str, object]]:
    manifest_path = _manifest_path(folder)
    if not manifest_path.exists():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def write_manifest(folder: Path, data: Dict[str, object]) -> None:
    manifest_path = _manifest_path(folder)
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)


def list_scenario_folders(settings: Settings) -> List[ScenarioFolder]:
    preset_dir, saved_dir = ensure_scenario_roots(settings)
    folders: List[ScenarioFolder] = []

    paths = getattr(settings, "paths", None)
    default_name = ""
    if paths is not None:
        default_name = str(getattr(paths, "default_preset", "")).strip()

    if default_name:
        default_path = preset_dir / default_name
        default_path.mkdir(parents=True, exist_ok=True)

    for kind, root in (("preset", preset_dir), ("saved", saved_dir)):
        if not root.exists():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            metadata = load_manifest(child)
            folders.append(
                ScenarioFolder(
                    name=child.name,
                    path=child,
                    kind=kind,
                    is_default=bool(default_name and child.name == default_name),
                    metadata=metadata,
                )
            )
    return folders


def create_scenario_folder(settings: Settings, name: str, kind: str = "saved") -> ScenarioFolder:
    preset_dir, saved_dir = ensure_scenario_roots(settings)
    root = preset_dir if kind == "preset" else saved_dir
    slug = _slugify(name)
    candidate = root / slug
    index = 1
    while candidate.exists():
        index += 1
        candidate = root / f"{slug}_{index}"
    candidate.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    manifest = {
        "name": name,
        "kind": kind,
        "created": now.strftime(ISO_FMT),
        "runs": [],
    }
    write_manifest(candidate, manifest)

    return ScenarioFolder(name=candidate.name, path=candidate, kind=kind, metadata=manifest)


def _to_forced_map(forced_input: Dict[str, ForcedStartInput]) -> Dict[str, Dict[str, Optional[int]]]:
    result: Dict[str, Dict[str, Optional[int]]] = {}
    for name, spec in forced_input.items():
        solver_spec = spec.to_solver_spec()
        if solver_spec:
            result[name] = solver_spec
    return result


def run_optimiser_for_scenario(
    settings: Settings,
    folder: ScenarioFolder,
    run_config: OptimiserRunConfig,
    progress_callback: Optional[Callable[[ProgressSnapshot], None]] = None,
    *,
    clean: bool = True,
) -> RunSummary:
    if folder.kind not in {"preset", "saved"}:
        raise ValueError(f"Unsupported folder kind: {folder.kind}")

    target = folder.path
    target.mkdir(parents=True, exist_ok=True)

    tracker = ProgressTracker(progress_callback)
    solver_core.set_progress_listener(tracker.listener)

    original_cache = solver_core.CACHE
    original_surplus = dict(solver_core.SURPLUS_OPTIONS_M)
    original_plus = list(solver_core.PLUSMINUS_LEVELS_M)
    original_run_flag = solver_core.RUN_BENEFIT_PLUSMINUS
    original_forced = dict(solver_core.FORCED_START)
    original_cfg = dict(solver_core.CFG)
    original_time_limit = solver_core.SOLVE_SECONDS
    original_dims = list(solver_core.PRIMARY_OBJECTIVE_DIMS_TO_RUN)
    original_prefix = solver_core.PKL_PREFIX

    started_at = datetime.now(timezone.utc)

    scenario_label = None
    if folder.metadata is not None:
        scenario_label = folder.metadata.get("name")
    if not scenario_label:
        scenario_label = folder.name
    folder_slug = _slugify(scenario_label or "")

    custom_prefix_raw = (run_config.run_prefix or "").strip()
    if custom_prefix_raw:
        custom_slug = re.sub(r"[^A-Za-z0-9_-]+", "_", custom_prefix_raw).strip("_")
        run_prefix = f"{custom_slug}_" if custom_slug else f"{folder_slug}_"
    else:
        run_prefix = f"{folder_slug}_"

    if clean and folder.kind == "saved":
        if run_prefix:
            pattern = f"{run_prefix}*.pkl"
        else:
            pattern = "*.pkl"
        for existing in target.glob(pattern):
            try:
                existing.unlink()
            except OSError:
                pass

    solver_core.CACHE = target
    target.mkdir(parents=True, exist_ok=True)

    solver_core.PKL_PREFIX = run_prefix

    solver_core.SURPLUS_OPTIONS_M = {str(k): float(v) for k, v in run_config.surplus_options_m.items()}
    solver_core.PLUSMINUS_LEVELS_M = [float(v) for v in run_config.plusminus_levels_m]
    solver_core.RUN_BENEFIT_PLUSMINUS = bool(run_config.run_plusminus)
    solver_core.FORCED_START = _to_forced_map(run_config.forced_start)
    solver_core.CFG["YEARS"] = int(run_config.years)
    solver_core.CFG["START_FY"] = int(run_config.start_fy)
    if run_config.time_limit:
        solver_core.SOLVE_SECONDS = int(run_config.time_limit)

    dims_to_use = list(run_config.objective_dims)
    solver_core.PRIMARY_OBJECTIVE_DIMS_TO_RUN = dims_to_use

    produced_files: List[Path] = []

    error: Optional[BaseException] = None
    try:
        for cost_type in run_config.cost_types:
            sheet_map = solver_core.BENEFIT_SCENARIOS
            for scenario_key in run_config.scenario_keys:
                scenario_sheet = sheet_map.get(scenario_key)
                if scenario_sheet is None:
                    raise ValueError(f"Unknown scenario key: {scenario_key}")
                tracker.listener(
                    "run_step",
                    {
                        "cost_type": cost_type,
                        "scenario_key": scenario_key,
                        "message": f"Running {cost_type} / {scenario_key}",
                    },
                )
                solver_core.optimise_family_for(cost_type, scenario_key, scenario_sheet)
        produced_files = sorted(target.glob("*.pkl"))
    except BaseException as exc:
        error = exc
        raise
    finally:
        solver_core.set_progress_listener(None)
        solver_core.CACHE = original_cache
        solver_core.SURPLUS_OPTIONS_M = original_surplus
        solver_core.PLUSMINUS_LEVELS_M = original_plus
        solver_core.RUN_BENEFIT_PLUSMINUS = original_run_flag
        solver_core.FORCED_START = original_forced
        solver_core.CFG.update(original_cfg)
        solver_core.SOLVE_SECONDS = original_time_limit
        solver_core.PRIMARY_OBJECTIVE_DIMS_TO_RUN = original_dims
        solver_core.PKL_PREFIX = original_prefix
        if error is not None:
            tracker.finalize("run_error")

    finished_at = datetime.now(timezone.utc)
    elapsed = (finished_at - started_at).total_seconds()

    tracker.finalize("run_complete")

    summary = RunSummary(
        folder=folder,
        started_at=started_at,
        finished_at=finished_at,
        elapsed_seconds=elapsed,
        output_files=produced_files,
    )

    manifest = load_manifest(target) or {
        "name": folder.name,
        "kind": folder.kind,
        "created": started_at.strftime(ISO_FMT),
        "runs": [],
    }
    runs = manifest.setdefault("runs", [])
    runs.append(
        {
            "started_at": summary.started_at.strftime(ISO_FMT),
            "finished_at": summary.finished_at.strftime(ISO_FMT),
            "elapsed_seconds": summary.elapsed_seconds,
            "output_files": [f.name for f in summary.output_files],
            "config": run_config.serializable(),
        }
    )
    manifest["last_run"] = runs[-1]
    write_manifest(target, manifest)

    return summary
